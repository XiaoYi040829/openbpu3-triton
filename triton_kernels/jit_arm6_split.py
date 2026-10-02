#!/usr/bin/env python3
"""
jit_arm6_split.py — 6-DOF 机械臂的球体轨迹与关节控制 kernel

机械臂一步分成多个小 kernel，由 host 依次调用：
    K1 sphere_step : 目标球体圆周浮动（整数相位 + 正弦查表）
    K2 逆运动学    : 见 jit_ik_micro.py（ik_geom + atan2）
    K3 joint_pd    : 关节 PD 控制 + 角度积分

拆成小 kernel 是为了让每个 kernel 同时活跃的值都放得进寄存器池，
且代码长度远低于参数区之前的上限（7680 字节）。

球体轨迹用整数相位查 sin 表：相位是确定性递推，用整数表示可以在
host 与 kernel 之间逐位一致。
"""
import math
import os
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
for p in [x for x in (os.environ.get("FLAGTREE_PY"), WS,
                      os.path.join(WS, "triton_kernels")) if x]:
    if p not in sys.path:
        sys.path.insert(0, p)

import triton                    # noqa: E402
import triton.language as tl     # noqa: E402

DT = tl.constexpr(0.004166666666)   # 控制周期 1/240 s
KP = tl.constexpr(120.0)            # PD 比例增益
KD = tl.constexpr(12.0)             # PD 微分增益

# 相位用整数定标表示（单位 = 1/TAB_BIG 个周期），索引 = 相位 % TAB_BIG，
# 最近邻查表、不插值。
# 表长 512（2KB）：参数页为 0x1000=4096 字节，1024 项 f32 会占满整页并越界
# 覆盖下一页。512 项最近邻误差 < 6.2e-3，机械臂控制回路可接受。
TAB_BIG = tl.constexpr(512)


@triton.jit
def _sin_tab_i(idx, tab_ptr):
    """整数索引查 sin 表（idx 单位 = 1/TAB_BIG 周期，最近邻）。"""
    k = idx % TAB_BIG
    return tl.load(tab_ptr + k)


@triton.jit
def _cos_tab_i(idx, tab_ptr):
    """cos = sin(idx + 1/4 周期)，1/4 周期 = TAB_BIG/4 = 128。"""
    return _sin_tab_i(idx + 128, tab_ptr)


# ============================ K1: 球体物理 ============================
@triton.jit
def sphere_step_kernel(state_ptr, iphase_ptr, tab_ptr, out_ptr):
    """目标球体沿圆周浮动一步。

    写法上的两点：
      1. 本核没有 lane 间通路，跨 lane 归约（tl.sum 等）会被编译器拒绝，
         所以所有标量都由 host 预先广播成 8 lane 相同的值。
      2. 相位用整数定标（单位 = 1/TAB_BIG 周期），host 按 omega*dt 换算好增量。

    state[0..7]  = radius，8 lane 均相同（host 广播）
    state[8..15] = z，     8 lane 均相同
    iphase[0..7] = 当前整数相位，8 lane 均相同
    iphase[8..15]= 每步整数增量，8 lane 均相同
    tab[0..511]  = sin 表（见 sin_table()）
    out[0..7]  = x' ; out[8..15] = y' ; out[16..23] = z（原样透传）
    """
    i = tl.arange(0, 8)
    rad = tl.load(state_ptr + i)          # 已广播
    z = tl.load(state_ptr + 8 + i)
    ph_i = tl.load(iphase_ptr + i)        # 已广播
    dph_i = tl.load(iphase_ptr + 8 + i)

    np_i = (ph_i + dph_i) % TAB_BIG
    nx = rad * _cos_tab_i(np_i, tab_ptr)
    ny = rad * _sin_tab_i(np_i, tab_ptr)

    tl.store(out_ptr + i, nx)
    tl.store(out_ptr + 8 + i, ny)
    tl.store(out_ptr + 16 + i, z)
    # 不回写新相位：相位是确定性递推，由 host 侧维护。


# ============ K2 逆运动学：见 jit_ik_micro.py ============
# 把 ik_geom 与四次 atan2 合在一个 kernel 里时，同时活跃的浮点值超过
# 浮点寄存器池（f8-f27，20 个），编译器报寄存器池耗尽；因此拆成
# ik_geom_kernel 与可复用的 atan2_kernel 两个 kernel。

# ============================ K3: 关节 PD 控制 ============================
@triton.jit
def joint_pd_kernel(cur_ptr, tgt_ptr, out_ptr):
    """关节 PD 控制 + 角度积分。

    数据布局为"每 lane 一个关节、同名量同偏移"，kernel 内只用固定偏移
    取数，不做跨 lane 的索引运算。

    cur[0..7] = q   (每 lane 一个关节的当前角)
    cur 第二页(+8) = dq  —— 由 host 放在同一数组的后 8 个元素
    tgt[0..7] = 目标角
    out[0..7] = q' ; out[8..15] = dq'
    """
    i = tl.arange(0, 8)
    q = tl.load(cur_ptr + i)
    dq = tl.load(cur_ptr + 8 + i)     # 固定偏移，不做取模
    tq = tl.load(tgt_ptr + i)

    acc = KP * (tq - q) - KD * dq
    ndq = dq + acc * DT
    nq = q + ndq * DT

    tl.store(out_ptr + i, nq)
    tl.store(out_ptr + 8 + i, ndq)


SPHERE_SIG = {"state_ptr": "*fp32", "iphase_ptr": "*i32",
              "tab_ptr": "*fp32", "out_ptr": "*fp32"}
PD_SIG = {"cur_ptr": "*fp32", "tgt_ptr": "*fp32", "out_ptr": "*fp32"}


def sin_table():
    """sphere_step_kernel 的 tab 参数：TAB_BIG 项 sin，覆盖 [0, 2pi)。"""
    n = int(TAB_BIG.value) if hasattr(TAB_BIG, "value") else 512
    return [math.sin(2 * math.pi * k / n) for k in range(n)]


def main():
    from jit_openbpu3 import jit_compile
    print("=" * 72)
    print("机械臂 kernel：K1 sphere_step 与 K3 joint_pd（K2 见 jit_ik_micro.py）")
    print("=" * 72)
    total = 0
    ok = True
    for name, fn, sig in (("K1 sphere_step", sphere_step_kernel, SPHERE_SIG),
                          ("K3 joint_pd", joint_pd_kernel, PD_SIG)):
        try:
            r, _ = jit_compile(fn, sig)
            total += r["n_inst"]
            print(f"\n  {name}: {r['n_inst']:4d} 条  "
                  f"映射 {r['num_mapped']}/跳过 {r['num_skipped']}  "
                  f".S一致={r['asm_matches_bin']}")
            print(f"      CTA 特征向量 {r['cta_snnspecvec']}")
        except Exception as e:
            import traceback
            print(f"\n  {name}: 编译失败 {type(e).__name__}: {e}")
            traceback.print_exc()
            ok = False
    print(f"\n  两个 kernel 合计 {total} 条")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
