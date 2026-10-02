#!/usr/bin/env python3
"""
jit_ik_micro.py — 逆运动学的微 kernel 分解（每个都在寄存器池容量内）

为什么拆：把 ik_geom 与四次 atan2 合在一个 kernel 里（532 条）时，同时活跃
的浮点值达 25 个，超过浮点寄存器池（f8-f27，20 个），编译器报寄存器池耗尽。

拆法（host 侧串联，各 kernel 压力都远低于池子）：
    M1 ik_geom  : 由目标点算 rr, ss, s3, c3, L3*s3, A2+L3*c3, reach（不含 atan2）
    M2 atan2    : 单次 atan2 —— **同一个编译产物复用 4 次**
                  q1 = atan2(py, px)
                  q3 = atan2(s3, c3)
                  t1 = atan2(rr, ss)
                  t2 = atan2(L3*s3, A2 + L3*c3)   -> q2 = t1 - t2

这也贴近真实 GPU 的做法：小 kernel 反复调用，而不是把一切塞进一个巨核。
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

D1 = tl.constexpr(0.20)
A2 = tl.constexpr(0.40)
L3 = tl.constexpr(0.42)          # a3 + d6，工具与前臂共线
HALF_PI = tl.constexpr(1.5707963267948966)
PI = tl.constexpr(3.141592653589793)


@triton.jit
def atan2_kernel(y_ptr, x_ptr, out_ptr):
    """单次 atan2，值域 (-pi, pi]，全程无分支（tl.where 落到掩码算术）。

    用多项式（只用 fmul/fadd），host 侧参考精度 1.66e-06。
    """
    i = tl.arange(0, 8)
    y = tl.load(y_ptr + i)
    x = tl.load(x_ptr + i)

    ax = tl.abs(x)
    ay = tl.abs(y)
    swap = ay > ax
    num = tl.where(swap, x, y)
    den = tl.where(swap, y, x)
    den_s = tl.where(den == 0.0, 1.0, den)
    t = num / den_s

    z = t * t
    p = -0.01172120
    p = 0.05265332 + z * p
    p = -0.11643287 + z * p
    p = 0.19354346 + z * p
    p = -0.33262347 + z * p
    p = 0.99997726 + z * p
    a = t * p

    a = tl.where(swap, tl.where(t >= 0.0, HALF_PI, -HALF_PI) - a, a)
    a = tl.where(x < 0.0, a + tl.where(y >= 0.0, PI, -PI), a)
    a = tl.where((x == 0.0) & (y == 0.0), 0.0, a)
    tl.store(out_ptr + i, a)


@triton.jit
def ik_geom_kernel(pos_ptr, out_ptr):
    """IK 的几何部分：由 TCP 目标点算出 atan2 所需的各分量。

    pos[0..7]=px, [8..15]=py, [16..23]=pz  （host 各自广播 8 lane）
    out[0..7]=rr  [8..15]=ss  [16..23]=s3  [24..31]=c3
       [32..39]=L3*s3  [40..47]=A2+L3*c3  [48..55]=reach

    关键简化：q3 = atan2(s3, c3) 中 c3 = ddc、s3 = -sqrt(1-ddc²)，
    而后续只需要 sin(q3)/cos(q3) —— 正是 s3/c3 本身（s3²+c3²=1）。
    因此整个 IK 不需要求 sin/cos。
    """
    i = tl.arange(0, 8)
    px = tl.load(pos_ptr + i)
    py = tl.load(pos_ptr + 8 + i)
    pz = tl.load(pos_ptr + 16 + i)

    rr = tl.sqrt(px * px + py * py)
    ss = pz - D1
    lsq = rr * rr + ss * ss
    dd = (lsq - A2 * A2 - L3 * L3) / (2.0 * A2 * L3)
    reach = tl.abs(dd) <= 1.0
    ddc = tl.minimum(tl.maximum(dd, -1.0), 1.0)
    s3 = -tl.sqrt(tl.maximum(0.0, 1.0 - ddc * ddc))
    c3 = ddc

    tl.store(out_ptr + i, rr)
    tl.store(out_ptr + 8 + i, ss)
    tl.store(out_ptr + 16 + i, s3)
    tl.store(out_ptr + 24 + i, c3)
    tl.store(out_ptr + 32 + i, L3 * s3)
    tl.store(out_ptr + 40 + i, A2 + L3 * c3)
    tl.store(out_ptr + 48 + i, tl.where(reach, 1.0, 0.0))


ATAN2_SIG = {"y_ptr": "*fp32", "x_ptr": "*fp32", "out_ptr": "*fp32"}
GEOM_SIG = {"pos_ptr": "*fp32", "out_ptr": "*fp32"}


def main():
    from jit_openbpu3 import jit_compile
    print("=" * 72)
    print("IK 微 kernel 分解：每个都在寄存器池容量内")
    print("=" * 72)
    ok = True
    for nm, fn, sig in (("M1 ik_geom", ik_geom_kernel, GEOM_SIG),
                        ("M2 atan2", atan2_kernel, ATAN2_SIG)):
        r, _ = jit_compile(fn, sig)
        exh = r["reg_exhausted"]
        if exh:
            ok = False
        print(f"\n  {nm}: {r['n_inst']:4d} 条  跳过 {r['num_skipped']}  "
              f".S一致 {r['asm_matches_bin']}")
        print(f"      寄存器压力 int={r['reg_pressure_int']} "
              f"flt={r['reg_pressure_flt']}  "
              f"耗尽={exh} {'<<< 不可信' if exh else 'OK'}")
        print(f"      CTA 特征向量 {r['cta_snnspecvec']}")
    print("\n" + "=" * 72)
    print("PASS: 两个微 kernel 均未耗尽寄存器池" if ok
          else "FAIL: 仍有 kernel 耗尽")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
