#!/usr/bin/env python3
"""
run_pybullet_openbpu3.py — 顶层启动脚本

PyBullet 调用 OpenBPU-3 做物理求解，完成仿真。

完整链路，全程无手写汇编：
    pybullet (x86 host: 建场景/管理刚体/世界状态)
        ↓  读刚体状态
    BackendOpenBPU3
        ↓  Triton IR (bullet_step.ttir)
        ↓  flagtree 官方 TTIR pass
        ↓  polaris C++ pass (tensorcore_mapping + riscv_codegen)
        ↓  真 RV64 + RV-SIMT 机器码
    OpenBPU-3 周期精确模拟器 (8 刚体 = 8 条 SIMT lane 并行)
        ↓  UART (MMIO 非缓存区) 回传
    写回 pybullet 世界 -> 下一步

与"旁路对比"的区别: PyBullet 自身重力被关闭(setGravity 0)，
世界状态由 OpenBPU-3 的求解结果驱动演化。
"""
import os
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
for p in [x for x in (os.environ.get("FLAGTREE_PY"), WS,
                      os.path.join(WS, "triton_kernels")) if x]:
    if p not in sys.path:
        sys.path.insert(0, p)

from bullet_openbpu3_backend import (  # noqa: E402
    BackendOpenBPU3, PyBulletOpenBPU3World, ref_step_f32,
    LANES, DT, G, RADIUS, REST,
)


def main():
    STEPS = int(os.environ.get("STEPS", "3"))
    cycles = int(os.environ.get("EMU_CYCLES", "70000"))

    print("=" * 72)
    print("PyBullet 调用 OpenBPU-3 做物理求解 (Triton + flagos 全链路)")
    print("=" * 72)

    be = BackendOpenBPU3(cycles=cycles)
    print(f"\n[1] flagos 编译")
    print(f"    {be.info()}")
    print(f"    输出区(第3个指针) = 0x{be.out_addr:08X}, 一次回读 {LANES*2} 个 f32")

    y0 = [0.252, 0.26, 0.30, 0.255, 0.40, 0.258, 0.35, 0.60]
    v0 = [-0.5, -0.8, -1.5, -0.3, -0.2, -1.0, -0.4, -0.1]

    print(f"\n[2] PyBullet 建场景: {LANES} 个球体 (radius={RADIUS})")
    world = PyBulletOpenBPU3World(be)
    world.add_spheres(y0, v0)
    ry, rv = world.read_state()
    print(f"    PyBullet 内 y = {[f'{t:.4f}' for t in ry]}")
    print(f"    PyBullet 内 v = {[f'{t:+.4f}' for t in rv]}")
    print(f"    PyBullet 自身重力已关闭 -> 积分由 OpenBPU-3 负责")

    print(f"\n[3] 逐步仿真 ({STEPS} 步; 每步: PyBullet读 -> OpenBPU3求解 -> 写回)")
    for s in range(STEPS):
        y, v = world.step()
        bounced = [i for i in range(LANES)
                   if abs(y[i] - RADIUS) < 1e-5 and v[i] > 0]
        print(f"    step{s+1}: y[0]={y[0]:.6f} v[0]={v[0]:+.6f}  反弹lane={bounced}")

    fy, fv = world.read_state()
    world.close()

    print(f"\n[4] 最终 PyBullet 世界状态 (由 OpenBPU-3 驱动)")
    print(f"    y = {[f'{t:.6f}' for t in fy]}")
    print(f"    v = {[f'{t:+.6f}' for t in fv]}")
    print(f"\n    模拟器: {be.n_sim} 次运行, {be.tot_instr} 条指令, "
          f"{be.tot_cycles} 周期")

    print(f"\n[5] 与 host f32 参考对比")
    ey_ref, ev_ref = ref_step_f32(y0, v0, STEPS)
    print(f"    参考 y = {[f'{t:.6f}' for t in ey_ref]}")
    print(f"    参考 v = {[f'{t:+.6f}' for t in ev_ref]}")
    ey = max(abs(fy[i] - ey_ref[i]) for i in range(LANES))
    ev = max(abs(fv[i] - ev_ref[i]) for i in range(LANES))
    # PyBullet 内部以 float32 存储位姿, 写回/读出会有一次量化, 故给 1e-6 余量
    tol = 1e-6
    print(f"    最大差异: y {ey:.3e}, v {ev:.3e}  (容差 {tol:.0e})")

    ok = ey <= tol and ev <= tol
    print("=" * 72)
    if ok:
        print("PASS: PyBullet 的物理求解已由 OpenBPU-3 完成")
        print("      Triton IR -> flagos -> 真 RV-SIMT 机器码 -> 周期精确模拟器")
        print("      8 刚体 SIMT 并行, 含重力积分 + 碰撞检测 + 冲量响应")
    else:
        print(f"DIFF: y {ey:.3e} / v {ev:.3e} 超容差")
        for i in range(LANES):
            print(f"  body{i}: ref y={ey_ref[i]:.6f} v={ev_ref[i]:+.6f} | "
                  f"got y={fy[i]:.6f} v={fv[i]:+.6f}")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
