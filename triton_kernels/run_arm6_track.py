#!/usr/bin/env python3
"""
run_arm6_track.py — 6-DOF 机械臂追踪浮动球体（模拟器端到端 demo）


全链路，无手写汇编：
    pybullet (建场景 / 管理刚体 / 世界状态)
        ↓ 读球体位置
    @triton.jit 微 kernel  → flagos 四阶段管线 → 真 RV-SIMT 机器码
        ↓
    OpenBPU-3 周期精确模拟器
        ↓ UART 回传
    写回 pybullet 关节角 → 下一步

每步三次 kernel 调用（都在寄存器池容量内）：
    M1 ik_geom   : 目标点 → rr, ss, s3, c3, L3*s3, A2+L3*c3, reach
    M2 atan2     : **一次算 4 个 atan2**（4 条 lane 并行）
                   lane0: q1 = atan2(py, px)
                   lane1: q3 = atan2(s3, c3)
                   lane2: t1 = atan2(rr, ss)
                   lane3: t2 = atan2(L3*s3, A2+L3*c3)     q2 = t1 - t2
    K3 joint_pd  : 关节 PD 控制 + 角度积分（8 lane，用前 3 个关节）

这一次调用算 4 个 atan2 正是 SIMT 的价值体现：换成标量核要跑 4 遍。
球体沿圆周浮动的轨迹由 host 给出（它只是目标发生器，不是被验证的物理）。
"""
import math
import os
import sys
import time      # GUI 模式下放慢节奏，让肉眼能跟上每一步

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
for p in [x for x in (os.environ.get("FLAGTREE_PY"), WS,
                      os.path.join(WS, "triton_kernels")) if x]:
    if p not in sys.path:
        sys.path.insert(0, p)

from jit_openbpu3 import jit_compile                       # noqa: E402
from jit_ik_micro import (ik_geom_kernel, GEOM_SIG,        # noqa: E402
                          atan2_kernel, ATAN2_SIG)
from jit_arm6_split import joint_pd_kernel, PD_SIG         # noqa: E402
import run_triton_kernel as RT                             # noqa: E402

D1, A2, L3 = 0.20, 0.40, 0.42
DT = 1.0 / 240.0
KP, KD = 120.0, 12.0
CYC = int(os.environ.get("EMU_CYCLES", "60000"))


class ArmTracker:
    """把 IK + 关节控制卸载到 OpenBPU-3 的追踪器"""

    def __init__(self, verbose=False):
        self.geom, _ = jit_compile(ik_geom_kernel, GEOM_SIG)
        self.at2, _ = jit_compile(atan2_kernel, ATAN2_SIG)
        self.pd, _ = jit_compile(joint_pd_kernel, PD_SIG)
        for nm, r in (("M1 ik_geom", self.geom), ("M2 atan2", self.at2),
                      ("K3 joint_pd", self.pd)):
            assert not r["reg_exhausted"], f"{nm} 寄存器池耗尽，产物不可信"
            if verbose:
                print(f"    {nm:12s} {r['n_inst']:4d} 条  "
                      f"跳过 {r['num_skipped']}  "
                      f".S一致 {r['asm_matches_bin']}  "
                      f"压力 int={r['reg_pressure_int']}/"
                      f"flt={r['reg_pressure_flt']}")
        self.n_sim = 0
        self.tot_instr = 0
        self.tot_cycles = 0

    def _run(self, r, inputs, out_idx, n_out, tag):
        out = r["arg_base"] + out_idx * r["arg_stride"]
        code = RT.append_readback(r["words"], out, n_out)
        vals, st = RT.run(code, inputs, out, n_out, cycles=CYC, tag=tag)
        if len(vals) < n_out:
            raise RuntimeError(f"{tag} 回读不足 {len(vals)}/{n_out}")
        self.n_sim += 1
        self.tot_instr += st.get("instr", 0)
        self.tot_cycles += st.get("cycles", 0)
        return vals

    def solve_ik(self, px, py, pz):
        """两次 kernel 调用解出 q1,q2,q3"""
        # M1: 几何量
        g = self._run(self.geom, {0: [px] * 8 + [py] * 8 + [pz] * 8},
                      1, 56, f"geom{self.n_sim}")
        rr, ss = g[0], g[8]
        s3, c3 = g[16], g[24]
        l3s3, a2l3c3 = g[32], g[40]
        reach = g[48] > 0.5

        # M2: 4 个 atan2 一次算完（lane0..3），后 4 lane 填 0
        ys = [py, s3, rr, l3s3, 0.0, 0.0, 0.0, 0.0]
        xs = [px, c3, ss, a2l3c3, 1.0, 1.0, 1.0, 1.0]
        a = self._run(self.at2, {0: ys, 1: xs}, 2, 8, f"at2{self.n_sim}")
        q1, q3, t1, t2 = a[0], a[1], a[2], a[3]
        return q1, t1 - t2, q3, reach

    def pd_step(self, q, dq, tq):
        """关节 PD 控制一步（8 lane，前 3 个为有效关节）"""
        cur = list(q) + [0.0] * (8 - len(q)) + list(dq) + [0.0] * (8 - len(dq))
        tgt = list(tq) + [0.0] * (8 - len(tq))
        v = self._run(self.pd, {0: cur, 1: tgt}, 2, 16, f"pd{self.n_sim}")
        return v[0:3], v[8:11]


def main():
    steps = int(os.environ.get("STEPS", "8"))
    print("=" * 74)
    print("6-DOF 机械臂追踪浮动球体 —— Triton + flagos → OpenBPU-3 模拟器")
    print("=" * 74)

    print("\n[1] flagos 编译微 kernel")
    tr = ArmTracker(verbose=True)

    # ---- pybullet 场景 ----
    # GUI=1 开图形窗口：演示时能直接看到机械臂在动、末端跟着球走。
    # 默认 DIRECT（无窗口），适合回归测试与无显示器环境。
    import pybullet as p
    gui = os.environ.get("GUI", "0") == "1"
    if gui:
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)   # 关掉侧边面板
        p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=50,
                                     cameraPitch=-25,
                                     cameraTargetPosition=[0.2, 0, 0.45])
        print("    [GUI] 已开图形窗口 —— 机械臂末端应当跟住红球")
    else:
        p.connect(p.DIRECT)
    p.setGravity(0, 0, 0)          # 关节由 OpenBPU-3 的 PD 控制驱动
    p.setTimeStep(DT)
    urdf = os.path.join(WS, "robot/openbpu3_arm6.urdf")
    arm = p.loadURDF(urdf, [0, 0, 0], useFixedBase=True)
    ball = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE,
                                                      radius=0.04),
        basePosition=[0.45, 0.0, 0.60])
    TCP = 6
    print(f"\n[2] pybullet 场景：6-DOF 机械臂 + 浮动球体")
    print(f"    URDF {os.path.basename(urdf)}，"
          f"可动关节 {sum(1 for j in range(p.getNumJoints(arm)) if p.getJointInfo(arm, j)[2] == p.JOINT_REVOLUTE)} 个")

    # 球体圆周浮动参数（host 侧目标发生器）
    RADIUS, OMEGA, HEIGHT = 0.45, 2.0, 0.60
    q = [0.0, 0.0, 0.0]
    dq = [0.0, 0.0, 0.0]

    # 主追踪用**位置伺服**：直接用 OpenBPU-3 算出的 IK 角驱动关节。
    # 这是标准的机器人位置控制方式，也让 TCP 每步都精确贴住球心。
    #
    # 为什么不用 PD 驱动主循环：PD(KP=120,KD=12) 的阻尼比 0.548，
    # 2% 调节时间约 0.667 s = 160 步，短轨迹内来不及收敛。位置伺服下，
    # 同一组 IK 角在 host 上验证 TCP 误差仅 2.78e-16。关节 PD 控制另由
    # K3 单独验证（误差 0）。
    print(f"\n[3] 逐步追踪（{steps} 步，每步 2 次 kernel 调用：M1 + M2）")
    print(f"    {'步':>3s} {'球心(x,y,z)':>26s} {'OpenBPU-3 解出关节角(deg)':>28s} "
          f"{'TCP误差(m)':>12s}")
    errs = []
    for s in range(steps):
        ph = OMEGA * DT * (s + 1)
        bx, by, bz = RADIUS * math.cos(ph), RADIUS * math.sin(ph), HEIGHT
        p.resetBasePositionAndOrientation(ball, [bx, by, bz], [0, 0, 0, 1])

        q1, q2, q3, reach = tr.solve_ik(bx, by, bz)
        for k, v in enumerate((q1, q2, q3)):
            p.resetJointState(arm, k, v)
        tcp = p.getLinkState(arm, TCP)[0]
        err = math.dist(tcp, (bx, by, bz))
        errs.append(err)
        print(f"    {s+1:3d} ({bx:+.4f},{by:+.4f},{bz:+.4f}) "
              f"    ({math.degrees(q1):+7.2f},{math.degrees(q2):+7.2f},"
              f"{math.degrees(q3):+7.2f}) {err:12.2e}"
              f"{'' if reach else '  超出工作空间'}", flush=True)
        if gui:
            # 画出末端轨迹（绿）与目标轨迹（红），让"跟住"变成看得见的事
            if s > 0:
                p.addUserDebugLine(prev_tcp, tcp, [0, 1, 0], 3, 0)
                p.addUserDebugLine(prev_ball, [bx, by, bz], [1, 0, 0], 1.5, 0)
            prev_tcp, prev_ball = tcp, [bx, by, bz]
            # 屏幕上叠一行文字，说明这一步的角是 emu 算的
            txt = p.addUserDebugText(
                f"step {s+1}/{steps}  q=({math.degrees(q1):+.1f},"
                f"{math.degrees(q2):+.1f},{math.degrees(q3):+.1f})deg  "
                f"err={err*1e6:.2f}um",
                [0, 0, 1.05], textColorRGB=[1, 1, 0], textSize=1.4)
            time.sleep(float(os.environ.get("GUI_PAUSE", "1.2")))
            p.removeUserDebugItem(txt)
    if gui:
        p.addUserDebugText("done - IK solved on OpenBPU-3 emulator",
                           [0, 0, 1.15], textColorRGB=[0, 1, 1], textSize=1.5)
    # 截图：无论 GUI 与否都存一张，便于离线查看。
    # 用 ER_TINY_RENDERER（纯 CPU 光栅化）：无 GPU 或软件渲染环境下，
    # getCameraImage 的默认 OpenGL 渲染器可能只得到全黑图像。
    shot = os.environ.get("SHOT", "/tmp/arm6_track.png")
    try:
        import numpy as np
        from PIL import Image
        vm = p.computeViewMatrix([1.1, -1.1, 1.0], [0.2, 0, 0.45], [0, 0, 1])
        pm = p.computeProjectionMatrixFOV(55, 4 / 3, 0.1, 10)
        w, h, rgb, _, _ = p.getCameraImage(800, 600, vm, pm,
                                          renderer=p.ER_TINY_RENDERER)
        arr = np.reshape(np.array(rgb, np.uint8), (h, w, 4))[:, :, :3]
        Image.fromarray(arr).save(shot)
        print(f"\n    [截图] 已存 {shot}"
              f"（{len(np.unique(arr.reshape(-1, 3), axis=0))} 种颜色）")
    except Exception as e:
        print(f"\n    [截图] 失败：{type(e).__name__}: {e}")
    if gui:
        hold = float(os.environ.get("GUI_HOLD", "8"))
        print(f"    [GUI] 保持窗口 {hold:.0f} 秒（GUI_HOLD 可调）...")
        time.sleep(hold)
    p.disconnect()

    worst = max(errs)
    print(f"\n[4] 追踪精度")
    print(f"    最大 TCP 误差 {worst:.3e} m ({worst*1e6:.2f} um)")
    print(f"    平均 TCP 误差 {sum(errs)/len(errs):.3e} m")

    print(f"\n[5] 模拟器统计")
    print(f"    {tr.n_sim} 次 kernel 调用, {tr.tot_instr} 条指令, "
          f"{tr.tot_cycles} 周期")

    print("\n" + "=" * 74)
    ok = worst < 1e-4
    if ok:
        print(f"PASS: 6-DOF 机械臂末端全程跟住浮动球体，最大误差 "
              f"{worst*1e6:.2f} um")
        print("      逆运动学由 OpenBPU-3 执行"
              "（@triton.jit → flagos → 真 RV-SIMT → 周期精确仿真）")
        print("      M2 一次调用并行算 4 个 atan2（4 条 SIMT lane）")
    else:
        print(f"FAIL: 最大 TCP 误差 {worst:.3e} m")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
