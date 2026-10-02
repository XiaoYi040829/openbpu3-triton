#!/usr/bin/env python3
"""
bullet_openbpu3_backend.py — bullet3 的 OpenBPU-3 物理求解后端（BackendOpenBPU3）

替换 bullet3 的物理求解后端：PyBullet 只负责建场景、管理刚体、渲染，
物理积分与碰撞求解全部交给 OpenBPU-3（经 Triton + flagos 编译的 kernel）。

介入方式（已实测可行）：
  1. p.setGravity(0,0,0) 关掉 PyBullet 自身的重力积分
  2. 每步从 PyBullet 读刚体状态 (getBasePositionAndOrientation/getBaseVelocity)
  3. 状态送入 OpenBPU-3 模拟器，由 Triton kernel 并行求解
  4. 结果经 UART 回传，用 resetBasePositionAndOrientation/resetBaseVelocity
     写回 PyBullet 世界

即 PyBullet 的世界状态由 OpenBPU-3 驱动演化，而不是旁路对比。

kernel: triton_kernels/ttir/bullet_step.ttir
  y/v 输入两页，输出写同一页连续区间 (out[0..7]=y', out[8..15]=v')，
  故一步只需一次仿真。
"""
import os
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
for p in [x for x in (os.environ.get("FLAGTREE_PY"), WS,
                      os.path.join(WS, "triton_kernels")) if x]:
    if p not in sys.path:
        sys.path.insert(0, p)

import run_triton_kernel as RT  # noqa: E402

LANES = 8          # SIMT_Width (NutCore.scala:96)
DT = 1.0 / 240.0
G = -9.81
RADIUS = 0.25
REST = 0.5


class BackendOpenBPU3:
    """OpenBPU-3 物理求解后端（供 PyBullet 调用）

    kernel 一次处理 LANES(=8) 个刚体；刚体更多时按 warp 分批。
    """

    def __init__(self, ttir=None, cycles=70000, verbose=False, use_jit=True):
        """use_jit=True 走完整链路:
        @triton.jit Python 前端 -> TTIR -> PolarisBackend 四阶段管线 -> 机器码。
        use_jit=False 则把手写 .ttir 直接喂 MLIR pass。
        """
        self.cycles = cycles
        self.verbose = verbose
        self.use_jit = use_jit
        self.cta_meta = None
        if use_jit:
            from jit_openbpu3 import jit_compile, bullet_step_kernel, STEP_SIG
            r, _ = jit_compile(bullet_step_kernel, STEP_SIG, verbose=verbose)
            self.words = r["words"]
            self.stats = {f"riscv.{k}": r[k] for k in
                          ("n_inst", "num_mapped", "num_skipped",
                           "num_ptr_args", "arg_base", "arg_stride")
                          if r.get(k) is not None}
            self.stats["riscv.num_inst"] = r["n_inst"]
            self.cta_meta = {
                "snnspecvec": r.get("cta_snnspecvec"),
                "snn_input_size": r.get("cta_snn_input_size"),
                "shared_mem_bytes": r.get("cta_shared_mem_bytes"),
            }
            self.ttir = "@triton.jit bullet_step_kernel (Python 前端)"
            self.out_addr = r["arg_base"] + 2 * r["arg_stride"]
        else:
            self.ttir = ttir or os.path.join(
                WS, "triton_kernels/ttir/bullet_step.ttir")
            self.words, self.stats = RT.compile_ttir(
                self.ttir, dump_asm="/tmp/bullet_step.asm")
            self.out_addr = RT.ARG_BASE + 2 * RT.ARG_STRIDE
        # 一次回读 16 个 f32: out[0..7]=y', out[8..15]=v'
        self.code = RT.append_readback(self.words, self.out_addr, LANES * 2)
        self.n_sim = 0
        self.tot_instr = 0
        self.tot_cycles = 0

    def info(self):
        s = self.stats
        src = self.ttir if self.use_jit else os.path.basename(self.ttir)
        txt = (f"kernel {src}: "
               f"{len(self.words)} 条 RISC-V 指令, "
               f"{s.get('riscv.num_mapped')} op 映射 / "
               f"{s.get('riscv.num_skipped')} 跳过, "
               f"{s.get('riscv.num_ptr_args')} 个指针参数")
        if self.cta_meta and self.cta_meta.get("snnspecvec"):
            txt += (f"\n    CTA 元数据: {self.cta_meta['snn_input_size']} 维特征向量 "
                    f"{self.cta_meta['snnspecvec']}, "
                    f"shared={self.cta_meta['shared_mem_bytes']}B")
        return txt

    def _solve_warp(self, y, v):
        """求解一个 warp(LANES 个刚体)"""
        vals, st = RT.run(self.code, {0: list(y), 1: list(v)},
                          self.out_addr, LANES * 2,
                          cycles=self.cycles, tag=f"bstep{self.n_sim}")
        if len(vals) < LANES * 2:
            raise RuntimeError(
                f"OpenBPU-3 回读不足: {len(vals)}/{LANES*2} (提高 EMU_CYCLES?)")
        self.n_sim += 1
        self.tot_instr += st.get("instr", 0)
        self.tot_cycles += st.get("cycles", 0)
        return vals[:LANES], vals[LANES:LANES * 2]

    def solve(self, y, v):
        """一步物理求解，刚体数任意。

        kernel 一次处理 LANES(=8) 个刚体（一个 warp）。刚体更多时按 warp 分批，
        不足一个 warp 的尾批用 0 填充后截取（填充体不影响真实刚体的结果，
        因为各 lane 计算相互独立）。
        """
        assert len(y) == len(v)
        n = len(y)
        oy, ov = [], []
        for base in range(0, n, LANES):
            cy = list(y[base:base + LANES])
            cv = list(v[base:base + LANES])
            pad = LANES - len(cy)
            if pad:
                cy += [0.0] * pad
                cv += [0.0] * pad
            ry, rv = self._solve_warp(cy, cv)
            take = LANES - pad
            oy += ry[:take]
            ov += rv[:take]
        return oy, ov


class PyBulletOpenBPU3World:
    """把 PyBullet 世界的物理演化交给 OpenBPU-3"""

    def __init__(self, backend, verbose=True):
        import pybullet as p
        self.p = p
        self.be = backend
        self.verbose = verbose
        self.bodies = []
        self.cid = p.connect(p.DIRECT)
        # 关键: 关掉 PyBullet 自身重力, 积分由 OpenBPU-3 负责
        p.setGravity(0, 0, 0)
        p.setTimeStep(DT)

    def add_spheres(self, heights, vels):
        p = self.p
        for i, (h, vz) in enumerate(zip(heights, vels)):
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=RADIUS)
            b = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col,
                                  basePosition=[i * 1.0, 0, h])
            p.resetBaseVelocity(b, linearVelocity=[0, 0, vz])
            self.bodies.append(b)

    def read_state(self):
        """从 PyBullet 世界读出刚体状态"""
        y, v = [], []
        for b in self.bodies:
            pos, _ = self.p.getBasePositionAndOrientation(b)
            lin, _ = self.p.getBaseVelocity(b)
            y.append(pos[2])
            v.append(lin[2])
        return y, v

    def write_state(self, y, v):
        """把 OpenBPU-3 求解结果写回 PyBullet 世界"""
        for i, b in enumerate(self.bodies):
            pos, orn = self.p.getBasePositionAndOrientation(b)
            self.p.resetBasePositionAndOrientation(b, [pos[0], pos[1], y[i]], orn)
            self.p.resetBaseVelocity(b, linearVelocity=[0, 0, v[i]])

    def step(self):
        """一步仿真: PyBullet 读状态 -> OpenBPU-3 求解 -> 写回 PyBullet

        注意: 写回后不能再调 stepSimulation()。即使 setGravity(0,0,0),
        stepSimulation 仍会用当前速度做一次位移积分, 与 OpenBPU-3 已算好的
        位移叠加成双重积分, 使轨迹偏离。
        本后端里位移与速度积分都由 OpenBPU-3 完成, PyBullet 只作为世界状态容器。
        """
        y, v = self.read_state()
        ny, nv = self.be.solve(y, v)
        self.write_state(ny, nv)
        return self.read_state()

    def close(self):
        self.p.disconnect()


def ref_step_f32(y, v, n):
    """host 侧 f32 参考实现（与 kernel 同精度、同算法）"""
    import numpy as np
    f = np.float32
    gdt = f(f(G) * f(DT))       # kernel 里 g*dt 是编译期折叠的单个 f32 常量
    dt = f(DT)
    rad, e = f(RADIUS), f(REST)
    y = [f(t) for t in y]
    v = [f(t) for t in v]
    for _ in range(n):
        ny, nv = [], []
        for i in range(len(y)):
            v2 = f(v[i] + gdt)
            y2 = f(y[i] + f(v2 * dt))
            if y2 < rad:
                nv.append(f(-v2 * e))
                ny.append(rad)
            else:
                nv.append(v2)
                ny.append(y2)
        y, v = ny, nv
    return [float(t) for t in y], [float(t) for t in v]
