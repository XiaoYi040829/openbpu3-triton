#!/usr/bin/env python3
"""
run_triton_kernel.py — 走 Triton + flagos 完整链路在 OpenBPU-3 模拟器上跑 kernel

链路（全部真实，无手写汇编）：
    Triton IR (.ttir)
      -> flagtree 官方 TTIR pass (inliner/combine/canonicalizer/cse)
      -> polaris C++ pass add_tensorcore_mapping (TensorCore 映射分析)
      -> polaris C++ pass add_riscv_codegen (发射真 RV64 + RV-SIMT 机器码)
      -> .bin
      -> build/rtl/emu (周期精确 RTL 仿真器) 执行
      -> 结果经 UART 回传 host

kernel 指针参数的裸机约定（PolarisRiscvCodegen.cpp 中 ARG_BASE/ARG_STRIDE）：
    第 i 个指针参数 = 0x80002000 + i*0x1000
host 侧按同一约定把输入写进镜像、从对应地址取回输出。

读回方式：codegen 只产出计算部分，本脚本在其后追加一段标量回读代码
（UART 是标量 MMIO 设备，无法直接打印 SIMT 向量寄存器的 8 条道）。
"""
import os
import re
import struct
import subprocess
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
FLAGTREE_PY = os.environ.get("FLAGTREE_PY", "")
for p in [x for x in (FLAGTREE_PY, WS, os.path.join(WS, "triton_kernels")) if x]:
    if p not in sys.path:
        sys.path.insert(0, p)

EMU = os.environ.get("OPENBPU3_EMU", "emu")  # Wenquxing 26 周期精确模拟器
ARG_BASE, ARG_STRIDE = 0x80002000, 0x1000
N = 8

from openbpu3_backend import (  # noqa: E402
    li32, ADDI, ADD, SLLI, SRLI, ANDI, LW, SB, JAL, R,
    UART_TX, emit_print_hex8,
)


def compile_ttir(ttir_path, dump_asm=None):
    """跑完整 flagos 链路，返回 (指令字列表, 统计)"""
    from triton._C.libtriton import ir, passes, polaris
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = ir.parse_mlir_module(ttir_path, ctx)
    mod.context = ctx

    pm = ir.pass_manager(ctx)
    passes.common.add_inliner(pm)
    passes.ttir.add_combine(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    pm.run(mod)

    pm2 = ir.pass_manager(ctx)
    polaris.passes.polarisgpu.add_tensorcore_mapping(pm2)
    pm2.run(mod)

    pm3 = ir.pass_manager(ctx)
    polaris.passes.polarisgpu.add_riscv_codegen(pm3)
    pm3.run(mod)

    s = str(mod)
    m = re.search(r"polaris\.riscv\.words = array<i32: ([^>]*)>", s)
    if not m:
        raise RuntimeError("codegen 未产出 polaris.riscv.words")
    words = [int(x) & 0xFFFFFFFF for x in m.group(1).split(",")]
    stats = {k: int(v) for k, v in re.findall(r"polaris\.(\w[\w.]*) = (\d+)", s)}
    # 从 codegen 导出的属性读参数布局, 避免 host/device 两侧硬编码不一致
    global ARG_BASE, ARG_STRIDE
    if "riscv.arg_base" in stats:
        ARG_BASE = stats["riscv.arg_base"]
    if "riscv.arg_stride" in stats:
        ARG_STRIDE = stats["riscv.arg_stride"]

    if dump_asm:
        with open(dump_asm + ".S", "w") as f:
            f.write(".text\n")
            for w in words:
                f.write("  .insn 4, 0x%08x\n" % w)
        try:
            subprocess.run(["riscv64-linux-gnu-as", "-march=rv64imafc_zicsr",
                            "-o", dump_asm + ".o", dump_asm + ".S"],
                           check=True, capture_output=True)
            r = subprocess.run(["riscv64-linux-gnu-objdump", "-d", dump_asm + ".o"],
                               check=True, capture_output=True, text=True)
            with open(dump_asm, "w") as f:
                f.write(r.stdout)
        except Exception:
            pass
    return words, stats


def append_readback(words, out_addr, n, status=True):
    """在 codegen 产出之后追加标量回读段：读 out_addr 的 n 个 f32 bit 并 UART 输出

    status=True 时在各段之间插入哨兵标记（见 kernel_status.py）。
    没有哨兵的话，"回读不足"这一个症状底下藏着至少四种原因
    （核没跑 / 主体崩了 / 回读地址错 / 结果真没写全），
    而 emu 对成功和崩溃给出的终止信息逐字相同、退出码都是 0，
    无法区分。哨兵把"执行推进到哪一步"变成可观测的。
    """
    from kernel_status import emit_sentinel

    code = list(words)
    # 哨兵 1：计算主体已执行完（此时还没碰回读地址，能把主体故障与
    # 回读故障分开）
    if status:
        emit_sentinel(code, "compute_done")
    code += li32(28, out_addr)
    code += li32(30, UART_TX)
    code.append(ADDI(21, 0, n))
    loop = len(code)
    code.append(LW(19, 28, 0))
    emit_print_hex8(code, 19, 30, 10, 11)
    code.append(ADDI(28, 28, 4))
    code.append(ADDI(21, 21, -1))
    off = (loop - len(code)) * 4
    BNE = (((off >> 12) & 1) << 31) | (((off >> 5) & 0x3F) << 25) | \
          (0 << 20) | ((21 & 0x1F) << 15) | (1 << 12) | \
          (((off >> 1) & 0xF) << 8) | (((off >> 11) & 1) << 7) | 0x63
    code.append(BNE)
    # 哨兵 2/3：回读循环正常退出 / 尾声完毕。
    # 两个哨兵之间没有别的指令，但仍分开发：若回读循环的 BNE 偏移算错
    # 导致落到循环外的错误位置，S2 和 S3 会缺一个而不是同时缺。
    if status:
        emit_sentinel(code, "readback_done", uart_reg=30)
        emit_sentinel(code, "epilogue_done", uart_reg=30)
    code.append(JAL(0, 0))
    return code


# 浮点常量池（只读）。codegen 的 CONST_BASE 与此一致：
#   [0]=0.0  [1]=1.0  [2]=0.5  [3]=2^23
# kernel prologue 把这四个常量 flw 进 f28-f31（不参与分配），供 floor、
# exp、fmax 等展开共用，避免每处都用 li + fmv.w.x 重新物化。
# 必须与 codegen 的 CONST_BASE 严格一致（PolarisRiscvCodegen.cpp）：
# 不一致时 kernel 会读到未初始化内存，floor/exp 等结果错误，且不报错。
# 地址放在 0x80020000，高于全部参数区（0x80002000 起、每个 0x1000、
# 最多 6 个）。改这个值时两处必须同步改。
CONST_BASE = 0x80020000
CONST_POOL = [0.0, 1.0, 0.5, 8388608.0]



def run(words, inputs, out_addr, n_out, cycles=60000, tag="tt"):
    """inputs: {arg_index: [float,...]} 按 ARG_BASE+i*ARG_STRIDE 放入镜像"""
    # 0x30000(192KB)：镜像必须覆盖到常量池（CONST_BASE = 0x80020000）之后。
    img = bytearray(0x30000)
    # 代码（含追加的回读段）不得伸进参数区，否则输入数据会覆盖指令
    if len(words) * 4 > ARG_BASE - 0x80000000:
        raise ValueError(f"镜像代码 {len(words) * 4} 字节超出参数区起点 "
                         f"{ARG_BASE:#x}，kernel 过大")
    for i, w in enumerate(words):
        struct.pack_into("<I", img, i * 4, w & 0xFFFFFFFF)
    # 常量池必须先填，kernel prologue 会 flw 它
    for j, v in enumerate(CONST_POOL):
        struct.pack_into("<f", img, CONST_BASE - 0x80000000 + j * 4, v)
    # 按元素类型打包：整数数组必须用 "<i"，不能统一按 "<f"。
    # 判据：Python int（非 bool/float）视为 i32。
    for idx, vals in inputs.items():
        base = ARG_BASE + idx * ARG_STRIDE - 0x80000000
        for j, v in enumerate(vals):
            if isinstance(v, bool):
                struct.pack_into("<i", img, base + j * 4, int(v))
            elif isinstance(v, int):
                struct.pack_into("<i", img, base + j * 4, v)
            else:
                struct.pack_into("<f", img, base + j * 4, float(v))
    path = f"/tmp/tt_{tag}.bin"
    with open(path, "wb") as f:
        f.write(bytes(img))

    cmd = ["nice", "-n", "15", EMU, "-i", path, "--no-diff", "-C", str(cycles),
           "-b", "99999999", "-e", "99999999"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    out = r.stdout + r.stderr

    # 走 kernel_status 统一解析：它会先剥 ANSI 转义再取值，
    # 直接用 re 找 <xxxxxxxx> 会被 emu 的颜色码干扰。
    from kernel_status import parse_uart, diagnose
    p = parse_uart(out)
    vals = [struct.unpack("<f", struct.pack("<I", w))[0] for w in p["words"]]

    st = {}
    if p["instr"] is not None:
        st["instr"] = p["instr"]
        st["cycles"] = p["cycles"]
    # 执行状态回传：调用方不看也不影响原有行为，但一旦结果不对，
    # st["fault"] 直接给出失败阶段而不用重新二分定位。
    #
    # diagnose 内部用 IPC 判据，不需要 host 提供指令数基线。
    code, why = diagnose(p, n_out, cycles_budget=cycles)
    st["fault"] = code
    st["fault_why"] = why
    st["stages"] = p["stages"]
    st["term"] = p["term"]
    return vals[:n_out], st


def main():
    ttir = os.path.join(WS, "triton_kernels/ttir/gravity.ttir")
    print("=" * 68)
    print("Triton + flagos -> OpenBPU-3: 重力积分 kernel")
    print("=" * 68)

    words, stats = compile_ttir(ttir, dump_asm="/tmp/tt_gravity.asm")
    print(f"flagos 编译: {len(words)} 条指令")
    print(f"  统计: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())
                                  if k.startswith("riscv") or k == "num_dot"))

    # kernel 签名: (y_ptr, v_ptr, oy_ptr, ov_ptr)
    # 参数按约定落在 ARG_BASE + i*0x1000
    dt, g = 1.0 / 240.0, -9.81
    y0 = [1.0 + i * 0.5 for i in range(N)]
    v0 = [0.0] * N
    OY = ARG_BASE + 2 * ARG_STRIDE      # 第 3 个参数 = 输出 y
    OV = ARG_BASE + 3 * ARG_STRIDE      # 第 4 个参数 = 输出 v

    # 输出 y 与 v 在不同页, 这里只回读 y
    code = append_readback(words, OY, N)
    print(f"追加标量回读段后: {len(code)} 条指令")

    print(f"\n输入 y = {[f'{t:.3f}' for t in y0]}")
    ref_v = [v0[i] + g * dt for i in range(N)]
    ref_y = [y0[i] + ref_v[i] * dt for i in range(N)]
    print(f"期望 y' = {[f'{t:.6f}' for t in ref_y]}")

    vals, st = run(code, {0: y0, 1: v0}, OY, N,
                   cycles=int(os.environ.get("EMU_CYCLES", "60000")),
                   tag="gravity")
    print(f"\nemu: {st.get('instr','?')} 条指令, {st.get('cycles','?')} 周期")
    print(f"UART 收到 {len(vals)} 个值")
    if len(vals) < N:
        print(f"未收齐, 收到: {vals}")
        return 1
    print(f"OpenBPU-3 y' = {[f'{t:.6f}' for t in vals]}")

    err = max(abs(vals[i] - ref_y[i]) for i in range(N))
    tol = 1e-5
    print(f"\n最大误差: {err:.3e}  (容差 {tol:.0e}, f32 相对精度 ~1e-7)")
    ok = err <= tol
    print("=" * 68)
    print("PASS: Triton kernel 经 flagos 编译, 在 OpenBPU-3 上算出正确结果"
          if ok else f"DIFF: 误差 {err:.3e}")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
