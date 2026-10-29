#!/usr/bin/env python3
"""
openbpu3_backend.py — OpenBPU-3 计算后端：把物理计算卸载到 PolarisBPU 模拟器

主机跑 PyBullet 主逻辑，物理计算核心（这里是重力积分与球体碰撞检测）交给
OpenBPU-3 执行，结果回传主机。

数据通路（已实测可用）：
  host -> 镜像内数据区(0x80002000 起) -> emu 加载 -> SIMT kernel 计算
       -> 各 lane store 到 0x80001000 -> 标量段串行读回 -> UART(0x40600004) 输出
       -> host 从 stdout 解析

为什么走 UART 而不是直接 dump 内存：
  仿真配置开了写回式缓存(Settings.scala:28,34)，该核无 CMO 指令，脏行不回写；
  用 golden 镜像 matrixsimt0832.bin 对照过，其 store 同样 dump 不出（差异 0 字节）。
  UART 属 MMIO 非缓存区(NutCore.scala:131-134)，emu.cpp:330-331 直接 printf 字符。

定点数约定：
  本模块的手写 kernel 用 Q16.16 定点（int32）表示物理量，
  即 real = fixed / 65536.0。加减法可直接整数运算，乘法需右移 16 位。
"""
import os
import re
import struct
import subprocess
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
EMU = os.environ.get("OPENBPU3_EMU", "emu")  # Wenquxing 26 周期精确模拟器
SIMT_WIDTH = 8               # NutCore.scala:96
UART_TX = 0x40600004         # AXI4UART.scala:33/39
CODE_BASE = 0x80000000
RESULT_BASE = 0x80001000
IN_BASE = 0x80002000         # 输入区起始（各数组间隔 0x1000）

FP_SHIFT = 16
FP_ONE = 1 << FP_SHIFT


def to_fp(x: float) -> int:
    """float -> Q16.16 定点"""
    return int(round(x * FP_ONE)) & 0xFFFFFFFF


def from_fp(v: int) -> float:
    """Q16.16 定点 -> float（按有符号解释）"""
    if v & 0x80000000:
        v -= 1 << 32
    return v / FP_ONE


# ---------------- 最小 RV64 汇编器（编码经 GNU as 验证） ----------------
def I(imm, rs1, f3, rd, op):
    return ((imm & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) | ((f3 & 7) << 12) | \
           ((rd & 0x1F) << 7) | (op & 0x7F)


def R(f7, rs2, rs1, f3, rd, op):
    return ((f7 & 0x7F) << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | \
           ((f3 & 7) << 12) | ((rd & 0x1F) << 7) | (op & 0x7F)


def S(imm, rs2, rs1, f3, op):
    return (((imm >> 5) & 0x7F) << 25) | ((rs2 & 0x1F) << 20) | \
           ((rs1 & 0x1F) << 15) | ((f3 & 7) << 12) | ((imm & 0x1F) << 7) | (op & 0x7F)


def U(imm20, rd, op):
    return ((imm20 & 0xFFFFF) << 12) | ((rd & 0x1F) << 7) | (op & 0x7F)


def li32(rd, v, signed=False):
    """载入 32 位常量到 64 位寄存器。

    两种语义, 必须区分, 否则算术会错:
      signed=False (默认): 载入无符号 32 位值, 高 32 位为 0。
          用于地址 (0x80001000 等), 走 addi+slli 构造避免 lui 符号扩展。
      signed=True: 载入有符号 32 位值, 负数需符号扩展满 64 位。
          用于定点常量 (如 g=-9.81 -> 0xFFF630A4), 走标准 lui+addi
          (lui 的符号扩展此时正是我们要的)。

    负定点常量必须用 signed=True：按无符号载入会得到 0x00000000FFF630A4 而非
    0xFFFFFFFFFFF630A4，MUL 结果高位错、SRAI 算出正数。
    """
    v &= 0xFFFFFFFF
    out = []
    if signed:
        # 标准 lui+addi: 结果为符号扩展的 32 位值
        hi = (v + 0x800) >> 12
        lo = v - (hi << 12)
        if lo & 0x800:
            lo -= 0x1000
        if hi & 0xFFFFF:
            out.append(U(hi & 0xFFFFF, rd, 0x37))
            if lo:
                out.append(I(lo & 0xFFF, rd, 0, rd, 0x13))
        else:
            out.append(I(v & 0xFFF, 0, 0, rd, 0x13))
        return out
    # 无符号: bit31=1 时用 addi+slli 逐段构造, 高位保持 0
    if v & 0x80000000:
        hi20 = v >> 12
        lo12 = v & 0xFFF
        # ADDI 立即数是 12 位有符号: lo12>=0x800 会被当负数(少 0x1000),
        # 必须在高位预先 +1 补偿。否则如 0xBD276C8C 会被装成 0xBD275C8C。
        if lo12 & 0x800:
            hi20 += 1
        out += li32(rd, hi20)
        out.append(I(12, rd, 1, rd, 0x13))      # slli rd,rd,12
        if lo12:
            out.append(I(lo12, rd, 0, rd, 0x13))  # addi (低 12 位按位传, 硬件按有符号解释)
        return out
    hi = (v + 0x800) >> 12
    lo = v - (hi << 12)
    if hi:
        out.append(U(hi, rd, 0x37))
        if lo & 0xFFF:
            out.append(I(lo, rd, 0, rd, 0x13))
    else:
        out.append(I(v, 0, 0, rd, 0x13))
    return out


ADDI = lambda rd, rs1, i: I(i & 0xFFF, rs1, 0, rd, 0x13)
ADD = lambda rd, rs1, rs2: R(0, rs2, rs1, 0, rd, 0x33)
SUB = lambda rd, rs1, rs2: R(0x20, rs2, rs1, 0, rd, 0x33)
MUL = lambda rd, rs1, rs2: R(0x01, rs2, rs1, 0, rd, 0x33)
SLLI = lambda rd, rs1, sh: I(sh, rs1, 1, rd, 0x13)
SRLI = lambda rd, rs1, sh: I(sh, rs1, 5, rd, 0x13)
SRAI = lambda rd, rs1, sh: I(0x400 | sh, rs1, 5, rd, 0x13)
ANDI = lambda rd, rs1, i: I(i & 0xFFF, rs1, 7, rd, 0x13)
SLT = lambda rd, rs1, rs2: R(0, rs2, rs1, 2, rd, 0x33)
LW = lambda rd, rs1, off: I(off & 0xFFF, rs1, 2, rd, 0x03)
SW = lambda rs2, rs1, off: S(off & 0xFFF, rs2, rs1, 2, 0x23)
SB = lambda rs2, rs1, off: S(off & 0xFFF, rs2, rs1, 0, 0x23)
JAL = lambda rd, off: (((off >> 20) & 1) << 31) | (((off >> 1) & 0x3FF) << 21) | \
    (((off >> 11) & 1) << 20) | (((off >> 12) & 0xFF) << 12) | ((rd & 0x1F) << 7) | 0x6F
BNE = lambda rs1, rs2, off: (((off >> 12) & 1) << 31) | (((off >> 5) & 0x3F) << 25) | \
    ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) | (1 << 12) | \
    (((off >> 1) & 0xF) << 8) | (((off >> 11) & 1) << 7) | 0x63
CSRW = lambda csr, rs1: I(csr, rs1, 1, 0, 0x73)
CSRR = lambda rd, csr: I(csr, 0, 2, rd, 0x73)
START_WARP = lambda: I(0x00C, 0, 0, 0, 0x7B)
END_WARP = lambda: I(0x00E, 0, 0, 0, 0x7B)
SYN_WARP = lambda: I(0x00D, 0, 0, 0, 0x7B)
TID = lambda rd: R(0x60, 0, 0, 0, rd, 0x33)


def emit_print_hex8(code, vreg, uart, ta, tb):
    """把 vreg 的 32 位以 8 位十六进制打到 UART，前后加 '<' '>' 定界"""
    code.append(ADDI(12, 0, 0x3C))
    code.append(SB(12, uart, 0))
    for shift in (28, 24, 20, 16, 12, 8, 4, 0):
        code.append(SRLI(ta, vreg, shift))
        code.append(ANDI(ta, ta, 0xF))
        code.append(ADDI(tb, 0, 10))
        code.append(SLT(tb, ta, tb))
        code.append(ADDI(ta, ta, 48))
        code.append(ADDI(tb, tb, -1))
        code.append(ANDI(tb, tb, 39))
        code.append(ADD(ta, ta, tb))
        code.append(SB(ta, uart, 0))
    code.append(ADDI(12, 0, 0x3E))
    code.append(SB(12, uart, 0))


def emit_readback_loop(code, n, uart_reg=30):
    """标量段：串行读 RESULT_BASE 的 n 个 int32 并 UART 打印

    为什么要标量段：UART 是标量 MMIO 设备，无法直接打印 SIMT 向量寄存器的
    8 条道；先让各 lane store 到内存，再由标量段读回逐个输出。
    """
    code += li32(28, RESULT_BASE)
    code += li32(uart_reg, UART_TX)
    code.append(ADDI(21, 0, n))
    loop = len(code)
    code.append(LW(19, 28, 0))
    emit_print_hex8(code, 19, uart_reg, 10, 11)
    code.append(ADDI(28, 28, 4))
    code.append(ADDI(21, 21, -1))
    code.append(BNE(21, 0, ((loop - len(code)) * 4) & 0x1FFF))


def run_kernel(code, data_arrays, n_out, cycles=40000, tag="kernel"):
    """组装镜像 -> emu 执行 -> 从 UART 解析 n_out 个 int32 结果

    data_arrays: {地址: [int32, ...]}
    """
    img = bytearray(0x8000)
    # 代码不得伸进输入区，否则输入数据会覆盖指令
    if len(code) * 4 > IN_BASE - CODE_BASE:
        raise ValueError(f"镜像代码 {len(code) * 4} 字节超出输入区起点 "
                         f"{IN_BASE:#x}，kernel 过大")
    for i, w in enumerate(code):
        struct.pack_into("<I", img, i * 4, w & 0xFFFFFFFF)
    for addr, vals in data_arrays.items():
        off = addr - CODE_BASE
        for i, v in enumerate(vals):
            struct.pack_into("<I", img, off + i * 4, v & 0xFFFFFFFF)

    path = f"/tmp/obpu3_{tag}.bin"
    with open(path, "wb") as f:
        f.write(bytes(img))

    # -b/-e 设到不可达周期以压掉逐周期调试日志。
    # 否则 4 个 PE 的 Decoder/LSU/CSR 日志从第 1 周期起刷屏, 会淹没 UART 输出,
    # 且数百 MB 输出会拖慢主机。
    cmd = ["nice", "-n", "15", EMU, "-i", path, "--no-diff", "-C", str(cycles),
           "-b", "99999999", "-e", "99999999"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    out = r.stdout + r.stderr

    seg = out.split("Core 0:")[0]
    if "flash instead" in seg:
        seg = seg[seg.rfind("flash instead") + 14:]
    # 先按原样提取(标量段输出时只有一个 PE 在写, 通常已是干净的 <xxxxxxxx>)。
    # 注意不能无条件压缩连续重复字符: 0x00022315 这类含 "00" 的合法值会被毁掉。
    toks = re.findall(r"<([0-9a-f]{8})>", seg)
    if not toks:
        # 退路: 多 PE 交错导致每字符重复 k 次时, 按固定倍数抽取
        for k in (4, 2, 8):
            cand = re.sub(r"(.)\1{%d}" % (k - 1), r"\1", seg)
            toks = re.findall(r"<([0-9a-f]{8})>", cand)
            if toks:
                break
    vals = [int(t, 16) for t in toks]

    ic = re.search(r"instrCnt = ([\d,]+), cycleCnt = ([\d,]+)", out)
    stats = {}
    if ic:
        stats["instr"] = int(ic.group(1).replace(",", ""))
        stats["cycles"] = int(ic.group(2).replace(",", ""))
    return vals[:n_out], stats, len(code)
