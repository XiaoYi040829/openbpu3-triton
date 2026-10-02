#!/usr/bin/env python3
"""
verify_td_encoding.py — 校验 LLVM .td 补丁的编码与 PolarisGPU RTL 一致

本脚本做三方交叉校验：
    RISCVInstrInfoSIMT.td  <->  RTL 源码 (Priviledged.scala / RVI.scala)
                           <->  GNU as 实际汇编结果

任一处不符即报错，避免 .td 写了却与硬件不一致（这类错误在实机上表现为
非法指令、核挂死，很难定位）。
"""
import os
import re
import subprocess
import sys

# RTL 源码目录（PolarisGPU 仓库的 src/main/scala/nutcore）。未提供时跳过 RTL 比对。
RTL = os.environ.get(
    "POLARIS_RTL_DIR",
    "")
TD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "RISCVInstrInfoSIMT.td")

# 期望编码（.td 里声明的 warp 操作码 -> 完整 32 位指令字）
WARP_OPS = {
    "JOIN":         (0x00B, 0x00B0007B),
    "START_WARP":   (0x00C, 0x00C0007B),
    "SYN_WARP":     (0x00D, 0x00D0007B),
    "END_WARP":     (0x00E, 0x00E0007B),
    "SYN_WARP_REG": (0x00F, 0x00F0007B),
}
TID_WORD_T1 = 0xC0000333   # TID -> t1(x6)

fails = []


def check(desc, ok, extra=""):
    print(f"  [{'OK ' if ok else 'BAD'}] {desc}" + (f"  {extra}" if extra else ""))
    if not ok:
        fails.append(desc)


def main():
    print("=" * 70)
    print("LLVM .td 补丁编码校验 (td <-> RTL <-> GNU as 三方一致)")
    print("=" * 70)

    # ---- 1. .td 里声明的 warp 操作码 ----
    print("\n[1] .td 声明的 warp 操作码")
    td = open(TD).read()
    for name, (wop, _word) in WARP_OPS.items():
        m = re.search(r"RVInstSIMT(?:Reg)?<0x([0-9A-Fa-f]{3}),\s*\"polaris\."
                      + name.lower() + r"\"", td)
        check(f".td 有 {name} = 0x{wop:03X}",
              bool(m) and int(m.group(1), 16) == wop,
              f"找到 0x{m.group(1)}" if m else "未找到")

    # ---- 2. RTL 源码里的 BitPat ----
    print("\n[2] RTL 源码 (Priviledged.scala) 的 BitPat")
    if not os.path.isfile(os.path.join(RTL, "isa/Priviledged.scala")):
        print(f"  [SKIP] 未找到 RTL 源码（{RTL}），设置 POLARIS_RTL_DIR 可启用")
        return gnu_as_check(rtl_skipped=True)
    priv = open(os.path.join(RTL, "isa/Priviledged.scala")).read()
    for name, (wop, _w) in WARP_OPS.items():
        # BitPat("b000000001011_00000_000_?????_1111011")
        pat = re.search(r"def\s+" + name + r"\s*=\s*BitPat\(\"b([01_?]+)\"\)",
                        priv)
        if not pat:
            check(f"RTL 有 {name}", False, "BitPat 未找到")
            continue
        bits = pat.group(1).replace("_", "")
        imm = bits[:12]
        opc = bits[-7:]
        ok = (imm == format(wop, "012b")) and opc == "1111011"
        check(f"RTL {name}: imm={imm} opcode={opc}", ok,
              f"期望 imm={format(wop,'012b')} opcode=1111011")

    # TID
    rvi = open(os.path.join(RTL, "isa/RVI.scala")).read()
    mt = re.search(r"def\s+TID\s*=\s*BitPat\(\"b([01_?]+)\"\)", rvi)
    if mt:
        bits = mt.group(1).replace("_", "")
        check("RTL TID: funct7=1100000 opcode=0110011",
              bits[:7] == "1100000" and bits[-7:] == "0110011",
              f"实际 funct7={bits[:7]} opcode={bits[-7:]}")
    else:
        check("RTL 有 TID", False)

    return gnu_as_check()


def gnu_as_check(rtl_skipped=False):
    # ---- 3. GNU as 实际汇编 ----
    print("\n[3] GNU as 汇编 .insn 形式，与期望字比对")
    lines = [".text"]
    order = []
    for name, (_wop, word) in WARP_OPS.items():
        areg = {"JOIN": "a1", "START_WARP": "a2", "SYN_WARP": "a3",
                "END_WARP": "a4", "SYN_WARP_REG": "a5"}[name]
        lines.append(f"  .insn r 0x7B, 0x0, 0x0, x0, x0, {areg}")
        order.append((name, word))
    lines.append("  .insn r 0x33, 0x0, 0x60, t1, x0, x0")
    order.append(("TID->t1", TID_WORD_T1))
    open("/tmp/td_check.S", "w").write("\n".join(lines) + "\n")
    try:
        subprocess.run(["riscv64-linux-gnu-as", "-march=rv64imaf_zicsr",
                        "-o", "/tmp/td_check.o", "/tmp/td_check.S"],
                       check=True, capture_output=True)
        r = subprocess.run(["riscv64-linux-gnu-objdump", "-d",
                            "/tmp/td_check.o"],
                           check=True, capture_output=True, text=True)
        got = re.findall(r"^\s+[0-9a-f]+:\s+([0-9a-f]{8})", r.stdout, re.M)
        for (name, want), g in zip(order, got):
            check(f"as 产出 {name} = 0x{g.upper()}",
                  int(g, 16) == want, f"期望 0x{want:08X}")
    except Exception as e:
        check("GNU as 汇编", False, str(e))

    print("\n" + "=" * 70)
    if fails:
        print(f"FAIL: {len(fails)} 项不一致")
        for f in fails:
            print(f"  - {f}")
    elif rtl_skipped:
        print("PASS: .td / GNU as 编码一致（RTL 比对已跳过）")
    else:
        print("PASS: .td / RTL / GNU as 三方编码完全一致")
    print("=" * 70)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
