#!/usr/bin/env python3
"""
verify_llvm_mc.py — 验证打了补丁的 LLVM 能汇编/反汇编 RV-SIMT 指令

本脚本用已编译的 llvm-mc（含 RISCVInstrInfoSIMT.td 补丁）做三项检查：
  1. 汇编：助记符 -> 机器码，编码与 RTL 一致
  2. 反汇编：机器码 -> 助记符
  3. 受控性：不带 -mattr=+xpolarissimt 时应报错（不污染标准 RISC-V）

补丁位置：llvm/lib/Target/RISCV/RISCVInstrInfoSIMT.td
         + RISCVFeatures.td 的 FeatureVendorXPolarisSIMT
         + RISCVInstrInfo.td 末尾的 include
"""
import os
import re
import subprocess
import sys

MC_CANDIDATES = [
    os.path.expanduser("~/llvm-project/build-riscv/bin/llvm-mc"),
]
ATTR = "+xpolarissimt"

# 助记符 -> 期望 32 位编码（取自 PolarisGPU RTL）
CASES = [
    ("polaris.join",            0x00B0007B),
    ("polaris.start_warp",      0x00C0007B),
    ("polaris.syn_warp",        0x00D0007B),
    ("polaris.end_warp",        0x00E0007B),
    ("polaris.tid\tt1",         0xC0000333),
]
fails = []


def check(desc, ok, extra=""):
    print(f"  [{'OK ' if ok else 'BAD'}] {desc}" + (f"  {extra}" if extra else ""))
    if not ok:
        fails.append(desc)


def find_mc():
    """优先用环境变量 LLVM_MC 指定的 llvm-mc，其次查默认构建位置。"""
    env = os.environ.get("LLVM_MC")
    if env:
        return env if os.path.exists(env) else None
    for p in MC_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def main():
    print("=" * 70)
    print("打了 RV-SIMT 补丁的 LLVM: 汇编 / 反汇编验证")
    print("=" * 70)
    mc = find_mc()
    if not mc:
        print("\n未找到打过补丁的 llvm-mc（设置 LLVM_MC 指定路径），跳过")
        return 2
    print(f"\nllvm-mc: {mc}")

    # ---- 1. 汇编 ----
    print("\n[1] 汇编: 助记符 -> 机器码")
    src = "\n".join(c[0] for c in CASES) + "\n"
    r = subprocess.run([mc, "-triple=riscv64", f"-mattr={ATTR}",
                        "-show-encoding"],
                       input=src, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        check("llvm-mc 汇编成功", False, r.stderr[:200])
    else:
        # encoding: [0x7b,0x00,0xb0,0x00]
        encs = re.findall(r"encoding: \[([^\]]+)\]", r.stdout)
        for (mnem, want), e in zip(CASES, encs):
            bs = [int(x.strip(), 16) for x in e.split(",")]
            got = bs[0] | (bs[1] << 8) | (bs[2] << 16) | (bs[3] << 24)
            name = mnem.split("\t")[0]
            check(f"{name:22s} = 0x{got:08X}", got == want,
                  f"期望 0x{want:08X}")

    # ---- 2. 反汇编 ----
    print("\n[2] 反汇编: 机器码 -> 助记符")
    words = [c[1] for c in CASES]
    hexs = " ".join(
        f"0x{(w >> (8 * i)) & 0xFF:02x}" for w in words for i in range(4))
    r2 = subprocess.run([mc, "-triple=riscv64", f"-mattr={ATTR}",
                         "-disassemble"],
                        input=hexs + "\n", capture_output=True, text=True,
                        timeout=120)
    out2 = r2.stdout
    for mnem, _w in CASES:
        name = mnem.split("\t")[0]
        check(f"反汇编出 {name}", name in out2)

    # ---- 3. 受控性 ----
    print("\n[3] 受控性: 不带 -mattr 应报错")
    r3 = subprocess.run([mc, "-triple=riscv64", "-show-encoding"],
                        input="polaris.start_warp\n", capture_output=True,
                        text=True, timeout=120)
    check("未启用扩展时拒绝该指令",
          "xpolarissimt" in r3.stderr or r3.returncode != 0,
          "(说明是受控 vendor 扩展，不污染标准 RISC-V)")

    print("\n" + "=" * 70)
    if fails:
        print(f"FAIL: {len(fails)} 项")
        for f in fails:
            print(f"  - {f}")
    else:
        print("PASS: LLVM 可汇编/反汇编全部 RV-SIMT 自定义指令，编码与 RTL 一致")
    print("=" * 70)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
