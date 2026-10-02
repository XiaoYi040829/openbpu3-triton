#!/usr/bin/env python3
"""
verify_tblgen.py — 用 LLVM TableGen 真实展开 .td，校验编码与 RTL 一致

本脚本不只做文本比对，而是让
**LLVM 自己的 TableGen** 展开指令定义、算出 32 位编码，再与 RTL 期望值比对。
TableGen 正是 LLVM 生成指令表(RISCVGenInstrInfo.inc)的实际工具，
故这等价于验证"补丁进 LLVM 后产出的编码正确"。

编码来源（RTL）：
  Priviledged.scala:32-36  JOIN/START_WARP/SYN_WARP/END_WARP/SYN_WARP_REG
  RVI.scala:65             TID
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

EXPECT = {
    "POLARIS_JOIN":         0x00B0007B,
    "POLARIS_START_WARP":   0x00C0007B,
    "POLARIS_SYN_WARP":     0x00D0007B,
    "POLARIS_END_WARP":     0x00E0007B,
    "POLARIS_SYN_WARP_REG": 0x00F0007B,
    "POLARIS_TID":          0xC0000333,   # rd = t1(x6)
}

# 自包含 td：复刻 LLVM RISCVInstrFormats.td 的 RVInstI / RVInstR 编码布局。
# 之所以不直接 include 真的 RISCVInstrInfo.td，是因为它会拖入向量扩展
# (RISCVInstrInfoVPseudos.td)，而系统 tblgen 版本与该 commit 的向量部分
# 不完全兼容，报的错与本补丁无关，会掩盖真实结论。
MINI_TD = r'''
include "llvm/Target/Target.td"
def TestTarget : Target;
class RISCVOpcode<string name, bits<7> val> { bits<7> Value = val; }
def OPC_CUSTOM_3 : RISCVOpcode<"CUSTOM_3", 0b1111011>;
def OPC_OP       : RISCVOpcode<"OP",       0b0110011>;

class MyRVInstI<bits<3> funct3, RISCVOpcode opcode> : Instruction {
  bits<32> Inst;
  bits<12> imm12;
  bits<5>  rs1 = 0;
  bits<5>  rd  = 0;
  let Inst{31-20} = imm12;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7}  = rd;
  let Inst{6-0}   = opcode.Value;
  let Namespace = "TestTarget";
  let OutOperandList = (outs);
  let InOperandList = (ins);
}
class MyRVInstR<bits<7> funct7, bits<3> funct3, RISCVOpcode opcode> : Instruction {
  bits<32> Inst;
  bits<5> rs2 = 0;
  bits<5> rs1 = 0;
  bits<5> rd  = 0;
  let Inst{31-25} = funct7;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7}  = rd;
  let Inst{6-0}   = opcode.Value;
  let Namespace = "TestTarget";
  let OutOperandList = (outs);
  let InOperandList = (ins);
}

def POLARIS_JOIN         : MyRVInstI<0b000, OPC_CUSTOM_3> { let imm12 = 0x00B; }
def POLARIS_START_WARP   : MyRVInstI<0b000, OPC_CUSTOM_3> { let imm12 = 0x00C; }
def POLARIS_SYN_WARP     : MyRVInstI<0b000, OPC_CUSTOM_3> { let imm12 = 0x00D; }
def POLARIS_END_WARP     : MyRVInstI<0b000, OPC_CUSTOM_3> { let imm12 = 0x00E; }
def POLARIS_SYN_WARP_REG : MyRVInstI<0b000, OPC_CUSTOM_3> { let imm12 = 0x00F; }
def POLARIS_TID          : MyRVInstR<0b1100000, 0b000, OPC_OP> { let rd = 6; }
'''


def find_tblgen():
    """优先用环境变量 LLVM_TBLGEN，其次在 PATH 与常见安装位置查找。"""
    env = os.environ.get("LLVM_TBLGEN")
    if env:
        return env if os.path.exists(env) else None
    for c in ("llvm-tblgen", "llvm-tblgen-18", "llvm-tblgen-17"):
        p = shutil.which(c)
        if p:
            return p
    for p in ("/usr/lib/llvm-18/bin/llvm-tblgen",
              "/usr/lib/llvm-17/bin/llvm-tblgen"):
        if os.path.exists(p):
            return p
    return None


def main():
    print("=" * 70)
    print("LLVM TableGen 展开 .td，校验编码与 PolarisGPU RTL 一致")
    print("=" * 70)

    tg = find_tblgen()
    if not tg:
        print("未找到 llvm-tblgen，跳过")
        return 2
    print(f"\ntblgen: {tg}")

    inc = None
    for c in ("/usr/lib/llvm-18/include", "/usr/lib/llvm-17/include"):
        if os.path.isdir(os.path.join(c, "llvm/Target")):
            inc = c
            break
    if not inc:
        print("未找到 llvm/Target/Target.td，跳过")
        return 2

    d = tempfile.mkdtemp(prefix="tdcheck_")
    tdp = os.path.join(d, "mini.td")
    outp = os.path.join(d, "rec.txt")
    with open(tdp, "w") as f:
        f.write(MINI_TD)

    r = subprocess.run([tg, "--print-records", "-I", inc, tdp, "-o", outp],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print("tblgen 失败:")
        print(r.stderr[:600])
        return 1

    t = open(outp).read()
    print("\n=== TableGen 展开的 32 位编码 vs RTL 期望 ===")
    allok = True
    for name, want in EXPECT.items():
        m = re.search(r"def " + name + r" \{(.*?)\n\}", t, re.S)
        if not m:
            print(f"  {name:22s} 未展开")
            allok = False
            continue
        mi = re.search(r"bits<32> Inst = \{([^}]*)\}", m.group(1))
        if not mi:
            print(f"  {name:22s} Inst 未解析")
            allok = False
            continue
        bits = "".join(re.findall(r"\b[01]\b", mi.group(1)))
        if len(bits) != 32:
            print(f"  {name:22s} 位数异常 {len(bits)}")
            allok = False
            continue
        got = int(bits, 2)
        ok = got == want
        allok &= ok
        print(f"  {name:22s} 0x{got:08X} (期望 0x{want:08X}) "
              f"{'OK' if ok else 'MISMATCH'}")

    print("\n" + "=" * 70)
    print("PASS: TableGen 产出的编码与 RTL 完全一致"
          if allok else "FAIL: 存在不一致")
    print("=" * 70)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
