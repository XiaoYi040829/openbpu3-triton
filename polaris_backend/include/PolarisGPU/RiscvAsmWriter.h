//===- RiscvAsmWriter.h - 把 RV64+RV-SIMT 指令字还原为 .S 汇编 --*- C++ -*-===//
//
// CodeGen 输出 .S 汇编（含 RV-SIMT 扩展指令），用 GNU binutils 汇编后与
// codegen 直出的二进制逐字节相同：
//     riscv64-linux-gnu-as -march=rv64imaf_zicsr -mno-relax kernel.S -o kernel.o
//     riscv64-linux-gnu-objcopy -O binary -j .text kernel.o kernel.bin
// -march 不能带 c：压缩指令会改变指令长度，产物与直出二进制不一致。
//
// 自定义 warp 控制指令（opcode 0x7B）与 TID 用 GNU as 的 `.insn` 通用编码
// 指示符表达，不依赖打过补丁的汇编器，标准 GNU as 即可汇编。
//
//===----------------------------------------------------------------------===//
#ifndef POLARISGPU_RISCV_ASM_WRITER_H
#define POLARISGPU_RISCV_ASM_WRITER_H

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace polaris {
namespace rv {

inline const char *xreg(uint32_t r) {
  static const char *n[32] = {
      "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
      "s0",   "s1", "a0", "a1", "a2", "a3", "a4", "a5",
      "a6",   "a7", "s2", "s3", "s4", "s5", "s6", "s7",
      "s8",   "s9", "s10", "s11", "t3", "t4", "t5", "t6"};
  return n[r & 31];
}

inline std::string freg(uint32_t r) { return "f" + std::to_string(r & 31); }

// 把一条 32 位指令字还原为汇编文本。
// 无法用标准助记符表达的（自定义 opcode）回落到 .insn 形式。
inline std::string disasmOne(uint32_t w) {
  char buf[128];
  uint32_t op = w & 0x7F;
  uint32_t rd = (w >> 7) & 0x1F;
  uint32_t f3 = (w >> 12) & 0x7;
  uint32_t rs1 = (w >> 15) & 0x1F;
  uint32_t rs2 = (w >> 20) & 0x1F;
  uint32_t f7 = (w >> 25) & 0x7F;
  int32_t immI = (int32_t)(w) >> 20;               // 符号扩展 12 位
  int32_t immS = (int32_t)(((w >> 25) << 5) | ((w >> 7) & 0x1F));
  if (immS & 0x800) immS -= 0x1000;

  switch (op) {
  case 0x37:  // LUI
    snprintf(buf, sizeof(buf), "lui %s, 0x%x", xreg(rd), (w >> 12) & 0xFFFFF);
    return buf;
  case 0x13:  // OP-IMM
    if (f3 == 0) { snprintf(buf, sizeof(buf), "addi %s, %s, %d", xreg(rd), xreg(rs1), immI); return buf; }
    if (f3 == 1) { snprintf(buf, sizeof(buf), "slli %s, %s, %u", xreg(rd), xreg(rs1), rs2 | ((f7 & 1) << 5)); return buf; }
    if (f3 == 5) {
      bool arith = (f7 & 0x20) != 0;
      snprintf(buf, sizeof(buf), "%s %s, %s, %u", arith ? "srai" : "srli",
               xreg(rd), xreg(rs1), rs2 | ((f7 & 1) << 5));
      return buf;
    }
    if (f3 == 7) { snprintf(buf, sizeof(buf), "andi %s, %s, %d", xreg(rd), xreg(rs1), immI); return buf; }
    break;
  case 0x33:  // OP
    if (f7 == 0x60 && f3 == 0) {   // TID (RVI.scala:65) —— 自定义
      snprintf(buf, sizeof(buf), ".insn r 0x33, 0x0, 0x60, %s, zero, zero", xreg(rd));
      return buf;
    }
    if (f7 == 0x00 && f3 == 0) { snprintf(buf, sizeof(buf), "add %s, %s, %s", xreg(rd), xreg(rs1), xreg(rs2)); return buf; }
    if (f7 == 0x20 && f3 == 0) { snprintf(buf, sizeof(buf), "sub %s, %s, %s", xreg(rd), xreg(rs1), xreg(rs2)); return buf; }
    if (f7 == 0x01 && f3 == 0) { snprintf(buf, sizeof(buf), "mul %s, %s, %s", xreg(rd), xreg(rs1), xreg(rs2)); return buf; }
    if (f7 == 0x00 && f3 == 2) { snprintf(buf, sizeof(buf), "slt %s, %s, %s", xreg(rd), xreg(rs1), xreg(rs2)); return buf; }
    if (f7 == 0x00 && f3 == 4) { snprintf(buf, sizeof(buf), "xor %s, %s, %s", xreg(rd), xreg(rs1), xreg(rs2)); return buf; }
    if (f7 == 0x00 && f3 == 6) { snprintf(buf, sizeof(buf), "or %s, %s, %s", xreg(rd), xreg(rs1), xreg(rs2)); return buf; }
    if (f7 == 0x00 && f3 == 7) { snprintf(buf, sizeof(buf), "and %s, %s, %s", xreg(rd), xreg(rs1), xreg(rs2)); return buf; }
    break;
  case 0x03:  // LOAD
    if (f3 == 2) { snprintf(buf, sizeof(buf), "lw %s, %d(%s)", xreg(rd), immI, xreg(rs1)); return buf; }
    if (f3 == 3) { snprintf(buf, sizeof(buf), "ld %s, %d(%s)", xreg(rd), immI, xreg(rs1)); return buf; }
    break;
  case 0x23:  // STORE
    if (f3 == 2) { snprintf(buf, sizeof(buf), "sw %s, %d(%s)", xreg(rs2), immS, xreg(rs1)); return buf; }
    if (f3 == 3) { snprintf(buf, sizeof(buf), "sd %s, %d(%s)", xreg(rs2), immS, xreg(rs1)); return buf; }
    if (f3 == 0) { snprintf(buf, sizeof(buf), "sb %s, %d(%s)", xreg(rs2), immS, xreg(rs1)); return buf; }
    break;
  case 0x07:  // FLW
    if (f3 == 2) { snprintf(buf, sizeof(buf), "flw %s, %d(%s)", freg(rd).c_str(), immI, xreg(rs1)); return buf; }
    break;
  case 0x27:  // FSW
    if (f3 == 2) { snprintf(buf, sizeof(buf), "fsw %s, %d(%s)", freg(rs2).c_str(), immS, xreg(rs1)); return buf; }
    break;
  case 0x73: { // SYSTEM (CSR)
    uint32_t csr = (w >> 20) & 0xFFF;
    if (f3 == 1) {
      if (rd == 0) { snprintf(buf, sizeof(buf), "csrw 0x%x, %s", csr, xreg(rs1)); return buf; }
      snprintf(buf, sizeof(buf), "csrrw %s, 0x%x, %s", xreg(rd), csr, xreg(rs1)); return buf;
    }
    if (f3 == 2) {
      if (rs1 == 0) { snprintf(buf, sizeof(buf), "csrr %s, 0x%x", xreg(rd), csr); return buf; }
      snprintf(buf, sizeof(buf), "csrrs %s, 0x%x, %s", xreg(rd), csr, xreg(rs1)); return buf;
    }
    break;
  }
  case 0x53: { // 浮点
    const char *rm = "";
    if (f7 == 0x00) { snprintf(buf, sizeof(buf), "fadd.s %s, %s, %s", freg(rd).c_str(), freg(rs1).c_str(), freg(rs2).c_str()); return buf; }
    if (f7 == 0x04) { snprintf(buf, sizeof(buf), "fsub.s %s, %s, %s", freg(rd).c_str(), freg(rs1).c_str(), freg(rs2).c_str()); return buf; }
    if (f7 == 0x08) { snprintf(buf, sizeof(buf), "fmul.s %s, %s, %s", freg(rd).c_str(), freg(rs1).c_str(), freg(rs2).c_str()); return buf; }
    if (f7 == 0x0C) { snprintf(buf, sizeof(buf), "fdiv.s %s, %s, %s", freg(rd).c_str(), freg(rs1).c_str(), freg(rs2).c_str()); return buf; }
    if (f7 == 0x2C) { snprintf(buf, sizeof(buf), "fsqrt.s %s, %s", freg(rd).c_str(), freg(rs1).c_str()); return buf; }
    if (f7 == 0x10 && f3 == 0) { snprintf(buf, sizeof(buf), "fsgnj.s %s, %s, %s", freg(rd).c_str(), freg(rs1).c_str(), freg(rs2).c_str()); return buf; }
    if (f7 == 0x10 && f3 == 1) { snprintf(buf, sizeof(buf), "fsgnjn.s %s, %s, %s", freg(rd).c_str(), freg(rs1).c_str(), freg(rs2).c_str()); return buf; }
    if (f7 == 0x14 && f3 == 0) { snprintf(buf, sizeof(buf), "fmin.s %s, %s, %s", freg(rd).c_str(), freg(rs1).c_str(), freg(rs2).c_str()); return buf; }
    if (f7 == 0x14 && f3 == 1) { snprintf(buf, sizeof(buf), "fmax.s %s, %s, %s", freg(rd).c_str(), freg(rs1).c_str(), freg(rs2).c_str()); return buf; }
    if (f7 == 0x50 && f3 == 0) { snprintf(buf, sizeof(buf), "fle.s %s, %s, %s", xreg(rd), freg(rs1).c_str(), freg(rs2).c_str()); return buf; }
    if (f7 == 0x50 && f3 == 1) { snprintf(buf, sizeof(buf), "flt.s %s, %s, %s", xreg(rd), freg(rs1).c_str(), freg(rs2).c_str()); return buf; }
    if (f7 == 0x50 && f3 == 2) { snprintf(buf, sizeof(buf), "feq.s %s, %s, %s", xreg(rd), freg(rs1).c_str(), freg(rs2).c_str()); return buf; }
    if (f7 == 0x78) { snprintf(buf, sizeof(buf), "fmv.w.x %s, %s", freg(rd).c_str(), xreg(rs1)); return buf; }
    if (f7 == 0x70) { snprintf(buf, sizeof(buf), "fmv.x.w %s, %s", xreg(rd), freg(rs1).c_str()); return buf; }
    if (f7 == 0x68) { snprintf(buf, sizeof(buf), "fcvt.s.w %s, %s", freg(rd).c_str(), xreg(rs1)); return buf; }
    if (f7 == 0x60) {
      // 必须显式带舍入模式: codegen 用 rtz(funct3=1) 做 float->int 截断,
      // 而 GNU as 的 "fcvt.w.s rd,rs" 默认 dyn(funct3=7), 会造成编码不一致。
      const char *rms = (f3 == 0x1) ? ", rtz" : (f3 == 0x0) ? ", rne"
                        : (f3 == 0x2) ? ", rdn" : (f3 == 0x3) ? ", rup"
                        : (f3 == 0x4) ? ", rmm" : "";
      snprintf(buf, sizeof(buf), "fcvt.w.s %s, %s%s", xreg(rd),
               freg(rs1).c_str(), rms);
      return buf;
    }
    (void)rm;
    break;
  }
  case 0x7B: { // PolarisGPU 自定义 warp 控制 (Priviledged.scala:32-36)
    uint32_t code = (w >> 20) & 0xFFF;
    const char *nm = "WARP?";
    const char *areg = "a1";
    switch (code) {
    case 0x00B: nm = "JOIN";         areg = "a1"; break;
    case 0x00C: nm = "START_WARP";   areg = "a2"; break;
    case 0x00D: nm = "SYN_WARP";     areg = "a3"; break;
    case 0x00E: nm = "END_WARP";     areg = "a4"; break;
    case 0x00F: nm = "SYN_WARP_REG"; areg = "a5"; break;
    }
    // 形如 .insn r 0x7B, 0x0, 0x0, x0, x0, aN
    snprintf(buf, sizeof(buf), ".insn r 0x7B, 0x0, 0x0, zero, zero, %s   # %s",
             areg, nm);
    return buf;
  }
  case 0x6F: { // JAL
    snprintf(buf, sizeof(buf), ".insn 4, 0x%08x   # jal", w);
    return buf;
  }
  case 0x63: { // BRANCH
    snprintf(buf, sizeof(buf), ".insn 4, 0x%08x   # branch", w);
    return buf;
  }
  }
  snprintf(buf, sizeof(buf), ".insn 4, 0x%08x", w);
  return buf;
}

// 产出完整 .S 文本（头部 START-WARP / 尾部 END-WARP 已由 codegen
// 在指令流里发射，这里只做文本化）
inline std::string toAsm(const std::vector<uint32_t> &words,
                         const std::string &kernelName) {
  std::string s;
  s += "# PolarisGPU / OpenBPU-3 kernel, generated by flagos polaris backend\n";
  s += "# 自定义 RV-SIMT 指令用 .insn 表达，标准 GNU as 即可汇编\n";
  s += "#   START_WARP 0x00C0007B  END_WARP 0x00E0007B  SYN_WARP 0x00D0007B\n";
  s += "#   JOIN       0x00B0007B  TID: funct7=0x60 opcode=0x33\n";
  s += "# 汇编: riscv64-linux-gnu-as -march=rv64imaf_zicsr -mno-relax kernel.S -o kernel.o\n";
  s += "#       riscv64-linux-gnu-objcopy -O binary -j .text kernel.o kernel.bin\n";
  s += "  .text\n";
  s += "  .globl " + kernelName + "\n";
  s += kernelName + ":\n";
  for (size_t i = 0; i < words.size(); ++i) {
    char pre[32];
    snprintf(pre, sizeof(pre), "  /* %04zu */ ", i * 4);
    s += pre;
    s += disasmOne(words[i]);
    s += "\n";
  }
  return s;
}

} // namespace rv
} // namespace polaris

#endif // POLARISGPU_RISCV_ASM_WRITER_H
