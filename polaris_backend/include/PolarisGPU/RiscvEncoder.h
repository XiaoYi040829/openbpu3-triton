//===- RiscvEncoder.h - RV64 + RV-SIMT 指令编码器 -------------*- C++ -*-===//
//
// 发射标准 RISC-V 编码与 RV-SIMT 扩展指令。RV-SIMT 编码值全部取自
// PolarisGPU RTL 源码，并与 GNU as、打补丁的 llvm-mc 的编码逐位一致：
//   START_WARP   0x00C0007B   nutcore/isa/Priviledged.scala:33
//   END_WARP     0x00E0007B   Priviledged.scala:35
//   JOIN         0x00B0007B   Priviledged.scala:32
//   SYN_WARP     0x00D0007B   Priviledged.scala:34  (rs1 可变)
//   SYN_WARP_REG 0x00F0007B   Priviledged.scala:36  (rs1 可变)
//   TID          funct7=0x60, funct3=0, opcode=0x33   isa/RVI.scala:65
//
// SIMT CSR (nutcore/backend/fu/SIMD_CSR.scala:116-127)：
//   0x00A SIMTMASK  0x00B SIMTRPC  0x00C WARP_ID
//   0x00D THREAD_NUM  0x00E THREAD_BASEID
//
//===----------------------------------------------------------------------===//
#ifndef POLARISGPU_RISCV_ENCODER_H
#define POLARISGPU_RISCV_ENCODER_H

#include <cstdint>
#include <vector>

namespace polaris {
namespace rv {

// ---- 基础 opcode (RISC-V 规范) ----
enum : uint32_t {
  OPC_LOAD   = 0x03,
  OPC_OPIMM  = 0x13,
  OPC_AUIPC  = 0x17,
  OPC_STORE  = 0x23,
  OPC_OP     = 0x33,
  OPC_LUI    = 0x37,
  OPC_OPIMM32= 0x1B,
  OPC_OP32   = 0x3B,
  OPC_BRANCH = 0x63,
  OPC_JALR   = 0x67,
  OPC_JAL    = 0x6F,
  OPC_SYSTEM = 0x73,
  OPC_WARP   = 0x7B,  // PolarisGPU 自定义 warp 控制 (Priviledged.scala)
};

// ---- SIMT CSR 地址 (SIMD_CSR.scala:116-127) ----
enum : uint32_t {
  CSR_SIMTMASK      = 0x00A,
  CSR_SIMTRPC       = 0x00B,
  CSR_WARP_ID       = 0x00C,
  CSR_THREAD_NUM    = 0x00D,
  CSR_THREAD_BASEID = 0x00E,
};

// ---- 常用寄存器号 ----
enum : uint32_t { X0 = 0, SP = 2, T0 = 5, T1 = 6, T2 = 7, A0 = 10 };

// ---- 指令格式编码 ----
inline uint32_t R(uint32_t f7, uint32_t rs2, uint32_t rs1, uint32_t f3,
                  uint32_t rd, uint32_t op) {
  return ((f7 & 0x7F) << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) |
         ((f3 & 0x7) << 12) | ((rd & 0x1F) << 7) | (op & 0x7F);
}

inline uint32_t I(uint32_t imm, uint32_t rs1, uint32_t f3, uint32_t rd,
                  uint32_t op) {
  return ((imm & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) | ((f3 & 0x7) << 12) |
         ((rd & 0x1F) << 7) | (op & 0x7F);
}

inline uint32_t S(uint32_t imm, uint32_t rs2, uint32_t rs1, uint32_t f3,
                  uint32_t op) {
  return (((imm >> 5) & 0x7F) << 25) | ((rs2 & 0x1F) << 20) |
         ((rs1 & 0x1F) << 15) | ((f3 & 0x7) << 12) | ((imm & 0x1F) << 7) |
         (op & 0x7F);
}

inline uint32_t U(uint32_t imm20, uint32_t rd, uint32_t op) {
  return ((imm20 & 0xFFFFF) << 12) | ((rd & 0x1F) << 7) | (op & 0x7F);
}

// ---- RV-SIMT warp 控制 (Priviledged.scala:32-36) ----
// 格式: imm[11:0]_rs1_000_rd_1111011; imm 即 warp 操作码
inline uint32_t warpCtrl(uint32_t code, uint32_t rs1 = 0, uint32_t rd = 0) {
  return I(code, rs1, 0, rd, OPC_WARP);
}
inline uint32_t JOIN()        { return warpCtrl(0x00B); }
inline uint32_t START_WARP()  { return warpCtrl(0x00C); }
inline uint32_t END_WARP()    { return warpCtrl(0x00E); }
// SYN_WARP: barrier 计数由 rs1/rd 字段拼成 (SIMD_CSR.scala:806)
inline uint32_t SYN_WARP(uint32_t cnt = 0) {
  return warpCtrl(0x00D, (cnt >> 5) & 0x1F, cnt & 0x1F);
}
inline uint32_t SYN_WARP_REG(uint32_t rs1) { return warpCtrl(0x00F, rs1); }

// ---- TID: 每 lane 写入自己的 lane 号 (RVI.scala:65, ALU.scala:246) ----
inline uint32_t TID(uint32_t rd) { return R(0x60, 0, 0, 0, rd, OPC_OP); }

// ---- CSR 访问 (RVZicsr) ----
inline uint32_t CSRRW(uint32_t rd, uint32_t csr, uint32_t rs1) {
  return I(csr, rs1, 0x1, rd, OPC_SYSTEM);
}
inline uint32_t CSRRS(uint32_t rd, uint32_t csr, uint32_t rs1) {
  return I(csr, rs1, 0x2, rd, OPC_SYSTEM);
}
inline uint32_t CSRW(uint32_t csr, uint32_t rs1) { return CSRRW(X0, csr, rs1); }
inline uint32_t CSRR(uint32_t rd, uint32_t csr) { return CSRRS(rd, csr, X0); }

// ---- 常用标量指令 ----
inline uint32_t ADDI(uint32_t rd, uint32_t rs1, int32_t imm) {
  return I((uint32_t)imm, rs1, 0x0, rd, OPC_OPIMM);
}
inline uint32_t ADD(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x00, rs2, rs1, 0x0, rd, OPC_OP);
}
inline uint32_t SUB(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x20, rs2, rs1, 0x0, rd, OPC_OP);
}
inline uint32_t MUL(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x01, rs2, rs1, 0x0, rd, OPC_OP);
}
inline uint32_t SLLI(uint32_t rd, uint32_t rs1, uint32_t sh) {
  return I(sh & 0x3F, rs1, 0x1, rd, OPC_OPIMM);
}
// 整数比较 (置 0/1)
inline uint32_t SLT(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x00, rs2, rs1, 0x2, rd, OPC_OP);
}
inline uint32_t SLTU(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x00, rs2, rs1, 0x3, rd, OPC_OP);
}

// 位运算 (用于无分支 select 的掩码算术)
inline uint32_t XOR(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x00, rs2, rs1, 0x4, rd, OPC_OP);
}
inline uint32_t AND(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x00, rs2, rs1, 0x7, rd, OPC_OP);
}
inline uint32_t OR(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x00, rs2, rs1, 0x6, rd, OPC_OP);
}
inline uint32_t LUI(uint32_t rd, uint32_t imm20) { return U(imm20, rd, OPC_LUI); }
inline uint32_t LW(uint32_t rd, uint32_t rs1, int32_t off) {
  return I((uint32_t)off, rs1, 0x2, rd, OPC_LOAD);
}
inline uint32_t LD(uint32_t rd, uint32_t rs1, int32_t off) {
  return I((uint32_t)off, rs1, 0x3, rd, OPC_LOAD);
}
inline uint32_t SW(uint32_t rs2, uint32_t rs1, int32_t off) {
  return S((uint32_t)off, rs2, rs1, 0x2, OPC_STORE);
}
inline uint32_t SD(uint32_t rs2, uint32_t rs1, int32_t off) {
  return S((uint32_t)off, rs2, rs1, 0x3, OPC_STORE);
}
// ---- B 型（条件分支）----
// 立即数是相对 pc 的字节偏移，必须偶数；编码把它打散成
//   imm[12] imm[10:5] | rs2 rs1 funct3 | imm[4:1] imm[11] | opcode
//
// 核支持全部 8 条分支/跳转指令（BEQ/BNE/BLT/BGE/BLTU/BGEU/JAL/JALR）。
// 当前 codegen 不发射分支：循环在 codegen 前完全展开，条件由 tl.where
// 降级为掩码合并。这些编码器供后续生成真循环使用。
inline uint32_t B(int32_t off, uint32_t rs2, uint32_t rs1,
                  uint32_t funct3, uint32_t opcode) {
  uint32_t u = (uint32_t)off;
  uint32_t imm12   = (u >> 12) & 0x1;
  uint32_t imm10_5 = (u >> 5) & 0x3F;
  uint32_t imm4_1  = (u >> 1) & 0xF;
  uint32_t imm11   = (u >> 11) & 0x1;
  return (imm12 << 31) | (imm10_5 << 25) | ((rs2 & 0x1F) << 20) |
         ((rs1 & 0x1F) << 15) | ((funct3 & 0x7) << 12) |
         (imm4_1 << 8) | (imm11 << 7) | (opcode & 0x7F);
}
inline uint32_t BEQ(uint32_t rs1, uint32_t rs2, int32_t off) {
  return B(off, rs2, rs1, 0x0, OPC_BRANCH);
}
inline uint32_t BNE(uint32_t rs1, uint32_t rs2, int32_t off) {
  return B(off, rs2, rs1, 0x1, OPC_BRANCH);
}
inline uint32_t BLT(uint32_t rs1, uint32_t rs2, int32_t off) {
  return B(off, rs2, rs1, 0x4, OPC_BRANCH);
}
inline uint32_t BGE(uint32_t rs1, uint32_t rs2, int32_t off) {
  return B(off, rs2, rs1, 0x5, OPC_BRANCH);
}

// ---- RVF 单精度浮点 (isa/RVF.scala:42-52) ----
enum : uint32_t { OPC_FP = 0x53, OPC_FLOAD = 0x07, OPC_FSTORE = 0x27 };
// rm=7 (dynamic rounding)
inline uint32_t FADD_S(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x00, rs2, rs1, 0x7, rd, OPC_FP);
}
inline uint32_t FSUB_S(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x04, rs2, rs1, 0x7, rd, OPC_FP);
}
inline uint32_t FMUL_S(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x08, rs2, rs1, 0x7, rd, OPC_FP);
}
inline uint32_t FDIV_S(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x0C, rs2, rs1, 0x7, rd, OPC_FP);
}
inline uint32_t FSQRT_S(uint32_t rd, uint32_t rs1) {
  return R(0x2C, 0, rs1, 0x7, rd, OPC_FP);
}
inline uint32_t FMIN_S(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x14, rs2, rs1, 0x0, rd, OPC_FP);
}
inline uint32_t FMAX_S(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x14, rs2, rs1, 0x1, rd, OPC_FP);
}
inline uint32_t FLW(uint32_t rd, uint32_t rs1, int32_t off) {
  return I((uint32_t)off, rs1, 0x2, rd, OPC_FLOAD);
}
inline uint32_t FSW(uint32_t rs2, uint32_t rs1, int32_t off) {
  return S((uint32_t)off, rs2, rs1, 0x2, OPC_FSTORE);
}
// 整浮点互转 (RVF.scala:58,65)：FCVT_S_W 为 rm=7（动态），FCVT_W_S 为 rm=1（截断）。
// codegen 的 sitofp/fptosi 降级直接编码显式舍入模式（RNE / RTZ）。
inline uint32_t FCVT_S_W(uint32_t rd, uint32_t rs1) {   // int -> float
  return R(0x68, 0x00, rs1, 0x7, rd, OPC_FP);
}
inline uint32_t FCVT_W_S(uint32_t rd, uint32_t rs1) {   // float -> int
  return R(0x60, 0x00, rs1, 0x1, rd, OPC_FP);
}
// 符号注入 (RVF.scala:53-54): fsgnj.s rd,rs,rs = fmv.s; fsgnjn.s rd,rs,rs = fneg.s
inline uint32_t FSGNJ_S(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x10, rs2, rs1, 0x0, rd, OPC_FP);
}
inline uint32_t FSGNJN_S(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x10, rs2, rs1, 0x1, rd, OPC_FP);
}
// fsgnjx.s rd,rs,rs = fabs.s (RVF.scala:55)
inline uint32_t FSGNJX_S(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x10, rs2, rs1, 0x2, rd, OPC_FP);
}
// 整数/浮点寄存器间按位搬运 (RVF.scala:60,67) —— 不做数值转换
// 用途: RV 无浮点立即数, f32 常量须先把 bit 模式装进整数寄存器再 fmv.w.x
inline uint32_t FMV_W_X(uint32_t rd, uint32_t rs1) {
  return R(0x78, 0x00, rs1, 0x0, rd, OPC_FP);
}
inline uint32_t FMV_X_W(uint32_t rd, uint32_t rs1) {
  return R(0x70, 0x00, rs1, 0x0, rd, OPC_FP);
}

// 浮点比较, 结果写整数寄存器 (RVF.scala:61-63)
inline uint32_t FLT_S(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x50, rs2, rs1, 0x1, rd, OPC_FP);
}
inline uint32_t FLE_S(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x50, rs2, rs1, 0x0, rd, OPC_FP);
}
inline uint32_t FEQ_S(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return R(0x50, rs2, rs1, 0x2, rd, OPC_FP);
}

// 载入 32 位常量: lui + addi（addi 立即数符号扩展，需在高位补偿）。
// RV64 上 lui 会把 bit31 符号扩展到 64 位, 故 lui 0x80002 得到的是
// 0xFFFFFFFF80002000 而不是 0x80002000 —— 地址落到 RAM 外。对 bit31=1 的
// 地址(即 0x8xxxxxxx)改用 "先装高 20 位再 slli 12" 的方式, 让高 32 位保持 0。
// 因此结果是零扩展的；i32 负数常量需要符号扩展，codegen 对其另补 sext.w。
inline void LI32(std::vector<uint32_t> &out, uint32_t rd, uint32_t v) {
  if (v & 0x80000000u) {
    uint32_t hi20 = v >> 12;          // < 2^20, 装载时无符号位问题
    uint32_t lo12 = v & 0xFFFu;
    // ADDI 的立即数是 12 位"有符号": lo12 >= 0x800 时会被当负数,
    // 结果少了 0x1000, 必须在高位预先 +1 补偿。
    if (lo12 & 0x800u) hi20 += 1;
    LI32(out, rd, hi20);
    out.push_back(SLLI(rd, rd, 12));
    if (lo12) out.push_back(ADDI(rd, rd, (int32_t)(int16_t)((lo12 ^ 0x800u) - 0x800u)));
    return;
  }
  uint32_t hi = (v + 0x800) >> 12;
  int32_t lo = (int32_t)(v - (hi << 12));
  if (hi) {
    out.push_back(LUI(rd, hi));
    if (lo) out.push_back(ADDI(rd, rd, lo));
  } else {
    out.push_back(ADDI(rd, X0, (int32_t)v));
  }
}

// 指令缓冲: 累积真 RISC-V 字, 序列化为 little-endian .bin
class RvBuffer {
public:
  void emit(uint32_t w) { words_.push_back(w); }
  void emitAll(const std::vector<uint32_t> &ws) {
    for (uint32_t w : ws) words_.push_back(w);
  }
  size_t size() const { return words_.size(); }
  const std::vector<uint32_t> &words() const { return words_; }
  /// 回填已发射的指令（供生成分支时回填目标偏移；当前 codegen 未使用）。
  void patch(size_t idx, uint32_t w) {
    if (idx < words_.size()) words_[idx] = w;
  }
  uint32_t at(size_t idx) const {
    return idx < words_.size() ? words_[idx] : 0u;
  }
  std::vector<uint8_t> toBytes() const {
    std::vector<uint8_t> b;
    b.reserve(words_.size() * 4);
    for (uint32_t w : words_) {
      b.push_back(w & 0xFF);
      b.push_back((w >> 8) & 0xFF);
      b.push_back((w >> 16) & 0xFF);
      b.push_back((w >> 24) & 0xFF);
    }
    return b;
  }

private:
  std::vector<uint32_t> words_;
};

} // namespace rv
} // namespace polaris

#endif // POLARISGPU_RISCV_ENCODER_H
