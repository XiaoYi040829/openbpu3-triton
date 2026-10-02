//===-- RISCVInstrInfoSIMT.td - PolarisGPU RV-SIMT 扩展 ----*- tablegen -*-===//
//
// LLVM RISCV 后端补丁：注册 PolarisGPU 的 RV-SIMT
// 自定义控制指令与 SIMT CSR，使 LLVM 的汇编器/反汇编器能识别它们。
//
// 编码全部取自 PolarisGPU RTL 源码，并已用 GNU as 交叉验证（逐位一致）：
//   src/main/scala/nutcore/isa/Priviledged.scala:32-36   5 条 warp 控制
//   src/main/scala/nutcore/isa/RVI.scala:65              TID
//   src/main/scala/nutcore/backend/fu/SIMD_CSR.scala:116-127  SIMT CSR
//
// 用法（打进 LLVM 源码树）：
//   1. 放到 llvm/lib/Target/RISCV/RISCVInstrInfoSIMT.td
//   2. 在 llvm/lib/Target/RISCV/RISCVInstrInfo.td 末尾加：
//        include "RISCVInstrInfoSIMT.td"
//   3. 在 RISCVFeatures.td 里加 FeatureVendorXPolarisSIMT（见下方；
//      第 2、3 步即 llvm-10dc3a8e.patch）
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 子目标特性
//===----------------------------------------------------------------------===//
// 放入 RISCVFeatures.td：
//
//   def FeatureVendorXPolarisSIMT
//       : RISCVExtension<"xpolarissimt", 1, 0,
//                        "'XPolarisSIMT' (PolarisGPU RV-SIMT)">;
//   def HasVendorXPolarisSIMT
//       : Predicate<"Subtarget->hasVendorXPolarisSIMT()">,
//         AssemblerPredicate<(all_of FeatureVendorXPolarisSIMT),
//                            "'XPolarisSIMT' (PolarisGPU RV-SIMT)">;

//===----------------------------------------------------------------------===//
// 指令格式
//===----------------------------------------------------------------------===//
// warp 控制指令复用 I 型编码，opcode = 0x7B (custom-3)：
//     imm[11:0] | rs1 | funct3=000 | rd | 1111011
// imm 字段是 warp 操作码（0x00B..0x00F），与 SIMD_CSR.scala 的
// START_WARP/END_WARP/SYN_WARP 常量同值（该 imm 会被 CSR 单元当作
// 内部选择码：SIMD_CSR.scala:372 `addr = src2(16,5)`）。

class RVInstSIMT<bits<12> wop, string opcodestr>
    : RVInstI<0b000, OPC_CUSTOM_3, (outs), (ins), opcodestr, ""> {
  let imm12 = wop;
  let rs1 = 0;
  let rd = 0;
  let hasSideEffects = 1;
  let mayLoad = 0;
  let mayStore = 0;
}

// 带寄存器操作数的变体（SYN_WARP 的 barrier 计数由 rs1/rd 字段拼成，
// 见 SIMD_CSR.scala:806 `Cat(instr(19,15), instr(11,7))`）
class RVInstSIMTReg<bits<12> wop, string opcodestr>
    : RVInstI<0b000, OPC_CUSTOM_3, (outs), (ins GPR:$rs1), opcodestr, "$rs1"> {
  let imm12 = wop;
  let rd = 0;
  let hasSideEffects = 1;
  // mayLoad/mayStore 必须显式给出: 该指令没有 SelectionDAG pattern,
  // TableGen 无法推断, 否则报 "Can't infer mayStore from patterns"。
  let mayLoad = 0;
  let mayStore = 0;
}

//===----------------------------------------------------------------------===//
// RV-SIMT warp 控制指令 (Priviledged.scala:32-36)
//===----------------------------------------------------------------------===//
let Predicates = [HasVendorXPolarisSIMT] in {

// JOIN — SIMT 分支重汇聚。编码 0x00B0007B
def POLARIS_JOIN       : RVInstSIMT<0x00B, "polaris.join">;

// START_WARP — 启动 warp（经 BoringUtils 广播给 CTA 调度器，
// 见 SIMD_CSR.scala:794 `${namespace}_StartWarp`）。编码 0x00C0007B
def POLARIS_START_WARP : RVInstSIMT<0x00C, "polaris.start_warp">;

// SYN_WARP — SM 内 warp 同步（barrier）。编码 0x00D0007B
// 注意：仅 SM 内有效，不支持跨 SM barrier。
def POLARIS_SYN_WARP   : RVInstSIMT<0x00D, "polaris.syn_warp">;

// END_WARP — 结束 warp。编码 0x00E0007B
def POLARIS_END_WARP   : RVInstSIMT<0x00E, "polaris.end_warp">;

// SYN_WARP_REG — barrier 计数来自寄存器。编码 0x00F0007B + rs1
def POLARIS_SYN_WARP_REG : RVInstSIMTReg<0x00F, "polaris.syn_warp_reg">;

//===----------------------------------------------------------------------===//
// TID — 每 lane 写入自己的 lane 号 (RVI.scala:65, ALU.scala:246)
//   funct7 = 0x60, funct3 = 000, opcode = 0110011 (OP)
//   rs1 = rs2 = 0，rd 为目标寄存器
//===----------------------------------------------------------------------===//
def POLARIS_TID : RVInstR<0b1100000, 0b000, OPC_OP, (outs GPR:$rd),
                          (ins), "polaris.tid", "$rd"> {
  let rs1 = 0;
  let rs2 = 0;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
}

} // Predicates = [HasVendorXPolarisSIMT]

//===----------------------------------------------------------------------===//
// 与 .insn 写法的对应
//===----------------------------------------------------------------------===//
// 未打补丁的汇编器需写成
//     .insn r 0x7B, 0x0, 0x0, x0, x0, a2      // START_WARP
// 打了本补丁后可直接写 polaris.start_warp，两者编码相同。

//===----------------------------------------------------------------------===//
// SIMT CSR (SIMD_CSR.scala:116-127)
//===----------------------------------------------------------------------===//
// 本补丁未注册这些 CSR 的名称，汇编时用 CSR 号（如 csrr a0, 0x00C）。
// 如需名称，可在 RISCVSystemOperands.td 加入：
//
//   def : SysReg<"simtmask",      0x00A>;  // SIMT 活跃线程掩码
//   def : SysReg<"simtrpc",       0x00B>;  // SIMT 重汇聚 PC
//   def : SysReg<"warp_id",       0x00C>;  // warp 号（只读）
//   def : SysReg<"thread_num",    0x00D>;  // 线程数
//   def : SysReg<"thread_baseid", 0x00E>;  // 线程基 ID
//
// 注意 0x00C/0x00D/0x00E 是双重身份：作为 CSR 时是 Warp_id/Thread_NUM/
// Thread_baseid；作为 0x7B 指令的 imm 时是 START/SYN/END_WARP 的选择码。
// 硬件靠 WarpType 区分（SIMD_CSR.scala:387）。
