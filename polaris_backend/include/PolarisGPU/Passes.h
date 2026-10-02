//===- Passes.h - PolarisGPU MLIR Passes ----------------------*- C++ -*-===//
//
// PolarisGPU 后端的 MLIR pass 声明。
// 这些 pass 在 TritonGPU IR 层做面向 PolarisGPU ISA 的分析与变换。
//
//===----------------------------------------------------------------------===//
#ifndef POLARISGPU_TRANSFORMS_PASSES_H
#define POLARISGPU_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace triton {
namespace polaris {

// 创建 PolarisGPU TensorCore 映射分析 pass：
// 遍历 TritonGPU IR 中的 tt.dot，分析其能否映射到 PolarisGPU 4x4 TensorCore
// (MAT4X4, opcode 0x2B)，并以 attribute 标注映射决策。
std::unique_ptr<mlir::Pass> createPolarisTensorCoreMappingPass();

// 创建 RV64 + RV-SIMT codegen pass：发射标准 RISC-V 编码与 RV-SIMT 扩展指令，
// 产出的 .bin 可由 cycle-accurate 模拟器与 QV19P 硬件直接执行。
// 结果挂在 polaris.riscv.words。
std::unique_ptr<mlir::Pass> createPolarisRiscvCodegenPass();

} // namespace polaris
} // namespace triton
} // namespace mlir

#endif // POLARISGPU_TRANSFORMS_PASSES_H
