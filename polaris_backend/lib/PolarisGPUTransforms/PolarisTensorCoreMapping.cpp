//===- PolarisTensorCoreMapping.cpp ---------------------------*- C++ -*-===//
//
// PolarisGPU TensorCore 映射分析 Pass
//
// 遍历 TritonGPU IR 中的 tt.dot 操作，分析其矩阵维度能否映射到 PolarisGPU
// 的 4x4 TensorCore（MAT4X4，opcode 0x2B）。对每个 tt.dot：
//   - 读取 M/N/K 维度
//   - 判断 M%4==0 && N%4==0 && K%4==0 是否成立（可整切为 4x4 tile）
//   - 用 attribute "polaris.tensorcore" = true/false 标注
//   - 标注所需 MAT4X4 tile 数 "polaris.mat4x4_tiles"
//
// 本 pass 只做分析与标注，不改变计算。RV64 + RV-SIMT codegen 不发射
// MAT4X4，含 tt.dot 的 kernel 在 codegen 阶段被拒绝编译。
//
//===----------------------------------------------------------------------===//
#include "PolarisGPU/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

namespace {

struct PolarisTensorCoreMappingPass
    : public PassWrapper<PolarisTensorCoreMappingPass,
                         OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PolarisTensorCoreMappingPass)

  StringRef getArgument() const final {
    return "polaris-tensorcore-mapping";
  }
  StringRef getDescription() const final {
    return "Map tt.dot ops to PolarisGPU 4x4 TensorCore (MAT4X4)";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    Builder builder(mod.getContext());
    int64_t numDot = 0, numMappable = 0, totalTiles = 0;

    mod.walk([&](triton::DotOp dotOp) {
      ++numDot;
      // 取 A (MxK) 与结果 D (MxN) 的张量类型
      auto aType =
          dyn_cast<RankedTensorType>(dotOp.getA().getType());
      auto dType =
          dyn_cast<RankedTensorType>(dotOp.getResult().getType());
      if (!aType || !dType || aType.getRank() != 2 || dType.getRank() != 2) {
        dotOp->setAttr("polaris.tensorcore",
                       builder.getBoolAttr(false));
        return;
      }
      int64_t M = dType.getShape()[0];
      int64_t N = dType.getShape()[1];
      int64_t K = aType.getShape()[1];

      bool mappable = (M % 4 == 0) && (N % 4 == 0) && (K % 4 == 0);
      dotOp->setAttr("polaris.tensorcore", builder.getBoolAttr(mappable));

      if (mappable) {
        ++numMappable;
        int64_t tiles = (M / 4) * (N / 4) * (K / 4);
        totalTiles += tiles;
        dotOp->setAttr("polaris.mat4x4_tiles",
                       builder.getI64IntegerAttr(tiles));
        dotOp->setAttr("polaris.mnk",
                       builder.getDenseI64ArrayAttr({M, N, K}));
      }
    });

    // 在 module 上记录统计（供 Python 侧读取）
    mod->setAttr("polaris.num_dot", builder.getI64IntegerAttr(numDot));
    mod->setAttr("polaris.num_tensorcore_mappable",
                 builder.getI64IntegerAttr(numMappable));
    mod->setAttr("polaris.total_mat4x4_tiles",
                 builder.getI64IntegerAttr(totalTiles));
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace polaris {

std::unique_ptr<mlir::Pass> createPolarisTensorCoreMappingPass() {
  return std::make_unique<PolarisTensorCoreMappingPass>();
}

} // namespace polaris
} // namespace triton
} // namespace mlir
