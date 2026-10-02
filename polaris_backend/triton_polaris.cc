//===- triton_polaris.cc - PolarisGPU backend pybind ----------*- C++ -*-===//
//
// 把 PolarisGPU 的 MLIR pass 通过 pybind11 注册到 libtriton 的
// polaris.passes 子模块，供 Python 侧 polaris 后端 add_stages 调用。
//
//===----------------------------------------------------------------------===//
#include "PolarisGPU/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_triton_polaris_passes(py::module &&m) {
  // 注册 PolarisGPU TensorCore 映射 pass
  m.def("add_tensorcore_mapping", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::polaris::createPolarisTensorCoreMappingPass());
  });
  // 注册 RV64 + RV-SIMT codegen pass（产出可由模拟器与 QV19P 直接执行的机器码）
  m.def("add_riscv_codegen", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::polaris::createPolarisRiscvCodegenPass());
  });
}

void init_triton_polaris(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_polaris_passes(passes.def_submodule("polarisgpu"));

  m.def("arch_info", []() {
    py::dict d;
    d["backend"] = "polaris";
    d["arch"] = "polaris_gpu";
    d["tensorcore"] = "MAT4X4";
    d["tensorcore_opcode"] = 0x2B;
    // 真实 RTL 参数: NutCore.scala:96 SIMT_Width=8, NutShell.scala:207-208 2 SM
    d["warp_size"] = 8;
    d["sm_count"] = 2;
    d["warps_per_sm"] = 4;   // Thread_Max/SIMT_Width = 32/8
    d["pe_per_sm"] = 2;      // NutShell.scala:102-103
    return d;
  });
}
