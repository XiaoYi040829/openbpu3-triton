# PolarisGPU FlagTree 后端 C++ 构建
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# PolarisGPU MLIR transform 库
add_triton_library(PolarisGPUTransforms
  lib/PolarisGPUTransforms/PolarisTensorCoreMapping.cpp
  lib/PolarisGPUTransforms/PolarisRiscvCodegen.cpp
  DEPENDS
  TritonTableGen
  LINK_LIBS PUBLIC
  TritonIR
  MLIRIR
  MLIRPass
  MLIRTransforms
  # scf.for 完全展开需要（本后端不发分支指令，循环必须在 codegen 前展开）
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRArithDialect
)

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonPolaris
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_polaris.cc
    LINK_LIBS PolarisGPUTransforms)
endif()
