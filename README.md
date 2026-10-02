# OpenBPU-3 Triton 后端

面向 OpenBPU-3（PolarisGPU，RV64 + RV-SIMT，8 lane × 2 SM）的 Triton 编译后端，基于 FlagTree（FlagOS 统一编译器）。`@triton.jit` kernel 经 Triton 前端和本后端编译为 RV64 + RV-SIMT 机器码，可在 OpenBPU-3 的周期精确 RTL 模拟器上执行。仓库包含一个端到端示例：PyBullet 中的 6-DOF 机械臂追踪浮动球体，逆运动学在模拟器上计算。

## 仓库结构

| 目录 | 内容 |
|---|---|
| `polaris_backend/` | 后端：Python 编译管线（`backend/`）与 C++ MLIR pass（RV64 + RV-SIMT codegen、TensorCore 映射分析） |
| `flagtree-patch/` | 对 FlagTree 23d0412 的补丁：注册 polaris 后端、for 循环边界检查 |
| `rv-simt-llvm-patch/` | 让 LLVM `llvm-mc` 识别 RV-SIMT 指令的补丁与编码校验脚本（可选） |
| `triton_kernels/` | kernel、模拟器运行器、回归测试、端到端示例 |
| `robot/` | 机械臂 URDF |

## 构建

已验证环境：Ubuntu 24.04，Python 3.12，x86-64。

### 1. 依赖

```bash
sudo apt install build-essential clang git binutils-riscv64-linux-gnu
python3 -m venv venv && . venv/bin/activate
pip install "setuptools>=79.0.1,<82" wheel "cmake>=3.18" "ninja>=1.11.1" pybind11 numpy
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install pybullet==3.2.6 pillow
```

需要 torch：FlagTree 自带的 NVIDIA 后端在 `import triton` 时会探测设备并导入 torch，CPU 版即可。

### 2. LLVM（10dc3a8e 预编译包）

```bash
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-10dc3a8e-ubuntu-x64.tar.gz
tar xf llvm-10dc3a8e-ubuntu-x64.tar.gz        # 下载约 1 GB，解压后约 3.6 GB
export LLVM_SYSPATH=$PWD/llvm-10dc3a8e-ubuntu-x64
```

### 3. googletest（构建时需要）

FlagTree 构建会编译 C++ 单元测试，cmake 先找 `~/.triton/googletest-release-1.12.1`，找不到才从 GitHub 克隆。网络不稳定时建议提前放好：

```bash
mkdir -p ~/.triton
git clone --depth 1 -b release-1.12.1 https://github.com/google/googletest.git ~/.triton/googletest-release-1.12.1
```

### 4. FlagTree + 本后端

```bash
git clone https://github.com/flagos-ai/FlagTree.git flagtree
cd flagtree
git checkout 23d04123b0060eb437594df01dfa1c8c4f3150ef
git apply <本仓库>/flagtree-patch/flagtree-23d0412-polaris.patch
mkdir -p third_party/polaris
cp -r <本仓库>/polaris_backend/. third_party/polaris/
cd python
pip install -e . --no-build-isolation
```

检查后端是否编入：

```bash
python -c "from triton._C.libtriton import polaris; print(dir(polaris.passes.polarisgpu))"
# 应包含 add_riscv_codegen 与 add_tensorcore_mapping
```

### 5. 模拟器

本仓库不含模拟器。执行 kernel 需要由 OpenBPU-3 RTL 编译出的 Verilator 模拟器（difftest 的 `emu`，使用 `-i <镜像> --no-diff -C <周期>`），用 `OPENBPU3_EMU` 指定其路径。

## 运行

```bash
export OPENBPU3_EMU=/path/to/emu
python triton_kernels/test_all_openbpu3.py          # 回归测试，18 组共 22 项
python triton_kernels/run_arm6_track.py             # 机械臂追踪示例
```

编译单个 kernel：

```python
import sys; sys.path.insert(0, "triton_kernels")
from jit_openbpu3 import jit_compile, vecadd_kernel, PTR3
r, _ = jit_compile(vecadd_kernel, PTR3)
print(r["n_inst"], r["asm_path"])                   # 机器码在 r["words"]
```

执行用 `triton_kernels/run_triton_kernel.py` 的 `run()`。本后端没有 Triton 运行时启动器，`kernel[grid](...)` 会报错。

### 环境变量

| 变量 | 作用 |
|---|---|
| `OPENBPU3_EMU` | 模拟器路径（默认在 PATH 中找 `emu`） |
| `EMU_CYCLES` | 回归测试的周期预算（默认 70000） |
| `SKIP_SLOW=1` | 跳过耗时的回归项 |
| `STEPS` | 机械臂示例的控制步数 |
| `LLVM_MC` | 打过 `rv-simt-llvm-patch` 的 `llvm-mc`；未提供时该项记为 SKIP |
| `LLVM_TBLGEN` | `llvm-tblgen`（系统自带即可，不需打补丁）；未提供时在 PATH 中查找，找不到记为 SKIP |
| `POLARIS_RTL_DIR` | OpenBPU-3 RTL 的 `src/main/scala/nutcore`；提供时额外比对 RTL 中的编码 |
| `POLARIS_STORE_PAD=N` | 每条 store 后插 N 个 nop（见局限性） |

回归测试的汇总把通过、失败、跳过分开计数，只有失败才算不通过。缺少可选工具时，相应项显示为 SKIP。

## 局限性

以下结果均在周期精确 RTL 模拟器上得到；这些 kernel 尚未在 FPGA（QV19P）上运行。

### 硬件缺陷与后端的处理

逐条用最小手写汇编 kernel 在模拟器上测得：

| 缺陷 | 后端处理 |
|---|---|
| `fsub.s` 在两操作数指数差恰为 1 且结果为负时丢失符号（0.6−1.0 得 +0.4） | 不发射 `fsub.s`，减法降级为 `fsgnjn.s` + `fadd.s` |
| `fmax.s` 返回较小值 | 改用 `-min(-a, -b)` |
| 浮点结果写入 `f0` 后，隔一条指令再读得到 NaN | 不分配 `f0`，生成代码中出现 `f0` 即编译失败 |
| `fadd.s` 在操作数或结果为 inf/NaN 时返回 +0 | 取决于运行时数值，编译期无法排除；kernel 须保证操作数为有限值 |
| `fmul.s` 乘积低于 2^-126 时得到错误的次正规数 | 同上；kernel 须保证乘积在正规数范围内（`exp` 的实现已对此做了处理） |

`fadd/fsub/fmul/fdiv` 对有限操作数、正规范围结果正确；`fmin`、`fsqrt`、比较、符号注入、`fmv.x.w`/`fmv.w.x`、`fcvt.w.s`/`fcvt.s.w` 正确。

### 编译器支持范围

不能正确降级的构造一律在编译期报错（以 `polaris backend:` 开头），不会静默产出错误代码：

- 张量元素总数必须为 8（每条 lane 一个元素）
- 只支持 32 位元素、不带掩码的 load/store
- 不支持跨 lane 归约（`tl.sum` 等）、`tl.dot`、整数扩宽/截断（`extsi/extui/trunci`）；整浮转换只支持 i32/u32 与 f32 之间
- 不生成分支：`tl.where` 降级为掩码合并，循环边界必须是 `tl.constexpr`（编译期完全展开，迭代次数不超过 512）
- 指针参数最多 6 个；kernel 代码不超过 7680 字节
- 不生成 spill 代码：同时活跃的值必须放得进寄存器池（整数 20 个、浮点 20 个），否则编译失败
- 对同一指针参数既读又写的 kernel 被拒绝：当前 SoC 把同一组 CTA 同时派发给两个 SM，读改写会被执行两次

Triton 官方 `language` 单元测试中（去掉测试自身跳过的，共 10694 例），能为本后端编译的有 78 例（0.73%），其余均为编译期报错。这只统计能否编译，未在模拟器上校验数值。

### 已知的板上问题

FPGA 上 WarpLSU 存在 store 地址/数据错位一拍的缺陷（模拟器上不出现），表现为个别 store 写到另一个 store 的地址，且结果不确定。`POLARIS_STORE_PAD` 插 nop 只能降低出错概率，不能保证正确；根治需修改 RTL。

### 精度

机械臂示例的末端最大误差约 1.5 µm，主要来自 `atan2` 的多项式近似，而不是代码生成。

## 许可证

本仓库采用 Apache License 2.0，见 `LICENSE`。例外：`flagtree-patch/` 中的补丁是对 FlagTree 源文件的修改，与 FlagTree 一致按 MIT 许可证发布（见 FlagTree 仓库的 `LICENSE`）。
