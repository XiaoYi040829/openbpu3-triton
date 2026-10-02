# RV-SIMT 的 LLVM 补丁

基于 LLVM `10dc3a8e916d73291269e5e2b82dd22681489aa1`（19.0.0git）。

```bash
git init llvm-project && cd llvm-project
git remote add origin https://github.com/llvm/llvm-project.git
git sparse-checkout set --no-cone /llvm/ /cmake/ /third-party/
git fetch --depth 1 --filter=blob:none origin 10dc3a8e916d73291269e5e2b82dd22681489aa1
git checkout FETCH_HEAD
cp <本目录>/RISCVInstrInfoSIMT.td llvm/lib/Target/RISCV/
git apply <本目录>/llvm-10dc3a8e.patch     # RISCVFeatures.td + RISCVInstrInfo.td
cmake -S llvm -B build-riscv -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_TARGETS_TO_BUILD=RISCV -DLLVM_ENABLE_PROJECTS=""
ninja -C build-riscv llvm-mc                # 8 核约 5 分钟
```

验证：`python3 verify_llvm_mc.py`（汇编、反汇编、未启用扩展时拒绝）。
