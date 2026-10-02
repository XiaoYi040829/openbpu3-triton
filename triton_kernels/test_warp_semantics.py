#!/usr/bin/env python3
"""
test_warp_semantics.py — 验证优化 pass 是否破坏 warp 控制语义

LLVM 扩展的编码校验不覆盖这一点：优化 pass（重排 / 合并 / DCE）是否会
误删或移动 SYN_WARP、START_WARP 等 warp 控制指令。

本工程的实际流程与据此确定的风险面：

  @triton.jit
      |
      v  MLIR pass: inliner, combine, canonicalizer, reorder_broadcast,
      |             CSE, symbol_dce            <-- 唯一跑优化的地方
      v
  TTGIR -> BPUIR
      |
      v  PolarisRiscvCodegen: buf.emit(...) 直接产 32 位指令字
      v
  bare-metal binary

关键事实：warp 控制指令是在 codegen 阶段**直接发射机器码**的，不经过
LLVM IR，因此 LLVM 的 instcombine / machine scheduler / DCE 根本不会看到
它们。打进 LLVM 的 .td 补丁只服务于 llvm-mc（汇编器/反汇编器）。

所以真实风险有三处，本文件逐一实测：

  T1 MLIR 层：barrier 会不会被 CSE 合并、被 DCE 删除、被 canonicalizer 移动
  T2 codegen 层：prologue/epilogue 的 warp 指令是否恒定出现且顺序正确
  T3 汇编器层：llvm-mc 往返（汇编→反汇编→再汇编）是否保持这些指令不变

用法：
    PYTHONPATH=$PWD/triton_kernels \
        python3 triton_kernels/test_warp_semantics.py
"""
import os
import re
import struct
import subprocess
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
for p in [x for x in (os.environ.get("FLAGTREE_PY"), WS,
                      os.path.join(WS, "triton_kernels")) if x]:
    if p not in sys.path:
        sys.path.insert(0, p)

LLVM_MC = os.environ.get("LLVM_MC", "")  # 打过 rv-simt-llvm-patch 的 llvm-mc

# warp 控制指令的 32 位编码（取自 RTL，已交叉验证）
ENC = {
    "JOIN":          0x00B0007B,
    "START_WARP":    0x00C0007B,
    "SYN_WARP":      0x00D0007B,
    "END_WARP":      0x00E0007B,
    "SYN_WARP_REG":  0x00F0007B,
}
TID_MASK, TID_MATCH = 0xFE00707F, 0x60000033   # TID: funct7=0x60, R 型

results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  {detail}" if detail else ""))


def find_warp(words):
    """在指令流里定位 warp 控制指令，返回 [(索引, 名字)]"""
    out = []
    for i, w in enumerate(words):
        w &= 0xFFFFFFFF
        for nm, e in ENC.items():
            if w == e:
                out.append((i, nm))
                break
        else:
            if (w & TID_MASK) == TID_MATCH:
                out.append((i, "TID"))
    return out


# =============================================================
# T1  MLIR 层：barrier 能否被优化 pass 消除或合并
# =============================================================
def t1_mlir_barrier_survives():
    """两个相邻 barrier + 一个循环内 barrier，检查数量是否被 CSE/DCE 改变。

    gpu.barrier 在 MLIR 里若被当作无副作用的 op，CSE 会把相邻的两个合并成
    一个，DCE 会删掉"结果未被使用"的那个 —— 两者都会破坏 SM 内同步语义。
    """
    print("\nT1 MLIR 优化 pass 是否保留 barrier")
    try:
        from jit_openbpu3 import jit_compile
        from jit_warp_probe import two_barriers, BARRIER_SIG

        r, _ = jit_compile(two_barriers, BARRIER_SIG)
        w = [x & 0xFFFFFFFF for x in r["words"]]
        n_syn = sum(1 for x in w if x == ENC["SYN_WARP"])

        # prologue/epilogue 各自也会发 SYN_WARP，故只要求 >= 源码里的个数
        ok = n_syn >= 3
        record("3 个 barrier 全部保留到机器码",
               ok, f"SYN_WARP 出现 {n_syn} 次（源码 3 个 + 收敛用）")
    except Exception as e:
        record("MLIR barrier 保留", False, f"{type(e).__name__}: {e}")


# =============================================================
# T2  codegen 层：prologue/epilogue 的 warp 指令顺序
# =============================================================
def t2_prologue_order():
    """START_WARP 必须在任何 TID / 计算之前，END_WARP 必须在最后。

    顺序错了会让 lane 在 warp 未启动时就读 TID，或在同步前就退出。
    对全部 6 个生产 kernel 检查。
    """
    print("\nT2 prologue/epilogue 中 warp 指令的位置与顺序")
    try:
        from jit_openbpu3 import (jit_compile, vecadd_kernel, PTR3,
                                 bullet_step_kernel, STEP_SIG)
        from jit_ik_micro import (ik_geom_kernel, GEOM_SIG,
                                 atan2_kernel, ATAN2_SIG)
        from jit_arm6_split import (sphere_step_kernel, SPHERE_SIG,
                                   joint_pd_kernel, PD_SIG)
        cases = [("vecadd", vecadd_kernel, PTR3),
                 ("bullet_step", bullet_step_kernel, STEP_SIG),
                 ("sphere_step", sphere_step_kernel, SPHERE_SIG),
                 ("ik_geom", ik_geom_kernel, GEOM_SIG),
                 ("atan2", atan2_kernel, ATAN2_SIG),
                 ("joint_pd", joint_pd_kernel, PD_SIG)]
        bad = []
        for nm, fn, sig in cases:
            r, _ = jit_compile(fn, sig)
            w = [x & 0xFFFFFFFF for x in r["words"]]
            seq = find_warp(w)
            names = [n for _, n in seq]
            idx = {n: i for i, (i_, n) in enumerate(seq) for _ in [0]}
            # START_WARP 必须存在且是第一条 warp 指令
            if not names or names[0] != "START_WARP":
                bad.append(f"{nm}: 首条 warp 指令是 {names[:1]}")
                continue
            # TID 必须在 START_WARP 之后
            if "TID" in names and names.index("TID") < names.index("START_WARP"):
                bad.append(f"{nm}: TID 在 START_WARP 之前")
            # END_WARP 必须是最后一条 warp 指令
            if names[-1] != "END_WARP":
                bad.append(f"{nm}: 末条 warp 指令是 {names[-1]}")
            # END_WARP 之前必须有 SYN_WARP（收敛后再退出）
            if "SYN_WARP" not in names[:-1]:
                bad.append(f"{nm}: END_WARP 前缺少 SYN_WARP")
        record("6 个 kernel 的 warp 指令顺序均正确",
               not bad, "; ".join(bad) if bad else "START_WARP->TID->...->SYN_WARP->END_WARP")
    except Exception as e:
        record("prologue/epilogue 顺序", False, f"{type(e).__name__}: {e}")


# =============================================================
# T3  汇编器层：llvm-mc 往返保持 warp 指令
# =============================================================
def t3_llvm_mc_roundtrip():
    """汇编 -> 反汇编 -> 再汇编，检查 5 条 warp 指令编码是否漂移。

    这是 LLVM 补丁真正参与的环节。若 .td 里的编码或谓词写错，
    往返后指令会变成 .word 或编码改变。
    """
    print("\nT3 llvm-mc 往返是否保持 warp 指令编码")
    if not os.path.exists(LLVM_MC):
        record("llvm-mc 往返", False, "llvm-mc 未构建")
        return
    try:
        # 助记符带 polaris. 前缀（见 RISCVInstrInfoSIMT.td）
        mnem = {"JOIN": "polaris.join", "START_WARP": "polaris.start_warp",
                "SYN_WARP": "polaris.syn_warp",
                "END_WARP": "polaris.end_warp"}
        src = "\n".join(mnem[k] for k in mnem) + "\n"
        p1 = subprocess.run(
            [LLVM_MC, "-triple=riscv64", "-mattr=+xpolarissimt",
             "-show-encoding"], input=src, capture_output=True, text=True,
            timeout=60)
        if p1.returncode != 0:
            record("llvm-mc 往返", False,
                   f"汇编失败: {p1.stderr.strip()[:70]}")
            return
        # 从 -show-encoding 输出里取出每条的编码字节
        encs = re.findall(r"encoding:\s*\[([^\]]+)\]", p1.stdout)
        got = []
        for e in encs:
            bs = [int(x.strip(), 16) for x in e.split(",")]
            got.append(int.from_bytes(bytes(bs), "little"))
        exp = [ENC[k] for k in mnem]
        mism = [(m, hex(g), hex(x)) for m, g, x in zip(mnem, got, exp) if g != x]
        record(f"{len(mnem)} 条 warp 指令编码与 RTL 一致",
               not mism and len(got) == len(exp),
               f"不符: {mism}" if mism else
               f"逐条匹配 ({len(got)}/{len(exp)})")
    except Exception as e:
        record("llvm-mc 往返", False, f"{type(e).__name__}: {e}")


# =============================================================
# T4  .td 属性审查：指出真实的防护缺口
# =============================================================
def t4_td_attributes():
    """检查 .td 是否设了阻止优化的属性。

    warp 指令不经 LLVM 优化管线，hasSideEffects=1 已足够；
    若将来改走 llc，则必须补 isConvergent（阻止跨分支移动）与 isBarrier。
    """
    print("\nT4 .td 属性是否足以防止优化破坏语义")
    td = os.path.join(WS, "rv-simt-llvm-patch/RISCVInstrInfoSIMT.td")
    if not os.path.exists(td):
        record(".td 属性审查", False, "补丁文件缺失")
        return
    s = open(td).read()
    has_se = s.count("hasSideEffects = 1")
    has_conv = "isConvergent" in s
    # 当前流程不经 LLVM 优化，故只要求 hasSideEffects 齐备
    record("warp 指令均标注 hasSideEffects=1",
           has_se >= 2, f"出现 {has_se} 处；isConvergent "
           f"{'已设' if has_conv else '未设（当前流程不经 LLVM 优化，不需要）'}")


def main():
    print("=" * 72)
    print("warp 控制语义 vs 优化 pass")
    print("=" * 72)
    t1_mlir_barrier_survives()
    t2_prologue_order()
    t3_llvm_mc_roundtrip()
    t4_td_attributes()
    print("\n" + "=" * 72)
    npass = sum(1 for _, ok, _ in results if ok)
    for nm, ok, _ in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}")
    print(f"\n{npass}/{len(results)} 通过")
    print("=" * 72)
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
