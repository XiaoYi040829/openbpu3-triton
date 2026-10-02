#!/usr/bin/env python3
"""
jit_openbpu3.py — 走 Triton Python 前端 + PolarisBackend 正规入口的完整链路

    Python 前端 (@triton.jit) -> Triton-IR -> CodeGen 后端 (openbpu3)

kernel 用 Python 写, 由 triton 的 AST->TTIR 前端转换, 再经 PolarisBackend.add_stages 的 ttir->ttgir->bpuir->bin 四阶段
正规管线编译成真 RV64 + RV-SIMT 机器码。

提供 jit_compile() 供上层(bullet3 后端)调用。
"""
import os
import struct
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
for p in [x for x in (os.environ.get("FLAGTREE_PY"), WS,
                      os.path.join(WS, "triton_kernels")) if x]:
    if p not in sys.path:
        sys.path.insert(0, p)

import triton                                    # noqa: E402
import triton.language as tl                     # noqa: E402
from triton.compiler.compiler import ASTSource   # noqa: E402
from triton.backends.compiler import GPUTarget   # noqa: E402
from triton._C.libtriton import ir               # noqa: E402


class _Meta:
    """add_stages 的 metadata 容器（stages 会往上写 name 等字段）"""
    def __init__(self):
        self.name = None


def jit_compile(fn, signature, verbose=False, constants=None):
    """把 @triton.jit 函数经 PolarisBackend 四阶段管线编译成机器码。

    返回 make_bin 的输出 dict:
        {bin_path, words, n_inst, arg_base, arg_stride, num_ptr_args, ...}
    """
    from triton.backends.polaris.compiler import PolarisBackend

    # 后端名 openbpu3（supports_target 同时接受 polaris）
    tgt = GPUTarget("openbpu3", "polaris_gpu", 8)
    if not PolarisBackend.supports_target(tgt):
        raise RuntimeError("PolarisBackend 不接受 target=openbpu3")
    be = PolarisBackend(tgt)
    opt = be.parse_options({})

    # ---- Python 前端: AST -> TTIR ----
    # constants: {参数名或下标: 值}，用于 tl.constexpr 参数。Triton 3.1 按
    # 参数下标取值（code_generator.py:425），这里统一把参数名换成下标。
    if constants:
        names = list(fn.arg_names)
        constants = {(names.index(k) if isinstance(k, str) else k): v
                     for k, v in constants.items()}
    src = ASTSource(fn=fn, signature=signature, constants=constants)
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = src.make_ir(opt, be.get_codegen_implementation(), ctx)
    mod.context = ctx
    if verbose:
        print(f"    [前端] TTIR {len(str(mod).splitlines())} 行")

    # ---- 后端正规入口: ttir -> ttgir -> bpuir -> bin ----
    stages = {}
    be.add_stages(stages, opt)
    meta = _Meta()
    cur = mod
    for name in ("ttir", "ttgir", "bpuir", "bin"):
        cur = stages[name](cur, meta)
        if verbose:
            kind = ("dict" if isinstance(cur, dict)
                    else "str" if isinstance(cur, str) else "module")
            print(f"    [stage] {name:6s} -> {kind}")
    if not isinstance(cur, dict) or "words" not in cur:
        raise RuntimeError(f"bin 阶段未产出机器码: {cur if isinstance(cur,dict) else type(cur)}")
    return cur, opt


# ---------------- 物理 kernel（Python 前端形式） ----------------

@triton.jit
def vecadd_kernel(a_ptr, b_ptr, c_ptr):
    """简单向量加法"""
    i = tl.arange(0, 8)
    a = tl.load(a_ptr + i)
    b = tl.load(b_ptr + i)
    tl.store(c_ptr + i, a + b)


@triton.jit
def bullet_step_kernel(y_ptr, v_ptr, out_ptr):
    """一步完整物理: 重力积分 + 地面碰撞 + 冲量响应

    每个刚体一条 SIMT lane。tl.where 在 codegen 里落到掩码算术，
    避免 warp 内 lane 分歧（该核 SYNWARP 仅 SM 内有效）。
    输出连续区间: out[0..7]=y', out[8..15]=v'
    """
    dt = 4.16666667e-03
    g = -9.81
    radius = 0.25
    e = 0.5

    i = tl.arange(0, 8)
    y = tl.load(y_ptr + i)
    v = tl.load(v_ptr + i)

    v2 = v + g * dt
    y2 = y + v2 * dt

    hit = y2 < radius
    v3 = tl.where(hit, -v2 * e, v2)
    y3 = tl.where(hit, radius, y2)

    tl.store(out_ptr + i, y3)
    tl.store(out_ptr + 8 + i, v3)


PTR3 = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32"}
STEP_SIG = {"y_ptr": "*fp32", "v_ptr": "*fp32", "out_ptr": "*fp32"}


def main():
    print("=" * 70)
    print("Triton Python 前端 (@triton.jit) + PolarisBackend 正规入口")
    print("=" * 70)

    for name, fn, sig in (("vecadd", vecadd_kernel, PTR3),
                          ("bullet_step", bullet_step_kernel, STEP_SIG)):
        print(f"\n--- {name} ---")
        try:
            r, opt = jit_compile(fn, sig, verbose=True)
            print(f"    产出 {r['n_inst']} 条真 RISC-V 指令 -> {r['bin_path']}")
            print(f"    op 映射 {r['num_mapped']} / 跳过 {r['num_skipped']}")
            print(f"    参数约定: {r['num_ptr_args']} 个指针, "
                  f"base=0x{r['arg_base']:08X}, stride=0x{r['arg_stride']:X}")
        except Exception as e:
            import traceback
            print(f"    失败: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
    print("\n" + "=" * 70)
    print("PASS: Python 前端 -> TTIR -> TTGIR -> BPUIR -> BIN 四阶段管线打通")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
