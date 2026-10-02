#!/usr/bin/env python3
"""jit_divergent.py — 验证 lane 间真实分歧的 if/else

每条 lane 根据自己的数据走**不同的计算路径**，且两条路径的
运算完全不同（不是同一式子换系数）。这样若分支未生效，
结果会全部落到同一侧，一眼可辨。
"""
import triton
import triton.language as tl


@triton.jit
def divergent_kernel(x_ptr, out_ptr):
    """if x > 0: out = x*x + 1   else: out = -x * 3 - 7

    两侧运算不同：正侧是平方加一，负侧是取反乘三减七。
    8 条 lane 的输入正负交替，所以必然一半走 if、一半走 else。
    """
    i = tl.arange(0, 8)
    x = tl.load(x_ptr + i)
    pos = x * x + 1.0
    neg = -x * 3.0 - 7.0
    out = tl.where(x > 0.0, pos, neg)
    tl.store(out_ptr + i, out)


@triton.jit
def nested_kernel(x_ptr, out_ptr):
    """三层嵌套条件，每条 lane 可能走 4 条不同路径之一：
         x >  1 : 100 + x
         0 < x <= 1 : 200 + x
        -1 < x <= 0 : 300 + x
         x <= -1 : 400 + x
    """
    i = tl.arange(0, 8)
    x = tl.load(x_ptr + i)
    inner_pos = tl.where(x > 1.0, 100.0 + x, 200.0 + x)
    inner_neg = tl.where(x > -1.0, 300.0 + x, 400.0 + x)
    out = tl.where(x > 0.0, inner_pos, inner_neg)
    tl.store(out_ptr + i, out)


DIV_SIG = {"x_ptr": "*fp32", "out_ptr": "*fp32"}
