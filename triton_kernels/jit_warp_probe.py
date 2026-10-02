#!/usr/bin/env python3
"""jit_warp_probe.py — warp 语义验证用的探针 kernel（供 test_warp_semantics 使用）

单独成模块的原因：@triton.jit 在编译时按**定义所在模块的全局命名空间**解析
tl / triton，写在测试函数内部的 kernel 会报 NameError('tl is not defined')。
"""
import triton
import triton.language as tl


@triton.jit
def two_barriers(a_ptr, out_ptr):
    """相邻两个 barrier + 间隔一个，用于检验 CSE/DCE 是否会合并或删除它们"""
    i = tl.arange(0, 8)
    a = tl.load(a_ptr + i)
    b = a * 2.0
    tl.debug_barrier()      # 第 1 个
    tl.debug_barrier()      # 第 2 个：与上一个相邻，CSE 的典型目标
    c = b + 1.0
    tl.debug_barrier()      # 第 3 个
    tl.store(out_ptr + i, c)


BARRIER_SIG = {"a_ptr": "*fp32", "out_ptr": "*fp32"}
