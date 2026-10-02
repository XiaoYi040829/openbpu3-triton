#!/usr/bin/env python3
"""jit_fp_probe.py — 浮点算术专项探针 kernel

单独成模块的原因：@triton.jit 在编译时按**定义所在模块的全局命名空间**
解析 tl / triton 等名字，写在测试函数内部的 kernel 会因局部 import
而报 NameError('tl is not defined')。

每个探针只做一种运算，8 lane 一批。这样一旦某批出错，失效点就唯一确定
到某条指令的 lowering 上，不用在复合表达式里二分定位。
"""
import triton
import triton.language as tl

SIG2 = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32"}
SIG1 = {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32"}


@triton.jit
def sub_probe(a_ptr, b_ptr, out_ptr):
    """纯减法，用于暴露 fsub.s 在指数差为 1 且结果为负时的符号缺陷"""
    i = tl.arange(0, 8)
    a = tl.load(a_ptr + i)
    b = tl.load(b_ptr + i)
    tl.store(out_ptr + i, a - b)


SUB_SIG = SIG2


@triton.jit
def add_probe(a_ptr, b_ptr, out_ptr):
    i = tl.arange(0, 8)
    a = tl.load(a_ptr + i)
    b = tl.load(b_ptr + i)
    tl.store(out_ptr + i, a + b)


@triton.jit
def mul_probe(a_ptr, b_ptr, out_ptr):
    i = tl.arange(0, 8)
    a = tl.load(a_ptr + i)
    b = tl.load(b_ptr + i)
    tl.store(out_ptr + i, a * b)


@triton.jit
def div_probe(a_ptr, b_ptr, out_ptr):
    i = tl.arange(0, 8)
    a = tl.load(a_ptr + i)
    b = tl.load(b_ptr + i)
    tl.store(out_ptr + i, a / b)


@triton.jit
def min_probe(a_ptr, b_ptr, out_ptr):
    i = tl.arange(0, 8)
    a = tl.load(a_ptr + i)
    b = tl.load(b_ptr + i)
    tl.store(out_ptr + i, tl.minimum(a, b))


@triton.jit
def max_probe(a_ptr, b_ptr, out_ptr):
    """fmax.s 在该核返回 min，codegen 必须用 -min(-a,-b) 绕开；本探针验证绕法"""
    i = tl.arange(0, 8)
    a = tl.load(a_ptr + i)
    b = tl.load(b_ptr + i)
    tl.store(out_ptr + i, tl.maximum(a, b))


@triton.jit
def abs_probe(a_ptr, b_ptr, out_ptr):
    i = tl.arange(0, 8)
    a = tl.load(a_ptr + i)
    b = tl.load(b_ptr + i)
    tl.store(out_ptr + i, tl.abs(a) + b * 0.0)


@triton.jit
def sqrt_probe(a_ptr, b_ptr, out_ptr):
    i = tl.arange(0, 8)
    a = tl.load(a_ptr + i)
    b = tl.load(b_ptr + i)
    tl.store(out_ptr + i, tl.sqrt(a) + b * 0.0)


@triton.jit
def select_probe(a_ptr, b_ptr, out_ptr):
    """分支消除路径：tl.where 降级为按位合并"""
    i = tl.arange(0, 8)
    a = tl.load(a_ptr + i)
    b = tl.load(b_ptr + i)
    tl.store(out_ptr + i, tl.where(a < b, a, b))


@triton.jit
def fma_probe(a_ptr, b_ptr, out_ptr):
    """a*b - a 组合：把乘法与相消减法串起来，覆盖两者的交互"""
    i = tl.arange(0, 8)
    a = tl.load(a_ptr + i)
    b = tl.load(b_ptr + i)
    tl.store(out_ptr + i, a * b - a)


# 探针注册表：名字 -> (kernel, host 参考函数)
# 参考值一律用 numpy.float32 逐步求值，与设备端运算顺序一致，
# 这样报出来的误差只反映 codegen/硬件，不含 host 侧的精度差异。
PROBES = {
    "add":    add_probe,
    "sub":    sub_probe,
    "mul":    mul_probe,
    "div":    div_probe,
    "min":    min_probe,
    "max":    max_probe,
    "abs":    abs_probe,
    "sqrt":   sqrt_probe,
    "select": select_probe,
    "fma":    fma_probe,
}
