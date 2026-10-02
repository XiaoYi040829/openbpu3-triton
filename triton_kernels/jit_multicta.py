#!/usr/bin/env python3
"""jit_multicta.py — 多 CTA kernel，用于验证跨 PE / 跨 SM 执行

本 kernel 用 tl.program_id 划分工作，每个 CTA 处理 8 个元素。
program_id 在后端降级为 csrr WARP_ID（SIMD_CSR.scala:119）。
模拟器上每个 SM 固定运行 8 个 CTA：PE0 跑 pid 0-3，PE1 跑 pid 4-7。
注意 SoC 顶层把同一组 CTA 同时派发给两个 SM，SM1 会重复 SM0 的工作，
因此对同一缓冲区做读改写的 kernel 结果会被重复累加。
"""
import triton
import triton.language as tl


@triton.jit
def multicta_kernel(a_ptr, b_ptr, c_ptr):
    """c[pid*8 + i] = a[pid*8 + i] + b[pid*8 + i]

    每个 CTA 读写自己的 8 元素段。host 给各段填不同的输入，输出因此
    可按段区分：能分清是真的多个 CTA 各自执行，还是同一个 CTA 跑了多次。
    """
    pid = tl.program_id(axis=0)
    i = tl.arange(0, 8)
    off = pid * 8 + i
    a = tl.load(a_ptr + off)
    b = tl.load(b_ptr + off)
    tl.store(c_ptr + off, a + b)


MCTA_SIG = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32"}
