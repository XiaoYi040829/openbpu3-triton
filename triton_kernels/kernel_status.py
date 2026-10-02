#!/usr/bin/env python3
"""
kernel_status.py — 内核执行状态与硬件异常回传

问题：当前判据是"UART 收够 n 个值就算成功"，任何失败都退化成同一个症状。
模拟器上的对照：

    正常 vecadd  : <40400000>x8 + "EXCEEDING CYCLE/INSTR LIMIT at pc = 0x0"
    非法指令     :               "EXCEEDING CYCLE/INSTR LIMIT at pc = 0x0"
    死循环       :               "EXCEEDING CYCLE/INSTR LIMIT at pc = 0x0"

三者的终止信息**逐字相同**，退出码一律 0。所以"回读不足"这一个症状底下
藏着至少四种完全不同的原因：kernel 没跑、跑了但没写结果、写了但地址不对、
回读段本身崩了。这几种情况表面上都是"输出全 0"，不加区分就只能从头二分定位。

做法是在指令流里插入**哨兵标记**，把执行推进到哪一步变成可观测的：

    [prologue][计算主体] S1 [结果回读] S2 [尾声] S3 [END_WARP]

host 按收到哪些哨兵判断失败阶段：
    收不到 S1        -> 计算主体内崩溃/挂死（最常见：非法指令、访存异常）
    S1 有 S2 无      -> 回读段崩溃（通常是回读地址算错，或 ECC 未初始化）
    S1/S2 有但值不全 -> 结果确实没写全（kernel 逻辑或寄存器分配问题）
    三个都有         -> 执行完整，数值问题另论

哨兵只用 UART 字节写（SB），不依赖任何算术或浮点单元 —— 那些单元本身
就是被怀疑的对象，用它们做诊断会自我循环。

哨兵之外再看 IPC（instrCnt / cycleCnt）：哨兵能定位阶段，但区分不了
"非法指令"和"死循环"（两者都停在同一个哨兵前）。死循环的 IPC 明显高于
正常值，卡死的 IPC 明显低于正常值，阈值见 IPC_NORMAL。
"""
import re

from openbpu3_backend import ADDI, SB, li32, UART_TX

# 哨兵字符：选可打印且不与十六进制回读格式 <xxxxxxxx> 冲突的符号。
# 用单字节而非多字节串：每个 SB 都是一次 MMIO 写，字节越少越不容易在
# 崩溃边缘被截断，从而误判阶段。
SENTINEL = {
    "compute_done":  0x21,   # '!' 计算主体执行完毕
    "readback_done": 0x40,   # '@' 结果回读完毕
    "epilogue_done": 0x23,   # '#' 尾声完毕，即将 END_WARP
}
# 反查表：host 侧从字符还原阶段名
_CHAR2STAGE = {chr(v): k for k, v in SENTINEL.items()}

# 阶段顺序，用于判断"推进到哪一步"
STAGE_ORDER = ["compute_done", "readback_done", "epilogue_done"]


def emit_sentinel(code, stage, uart_reg=None, tmp=12):
    """发射一个哨兵标记。

    uart_reg=None 时自行用 li32 装载 UART 地址（哨兵可能插在任何位置，
    不能假定调用方已经把 UART 地址放在某个寄存器里）。给了 uart_reg 就复用，
    省 2~4 条指令。

    tmp 默认 x12：哨兵只插在 kernel 主体执行完之后（见 append_readback），
    此时分配池里的值都已不再使用，借用 x12 不会破坏计算结果。
    """
    if stage not in SENTINEL:
        raise ValueError(f"未知哨兵阶段 {stage}，可用 {list(SENTINEL)}")
    if uart_reg is None:
        uart_reg = 31            # x31: 分配池外的高位寄存器
        code += li32(uart_reg, UART_TX)
    code.append(ADDI(tmp, 0, SENTINEL[stage]))
    code.append(SB(tmp, uart_reg, 0))
    return code


def parse_uart(out):
    """从 emu/UART 原始输出里分离哨兵与数据值。

    必须先剥掉 ANSI 转义序列：emu 用颜色码包裹终止信息
    （如 \\x1b[33mEXCEEDING...），其中的 'm' 和数字会混进解析。

    返回 dict:
        stages     : 收到的哨兵阶段列表（按出现顺序）
        words      : <xxxxxxxx> 形式的 32 位值（int）
        reached    : 推进到的最远阶段（None 表示连第一个哨兵都没到）
        term       : emu 的终止原因原文（若有）
        instr      : 执行的指令数（若 emu 报了）
        cycles     : 周期数
    """
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", out)

    stages = []
    for ch in clean:
        if ch in _CHAR2STAGE:
            stages.append(_CHAR2STAGE[ch])

    words = [int(t, 16) for t in re.findall(r"<([0-9a-fA-F]{8})>", clean)]

    reached = None
    for s in STAGE_ORDER:
        if s in stages:
            reached = s
        else:
            break        # 阶段是顺序推进的，断了就停

    term = None
    m = re.search(r"(EXCEEDING CYCLE/INSTR LIMIT at pc = 0x[0-9a-fA-F]+)",
                  clean)
    if m:
        term = m.group(1)
    else:
        m = re.search(r"(HIT (?:GOOD|BAD) TRAP[^\n]*)", clean)
        if m:
            term = m.group(1)

    instr = cycles = None
    m = re.search(r"instrCnt = ([\d,]+), cycleCnt = ([\d,]+)", clean)
    if m:
        instr = int(m.group(1).replace(",", ""))
        cycles = int(m.group(2).replace(",", ""))

    return {"stages": stages, "words": words, "reached": reached,
            "term": term, "instr": instr, "cycles": cycles}


# 失败模式。诊断文本直接给出下一步该查什么，而不是只说"失败了"。
FAULT = {
    "OK":              "执行完整，结果齐全",
    "NO_START":        "核未取指或 prologue 即崩溃",
    "COMPUTE_FAULT":   "计算主体内崩溃或挂死",
    "COMPUTE_HANG":    "计算主体死循环（指令数顶到周期上限）",
    "READBACK_FAULT":  "回读段崩溃",
    "PARTIAL_RESULT":  "执行完整但结果数量不足",
    "EPILOGUE_FAULT":  "尾声段崩溃（END_WARP 前）",
    "CYCLE_BUDGET":    "周期预算不足，kernel 未跑完",
}

_HINT = {
    "NO_START": "查镜像是否写到 0x80000000、_start 地址、CONST_POOL 是否已填",
    "COMPUTE_FAULT": "查非法指令（codegen 是否发射了未支持的 op）、"
                     "访存越界、读未初始化内存",
    "COMPUTE_HANG": "查分支目标是否算错形成自跳；或周期预算不足",
    "READBACK_FAULT": "查回读地址（arg_base + i*arg_stride）与 n_out",
    "PARTIAL_RESULT": "查 kernel 是否真写满 n 个元素；"
                      "寄存器耗尽（reg_exhausted）会让部分 lane 结果丢失",
    "EPILOGUE_FAULT": "查回读循环的 BNE 偏移与 END_WARP 发射",
    "CYCLE_BUDGET": "提高 cycles 参数后重试",
}


# IPC 判据的阈值（vecadd + 故障注入标定）：
#     正常 60k / 90k 周期 : 0.0844 / 0.0840   <- 与周期预算无关，很稳
#     计算主体死循环      : 0.2352 / 0.2426   <- 约 3 倍
#     周期预算不足        : 0.0340
#     回读段卡死          : 0.0025            <- 约 1/34
#
# 为什么用 IPC 而不是指令数：指令数受周期预算直接影响（同一 kernel 在
# 60k/90k 预算下是 5063/7563 条），任何绝对阈值都会随预算漂移；instr 里
# 还有约 7000 条与 kernel 本身无关的固定开销（n_out=1 时为 7103 条）。
# IPC 是比值，把这两个干扰都消掉了。
IPC_NORMAL = 0.084
IPC_HANG_RATIO = 2.0     # 高于正常 2 倍 -> 死循环
IPC_STALL_RATIO = 0.15   # 低于正常 15% -> 卡死在某处


def diagnose(parsed, n_expect, cycles_budget=None):
    """把 parse_uart 的观测归因到具体失败模式。

    n_expect        : 期望的结果值个数
    cycles_budget   : 传给 emu 的 -C 值，用于区分"周期不够"与"真的挂死"

    返回 (fault_code, 人类可读说明)
    """
    reached, words = parsed["reached"], parsed["words"]
    instr, cyc = parsed["instr"], parsed["cycles"]

    # 周期耗尽的判据：cycleCnt 撞到预算上限。emu 报的是 budget-1
    # （实测 -C 20000 时 cycleCnt = 19,999），所以不能用相等判断。
    cyc_exhausted = (cycles_budget is not None and cyc is not None
                     and cyc >= cycles_budget - 2)
    ipc = (instr / cyc) if (instr and cyc) else None
    hang = ipc is not None and ipc > IPC_NORMAL * IPC_HANG_RATIO
    stalled = ipc is not None and ipc < IPC_NORMAL * IPC_STALL_RATIO

    if reached is None:
        # 连第一个哨兵都没到。这里有三种情况必须分开，否则会把良性的
        # "预算给小了"误报成 codegen 有 bug（例如 -C 4000 时只跑 136 条指令
        # 就到点，这是预算不足，不是计算主体崩溃）。
        if instr is not None and instr < 64:
            return "NO_START", (
                f"{FAULT['NO_START']}（仅执行 {instr} 条指令）。"
                f"{_HINT['NO_START']}")
        # 死循环 vs 预算不足：两者都是周期耗尽且没到哨兵，用 IPC 区分。
        # 自跳循环里每周期都有指令退休，IPC 约为正常的 3 倍；
        # 预算不足则是正常代码被拦腰截断，IPC 偏低。
        if cyc_exhausted:
            if hang:
                return "COMPUTE_HANG", (
                    f"{FAULT['COMPUTE_HANG']}（IPC {ipc:.3f}，"
                    f"正常约 {IPC_NORMAL:.3f}，执行 {instr} 条后周期耗尽）。"
                    f"{_HINT['COMPUTE_HANG']}")
            return "CYCLE_BUDGET", (
                f"{FAULT['CYCLE_BUDGET']}（执行 {instr} 条指令，"
                f"IPC {ipc:.3f} 未见异常，周期 {cyc} 已达预算 "
                f"{cycles_budget}）。{_HINT['CYCLE_BUDGET']}")
        return "COMPUTE_FAULT", (
            f"{FAULT['COMPUTE_FAULT']}"
            + (f"（执行 {instr} 条指令后停止）" if instr else "")
            + f"。{_HINT['COMPUTE_FAULT']}")

    if reached == "compute_done":
        # 主体跑完了但回读没完成。周期耗尽有两种解释，不能一律归到预算：
        #   * 指令数接近正常值 -> 确实是预算差一点，加大 cycles 就好
        #   * 指令数远低于正常值 -> 主体刚跑完就卡住了，多给周期也没用，
        #     是回读段自己的问题（例如截断回读段后 instr=147，正常为 5063）
        if cyc_exhausted and len(words) < n_expect:
            if stalled:
                return "READBACK_FAULT", (
                    f"{FAULT['READBACK_FAULT']}"
                    f"（收到 {len(words)}/{n_expect} 个值，IPC {ipc:.4f} 远低于"
                    f"正常 {IPC_NORMAL:.3f} —— 核已停止推进，"
                    f"给更多周期也无用）。{_HINT['READBACK_FAULT']}")
            return "CYCLE_BUDGET", (
                f"{FAULT['CYCLE_BUDGET']}"
                f"（回读了 {len(words)}/{n_expect} 个，"
                f"周期 {cyc} 已达预算）。{_HINT['CYCLE_BUDGET']}")
        return "READBACK_FAULT", (
            f"{FAULT['READBACK_FAULT']}（收到 {len(words)}/{n_expect} 个值）。"
            f"{_HINT['READBACK_FAULT']}")

    if reached == "readback_done":
        if len(words) < n_expect:
            return "PARTIAL_RESULT", (
                f"{FAULT['PARTIAL_RESULT']}（{len(words)}/{n_expect}）。"
                f"{_HINT['PARTIAL_RESULT']}")
        return "EPILOGUE_FAULT", (
            f"{FAULT['EPILOGUE_FAULT']}。{_HINT['EPILOGUE_FAULT']}")

    # 走到 epilogue_done：执行路径完整
    if len(words) < n_expect:
        return "PARTIAL_RESULT", (
            f"{FAULT['PARTIAL_RESULT']}（{len(words)}/{n_expect}）。"
            f"{_HINT['PARTIAL_RESULT']}")
    return "OK", FAULT["OK"]
