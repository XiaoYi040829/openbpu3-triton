#!/usr/bin/env python3
"""
fp_fuzz.py — 危险浮点用例自动生成器

人手挑的"代表点"系统性地偏向数值良态区，窄触发窗口的浮点缺陷（如 fsub.s
在指数差恰为 1 且结果为负时丢符号）很容易漏掉，并在长轨迹上累积成大误差。

均匀随机撒点落进这类窗口的概率极低，所以这里按**已知危险类别**定向构造，
每类都对应一个具体的失效机理：

  cancel      a≈b，相减后有效位大量丢失；其中定向构造指数差恰为 1 的组合（fsub.s 缺陷的触发条件）。
  sign_flip   结果符号与两操作数都不同，检查符号位是否被正确推导。
  tiny        次正规数与极小值，检查是否被错误地清零。
  huge        接近 f32 上限，检查是否提前饱和。
  ratio       量级相差 2^20 以上，检查对阶时小数是否被整体丢弃。
  exact       可精确表示的值（2 的幂及其和），任何误差都必然是实现错误。
  boundary    0 / ±1 / 0.5 等特殊值，通常是 lowering 里 if 分支的边界。
  near_pow2   刚好跨过 2 的幂，指数递增处舍入行为易错。

每类生成的用例都记录**为什么危险**，失败时能直接读出触发条件，
不用再回头猜是哪种输入引发的。

结果写入 fp_fuzz.json（路径见 OUT）
"""
import argparse
import json
import math
import os
import random
import struct
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根
for p in [x for x in (os.environ.get("FLAGTREE_PY"), WS,
                      os.path.join(WS, "triton_kernels")) if x]:
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.environ.get("FP_FUZZ_OUT", "fp_fuzz.json")
N_LANE = 8


def f32(x):
    """截到 f32，避免 host 侧用 f64 算出设备达不到的精度。

    超出 f32 范围时返回 inf 而不是抛 OverflowError：huge 类的相加本来就会
    溢出，那是要观测的行为（设备是否也给 inf），不是脚本错误。
    """
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.inf if x > 0 else -math.inf


def ulp(x):
    """x 处的 f32 ulp 间距，用于构造"只差几个 ulp"的相消对"""
    if x == 0.0:
        return 1.4e-45
    b = struct.unpack("<I", struct.pack("<f", abs(x)))[0]
    return abs(f32(struct.unpack("<f", struct.pack("<I", b + 1))[0]) - abs(x))


# ---------------------------------------------------------------- 危险类别
def expof(x):
    """f32 的偏置指数字段（bits 30:23）"""
    if x == 0.0:
        return 0
    return (struct.unpack("<I", struct.pack("<f", x))[0] >> 23) & 0xFF


def gen_cancel(rng, n):
    """a≈b 且**指数不同**：需要对阶的减法。

    fsub.s 缺陷的触发条件是对阶，与尾数相消程度无关：
        0.6-1.0  (exp 126 vs 127, 差 1) -> +0.4  错
        0.6-0.9  (exp 126 vs 126, 差 0) -> -0.3  对
    也就是说这是**对阶路径**上的符号处理错误，与相消深度无关。
    注意 a=1.0、b∈[1.05,1.8] 这样的扫描抓不到它：两者指数都是 127，
    指数差恒为 0，整条扫描线都落在安全区。

    用 a=1.5, b=1.5*2^k 扫描可见，触发窗口为**指数差恰好为 1**：
        expdiff=1  1.5-3.0   -> +1.5    错
        expdiff=2  1.5-6.0   -> -4.5    对
        expdiff>=2 全部正确
    窗口只有一格宽，正是人手挑代表点几乎必然漏掉的形状。所以本类
    定向构造 expdiff==1。
    """
    out = []
    for _ in range(n):
        a = f32(rng.uniform(0.1, 10.0))
        # 取 b 落在 a 的下一个二进制量级内，使 expof(b)-expof(a) 恰为 1
        lo = f32(2.0 ** (expof(a) - 127 + 1))
        b = f32(rng.uniform(lo, lo * 1.999))
        if expof(b) - expof(a) != 1:      # 舍入把 b 顶进下一档，重取
            b = f32(lo * 1.5)
        why = f"exp 差 {expof(b) - expof(a)}，需对阶且结果为负"
        if rng.random() < 0.35:
            a, b = b, a
            why = f"exp 差 {expof(a) - expof(b)}，需对阶且结果为正"
        out.append((a, b, why))
    return out


def gen_same_exp(rng, n):
    """a≈b 但**指数相同**：对照组。

    单列一类是为了把"相消"和"对阶"两个因素分开。如果只有 cancel 类失败
    而本类全对，就能断定缺陷在对阶而非相消 —— 否则两者混在一起，
    看到失败也说不清是哪个原因。
    """
    out = []
    for _ in range(n):
        a = f32(rng.uniform(1.0, 1.9))
        s = 2.0 ** rng.randint(-6, 6)
        a = f32(a * s)
        # 同指数区间内取 b，尾数相消但无需对阶
        base = f32(2.0 ** (expof(a) - 127))
        b = f32(rng.uniform(base, base * 1.999))
        if expof(b) != expof(a):
            b = a
        out.append((a, b, "exp 相同，仅相消不对阶（对照组）"))
    return out


def gen_sign_flip(rng, n):
    """结果符号与两个操作数都不同 —— 符号位必须由大小关系推出而非照抄。

    刻意让 |b| 跨过 a 的量级，与 cancel 类同样保证指数差非零：
    符号推导和对阶在硬件里走同一条路径，分开测才知道错在哪一步。
    """
    out = []
    for _ in range(n):
        a = f32(rng.uniform(0.1, 5.0))
        lo = f32(2.0 ** (expof(a) - 127 + 1))
        b = f32(rng.uniform(lo, lo * 1.999))        # expdiff == 1
        if rng.random() < 0.5:
            out.append((a, b, "正-正得负，需对阶且符号取自 b"))
        else:
            out.append((f32(-a), f32(-b), "负-负得正，需对阶且符号取自 b"))
    return out


def gen_tiny(rng, n):
    """次正规数与极小值：容易被错误清零，或在乘除中提前下溢"""
    out = []
    for _ in range(n):
        e = rng.randint(-140, -120)
        a = f32(2.0 ** e * rng.uniform(1.0, 2.0))
        b = f32(2.0 ** (e + rng.randint(-2, 2)))
        out.append((a, b, f"次正规区 2^{e}，检查是否被清零"))
    return out


def gen_huge(rng, n):
    """接近 f32 上限：检查是否提前饱和到 inf"""
    out = []
    for _ in range(n):
        e = rng.randint(120, 127)
        a = f32(2.0 ** e * rng.uniform(1.0, 1.9))
        b = f32(2.0 ** (e - rng.randint(0, 3)))
        out.append((a, b, f"接近上限 2^{e}，检查是否提前饱和"))
    return out


def gen_ratio(rng, n):
    """量级差 2^20 以上：对阶时小数可能被整体丢弃"""
    out = []
    for _ in range(n):
        a = f32(rng.uniform(1.0, 100.0))
        k = rng.randint(20, 30)
        b = f32(a / (2.0 ** k))
        out.append((a, b, f"量级差 2^{k}，对阶后小数应仍有贡献"))
    return out


def gen_exact(rng, n):
    """可精确表示的值：任何非零误差都必然是实现错误，不能推给精度"""
    pool = [0.5, 1.0, 2.0, 4.0, 8.0, 0.25, 0.125, 3.0, 6.0, 12.0,
            1.5, 2.5, 3.5, 1024.0, 4096.0]
    out = []
    for _ in range(n):
        a = rng.choice(pool)
        b = rng.choice(pool)
        out.append((f32(a), f32(b), "两者均可精确表示，误差必须为 0"))
    return out


def gen_boundary(rng, n):
    """0 / ±1 / 0.5 等特殊值：lowering 里 if 分支的边界"""
    pool = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0]
    out = []
    for _ in range(n):
        a = rng.choice(pool)
        b = rng.choice(pool)
        out.append((f32(a), f32(b), "特殊值组合，覆盖 lowering 分支边界"))
    return out


def gen_near_pow2(rng, n):
    """刚好跨过 2 的幂：指数递增处的舍入最易出错"""
    out = []
    for _ in range(n):
        e = rng.randint(-8, 8)
        p = 2.0 ** e
        a = f32(p - ulp(p))
        b = f32(p + ulp(p) * 2)
        out.append((a, b, f"跨越 2^{e}，指数递增处舍入"))
    return out


GENERATORS = {
    "cancel":    gen_cancel,
    "same_exp":  gen_same_exp,
    "sign_flip": gen_sign_flip,
    "tiny":      gen_tiny,
    "huge":      gen_huge,
    "ratio":     gen_ratio,
    "exact":     gen_exact,
    "boundary":  gen_boundary,
    "near_pow2": gen_near_pow2,
}


# ---------------------------------------------------------------- 参考实现
# 全部用 f32 逐步求值，运算顺序与设备一致。
def ref_add(a, b):
    return f32(a + b)


def ref_sub(a, b):
    return f32(a - b)


def ref_mul(a, b):
    return f32(a * b)


def ref_div(a, b):
    return None if b == 0.0 else f32(a / b)


def ref_min(a, b):
    return f32(min(a, b))


def ref_max(a, b):
    return f32(max(a, b))


def ref_abs(a, b):
    return f32(abs(a) + f32(b * 0.0))


def ref_sqrt(a, b):
    return None if a < 0.0 else f32(math.sqrt(a) + f32(b * 0.0))


def ref_select(a, b):
    return f32(a if a < b else b)


def ref_fma(a, b):
    return f32(f32(a * b) - a)


REFS = {"add": ref_add, "sub": ref_sub, "mul": ref_mul, "div": ref_div,
        "min": ref_min, "max": ref_max, "abs": ref_abs, "sqrt": ref_sqrt,
        "select": ref_select, "fma": ref_fma}


# 已知硬件缺陷：本生成器发现的 fadd.s 溢出返 0（而非 +inf）。
# 归为"已知"而非"失败"，否则每次运行都有 10 个红色，真正的新问题会被淹没。
# 判据写成函数而不是硬编码用例列表，这样换 seed 也仍然认得出来。
def is_known_defect(op, ref, got):
    """fadd 路径上的溢出：期望 ±inf，设备给 0"""
    if got != 0.0 or not math.isinf(ref):
        return False
    # sub 走 a+(-b)，fma 的 a*b 先溢出再走 fadd，都经过同一条加法路径
    return op in ("add", "sub", "fma")


def rel_err(got, ref):
    """相对误差；ref 为 0 时退化为绝对误差。

    不能只看绝对误差：huge 类的量级在 1e38，绝对误差天然巨大；
    也不能只看相对误差：cancel 类的 ref 可能接近 0，相对误差会爆炸。
    两者取小，才能对所有类别用同一个阈值。

    inf/nan 单独处理：这两者参与减法后相对误差无意义，只能判"是否一致"。
    """
    if math.isnan(ref) or math.isnan(got):
        return 0.0 if (math.isnan(ref) and math.isnan(got)) else math.inf
    if math.isinf(ref) or math.isinf(got):
        return 0.0 if (ref == got) else math.inf
    if ref == 0.0:
        return abs(got)
    return min(abs(got - ref) / abs(ref), abs(got - ref))


# ---------------------------------------------------------------- 执行
def build_cases(ops, cats, per_cat, seed):
    """为每个 (op, cat) 生成 per_cat 个用例，按 8 lane 打包成批"""
    rng = random.Random(seed)
    batches = []
    for op in ops:
        for cat in cats:
            cases = GENERATORS[cat](rng, per_cat)
            # div/sqrt 的定义域限制：过滤掉参考值为 None 的项，
            # 否则会把"host 也算不出来"误判成设备错误
            cases = [c for c in cases if REFS[op](c[0], c[1]) is not None]
            for i in range(0, len(cases), N_LANE):
                chunk = cases[i:i + N_LANE]
                if len(chunk) < N_LANE:      # 补齐到 8 lane
                    chunk += [chunk[-1]] * (N_LANE - len(chunk))
                batches.append({"op": op, "cat": cat, "cases": chunk})
    return batches


def run_batch(batch, cycles, tag):
    """编译并在 emu 上跑一批 8 个用例，返回设备端 8 个结果"""
    from jit_openbpu3 import jit_compile
    from jit_fp_probe import PROBES, SIG2
    import run_triton_kernel as RT

    r, _ = jit_compile(PROBES[batch["op"]], SIG2)
    out = r["arg_base"] + 2 * r["arg_stride"]
    A = [c[0] for c in batch["cases"]]
    B = [c[1] for c in batch["cases"]]
    v, st = RT.run(RT.append_readback(r["words"], out, N_LANE),
                   {0: A, 1: B}, out, N_LANE, cycles=cycles, tag=tag)
    return v, st, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default="sub,add,mul,max,select,fma",
                    help="逗号分隔的探针名，默认覆盖历史失效点相关的运算")
    ap.add_argument("--cats", default=",".join(GENERATORS),
                    help="逗号分隔的危险类别")
    ap.add_argument("--per-cat", type=int, default=8,
                    help="每个 (op,cat) 生成的用例数")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="相对/绝对误差阈值（取小者）")
    ap.add_argument("--cycles", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--max-batches", type=int, default=0,
                    help="限制批数（调试用，0=不限）")
    ap.add_argument("--strict", action="store_true",
                    help="不豁免已知硬件缺陷，任何偏差都算失败")
    a = ap.parse_args()

    ops = [x for x in a.ops.split(",") if x]
    cats = [x for x in a.cats.split(",") if x]
    for op in ops:
        if op not in REFS:
            print(f"未知探针 {op}，可用: {', '.join(REFS)}")
            return 2
    for c in cats:
        if c not in GENERATORS:
            print(f"未知类别 {c}，可用: {', '.join(GENERATORS)}")
            return 2

    batches = build_cases(ops, cats, a.per_cat, a.seed)
    if a.max_batches:
        batches = batches[:a.max_batches]

    print("=" * 74)
    print("危险浮点用例自动生成器")
    print("=" * 74)
    print(f"运算 {len(ops)} 种 x 类别 {len(cats)} 种 -> {len(batches)} 批 "
          f"({len(batches) * N_LANE} 个用例), 阈值 {a.tol:.0e}")
    print(f"seed={a.seed}  每批一次 emu 运行, 单批约 10~20 秒\n")

    res = {"config": {"ops": ops, "cats": cats, "per_cat": a.per_cat,
                      "tol": a.tol, "seed": a.seed, "cycles": a.cycles,
                      "strict": a.strict},
           "batches": [], "failures": [], "known": []}
    n_case = n_fail = n_known = 0
    per_cat_stat = {}

    for bi, b in enumerate(batches):
        key = f"{b['op']}/{b['cat']}"
        try:
            v, st, r = run_batch(b, a.cycles, f"fz{bi}")
        except Exception as e:
            print(f"[{bi + 1}/{len(batches)}] {key:20s} 编译/运行失败 "
                  f"{type(e).__name__}: {e}")
            res["batches"].append({"op": b["op"], "cat": b["cat"],
                                   "error": f"{type(e).__name__}: {e}"})
            continue
        if len(v) < N_LANE:
            print(f"[{bi + 1}/{len(batches)}] {key:20s} 回读不足 "
                  f"{len(v)}/{N_LANE}")
            res["batches"].append({"op": b["op"], "cat": b["cat"],
                                   "error": f"readback {len(v)}/{N_LANE}"})
            continue

        # tiny/huge 的偏差多半是硬件的次正规/溢出处理，不是 codegen 错误。
        # 用同一个 1e-6 判它们会淹掉真正的缺陷信号，所以放宽并单独报告。
        tol = a.tol * (1e3 if b["cat"] in ("tiny", "huge") else 1.0)
        worst, bad, known = 0.0, [], []
        for k, (x, y, why) in enumerate(b["cases"]):
            ref = REFS[b["op"]](x, y)
            e = rel_err(v[k], ref)
            n_case += 1
            if e <= tol:
                worst = max(worst, e)
                continue
            d = {"lane": k, "a": x, "b": y, "why": why,
                 "ref": ref, "got": v[k], "err": e}
            if not a.strict and is_known_defect(b["op"], ref, v[k]):
                known.append(d)
                continue
            worst = max(worst, e)
            n_fail += 1
            bad.append(d)
        st_c = per_cat_stat.setdefault(b["cat"], {"n": 0, "bad": 0,
                                                  "known": 0, "worst": 0.0})
        st_c["n"] += N_LANE
        st_c["bad"] += len(bad)
        st_c["known"] += len(known)
        st_c["worst"] = max(st_c["worst"], worst)
        n_known += len(known)

        res["batches"].append({"op": b["op"], "cat": b["cat"],
                               "worst_err": worst, "n_bad": len(bad),
                               "n_known": len(known),
                               "tol": tol, "cycles": st.get("cycles"),
                               "n_inst": r["n_inst"]})
        res["failures"].extend(
            dict(op=b["op"], cat=b["cat"], **d) for d in bad)
        res["known"].extend(
            dict(op=b["op"], cat=b["cat"], **d) for d in known)
        flag = f"BAD {len(bad)}/{N_LANE}" if bad else (
            f"known {len(known)}" if known else "OK  ")
        print(f"[{bi + 1}/{len(batches)}] {key:20s} 最大误差 {worst:.2e}  "
              f"{flag}")
        if bad or known:
            d = (bad or known)[0]
            print(f"      例: {d['a']:.9g} op {d['b']:.9g} -> "
                  f"{d['got']:.9g}, 期望 {d['ref']:.9g}   [{d['why']}]")

    res["summary"] = {"n_case": n_case, "n_fail": n_fail,
                      "n_known": n_known, "per_cat": per_cat_stat,
                      "all_ok": n_fail == 0}
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 74)
    print(f"{'类别':<12s}{'用例':>6s}{'新失败':>8s}{'已知缺陷':>10s}"
          f"{'最大误差':>14s}")
    for c in cats:
        s = per_cat_stat.get(c)
        if not s:
            continue
        print(f"{c:<12s}{s['n']:>6d}{s['bad']:>8d}{s['known']:>10d}"
              f"{s['worst']:>14.2e}")
    print("-" * 74)
    print(f"{n_case} 个用例: {n_fail} 个新失败, {n_known} 个已知硬件缺陷"
          f"（fadd.s 溢出返 0）, 阈值 {a.tol:.0e}"
          f" -> {'PASS' if n_fail == 0 else 'FAIL'}")
    print(f"已写入 {OUT}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
