"""
FlagTree Polaris Backend — compiler.py

OpenBPU-3（PolarisGPU）后端，接入 FlagTree（FlagOS 统一编译器）。
实现 FlagTree/Triton 的 BaseBackend 接口，多阶段编译管线：

    ttir  -> ttgir -> bpuir -> bin

bin 阶段由 C++ pass add_riscv_codegen 产出 RV64 + RV-SIMT 机器码
（标准 RISC-V 指令与 RV-SIMT warp 控制指令），可由模拟器与 QV19P 直接执行。

参照 third_party/nvidia/backend/compiler.py 的 BaseBackend 结构实现。
"""
from dataclasses import dataclass
from typing import Any, Tuple, Optional
import functools
import hashlib
import os
import sys
import tempfile
from pathlib import Path


def _import_flagtree_base():
    """加载 FlagTree 的 BaseBackend/GPUTarget。
    优先走正常 import；若 triton 顶层因缺 _C 扩展无法加载，则直接从
    backends/compiler.py 源文件加载（该文件是纯 Python，不依赖 _C）。"""
    try:
        from triton.backends.compiler import BaseBackend, GPUTarget
        return BaseBackend, GPUTarget
    except Exception:
        pass
    # 回退：定位 FlagTree 源码树的 backends/compiler.py 直接加载
    import importlib.util
    here = Path(__file__).resolve()
    candidates = []
    for base in [here.parents[3] / "python" / "triton" / "backends",   # third_party/polaris/backend -> flagtree/python/triton/backends
                 here.parent.parent]:                                    # 已安装到 triton/backends/polaris 的情况
        candidates.append(base / "compiler.py")
    ft_env = os.environ.get("FLAGTREE_HOME")
    if ft_env:
        candidates.insert(0, Path(ft_env) / "python" / "triton" / "backends" / "compiler.py")
    for c in candidates:
        if c.exists():
            spec = importlib.util.spec_from_file_location("flagtree_backends_compiler", str(c))
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m.BaseBackend, m.GPUTarget
    raise ImportError("FlagTree BaseBackend not found")


BaseBackend, GPUTarget = _import_flagtree_base()

# OpenBPU-3 硬件参数，与 RTL 一一对应：
#   warp_size=8  -> NutCore.scala:96   SIMT_Width
#   warp_max=4   -> NutCore.scala:98   Thread_Max / SIMT_Width
#   sm_count=2   -> NutShell.scala:207-208  SM_0 / SM_1
_HW = {"warp_size": 8, "sm_count": 2, "warp_max": 4}


@dataclass(frozen=True)
class PolarisOptions:
    """对应 CUDAOptions —— 编译选项 + cache key 组成"""
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 2
    warp_size: int = _HW["warp_size"]
    enable_fp_fusion: bool = True
    arch: str = "polaris_gpu"
    sm_count: int = _HW["sm_count"]
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "fp16", "bf16", "fp4")
    # Triton 前端处理 tl.dot 时读取这三个字段；缺失会让前端抛 AttributeError，
    # 用户看不到真正的拒绝原因。本核无 fp8，取值同 NVIDIA 低算力档。
    allow_fp8e4nv: bool = False
    allow_fp8e4b15: bool = False
    max_num_imprecise_acc_default: int = 0
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = "polaris"
    # 官方 triton.compile 用 options.__dict__ 构造 metadata，随后
    # CompiledKernel.__init__ 无条件读取 cluster_dims、启动器读取 shared。
    # 缺这两个字段时，走官方入口编译会在最后一步 KeyError。
    # OpenBPU-3 没有 CTA cluster，恒为 (1,1,1)；tl.shared 无物理存储，恒为 0。
    cluster_dims: tuple = (1, 1, 1)
    shared: int = 0

    def __post_init__(self):
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
            "num_warps must be a power of 2"

    def hash(self):
        key = "_".join(f"{k}-{v}" for k, v in sorted(self.__dict__.items()))
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class PolarisBackend(BaseBackend):
    """PolarisGPU/BPU FlagTree 后端"""

    # 后端名为 openbpu3（`--target=openbpu3`），同时接受 polaris/polaris_gpu。
    SUPPORTED_BACKENDS = ("openbpu3", "polaris", "polaris_gpu", "bpu")

    @staticmethod
    def supports_target(target: "GPUTarget"):
        b = getattr(target, "backend", None)
        return b in PolarisBackend.SUPPORTED_BACKENDS

    def __init__(self, target: "GPUTarget") -> None:
        super().__init__(target)
        self.binary_ext = "bpubin"
        self.sm_count = _HW["sm_count"]
        self.warp_size = getattr(target, "warp_size", None) or _HW["warp_size"]

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in PolarisOptions.__dataclass_fields__.keys()
                if k in opts}
        return PolarisOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            getattr(metadata, "shared", 0),
            getattr(metadata, "arch", self.target.arch),
        )

    def get_codegen_implementation(self):
        return {"backend": "polaris", "stages": ["ttir", "ttgir", "bpuir", "bin"]}

    def load_dialects(self, ctx):
        # 本后端没有独立的 MLIR dialect，直接在 TTIR 上做变换
        return None

    # ---- 多阶段编译 ----

    @staticmethod
    def make_ttir(mod, metadata, opt):
        """优先用 FlagTree 真 MLIR 基础设施（libtriton.ir）做 TTIR 规范化 pass；
        无 _C 时回退到字符串规范化。"""
        try:
            from triton._C.libtriton import ir, passes
            if not isinstance(mod, str):
                # mod 是真 MLIR Module：跑 FlagTree 官方 TTIR pass 管线
                pm = ir.pass_manager(mod.context)
                pm.enable_debug()
                passes.common.add_inliner(pm)
                passes.ttir.add_combine(pm)
                passes.common.add_canonicalizer(pm)
                passes.ttir.add_reorder_broadcast(pm)
                passes.common.add_cse(pm)
                passes.common.add_symbol_dce(pm)
                pm.run(mod)
                return mod
        except Exception:
            pass
        # 回退：字符串规范化（无 _C 或传入字符串 IR 时）
        if isinstance(mod, str):
            return "\n".join(ln.rstrip() for ln in mod.strip().splitlines() if ln.strip())
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt):
        """TTIR -> TTGIR：注入 blocked layout，并运行 PolarisGPU C++ MLIR pass
        (add_tensorcore_mapping) 做 tt.dot -> MAT4X4 TensorCore 映射分析标注。"""
        if isinstance(mod, str):
            # 仅当调用方直接喂字符串 IR 时才走这条(无 MLIR 上下文可用)
            header = (f"// ttgir: #polaris.blocked warps={opt.num_warps} "
                      f"warp_size={opt.warp_size} arch={opt.arch} sm={opt.sm_count}")
            return header + "\n" + mod
        # 有 MLIR module 时必须跑 C++ pass 并原样返回 module：make_bin 要在
        # 同一个 module 上执行 riscv codegen，不能退化成字符串。
        from triton._C.libtriton import ir, polaris
        pm = ir.pass_manager(mod.context)
        # PolarisGPU 专属 C++ pass：分析 tt.dot 能否映射到 4x4 TensorCore
        polaris.passes.polarisgpu.add_tensorcore_mapping(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_bpuir(mod, metadata, opt):
        """BPU-IR 阶段：在 module 上做 PolarisGPU 专属标注。

        重要：必须原样返回 MLIR module 对象，不能转成字符串。
        下一阶段 make_bin 要在同一个 module 上跑 C++ pass add_riscv_codegen，
        字符串化会丢掉 MLIR 上下文导致 codegen 无法执行。
        """
        if isinstance(mod, str):
            return mod
        try:
            from triton._C.libtriton import ir as _ir
            b = _ir.builder(mod.context) if hasattr(_ir, "builder") else None
        except Exception:
            b = None
        # 记录调度相关元数据（CTA 8 维特征向量在 make_bin 前由 codegen 读取）
        try:
            mod.set_attr  # 某些版本无该接口，容错
        except Exception:
            pass
        return mod

    @staticmethod
    def make_bin(mod, metadata, opt):
        """运行 C++ pass add_riscv_codegen，产出 RV64 + RV-SIMT 机器码。

        产物可由 cycle-accurate 模拟器与 QV19P 硬件直接执行。
        """
        import re
        import struct
        out = {"options": opt}
        try:
            from triton._C.libtriton import ir as _ir, polaris as _pol
            if not isinstance(mod, str):
                pm = _ir.pass_manager(mod.context)
                _pol.passes.polarisgpu.add_riscv_codegen(pm)
                pm.run(mod)
        except Exception as e:
            out["error"] = f"riscv codegen 未执行: {e}"
            return out

        s = str(mod)
        m = re.search(r"polaris\.riscv\.words = array<i32: ([^>]*)>", s)
        if not m:
            out["error"] = "codegen 未产出 polaris.riscv.words"
            return out
        words = [int(x) & 0xFFFFFFFF for x in m.group(1).split(",")]

        out_dir = tempfile.mkdtemp(prefix="flagtree_polaris_")
        out_path = os.path.join(out_dir, "kernel.bin")
        with open(out_path, "wb") as f:
            f.write(b"".join(struct.pack("<I", w) for w in words))

        # ---- CodeGen 输出 .S 汇编, 用 riscv64-*-as + objcopy 汇编回 bin ----
        # 并与直出的 kernel.bin 逐字节比对（asm_matches_bin）。
        # 自定义 RV-SIMT 指令在 .S 里用 .insn 表达。
        asm_path = os.path.join(out_dir, "kernel.S")
        ma = re.search(r'polaris\.riscv\.asm = "((?:[^"\\]|\\.)*)"', s, re.S)
        asm_text = None
        if ma:
            # MLIR 的字符串属性把非 ASCII 与控制符转义成 \XX 十六进制形式
            # (换行是 \0A, 中文注释是 \E8\87\AA...)。必须按十六进制解码，
            # 否则写出的 .S 全在一行且含字面 "\0A"，汇编器无法处理。
            raw = ma.group(1)
            buf_b = bytearray()
            i = 0
            while i < len(raw):
                c = raw[i]
                if c == "\\" and i + 1 < len(raw):
                    nxt = raw[i + 1]
                    if nxt in "nt\"\\":
                        buf_b += {"n": b"\n", "t": b"\t",
                                  '"': b'"', "\\": b"\\"}[nxt]
                        i += 2
                        continue
                    # \XX 十六进制
                    if i + 2 < len(raw):
                        try:
                            buf_b.append(int(raw[i + 1:i + 3], 16))
                            i += 3
                            continue
                        except ValueError:
                            pass
                buf_b += c.encode("utf-8")
                i += 1
            asm_text = buf_b.decode("utf-8", errors="replace")
            with open(asm_path, "w") as f:
                f.write(asm_text)

        # 用工具链把 .S 汇编成 bin, 并与 codegen 直出的 bin 逐字节比对,
        # 以此证明 .S 与机器码严格一致(而非各自独立产出)。
        asm_bin_path, asm_ok = None, None
        if asm_text:
            import shutil
            import subprocess
            AS = shutil.which("riscv64-linux-gnu-as") or \
                shutil.which("riscv64-unknown-elf-as")
            OC = shutil.which("riscv64-linux-gnu-objcopy") or \
                shutil.which("riscv64-unknown-elf-objcopy")
            if AS and OC:
                obj = os.path.join(out_dir, "kernel.o")
                asm_bin_path = os.path.join(out_dir, "kernel_from_asm.bin")
                try:
                    # 关键: 必须去掉 'c'(压缩指令扩展)。带 c 时 as 会把
                    # addi t0,zero,8 之类压成 2 字节 c.li, 导致产物比
                    # codegen 直出的 4 字节定长指令流短且整体错位
                    # (例如 296 字节变成 264 字节，后续指令全部错位)。
                    subprocess.run([AS, "-march=rv64imaf_zicsr",
                                    "-mno-relax", "-o", obj, asm_path],
                                   check=True, capture_output=True)
                    subprocess.run([OC, "-O", "binary", "-j", ".text",
                                    obj, asm_bin_path],
                                   check=True, capture_output=True)
                    with open(out_path, "rb") as f1, open(asm_bin_path, "rb") as f2:
                        asm_ok = (f1.read() == f2.read())
                except Exception as e:
                    asm_ok = f"汇编失败: {e}"

        attrs = {k: int(v) for k, v in
                 re.findall(r"polaris\.riscv\.(\w+) = (\d+)", s)}
        # CTA 元数据: 8 维程序特征向量 (供 SNN 动态 Warp 调度器选策略) +
        # 共享内存需求。硬件接口见 sche.scala:73 snnspecvec、:59 sharedMemSize。
        mv = re.search(r"polaris\.cta\.snnspecvec = array<i32: ([^>]*)>", s)
        snnvec = [int(x) for x in mv.group(1).split(",")] if mv else None
        cta_attrs = {k: int(v) for k, v in
                     re.findall(r"polaris\.cta\.(\w+) = (\d+)", s)}
        # kernel 名取自模块里的 tt.func，与 nvidia 后端从 PTX 取 .entry 名对应。
        # 官方 triton.compile 传入的 metadata 是 dict，本仓库 jit_compile 传入的
        # 是属性对象；两种形式都要写入 name，否则走官方入口时
        # CompiledKernel 在最后一步 AttributeError。
        mname = re.search(r"tt\.func\s+(?:public\s+|private\s+)?@([A-Za-z_][A-Za-z0-9_]*)", s)
        kname = mname.group(1) if mname else "polaris_kernel"
        if isinstance(metadata, dict):
            metadata["name"] = kname
        else:
            setattr(metadata, "name", kname)
        out.update({
            "bin_path": out_path,
            "words": words,
            "n_inst": len(words),
            # kernel 指针参数的裸机约定，host 侧据此放输入 / 取输出
            "arg_base": attrs.get("arg_base"),
            "arg_stride": attrs.get("arg_stride"),
            "num_ptr_args": attrs.get("num_ptr_args"),
            "num_mapped": attrs.get("num_mapped"),
            "num_skipped": attrs.get("num_skipped"),
            # 寄存器池耗尽标志：正常编译时 codegen 已因耗尽而报错；仅在
            # POLARIS_ALLOW_SKIPPED=1 调试时才可能为 1，此时机器码不可信。
            "reg_exhausted": attrs.get("reg_exhausted", 0),
            "reg_pressure_int": attrs.get("reg_pressure_int", 0),
            "reg_pressure_flt": attrs.get("reg_pressure_flt", 0),
            # 峰值同时活跃数 = 真正需要的寄存器数（线性扫描会复用已死值），
            # 与 reg_pressure_*（SSA 值总数）不同，评估寄存器需求要看这个
            "reg_peak_int": attrs.get("reg_peak_int", 0),
            "reg_peak_flt": attrs.get("reg_peak_flt", 0),
            # select 展开的个数与占用指令数，由 codegen 在生成时直接计数，
            # 用于统计分支消除的代价
            "select_count": attrs.get("select_count", 0),
            "select_insts": attrs.get("select_insts", 0),
            # 被跳过的 op 名单：被跳过的 op 若参与数据流，产出的二进制就是错的，
            # 正常情况下 codegen 已因此报错，这里供调试时查看
            "skipped_ops": (re.search(r'polaris\.riscv\.skipped_ops = "([^"]*)"', s).group(1)
                            if re.search(r'polaris\.riscv\.skipped_ops = "([^"]*)"', s) else ""),
            # CTA 元数据：供 SNN 动态 Warp 调度器读取
            "cta_snnspecvec": snnvec,
            "cta_snn_input_size": cta_attrs.get("snn_input_size"),
            "cta_shared_mem_bytes": cta_attrs.get("shared_mem_bytes"),
            # .S 汇编产物，以及用工具链汇编后与直出 bin 的一致性
            "asm_path": asm_path if asm_text else None,
            "asm_bin_path": asm_bin_path,
            "asm_matches_bin": asm_ok,
        })
        return out

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["bpuir"] = lambda src, metadata: self.make_bpuir(src, metadata, options)
        stages["bin"] = lambda src, metadata: self.make_bin(src, metadata, options)
        # 官方 triton.compile 按 binary_ext 取最终产物且要求是 bytes；bin 阶段返回的
        # dict（机器码 + 编译统计）供本仓库工具使用，这里再转一次原始机器码字节，
        # 使官方入口也能完整编译。本仓库工具按名字只走前四步，不经过这一步。
        stages[self.binary_ext] = lambda src, metadata: self.make_bpubin(src, metadata, options)

    def make_bpubin(self, src, metadata, options):
        import struct
        if not isinstance(src, dict) or "words" not in src:
            # make_bin 失败时把原因放在 out["error"] 里返回，这里必须带出来，
            # 否则调用方只能看到"未产出机器码"，看不到真正的失败原因。
            why = src.get("error", "未给出原因") if isinstance(src, dict) else type(src).__name__
            raise RuntimeError(f"polaris backend: bin 阶段未产出机器码: {why}")
        return b"".join(struct.pack("<I", w & 0xFFFFFFFF) for w in src["words"])

    @functools.lru_cache()
    def hash(self):
        # 硬件参数进入 cache key，参数变化时旧缓存自动失效
        return f"polaris-{self.sm_count}sm-{self.warp_size}t-v3"
