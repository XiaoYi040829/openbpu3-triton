"""
FlagTree Polaris Backend — driver.py

OpenBPU-3（PolarisGPU）设备驱动，接入 FlagTree。实现 DriverBase 接口：
  - is_active()           : 后端是否可用（不依赖实体硬件，始终可用）
  - get_current_target()  : 返回 GPUTarget(polaris, polaris_gpu, warp_size=8)
  - utils                 : 设备属性
  - launcher_cls          : kernel 启动器（见 PolarisLauncher）

本后端没有 Triton 运行时启动器：`kernel[grid](...)` 会报错。
编译用 triton_kernels/jit_openbpu3.py 的 jit_compile()，
执行用 triton_kernels/run_triton_kernel.py 的 run()（周期精确 RTL 模拟器）。

参照 third_party/nvidia/backend/driver.py 实现。
"""
import os
from dataclasses import dataclass
from pathlib import Path


def _import_flagtree_driver_base():
    """加载 FlagTree 的 DriverBase（纯 Python 抽象基类，不依赖 _C/torch）。
    GPUDriver.__init__ 依赖 torch.cuda，无 GPU 环境不可用，
    故 polaris 直接继承 DriverBase。"""
    try:
        from triton.backends.driver import DriverBase
        return DriverBase
    except Exception:
        pass
    import importlib.util
    here = Path(__file__).resolve()
    candidates = []
    ft_env = os.environ.get("FLAGTREE_HOME")
    if ft_env:
        candidates.append(Path(ft_env) / "python" / "triton" / "backends" / "driver.py")
    candidates += [here.parents[3] / "python" / "triton" / "backends" / "driver.py",
                   here.parent.parent / "driver.py"]
    for c in candidates:
        if c.exists():
            spec = importlib.util.spec_from_file_location("flagtree_backends_driver", str(c))
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m.DriverBase
    raise ImportError("FlagTree DriverBase not found")


DriverBase = _import_flagtree_driver_base()


@dataclass(frozen=True)
class GPUTarget:
    backend: str
    arch: str
    warp_size: int


class PolarisUtils:
    """设备属性。数值取自 OpenBPU-3 RTL（NutCore.scala / NutShell.scala）。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_device_properties(self, device):
        return {
            "max_shared_mem": 0,             # SM 内无共享内存实体
            "multiprocessor_count": 2,       # NutShell.scala: SM_0 / SM_1
            "warpSize": 8,                   # NutCore.scala: SIMT_Width = 8
            "sm_clock_rate": 20_000,         # kHz；QV19P 上核时钟 20 MHz
            "mem_clock_rate": 0,             # 未标定
            "mem_bus_width": 64,             # 核 AXI 数据位宽
        }

    def load_binary(self, name, kernel, shared, device):
        # (module, function_handle, n_regs, n_spills)；本后端不生成 spill 代码
        return (None, name, 32, 0)


class PolarisLauncher:
    """本后端没有 Triton 运行时启动器。

    调用时直接报错，而不是静默返回：否则 `kernel[grid](...)` 看似成功，
    实际什么都没执行。
    """

    def __init__(self, src, metadata):
        self.src = src
        self.metadata = metadata

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "polaris backend: kernel[grid](...) is not supported (no runtime "
            "launcher). Compile with triton_kernels/jit_openbpu3.jit_compile() "
            "and execute with triton_kernels/run_triton_kernel.run() on the "
            "cycle-accurate RTL simulator.")


class PolarisDriver(DriverBase):
    """OpenBPU-3 FlagTree 驱动（继承 FlagTree DriverBase）"""

    def __init__(self):
        self.utils = PolarisUtils()
        self.launcher_cls = PolarisLauncher
        self.binary_ext = "bpubin"
        self._device = 0

    @staticmethod
    def is_active():
        # 不依赖实体硬件，始终可用
        return True

    def get_current_target(self):
        return GPUTarget("polaris", "polaris_gpu", 8)

    def get_current_device(self):
        return self._device

    def set_current_device(self, device):
        self._device = device

    def get_current_stream(self, device=None):
        return 0

    def get_device_capability(self, device=0):
        return (1, 0)

    def get_benchmarker(self):
        import time

        def bench(fn, *args, **kwargs):
            t0 = time.perf_counter()
            fn()
            return (time.perf_counter() - t0) * 1e3
        return bench

    def __repr__(self):
        return "PolarisDriver(active=True, target=polaris_gpu)"
