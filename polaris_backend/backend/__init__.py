"""
FlagTree Polaris Backend — OpenBPU-3（PolarisGPU）

FlagOS 统一编译器（FlagTree）的 OpenBPU-3 后端入口。
FlagTree 通过 compiler.py / driver.py 自动发现 Backend / Driver 类。
"""
from .compiler import PolarisBackend, PolarisOptions
from .driver import PolarisDriver

__all__ = ["PolarisBackend", "PolarisOptions", "PolarisDriver"]
