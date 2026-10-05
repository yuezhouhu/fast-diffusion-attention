# Copyright (c) 2026, fast-diffusion-attention authors.
# Build script adapted from FlashAttention-3 (hopper/setup.py).
#
# Environment variables controlling the build:
#   FDA_HDIMS              comma separated head dimensions to build (default: 64,96,128,192,256)
#   FDA_DISABLE_FP16       do not build the FP16 kernels
#   FDA_DISABLE_FP8        do not build the FP8 (e4m3) forward kernels
#   FDA_DISABLE_SOFTCAP    do not build the softcapping kernels
#   FDA_DISABLE_BACKWARD   do not build the backward kernels
#   FDA_LINEINFO=1         compile with -lineinfo (for profiling)
#   FDA_SKIP_CUDA_BUILD=1  only install the Python package
#   NVCC_THREADS           number of threads per nvcc invocation (default 4)
#   MAX_JOBS               number of parallel compilation jobs

import os
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup

this_dir = Path(__file__).resolve().parent
PACKAGE_NAME = "fast_diffusion_attention"
SRC_DIR = this_dir / "csrc" / "fda"
CUTLASS_DIR = this_dir / "csrc" / "cutlass"


def env_flag(name: str) -> bool:
    return os.getenv(name, "0").upper() in ["1", "ON", "YES", "TRUE", "Y"]


def get_version() -> str:
    version = {}
    exec((this_dir / PACKAGE_NAME / "version.py").read_text(), version)
    return version["__version__"]


ALL_HDIMS = [64, 96, 128, 192, 256]
HDIMS = [int(h) for h in os.getenv("FDA_HDIMS", "64,96,128,192,256").split(",") if h.strip()]
for h in HDIMS:
    assert h in ALL_HDIMS, f"Unsupported head dimension {h}, must be one of {ALL_HDIMS}"
DISABLE_FP16 = env_flag("FDA_DISABLE_FP16")
DISABLE_FP8 = env_flag("FDA_DISABLE_FP8")
DISABLE_SOFTCAP = env_flag("FDA_DISABLE_SOFTCAP")
DISABLE_BACKWARD = env_flag("FDA_DISABLE_BACKWARD")


def kernel_sources():
    dtypes_fwd = ["bf16"] + ([] if DISABLE_FP16 else ["fp16"]) + ([] if DISABLE_FP8 else ["e4m3"])
    dtypes_bwd = ["bf16"] + ([] if DISABLE_FP16 else ["fp16"])
    softcaps = [""] + ([] if DISABLE_SOFTCAP else ["_softcap"])
    srcs = []
    for hdim in HDIMS:
        for dtype in dtypes_fwd:
            for sc in softcaps:
                srcs.append(f"instantiations/flash_fwd_hdim{hdim}_{dtype}{sc}_sm90.cu")
        if not DISABLE_BACKWARD:
            for dtype in dtypes_bwd:
                for sc in softcaps:
                    srcs.append(f"instantiations/flash_bwd_hdim{hdim}_{dtype}{sc}_sm90.cu")
    return srcs


def feature_args():
    args = []
    for h in ALL_HDIMS:
        if h not in HDIMS:
            args.append(f"-DFLASHATTENTION_DISABLE_HDIM{h}")
    flags = {
        "FLASHATTENTION_DISABLE_BACKWARD": DISABLE_BACKWARD,
        "FLASHATTENTION_DISABLE_SOFTCAP": DISABLE_SOFTCAP,
        "FLASHATTENTION_DISABLE_FP16": DISABLE_FP16,
        "FLASHATTENTION_DISABLE_FP8": DISABLE_FP8,
    }
    args += [f"-D{k}" for k, v in flags.items() if v]
    return args


ext_modules = []
cmdclass = {}
if not env_flag("FDA_SKIP_CUDA_BUILD"):
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

    assert CUDA_HOME is not None, "nvcc not found, please set CUDA_HOME"
    if not (CUTLASS_DIR / "include").exists():
        subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], cwd=this_dir, check=True)
    subprocess.run([sys.executable, str(SRC_DIR / "generate_kernels.py")], check=True)
    nvcc_version = subprocess.check_output([os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"]).decode()
    if "release 13." in nvcc_version:
        # CUDA 13 ships CCCL (cub / thrust / libcu++) in include/cccl
        cccl_include = os.path.join(CUDA_HOME, "include", "cccl")
        for env_var in ["CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"]:
            current = os.environ.get(env_var, "")
            os.environ[env_var] = cccl_include + (":" + current if current else "")

    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--ftemplate-backtrace-limit=0",
        "--use_fast_math",
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",  # Necessary for the WGMMA shapes that we use
        "-DCUTLASS_ENABLE_GDC_FOR_SM90",  # For PDL
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "-DNDEBUG",  # Important, otherwise performance is severely impacted
        "-gencode", "arch=compute_90a,code=sm_90a",
        "--threads", os.getenv("NVCC_THREADS", "4"),
    ]
    if env_flag("FDA_LINEINFO"):
        nvcc_flags.append("-lineinfo")
    if env_flag("FDA_RESOURCE_USAGE"):
        nvcc_flags.append("--resource-usage")
    sources = ["flash_api.cpp", "flash_prepare_scheduler.cu"] + kernel_sources()
    ext_modules.append(
        CUDAExtension(
            name=f"{PACKAGE_NAME}._C",
            sources=[str(SRC_DIR / s) for s in sources],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + feature_args(),
                "nvcc": nvcc_flags + feature_args(),
            },
            include_dirs=[SRC_DIR, CUTLASS_DIR / "include"],
        )
    )
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="fast-diffusion-attention",
    version=get_version(),
    description="FlashAttention-3 class Hopper attention kernels for diffusion language models",
    long_description=(this_dir / "README.md").read_text() if (this_dir / "README.md").exists() else "",
    long_description_content_type="text/markdown",
    packages=find_packages(include=[PACKAGE_NAME, f"{PACKAGE_NAME}.*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
    install_requires=["torch"],
)
