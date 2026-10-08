# Copyright (c) 2023, Tri Dao.
# Copyright (c) 2026, fast-diffusion-attention authors.
# Build script adapted from FlashAttention-2 (setup.py).
#
# Environment variables controlling the build:
#   FDA_CUDA_ARCHS         semicolon separated compute capabilities to generate code for (default "80": sm80 SASS,
#                          which runs on sm80 / sm86 / sm87 / sm89, + compute_80 PTX), as FLASH_ATTN_CUDA_ARCHS
#   FDA_HDIMS              comma separated head dimensions to build (default: 32,64,96,128,192,256)
#   FDA_DISABLE_FP16       do not build the FP16 kernels
#   FDA_DISABLE_DROPOUT    do not build the dropout kernels
#   FDA_DISABLE_SOFTCAP    do not build the softcapping kernels
#   FDA_DISABLE_BACKWARD   do not build the backward kernels
#   FDA_TUNING=1           also compile the candidate kernel configurations of csrc/fda/src/tuning.h, selected at
#                          run time with FDA_TUNE_FWD / FDA_TUNE_BWD (benchmarks/tune_configs.py)
#   FDA_LINEINFO=1         compile with -lineinfo (for profiling)
#   FDA_RESOURCE_USAGE=1   print the registers / spills of every kernel (ptxas -v)
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


ALL_HDIMS = [32, 64, 96, 128, 192, 256]
HDIMS = [int(h) for h in os.getenv("FDA_HDIMS", ",".join(map(str, ALL_HDIMS))).split(",") if h.strip()]
for h in HDIMS:
    assert h in ALL_HDIMS, f"Unsupported head dimension {h}, must be one of {ALL_HDIMS}"
DISABLE_FP16 = env_flag("FDA_DISABLE_FP16")
DISABLE_DROPOUT = env_flag("FDA_DISABLE_DROPOUT")
DISABLE_SOFTCAP = env_flag("FDA_DISABLE_SOFTCAP")
DISABLE_BACKWARD = env_flag("FDA_DISABLE_BACKWARD")
CUDA_ARCHS = [a.strip() for a in os.getenv("FDA_CUDA_ARCHS", "80").split(";") if a.strip()]


TUNING = env_flag("FDA_TUNING")
NUM_TUNING_CANDIDATES = {"fwd": 19, "bwd": 34}  # as in csrc/fda/src/tuning.h, generate_kernels.py, tune_configs.py


def kernel_sources():
    dtypes = ["bf16"] + ([] if DISABLE_FP16 else ["fp16"])
    # fwd_split: the split-KV forward (FlashAttention-2's flash_fwd_split_hdim* files)
    directions = ["fwd", "fwd_split"] + ([] if DISABLE_BACKWARD else ["bwd"])
    srcs = []
    for hdim in HDIMS:
        for dtype in dtypes:
            for direction in directions:
                srcs.append(f"src/flash_{direction}_hdim{hdim}_{dtype}_sm80.cu")
                if TUNING:
                    srcs += [f"src/tuning/flash_{direction}_hdim{hdim}_{dtype}_tune{i}_sm80.cu"
                             for i in range(NUM_TUNING_CANDIDATES.get(direction, 0))]
    return srcs


def feature_args():
    args = [f"-DFLASHATTENTION_DISABLE_HDIM{h}" for h in ALL_HDIMS if h not in HDIMS]
    flags = {
        "FLASHATTENTION_DISABLE_BACKWARD": DISABLE_BACKWARD,
        "FLASHATTENTION_DISABLE_DROPOUT": DISABLE_DROPOUT,
        "FLASHATTENTION_DISABLE_SOFTCAP": DISABLE_SOFTCAP,
        "FLASHATTENTION_DISABLE_FP16": DISABLE_FP16,
        "FDA_TUNING": TUNING,
    }
    args += [f"-D{k}" for k, v in flags.items() if v]
    return args


def cuda_gencode_args():
    # As FlashAttention-2: SASS for every requested architecture, PTX of the last one. The sm80 SASS runs on all the
    # sm8x GPUs (A100 / A800 / A30, A10 / A40 / RTX 30xx / A6000, Jetson Orin, L4 / L40 / RTX 40xx).
    args = []
    for arch in CUDA_ARCHS:
        args += ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]
    args += ["-gencode", f"arch=compute_{CUDA_ARCHS[-1]},code=compute_{CUDA_ARCHS[-1]}"]
    return args


ext_modules = []
cmdclass = {}
if not env_flag("FDA_SKIP_CUDA_BUILD"):
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

    assert CUDA_HOME is not None, "nvcc not found, please set CUDA_HOME"
    if not (CUTLASS_DIR / "include").exists():
        subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], cwd=this_dir, check=True)
    subprocess.run([sys.executable, str(SRC_DIR / "src" / "generate_kernels.py")], check=True)
    nvcc_version = subprocess.check_output([os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"]).decode()
    if "release 13." in nvcc_version:
        # CUDA 13 ships CCCL (cub / thrust / libcu++) in include/cccl
        cccl_include = os.path.join(CUDA_HOME, "include", "cccl")
        for env_var in ["CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"]:
            current = os.environ.get(env_var, "")
            os.environ[env_var] = cccl_include + (":" + current if current else "")

    # PyTorch 2.13+ requires C++20 for extensions that include ATen headers (as FlashAttention-2)
    torch_major, torch_minor = (int(x) for x in torch.__version__.split(".")[:2])
    cxx_standard = "c++20" if (torch_major, torch_minor) >= (2, 13) else "c++17"

    nvcc_flags = [
        "-O3",
        f"-std={cxx_standard}",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads", os.getenv("NVCC_THREADS", "4"),
    ] + cuda_gencode_args()
    if env_flag("FDA_LINEINFO"):
        nvcc_flags.append("-lineinfo")
    if env_flag("FDA_RESOURCE_USAGE"):
        nvcc_flags.append("--resource-usage")
    sources = ["flash_api.cpp"] + kernel_sources()
    ext_modules.append(
        CUDAExtension(
            name="fda_2_cuda",
            sources=[str(SRC_DIR / s) for s in sources],
            extra_compile_args={
                "cxx": ["-O3", f"-std={cxx_standard}"] + feature_args(),
                "nvcc": nvcc_flags + feature_args(),
            },
            include_dirs=[SRC_DIR, SRC_DIR / "src", CUTLASS_DIR / "include"],
        )
    )
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="fast-diffusion-attention",
    version=get_version(),
    description="FlashAttention-2 class attention kernels for diffusion language models (sm80 / sm86 / sm89)",
    long_description=(this_dir / "README.md").read_text() if (this_dir / "README.md").exists() else "",
    long_description_content_type="text/markdown",
    packages=find_packages(include=[PACKAGE_NAME, f"{PACKAGE_NAME}.*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
    install_requires=["torch"],
)
