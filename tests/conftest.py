# Copyright (c) 2026, fast-diffusion-attention authors.
"""With pytest-xdist (-n N), FDA_TEST_GPUS="0,1,2,3" spreads the workers over these GPUs: it replaces the
CUDA_VISIBLE_DEVICES of every worker, i.e. it takes the indices of the machine (not of CUDA_VISIBLE_DEVICES)."""

import os

_worker = os.environ.get("PYTEST_XDIST_WORKER")
_gpus = os.environ.get("FDA_TEST_GPUS")
if _worker is not None and _gpus:
    gpus = [g for g in _gpus.split(",") if g.strip()]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[int(_worker.lstrip("gw")) % len(gpus)]
