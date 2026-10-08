# Copyright (c) 2026, fast-diffusion-attention authors.
"""With pytest-xdist (-n N), FDA_TEST_GPUS="0,1,2,3" spreads the workers over these GPUs: it replaces the
CUDA_VISIBLE_DEVICES of every worker, i.e. it takes the indices of the machine (not of CUDA_VISIBLE_DEVICES).
The fixture large_memory runs the memory-heavy tests one at a time on every GPU."""

import fcntl
import os
import tempfile

import pytest

_worker = os.environ.get("PYTEST_XDIST_WORKER")
_gpus = os.environ.get("FDA_TEST_GPUS")
if _worker is not None and _gpus:
    gpus = [g for g in _gpus.split(",") if g.strip()]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[int(_worker.lstrip("gw")) % len(gpus)]


@pytest.fixture
def large_memory():
    """Serializes the memory-heavy tests (float32 references of long sequences: ~20 GB) between the workers that share
    a GPU, and gives the memory back to the GPU afterwards (the caching allocator of a worker keeps it otherwise)."""
    import torch
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "_")
    with open(os.path.join(tempfile.gettempdir(), f"fda_test_large_gpu{gpu}.lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            torch.cuda.empty_cache()
            fcntl.flock(f, fcntl.LOCK_UN)
