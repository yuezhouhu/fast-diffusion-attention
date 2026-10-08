# Copyright (c) 2026, fast-diffusion-attention authors.
"""Ranks the candidate configurations of a GPU class from the outputs of tune_configs.py on the GPUs of the class.

    python benchmarks/tune_configs.py --headdim 128 --direction fwd --min-ms 300 > a6000/fwd_hd128.txt   # every GPU
    python benchmarks/select_configs.py a6000/fwd_hd128.txt l40s/fwd_hd128.txt --reference 1

For every candidate: the geometric mean of its throughput relative to FlashAttention-2 over all the shapes of all the
files (every GPU of the class weighs the same), its minimum, and the geometric mean per GPU. With --reference (the
candidate that is FlashAttention-2's configuration for this class), the margin of every candidate over it: the
dispatch keeps FlashAttention-2's configuration unless another one is at least 3% faster (flash_fwd_launch_template.h).
"""

import argparse
import math
import re


def parse(path):
    """Output of tune_configs.py -> {candidate: {shape: relative throughput}} (complete rows only)."""
    names, res = None, {}
    for line in open(path):
        if line.strip().startswith("shape"):
            names = line.split()[1:]
            continue
        m = re.match(r"\s*(L=\d+ bs=\d+)\s+(.*)", line)
        if m and names:
            values = m.group(2).split()
            if len(values) != len(names):
                continue
            for name, v in zip(names, values):
                res.setdefault(name, {})[m.group(1)] = float(v.rstrip("*"))
    return res


def geomean(values):
    return math.exp(sum(map(math.log, values)) / len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="tune_configs.py outputs of one head dimension and direction")
    parser.add_argument("--reference", default=None, help="candidate index of FlashAttention-2's configuration")
    args = parser.parse_args()
    per_gpu = [parse(f) for f in args.files]
    names = set.intersection(*[set(p) for p in per_gpu])
    stats = {}
    for name in names:
        values = [v for p in per_gpu for v in p[name].values()]
        stats[name] = (geomean(values), min(values), [geomean(list(p[name].values())) for p in per_gpu])
    ref = f"c{args.reference}" if args.reference is not None else None
    if ref is not None and ref not in stats:
        raise SystemExit(f"candidate {args.reference} is not in all the files")
    print(f"{'candidate':>10s} {'geomean':>8s} {'min':>6s}  per GPU" + ("   vs reference" if ref else ""))
    for name in sorted(stats, key=lambda nm: -stats[nm][0]):
        g, lo, gpus = stats[name]
        margin = f"   {100 * (g / stats[ref][0] - 1):+5.1f}%" if ref else ""
        print(f"{name:>10s} {g:8.3f} {lo:6.2f}  " + " ".join(f"{x:.3f}" for x in gpus) + margin)


if __name__ == "__main__":
    main()
