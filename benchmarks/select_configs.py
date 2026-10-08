# Copyright (c) 2026, fast-diffusion-attention authors.
"""Selects the configuration of a GPU class from the outputs of tune_configs.py on the GPUs of the class.

    python benchmarks/tune_configs.py --headdim 128 --direction fwd --min-ms 300 > a6000/fwd_hd128.txt
    python benchmarks/select_configs.py a6000/fwd_hd128.txt      # the GPU of the class (sm86: RTX A6000)

Every candidate is ranked by the geometric mean of its throughput relative to FlashAttention-2 over all the shapes of
all the files: one per GPU of the class, every GPU weighing the same (the classes are tuned on one GPU each: sm80
A800, sm86 RTX A6000, sm89 L40S). The decision follows the rules of the dispatch
(flash_fwd_launch_template.h / flash_bwd_launch_template.h; the files must contain the "dispatch" column, i.e. come
from a run without --no-dispatch):
- the configuration of the dispatch unless the best candidate is at least 3% faster;
- a split by sequence length (one candidate up to FlashAttention-2's L, another above) only if it is at least 3%
  faster than the best single choice;
- the choice must not be more than 1% slower than the dispatch on any GPU of the class;
- margins within 1.5% of the 3% threshold are flagged: the sweeps have about 2% of noise (the dispatch against itself),
  confirm them with a head-to-head run (tune_configs.py --candidates ... --rounds 9).
"""

import argparse
import math
import re

THRESHOLD = 1.03   # gain needed to leave the dispatch, or to split by length
GUARD = 0.99       # no GPU of the class more than 1% slower than with the dispatch
NEAR = 0.015       # margins this close to the threshold need a head-to-head run
SPLITS = (512, 1024, 2048, 4096, 8192)


def parse(path):
    """Output of tune_configs.py -> {candidate: {(L, block size): relative throughput}} (complete rows only)."""
    names, res = None, {}
    for line in open(path):
        if line.strip().startswith("shape"):
            names = line.split()[1:]
            continue
        m = re.match(r"\s*L=(\d+) bs=(\d+)\s+(.*)", line)
        if m and names:
            values = m.group(3).split()
            if len(values) != len(names):
                continue
            for name, v in zip(names, values):
                res.setdefault(name, {})[(int(m.group(1)), int(m.group(2)))] = float(v.rstrip("*"))
    return res


def geomean(values):
    return math.exp(sum(map(math.log, values)) / len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="tune_configs.py outputs of one head dimension and direction, one per GPU")
    parser.add_argument("--top", type=int, default=8, help="number of candidates in the ranking")
    args = parser.parse_args()
    tables = [parse(f) for f in args.files]
    names = set.intersection(*[set(t) for t in tables])
    shapes = sorted(set.intersection(*[set(t[nm]) for t in tables for nm in names]))

    def score(choice, gpus=None, sel=lambda s: True):
        """geometric mean of the choice (a function shape -> candidate) over the GPUs and the shapes sel"""
        ts = tables if gpus is None else [tables[i] for i in gpus]
        return geomean([t[choice(s)][s] for t in ts for s in shapes if sel(s)])

    ranking = sorted(names, key=lambda nm: -score(lambda s, nm=nm: nm))
    print(f"{'candidate':>10s} {'geomean':>8s}  per GPU")
    for nm in ranking[:args.top]:
        print(f"{nm:>10s} {score(lambda s, nm=nm: nm):8.3f}  "
              + " ".join(f"{score(lambda s, nm=nm: nm, [i]):.3f}" for i in range(len(tables))))
    if "dispatch" not in names:
        raise SystemExit("no dispatch column (a run without --no-dispatch is needed for the decision)")

    dispatch = lambda s: "dispatch"
    base = score(dispatch)
    single = ranking[0]
    best_single = score(lambda s: single)
    splits = []
    for T in SPLITS:
        a = max(names, key=lambda nm: score(lambda s, nm=nm: nm, sel=lambda s: s[0] <= T))
        b = max(names, key=lambda nm: score(lambda s, nm=nm: nm, sel=lambda s: s[0] > T))
        choice = lambda s, a=a, b=b, T=T: a if s[0] <= T else b
        splits.append((score(choice), T, a, b, choice))
    best_split, T, a, b, split_choice = max(splits, key=lambda x: x[0])
    single_wins = single != "dispatch" and best_single >= THRESHOLD * base
    ref = best_single if single_wins else base
    if a != b and best_split >= THRESHOLD * ref:
        decision, choice = f"L <= {T}: {a}, above: {b}", split_choice
    elif single_wins:
        decision, choice = single, (lambda s: single)
    else:
        decision, choice = "dispatch", dispatch
    gains = [score(choice, [i]) / score(dispatch, [i]) for i in range(len(tables))]
    print(f"decision: {decision} ({100 * (score(choice) / base - 1):+.1f}%; per GPU "
          + " ".join(f"{100 * (g - 1):+.1f}%" for g in gains) + ")"
          + f"; best single {single} {100 * (best_single / base - 1):+.1f}%, best split {100 * (best_split / base - 1):+.1f}%")
    if min(gains) < GUARD:
        print("rejected: more than 1% slower than the dispatch on a GPU of the class")
    if any(abs(m - THRESHOLD) <= NEAR for m in (best_single / base, best_split / max(base, best_single))):
        print("near the threshold: confirm with a head-to-head run (--candidates ... --rounds 9)")


if __name__ == "__main__":
    main()
