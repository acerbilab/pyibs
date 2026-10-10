"""List the run's trials that could carry scenes 2 to 4: a likely move (count 2 to 4) and a surprising one (count 24 to 45).

For each, the probability of the person's move under the model is estimated from M fresh simulations (seeded), so that
the chosen count can be read against it. Run like run_ibs.py, in the model's environment:

    PYTHONPATH=<ninarow>/model_fitting python -u candidates.py TRACE.json [--m 2000] [--seed S]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from fourbynine import fourbynine_board, fourbynine_pattern
from tree_search import TreeSearch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--m", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    tr = json.loads(Path(a.trace).read_text())
    model = TreeSearch(verbose=False)
    params = np.array(
        [tr["model"]["params"][n] for n in model.param_names], dtype=float
    )
    model.set_params(params)
    rng = np.random.default_rng(a.seed)
    for i, t in enumerate(tr["trials"]):
        if not (2 <= t["K"] <= 4 or 24 <= t["K"] <= 45):
            continue
        board = fourbynine_board(
            fourbynine_pattern(t["black"]), fourbynine_pattern(t["white"])
        )
        model.heuristic.seed_generator(int(rng.integers(2**63)))
        moves = np.array([model.predict(board) for _ in range(a.m)])
        p = float(np.mean(moves == t["move"]))
        top = np.bincount(moves, minlength=36)
        pieces = bin(t["black"]).count("1") + bin(t["white"]).count("1")
        print(
            f"trial {i:3d} line {t['line']:5d} K {t['K']:3d} p {p:.4f} (1/p {1 / p if p else float('inf'):7.1f}) "
            f"pieces {pieces:2d} move {t['move']:2d} model's top squares {np.argsort(-top)[:3].tolist()} "
            f"with {np.sort(top)[::-1][:3].tolist()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
