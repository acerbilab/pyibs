"""The film's recorded run: one PyIBS estimate of the four-in-a-row model's log-likelihood on real positions.

The trials are N positions drawn at random, without replacement, from the human-versus-human games in data_hvh.txt
(5,482 positions; the set of [1] Section 5.4), each with the move the person played as the response. The model is the
Wei Ji Ma lab's heuristic search (WeiJiMaLab/ninarow, model_fitting/tree_search.py), with the parameters of --params
(a JSON file whose "params" name every parameter of TreeSearch) or, without it, the build's defaults
(model_fitting/config.yaml, the initial values). PyIBS runs one repeat with vectorized=False, so that each call of the
simulator is one round: one simulation for every trial that has not yet matched. A wrapper logs every call (which
trials, which simulated moves) without drawing from the run's generator; the counts, the rounds and the estimate are
recomputed from the log and checked against PyIBS's own result.

Run in the model's environment, with the model's build and PyIBS on the path:

    PYTHONPATH=<ninarow>/model_fitting:<pyibs> python -u run_ibs.py DATA OUT.json [--params P.json] [--n 100] [--select-seed S] [--seed S]

The record names the revisions of PyIBS and of the model; the data file is named by its SHA-256.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import fourbynine
import numpy as np
from fourbynine import fourbynine_board, fourbynine_pattern
from tree_search import TreeSearch

import pyibs
from pyibs import IBS


def git_rev(path):
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def load_positions(path):
    rows = []
    for k, line in enumerate(Path(path).read_text().splitlines()):
        f = line.split("\t")
        black, white, color, move = int(f[0]), int(f[1]), f[2], int(f[3])
        rows.append(
            {
                "line": k + 1,
                "black": black,
                "white": white,
                "color": color,
                "move": move.bit_length() - 1,
            }
        )
    return rows


class Simulator:
    """The model as a PyIBS simulator. A design row is (trial, black, white); a response is a square, 0 to 35.

    Each call seeds the model's own generator from PyIBS's rng, so that the run repeats from its seed, and logs the
    trials it was asked for and the moves it returned.
    """

    def __init__(self, model, params):
        self.model = model
        self.model.set_params(params)
        self.log = []

    def __call__(self, params, design_rows, rng):
        self.model.heuristic.seed_generator(int(rng.integers(2**63)))
        moves = np.empty(len(design_rows), dtype=np.int64)
        for j, (_, black, white) in enumerate(
            np.asarray(design_rows, dtype=np.int64)
        ):
            board = fourbynine_board(
                fourbynine_pattern(int(black)), fourbynine_pattern(int(white))
            )
            moves[j] = self.model.predict(board)
        self.log.append(
            {
                "trials": [int(t) for t in np.asarray(design_rows)[:, 0]],
                "moves": [int(m) for m in moves],
            }
        )
        return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--select-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--params")
    a = ap.parse_args()

    rows = load_positions(a.data)
    pick = np.random.default_rng(a.select_seed).choice(
        len(rows), size=a.n, replace=False
    )
    trials = [rows[i] for i in pick]
    for (
        t
    ) in (
        trials
    ):  # every position is legal and it is the recorded player's turn
        board = fourbynine_board(
            fourbynine_pattern(t["black"]), fourbynine_pattern(t["white"])
        )
        assert board.active_player() == (t["color"].lower() == "white"), t
        assert not ((t["black"] | t["white"]) >> t["move"]) & 1, t

    model = TreeSearch(verbose=False)
    if a.params:
        given = json.loads(Path(a.params).read_text())["params"]
        assert set(given) == set(model.param_names), (
            sorted(given),
            model.param_names,
        )
        params = np.array(
            [given[name] for name in model.param_names], dtype=float
        )
    else:
        params = model.initial_params.astype(float)
    sim = Simulator(model, params)
    design = np.array(
        [[i, t["black"], t["white"]] for i, t in enumerate(trials)],
        dtype=np.int64,
    )
    responses = np.array([t["move"] for t in trials], dtype=np.int64)

    ibs = IBS(sim, responses, design, vectorized=False, random_seed=a.seed)
    started = time.time()
    res = ibs(params, num_reps=1, additional_output="full")
    seconds = time.time() - started

    # The counts from the log: trial i's simulations, in call order, until its first match.
    K = [None] * a.n
    seen = [0] * a.n
    for call in sim.log:
        for t, m in zip(call["trials"], call["moves"]):
            if K[t] is None:
                seen[t] += 1
                if m == trials[t]["move"]:
                    K[t] = seen[t]
    assert all(k is not None for k in K)
    est = lambda k: -sum(
        1.0 / j for j in range(1, k)
    )  # noqa: E731  [1] Equation 14
    var = lambda k: sum(
        1.0 / j**2 for j in range(1, k)
    )  # noqa: E731  [1] Equation 16
    L = sum(est(k) for k in K)
    V = sum(var(k) for k in K)
    neg, neg_sd, calls = (
        float(res["neg_logl"]),
        float(res["neg_logl_std"]),
        int(res["fun_count"]),
    )
    assert calls == len(sim.log), (calls, len(sim.log))
    print(
        f"IBS: {neg!r}; from the log: {-L!r}; sd {np.sqrt(V):.3f}; {len(sim.log)} rounds; "
        f"{sum(len(c['trials']) for c in sim.log)} simulations; {seconds:.1f} s",
        flush=True,
    )

    record = {
        "what": "One PyIBS estimate of the log-likelihood of the four-in-a-row heuristic-search model on real positions",
        "data": {
            "file": Path(a.data).name,
            "sha256": hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
            "source": "https://raw.githubusercontent.com/basvanopheusden/fourinarow/master/data_hvh.txt",
        },
        "select_seed": a.select_seed,
        "seed": a.seed,
        "n": a.n,
        "model": {
            "repo": "https://github.com/WeiJiMaLab/ninarow",
            "revision": git_rev(Path(fourbynine.__file__).parent),
            "params": dict(zip(model.param_names, params.tolist())),
            "params_file": a.params and Path(a.params).name,
        },
        "pyibs": {
            "version": pyibs.__version__,
            "revision": git_rev(Path(pyibs.__file__).parent),
        },
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "settings": {
            "vectorized": False,
            "num_reps": 1,
            "neg_logl_threshold": None,
        },
        "trials": [{**t, "K": k} for t, k in zip(trials, K)],
        "rounds": [len(c["trials"]) for c in sim.log],
        "log": sim.log,
        "estimate": {
            "loglik": L,
            "sd": float(np.sqrt(V)),
            "pyibs_neg_logl": neg,
            "pyibs_neg_logl_std": neg_sd,
        },
        "seconds": seconds,
    }
    Path(a.out).write_text(json.dumps(record))
    Ks = np.array(K)
    print(
        "K: min",
        Ks.min(),
        "median",
        int(np.median(Ks)),
        "max",
        Ks.max(),
        "| K in 2..4:",
        int(((Ks >= 2) & (Ks <= 4)).sum()),
        "| K in 25..40:",
        int(((Ks >= 25) & (Ks <= 40)).sum()),
        "| rounds",
        len(sim.log),
        flush=True,
    )


if __name__ == "__main__":
    main()
