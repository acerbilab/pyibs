"""Turn the recorded run into the film's data: run/run_data.js, which film.html loads as the global RUN_DATA.

From the trace (run_ibs.py): every trial's count, the size of each round, the estimate and its SD, and the simulated
moves of the two trials that carry scenes 2 to 4 (--likely, --surprising, by trial number). From fresh simulations of
the model on those two positions, seeded: the probability of the person's move (--m simulations each), forty
simulated moves of the likely position for line 2.4 and twenty for line 3.1. Line 3.2's twenty simulations of the
surprising position are the first twenty of its recorded IBS row, all misses when its count exceeds twenty. Run in
the model's environment:

    PYTHONPATH=<ninarow>/model_fitting python -u film_data.py TRACE.json OUT.js --likely I --surprising J [--m 2000] [--seed S]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from fourbynine import fourbynine_board, fourbynine_pattern
from tree_search import TreeSearch


def rows_of(trace):
    """Each trial's simulated moves in its IBS row, from the log, up to and including its first match."""
    rows = [[] for _ in trace["trials"]]
    done = [False] * len(rows)
    for call in trace["log"]:
        for t, m in zip(call["trials"], call["moves"]):
            if not done[t]:
                rows[t].append(m)
                done[t] = m == trace["trials"][t]["move"]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("out")
    ap.add_argument("--likely", type=int, required=True)
    ap.add_argument("--surprising", type=int, required=True)
    ap.add_argument(
        "--column",
        default="0,1,2",
        help="three more trials for the column of line 2.2",
    )
    ap.add_argument("--m", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    tr = json.loads(Path(a.trace).read_text())
    rows = rows_of(tr)
    for t, r in zip(tr["trials"], rows):
        assert (
            len(r) == t["K"] and r[-1] == t["move"] and t["move"] not in r[:-1]
        )

    model = TreeSearch(verbose=False)
    model.set_params(
        np.array(
            [tr["model"]["params"][n] for n in model.param_names], dtype=float
        )
    )
    rng = np.random.default_rng(a.seed)

    def simulate(t, n):
        board = fourbynine_board(
            fourbynine_pattern(t["black"]), fourbynine_pattern(t["white"])
        )
        model.heuristic.seed_generator(int(rng.integers(2**63)))
        return [int(model.predict(board)) for _ in range(n)]

    def position(i):
        t = tr["trials"][i]
        return {
            k: t[k] for k in ("line", "black", "white", "color", "move", "K")
        } | {"trial": i, "row": rows[i]}

    A, B = position(a.likely), position(a.surprising)
    for P in (A, B):
        P["p"] = float(
            np.mean(
                np.array(simulate(tr["trials"][P["trial"]], a.m)) == P["move"]
            )
        )
        P["p_sims"] = a.m
    A["sims40"] = simulate(tr["trials"][a.likely], 40)
    A["fixed20"] = simulate(tr["trials"][a.likely], 20)
    assert (
        B["K"] > 20
    ), "line 3.2 needs twenty misses from the surprising position's row"
    B["fixed20"] = B["row"][:20]

    data = {
        "source": "run/"
        + Path(a.trace).name
        + ", made into film data by run/film_data.py",
        "seed": a.seed,
        "A": A,
        "B": B,
        "column": [position(int(i)) for i in a.column.split(",")],
        "K": [t["K"] for t in tr["trials"]],
        "rounds": tr["rounds"],
        "loglik": tr["estimate"]["loglik"],
        "sd": tr["estimate"]["sd"],
        "simulations": sum(tr["rounds"]),
    }
    Path(a.out).write_text(
        "// Written by run/film_data.py from the recorded run; do not edit by hand.\nwindow.RUN_DATA = "
        + json.dumps(data)
        + ";\n"
    )
    print(
        f"likely: trial {A['trial']}, K {A['K']}, p {A['p']:.3f}, fixed20 matches {sum(m == A['move'] for m in A['fixed20'])}, "
        f"sims40 matches {sum(m == A['move'] for m in A['sims40'])}"
    )
    print(f"surprising: trial {B['trial']}, K {B['K']}, p {B['p']:.4f}")
    print(
        f"run: L {data['loglik']:.2f} sd {data['sd']:.2f}, {len(data['rounds'])} rounds, {data['simulations']} simulations"
    )


if __name__ == "__main__":
    main()
