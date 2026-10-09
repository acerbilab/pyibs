"""Wall time of one IBS estimate: PyIBS 1.5 against PyIBS 0.1.0.

Phase 3, step 3, of ``dev/plans/pyibs-1.5.md``. A cell is a version, a
number of trials N (100, 1,000 or 10,000), ``num_reps`` (10 or 100) and a
simulator: the example model's (the orientation discrimination model at
``ibs_example.m``'s generating parameters, with N orientations drawn as in
``ibs_example.m``), fast, or slowed by a sleep of 0.2 s per call. Each
version runs at its defaults, with a new ``IBS`` object per estimate, so
that ``vectorized=None`` decides at every estimate as it does at an
object's first call; 0.1.0 runs with ``max_iter=10**5``, an integer (its
default is 15), so that both versions sample every repeat to completion.
Both versions call the same two-argument simulator (0.1.0 passes no
generator), which counts its calls and the rows it simulates; its draws
come from a generator of its own, seeded per estimate.

A worker process runs cells with the interpreter of its version's venv:
``.venv`` for 1.5 and ``.venv-0.1`` for 0.1.0 (``uv venv --python 3.12
.venv-0.1``, ``uv pip install --python .venv-0.1 pyibs==0.1.0``). It runs
with ``-P``, which leaves the script's directory off ``sys.path``, and
reports the version and the location of the ``pyibs`` that it imports:
the checkout for 1.5, through the editable install, and the venv's
installed package for 0.1.0. The
coordinator first runs the fast cells, one worker per version, one after
the other and alone; then the slow cells, one worker each, at most
``--slow-jobs`` at a time (default 1). A slow cell sleeps for nearly all
of its time, so slow cells running together, or beside other work, change
its time by the little that its computation takes longer.

``--smoke`` times one estimate of each fast cell, and replaces each slow
cell by a dry run: the fast simulator on the schedule that the slow one
gets, ``vectorized=False`` (both versions decide False for a simulator
that takes 0.1 s or more), with the projected time ``0.2 s * calls`` plus
the dry run's own time. It prints the projected runtime of the full run.

Run from the repository root, unbuffered, logged:

    .venv/bin/python -u dev/scripts/timing.py --smoke \\
        --out dev/scripts/runs/timing_smoke_$(date +%s).json \\
        > dev/scripts/runs/timing_smoke_$(date +%s).log 2>&1
"""

import argparse
import datetime
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA_SEED = 20261009
RUN_SEED = 20261011
TRIALS = (100, 1000, 10000)
NUM_REPS = (10, 100)
SIMULATORS = ("fast", "slow")
SLOW_DELAY = 0.2
THETA = (0.0, 0.2, 0.03)  # ibs_example.m: log(sigma), bias, lapse
FAST_MIN_SECONDS = 2.0
FAST_MIN_ESTIMATES = 5
FAST_MAX_ESTIMATES = 50
MARKER = "@@RESULT "

VERSIONS = {
    "1.5": ".venv",
    "0.1.0": ".venv-0.1",
}


def interpreter(venv):
    if os.name == "nt":
        return str(Path(venv) / "Scripts" / "python.exe")
    return str(Path(venv) / "bin" / "python")


def all_cells():
    cells = []
    for version in VERSIONS:
        for sim in SIMULATORS:
            for n_trials in TRIALS:
                for num_reps in NUM_REPS:
                    cells.append(
                        dict(
                            index=len(cells),
                            version=version,
                            sim=sim,
                            trials=n_trials,
                            num_reps=num_reps,
                        )
                    )
    return cells


def cell_name(c):
    return f"{c['version']}_{c['sim']}_N{c['trials']}_n{c['num_reps']}"


# --------------------------------------------------------------------------
# The worker, which runs under either version's interpreter.


class Simulator:
    """The orientation discrimination model, two arguments, counting calls.

    The same code as ``psycho_generator`` of ``pyibs.examples``, drawing
    from the generator ``rng`` of the object.
    """

    def __init__(self, delay, rng):
        self.delay = delay
        self.rng = rng
        self.calls = 0
        self.rows = 0

    def __call__(self, theta, S):
        import numpy as np

        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        S = np.asarray(S, dtype=float)
        self.rows += S.shape[0]
        sigma, bias, lapse = np.exp(theta[0]), theta[1], theta[2]
        X = S + sigma * self.rng.standard_normal(S.shape)
        R = np.where(X >= bias, 1.0, -1.0)
        lapse_idx = self.rng.random(S.shape) < lapse
        R[lapse_idx] = (
            2.0 * self.rng.integers(2, size=np.count_nonzero(lapse_idx)) - 1
        )
        return R


def data(n_trials):
    """N orientations and the responses at the generating parameters."""
    import numpy as np

    rng = np.random.default_rng([DATA_SEED, n_trials])
    S = 3 * rng.standard_normal(n_trials)
    R = Simulator(0, rng)(np.array(THETA), S)
    return S, R


def one_estimate(cell, j, delay, vectorized):
    import numpy as np

    from pyibs import IBS

    S, R = data(cell["trials"])
    rng = np.random.default_rng([RUN_SEED, cell["index"], j])
    sim = Simulator(delay, rng)
    kwargs = {}
    if cell["version"] == "0.1.0":
        kwargs["max_iter"] = 10**5
    ibs = IBS(sim, R, S, vectorized=vectorized, **kwargs)
    t0 = time.perf_counter()
    value, var = ibs(
        np.array(THETA), num_reps=cell["num_reps"], additional_output="var"
    )
    seconds = time.perf_counter() - t0
    return dict(
        seconds=seconds,
        calls=sim.calls,
        rows=sim.rows,
        neg_logl=float(value),
        var=float(var),
    )


def run_worker_cell(cell, smoke, slow_estimates):
    out = dict(cell)
    out["name"] = cell_name(cell)
    if cell["sim"] == "fast":
        runs = []
        target = 1 if smoke else FAST_MIN_ESTIMATES
        t_start = time.perf_counter()
        while len(runs) < (1 if smoke else FAST_MAX_ESTIMATES):
            runs.append(one_estimate(cell, len(runs), 0, None))
            elapsed = time.perf_counter() - t_start
            if len(runs) >= target and (smoke or elapsed >= FAST_MIN_SECONDS):
                break
        out["mode"] = "smoke" if smoke else "full"
    elif smoke:
        # The slow simulator's schedule, without its sleeps.
        runs = [one_estimate(cell, 0, 0, False)]
        for r in runs:
            r["dry_seconds"] = r["seconds"]
            r["seconds"] = r["seconds"] + SLOW_DELAY * r["calls"]
        out["mode"] = "dry"
    else:
        runs = [
            one_estimate(cell, j, SLOW_DELAY, None)
            for j in range(slow_estimates)
        ]
        out["mode"] = "full"
    secs = sorted(r["seconds"] for r in runs)
    out["runs"] = runs
    out["estimates"] = len(runs)
    out["median_seconds"] = statistics.median(secs)
    out["min_seconds"] = secs[0]
    out["mean_seconds"] = sum(secs) / len(secs)
    out["mean_calls"] = sum(r["calls"] for r in runs) / len(runs)
    out["mean_rows_per_trial"] = sum(r["rows"] for r in runs) / (
        len(runs) * cell["trials"]
    )
    return out


def worker(spec):
    import importlib.metadata

    import numpy
    import scipy

    import pyibs

    info = dict(
        version_label=spec["version"],
        pyibs=importlib.metadata.version("pyibs"),
        pyibs_file=pyibs.__file__,
        python=sys.version.split()[0],
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        executable=sys.executable,
    )
    print(MARKER + json.dumps(dict(kind="info", info=info)), flush=True)
    for cell in spec["cells"]:
        result = run_worker_cell(cell, spec["smoke"], spec["slow_estimates"])
        print(MARKER + json.dumps(dict(kind="cell", cell=result)), flush=True)


# --------------------------------------------------------------------------
# The coordinator.


def provenance():
    def git(*args):
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    return dict(
        date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        commit=git("rev-parse", "HEAD"),
        clean=git("status", "--porcelain") == "",
        platform=platform.platform(),
        processor=platform.processor() or platform.machine(),
        cpu_count=os.cpu_count(),
        argv=sys.argv,
        data_seed=DATA_SEED,
        run_seed=RUN_SEED,
        slow_delay=SLOW_DELAY,
    )


def write(path, payload):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only", help="run only the cells matching REGEX")
    parser.add_argument("--skip", help="skip the cells matching REGEX")
    parser.add_argument("--slow-estimates", type=int, default=1)
    parser.add_argument("--slow-jobs", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        worker(json.loads(args.worker))
        return 0
    if args.out is None:
        parser.error("--out is required")

    cells = all_cells()
    if args.only:
        cells = [c for c in cells if re.search(args.only, cell_name(c))]
    if args.skip:
        cells = [c for c in cells if not re.search(args.skip, cell_name(c))]
    payload = dict(meta=provenance(), versions={}, cells={})
    payload["meta"]["slow_jobs"] = args.slow_jobs
    print("PyIBS timing", json.dumps(payload["meta"], indent=1), flush=True)
    write(args.out, payload)
    script = str(Path(__file__).resolve())
    lock = threading.Lock()

    def run_group(version, group):
        spec = dict(
            version=version,
            cells=group,
            smoke=args.smoke,
            slow_estimates=args.slow_estimates,
        )
        cmd = [
            interpreter(VERSIONS[version]),
            "-P",
            "-u",
            script,
            "--worker",
            json.dumps(spec),
        ]
        names = ", ".join(cell_name(c) for c in group)
        with lock:
            print(f"worker {version} ({names}): {cmd[0]}", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        for raw in proc.stdout:
            with lock:
                if not raw.startswith(MARKER):
                    print(f"  [{version}] {raw.rstrip()}", flush=True)
                    continue
                msg = json.loads(raw[len(MARKER) :])
                if msg["kind"] == "info":
                    payload["versions"][version] = msg["info"]
                    print(f"  {json.dumps(msg['info'])}", flush=True)
                else:
                    c = msg["cell"]
                    payload["cells"][c["name"]] = c
                    print(
                        f"  {c['name']:<26} {c['mode']:<5} "
                        f"median {c['median_seconds']:10.3f} s over "
                        f"{c['estimates']:>2} estimates; calls "
                        f"{c['mean_calls']:9.1f}, rows per trial "
                        f"{c['mean_rows_per_trial']:9.2f}",
                        flush=True,
                    )
                write(args.out, payload)
        if proc.wait() != 0:
            raise RuntimeError(f"worker {version} failed: {proc.returncode}")

    # The fast cells, one worker per version, alone.
    for version in VERSIONS:
        fast = [c for c in cells if c["version"] == version]
        fast = [c for c in fast if c["sim"] == "fast"]
        if fast:
            fast.sort(key=lambda c: (c["num_reps"], c["trials"]))
            run_group(version, fast)
    print("fast cells done", flush=True)
    # The slow cells, one worker each, the costliest first.
    slow = [c for c in cells if c["sim"] == "slow"]
    slow.sort(key=lambda c: (-c["num_reps"], -c["trials"], c["version"]))
    with ThreadPoolExecutor(max_workers=args.slow_jobs) as pool:
        futures = [pool.submit(run_group, c["version"], [c]) for c in slow]
        for fut in futures:
            fut.result()
    print("slow cells done", flush=True)

    # The comparison, and the projection of a full run from a smoke pass.
    print(flush=True)
    print(
        f"{'cell':<20} {'1.5 (s)':>11} {'0.1.0 (s)':>11} {'ratio':>8}",
        flush=True,
    )
    for sim in SIMULATORS:
        for n_trials in TRIALS:
            for num_reps in NUM_REPS:
                key = f"{sim}_N{n_trials}_n{num_reps}"
                new = payload["cells"].get(f"1.5_{key}")
                old = payload["cells"].get(f"0.1.0_{key}")
                if not new and not old:
                    continue
                t_new = new["median_seconds"] if new else math.nan
                t_old = old["median_seconds"] if old else math.nan
                print(
                    f"{key:<20} {t_new:11.3f} {t_old:11.3f} "
                    f"{t_old / t_new:8.2f}",
                    flush=True,
                )
    if args.smoke:
        total = 0.0
        for c in payload["cells"].values():
            if c["sim"] == "fast":
                t = c["median_seconds"]
                n = min(
                    FAST_MAX_ESTIMATES,
                    max(FAST_MIN_ESTIMATES, math.ceil(FAST_MIN_SECONDS / t)),
                )
                total += n * t
            else:
                total += args.slow_estimates * c["median_seconds"]
        payload["projected_seconds"] = total
        write(args.out, payload)
        print(
            f"projected full run ({args.slow_estimates} estimate(s) per "
            f"slow cell): {total:.0f} s = {total / 3600:.2f} h, one cell at a "
            "time",
            flush=True,
        )
        for c in sorted(
            payload["cells"].values(), key=lambda c: -c["median_seconds"]
        )[:8]:
            print(
                f"  {c['name']:<26} {c['median_seconds'] / 60:8.1f} min "
                "per estimate",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
