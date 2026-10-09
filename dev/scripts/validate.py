"""Statistical validation of PyIBS's estimates against exact log-likelihoods.

Phase 3, step 1, of ``dev/plans/pyibs-1.5.md``. Every model below has a
known matching probability ``p_i`` for each trial, so its exact
log-likelihood is ``sum_i w_i log p_i``. A cell is a model with one setting
of ``vectorized`` (True, False, None), ``num_reps`` (1, 10, 100) and the
likelihood threshold (off, or at the chance level for the Bernoulli,
weighted and categorical models). Each estimate of a cell comes from a new
``IBS`` object with its own seed, so that the cells of ``vectorized=None``
sample the object's deciding call, whose first round is the timing call.

Per cell, over the estimates e of the negative log-likelihood, with err =
e - exact:

- the bias, ``mean(err)``, in standard errors of the mean; with the
  threshold, the mean is compared with the expected value of a thresholded
  estimate, ``-mean(max(Y, -T))`` over exact repeats Y
  (:func:`pyibs.testing._exact.exact_draw`), combining both standard
  errors;
- the mean squared z-score, ``mean(err**2 / var_estimate)``, in standard
  errors from 1, and the same statistic of exact IBS estimates drawn from
  geometric counts, as a reference;
- the 95% coverage of ``e +- 1.96 sqrt(var_estimate)``;
- the ratio of the estimates' SD to the exact SD,
  ``sqrt(exact_var / num_reps)`` (no threshold);
- the samples per trial against ``num_reps * mean_i(1 / p_i)``: at
  ``vectorized=False`` (no surplus) in standard errors, otherwise as a
  ratio, surplus included.

A cell passes when its bias is within 4.5 standard errors and, at
``num_reps`` of 10 or more without the threshold, its mean squared z-score
is within 4.5 standard errors of 1. In the models whose trials all match
with probability 0.999 (``CALIBRATION_REPORTED_ONLY``), the mean squared
z-score is reported instead, since exact IBS fails that gate there too:
every count of an estimate is 1 with a probability of 0.999**(100 n), which
makes the variance estimate 0, and with n = 100 the exact draws' mean
squared z-score is about 1.28. Their gate is instead the share of
estimates whose variance estimate is 0, within 4.5 standard errors of its
exact probability, ``prod_i p_i**n``. The other statistics are reported.

Separately, ``--zero-check`` verifies that a model whose trials all match at
the first sample returns a value and a variance of exactly 0.

Run from the repository root with the venv, unbuffered, logged:

    .venv/bin/python -u dev/scripts/validate.py --estimates 100 \\
        --out dev/scripts/runs/validate_smoke_$(date +%s).json \\
        > dev/scripts/runs/validate_smoke_$(date +%s).log 2>&1

``--project 2000`` (the default) prints the runtime that 2,000 estimates
per cell would take, from the cells' measured times.
"""

import argparse
import datetime
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

import numpy as np

DATA_SEED = 20261009
"""The seed of the models' data: responses, probabilities and weights."""

RUN_SEED = 20261010
"""The root seed of the estimates and of the exact references."""

BERNOULLI_P = (0.001, 0.01, 0.1, 0.5, 0.9, 0.999)
CATEGORICAL_K = (2, 4, 8)
VECTORIZED = (True, False, None)
NUM_REPS = (1, 10, 100)
TOL = 4.5
"""The tolerance of the gates, in standard errors."""

CALIBRATION_REPORTED_ONLY = ("bernoulli_p0.001", "bernoulli_p0.999")
"""The models whose calibration is reported and not gated (the PI's
ruling of 2026-10-09 on the smoke pass): their data hold no rare response,
so all their trials match with probability 0.999."""

THRESHOLD_REPEATS = 10**5
"""Exact repeats behind the expected value of a thresholded estimate."""

REFERENCE_BUDGET = 5 * 10**7
"""Geometric counts behind the exact reference of a model and num_reps."""

REFERENCE_MAX = 20_000
"""The most exact estimates in a reference."""

TEXT_LABELS = np.array(["left", "right", "up", "down"])

# The example model's data set and parameter vectors, as in
# pyibs/testing/test_examples.py: ibs_example.m's generating parameters
# (log(sigma), bias, lapse) and two others.
PSYCHO_TRIALS = 600
PSYCHO_THETAS = (
    ("true", (math.log(1.0), 0.2, 0.03)),
    ("narrow", (math.log(0.5), -0.5, 0.1)),
    ("wide", (math.log(3.0), 1.0, 0.01)),
)


# --------------------------------------------------------------------------
# Simulators. Classes rather than closures, so that they pickle.


class BernoulliSim:
    """Respond 1 with probability q, else 0, for every requested trial."""

    def __init__(self, q):
        self.q = q

    def __call__(self, theta, idx, rng):
        return (rng.random(np.shape(idx)[0]) < self.q).astype(float)


def _categorical(cdf, size, rng):
    """Outcomes 0..K-1 drawn by inverting the cumulative probabilities."""
    out = np.searchsorted(cdf, rng.random(size), side="right")
    return np.minimum(out, cdf.size - 1)


class CategoricalSim:
    """Draw one of K outcomes with fixed probabilities, as labels if any."""

    def __init__(self, probs, labels=None):
        self.cdf = np.cumsum(probs)
        self.labels = labels

    def __call__(self, theta, idx, rng):
        out = _categorical(self.cdf, np.shape(idx)[0], rng)
        if self.labels is None:
            return out.astype(float)
        return self.labels[out]


class TwoColumnSim:
    """Two independent columns: a per-trial Bernoulli and a categorical."""

    def __init__(self, q1, probs2):
        self.q1 = q1
        self.cdf2 = np.cumsum(probs2)

    def __call__(self, theta, idx, rng):
        idx = np.asarray(idx)
        col1 = (rng.random(idx.size) < self.q1[idx]).astype(float)
        col2 = _categorical(self.cdf2, idx.size, rng).astype(float)
        return np.column_stack((col1, col2))


class PsychoSim:
    """The example model's simulator, with its ``rng`` parameter."""

    def __call__(self, theta, S, rng):
        from pyibs.examples.psycho_model import psycho_generator

        return psycho_generator(theta, S, rng)


class Echo:
    """Return the observed response of every requested trial."""

    def __init__(self, responses):
        self.responses = responses

    def __call__(self, theta, idx, rng):
        return self.responses[np.asarray(idx)]


# --------------------------------------------------------------------------
# Models.


@dataclass
class Model:
    """A simulator with its data and the exact matching probabilities.

    ``p`` holds each trial's probability of matching its response at
    ``theta``; ``threshold`` is the chance level, or None when the model's
    cells run without the threshold only.
    """

    name: str
    sim: object
    responses: np.ndarray
    design: object
    theta: np.ndarray
    p: np.ndarray
    weights: object = None
    threshold: object = None
    info: dict = field(default_factory=dict)

    @property
    def w(self):
        if self.weights is None:
            return np.ones(self.p.size)
        return np.asarray(self.weights, dtype=float)

    @property
    def exact_nll(self):
        return float(-np.sum(self.w * np.log(self.p)))

    @property
    def exact_var(self):
        from pyibs.testing._exact import exact_var

        return exact_var(self.p, self.weights)


def _data_rng(*key):
    return np.random.default_rng([DATA_SEED, *key])


def build_models():
    """Every model of the validation, in a fixed order."""
    from scipy.special import ndtr

    from pyibs.examples.psycho_model import psycho_neg_logl

    models = []
    zero = np.zeros(1)

    # Bernoulli: 100 trials at each p, responses drawn at that p.
    for k, q in enumerate(BERNOULLI_P):
        R = (_data_rng(1, k).random(100) < q).astype(float)
        p = np.where(R == 1, q, 1 - q)
        models.append(
            Model(
                f"bernoulli_p{q:g}",
                BernoulliSim(q),
                R,
                None,
                zero,
                p,
                threshold=100 * math.log(2),
                info={"q": q, "ones": int(R.sum())},
            )
        )

    # Weights: the Bernoulli model at p = 0.5.
    base = models[BERNOULLI_P.index(0.5)]
    w_int = _data_rng(2, 0).integers(1, 4, size=100).astype(float)
    w_frac = _data_rng(2, 1).uniform(0.2, 2.0, size=100)
    for label, w in (("int", w_int), ("frac", w_frac)):
        models.append(
            Model(
                f"weights_{label}",
                base.sim,
                base.responses,
                None,
                zero,
                base.p,
                weights=w,
                threshold=float(np.sum(w) * math.log(2)),
                info={"q": 0.5, "w_min": w.min(), "w_max": w.max()},
            )
        )

    # Categorical: 200 trials, outcome probabilities from a Dirichlet.
    cat4 = None
    for k, K in enumerate(CATEGORICAL_K):
        rng = _data_rng(3, k)
        probs = rng.dirichlet(np.ones(K))
        R = _categorical(np.cumsum(probs), 200, rng)
        models.append(
            Model(
                f"categorical_k{K}",
                CategoricalSim(probs),
                R.astype(float),
                None,
                zero,
                probs[R],
                threshold=200 * math.log(K),
                info={"probs": probs.tolist()},
            )
        )
        if K == 4:
            cat4 = (probs, R)

    # Text: the categorical model with 4 outcomes, as strings.
    probs, R = cat4
    models.append(
        Model(
            "text_k4",
            CategoricalSim(probs, TEXT_LABELS),
            TEXT_LABELS[R],
            None,
            zero,
            probs[R],
            info={"labels": TEXT_LABELS.tolist()},
        )
    )

    # Two response columns: a Bernoulli column whose probability varies
    # over the trials, and an independent column of 3 outcomes.
    rng = _data_rng(4, 0)
    q1 = np.linspace(0.1, 0.9, 100)
    probs2 = rng.dirichlet(np.ones(3))
    col1 = (rng.random(100) < q1).astype(float)
    col2 = _categorical(np.cumsum(probs2), 100, rng)
    R2 = np.column_stack((col1, col2.astype(float)))
    p2 = np.where(col1 == 1, q1, 1 - q1) * probs2[col2]
    models.append(
        Model(
            "two_columns",
            TwoColumnSim(q1, probs2),
            R2,
            None,
            zero,
            p2,
            info={"probs2": probs2.tolist()},
        )
    )

    # The example model, ibs_example.m's data set, at three vectors.
    from pyibs.examples.psycho_model import psycho_generator

    rng = np.random.default_rng(DATA_SEED)
    S = 3 * rng.standard_normal((PSYCHO_TRIALS, 1))
    R = psycho_generator(np.array(PSYCHO_THETAS[0][1]), S, rng)
    for label, theta in PSYCHO_THETAS:
        theta = np.array(theta)
        sigma, bias, lapse = np.exp(theta[0]), theta[1], theta[2]
        z = (S[:, 0] - bias) / sigma
        right = R[:, 0] == 1
        p = lapse / 2 + (1 - lapse) * np.where(right, ndtr(z), ndtr(-z))
        closed = psycho_neg_logl(theta, S, R)
        assert math.isclose(-np.sum(np.log(p)), closed, rel_tol=1e-12)
        models.append(
            Model(
                f"psycho_{label}",
                PsychoSim(),
                R,
                S,
                theta,
                p,
                info={"theta": theta.tolist()},
            )
        )
    return models


_MODELS = None


def models_by_name():
    global _MODELS
    if _MODELS is None:
        _MODELS = {m.name: m for m in build_models()}
    return _MODELS


# --------------------------------------------------------------------------
# Cells.


@dataclass(frozen=True)
class Cell:
    index: int
    model: str
    vectorized: object
    num_reps: int
    threshold: bool

    @property
    def name(self):
        v = {True: "T", False: "F", None: "N"}[self.vectorized]
        thr = "_thr" if self.threshold else ""
        return f"{self.model}_v{v}_n{self.num_reps}{thr}"


def all_cells():
    """Every cell, in a fixed order that fixes each cell's seed."""
    cells = []
    for model in build_models():
        for threshold in (False, True):
            if threshold and model.threshold is None:
                continue
            for vectorized in VECTORIZED:
                for num_reps in NUM_REPS:
                    cells.append(
                        Cell(
                            len(cells),
                            model.name,
                            vectorized,
                            num_reps,
                            threshold,
                        )
                    )
    return cells


def run_cell(cell, n_estimates):
    """Draw the cell's estimates; return its raw arrays and wall time."""
    from pyibs import IBS

    model = models_by_name()[cell.model]
    T = model.threshold if cell.threshold else np.inf
    est = np.empty(n_estimates)
    var = np.empty(n_estimates)
    samples = np.empty(n_estimates)
    calls = np.empty(n_estimates, dtype=np.int64)
    flags = np.empty(n_estimates, dtype=np.int64)
    decided = np.zeros(n_estimates, dtype=bool)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        # The zero-variance warning and that of vectorized=True with
        # num_reps=1; both are visible in the arrays.
        warnings.simplefilter("ignore", UserWarning)
        for j in range(n_estimates):
            seed = np.random.SeedSequence(RUN_SEED, spawn_key=(cell.index, j))
            ibs = IBS(
                model.sim,
                model.responses,
                model.design,
                vectorized=cell.vectorized,
                neg_logl_threshold=T,
                random_seed=seed,
            )
            res = ibs(
                model.theta,
                num_reps=cell.num_reps,
                trial_weights=model.weights,
                additional_output="full",
            )
            est[j] = res.neg_logl
            var[j] = res.neg_logl_var
            samples[j] = res.num_samples_per_trial
            calls[j] = res.fun_count
            flags[j] = res.exit_flag
            decided[j] = bool(ibs.vectorized)
    seconds = time.perf_counter() - t0
    raw = dict(
        est=est,
        var=var,
        samples=samples,
        calls=calls,
        flags=flags,
        decided=decided,
    )
    return cell, raw, seconds


# --------------------------------------------------------------------------
# Exact references.


def reference_calibration(model_name, num_reps, index):
    """Mean squared z-score and coverage of exact IBS estimates."""
    from pyibs.testing._exact import exact_draw

    model = models_by_name()[model_name]
    M = int(min(REFERENCE_MAX, REFERENCE_BUDGET // (num_reps * model.p.size)))
    rng = np.random.default_rng(
        np.random.SeedSequence(RUN_SEED, spawn_key=(10**6, index))
    )
    d = exact_draw(model.p, num_reps * M, rng, model.weights)
    loglik = d.values.reshape(M, num_reps).mean(axis=1)
    var = d.var_estimates.reshape(M, num_reps).sum(axis=1) / num_reps**2
    err = -loglik - model.exact_nll
    stats = _calibration(err, var)
    stats["n"] = M
    return ("calibration", model_name, num_reps), stats


def reference_threshold(model_name, index):
    """Expected value of a thresholded estimate, from exact repeats."""
    from pyibs.testing._exact import exact_draw

    model = models_by_name()[model_name]
    rng = np.random.default_rng(
        np.random.SeedSequence(RUN_SEED, spawn_key=(2 * 10**6, index))
    )
    d = exact_draw(model.p, THRESHOLD_REPEATS, rng, model.weights)
    T = model.threshold
    clipped = -np.maximum(d.values, -T)
    below = d.values < -T
    stats = dict(
        n=THRESHOLD_REPEATS,
        nll_mean=float(clipped.mean()),
        nll_se=float(clipped.std(ddof=1) / math.sqrt(clipped.size)),
        prob_below=float(below.mean()),
    )
    return ("threshold", model_name), stats


# --------------------------------------------------------------------------
# Statistics.


def _mean_se(x):
    x = np.asarray(x, dtype=float)
    if x.size < 2 or not np.all(np.isfinite(x)):
        return float(np.mean(x)), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size))


def _calibration(err, var):
    """Mean squared z-score and 95% coverage of estimates."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = np.where(var > 0, err**2 / var, np.where(err == 0, 0.0, np.inf))
    msz, msz_se = _mean_se(z2)
    cover = np.abs(err) <= 1.96 * np.sqrt(var)
    return dict(
        msz=msz,
        msz_se=msz_se,
        coverage=float(cover.mean()),
        zero_var=int(np.sum(var == 0)),
    )


def cell_stats(cell, raw, refs):
    model = models_by_name()[cell.model]
    est, var = raw["est"], raw["var"]
    n = est.size
    exact = model.exact_nll
    err = est - exact
    out = dict(n=n, exact_nll=exact)
    mean_err, se_err = _mean_se(err)
    out["mean_err"] = mean_err
    out["se_err"] = se_err
    if cell.threshold:
        ref = refs[("threshold", cell.model)]
        expected = ref["nll_mean"]
        se = math.sqrt(se_err**2 + ref["nll_se"] ** 2)
        out["expected_nll"] = expected
        out["bias_z"] = (float(est.mean()) - expected) / se
        out["frac_flag1"] = float(np.mean(raw["flags"] == 1))
        out["expected_frac_flag1"] = 1 - (1 - ref["prob_below"]) ** (
            cell.num_reps
        )
    else:
        out["expected_nll"] = exact
        out["bias_z"] = mean_err / se_err if se_err > 0 else float("nan")
        exact_sd = math.sqrt(model.exact_var / cell.num_reps)
        out["sd_ratio"] = float(np.std(err, ddof=1) / exact_sd)
        out["frac_flag1"] = float(np.mean(raw["flags"] == 1))
    out.update(_calibration(err, var))
    out["msz_z"] = (
        (out["msz"] - 1) / out["msz_se"]
        if out["msz_se"] > 0
        else float("inf")
        if out["msz"] != 1
        else 0.0
    )
    if not cell.threshold:
        ref = refs.get(("calibration", cell.model, cell.num_reps))
        if ref is not None:
            out["ref_msz"] = ref["msz"]
            out["ref_msz_se"] = ref["msz_se"]
            out["ref_coverage"] = ref["coverage"]
            out["ref_zero_var"] = ref["zero_var"]
            out["ref_n"] = ref["n"]
    # Samples per trial against num_reps * mean(1 / p).
    expected_samples = cell.num_reps * float(np.mean(1 / model.p))
    out["samples_mean"] = float(raw["samples"].mean())
    out["samples_expected"] = expected_samples
    out["samples_ratio"] = out["samples_mean"] / expected_samples
    sd_one = math.sqrt(
        cell.num_reps * float(np.sum((1 - model.p) / model.p**2))
    ) / (model.p.size)
    out["samples_z"] = (
        (out["samples_mean"] - expected_samples) / (sd_one / math.sqrt(n))
        if sd_one > 0
        else float("nan")
    )
    # The share of zero variance estimates against its exact probability:
    # a variance estimate is 0 when every count of trials of positive
    # weight is 1.
    positive = model.w > 0
    p_zero = float(np.prod(model.p[positive]) ** cell.num_reps)
    k_zero = out["zero_var"]
    out["zero_var_expected"] = n * p_zero
    sd_zero = math.sqrt(n * p_zero * (1 - p_zero))
    out["zero_var_z"] = (
        (k_zero - n * p_zero) / sd_zero
        if sd_zero > 0
        else 0.0
        if k_zero == n * p_zero
        else float("inf")
    )
    out["calls_mean"] = float(raw["calls"].mean())
    out["decided_true"] = float(raw["decided"].mean())
    # The gates.
    out["pass_bias"] = bool(abs(out["bias_z"]) <= TOL)
    reported_only = cell.model in CALIBRATION_REPORTED_ONLY
    gated_cal = (
        cell.num_reps >= 10 and not cell.threshold and not reported_only
    )
    out["pass_cal"] = bool(abs(out["msz_z"]) <= TOL) if gated_cal else None
    out["pass_zero"] = (
        bool(abs(out["zero_var_z"]) <= TOL) if reported_only else None
    )
    out["passed"] = (
        out["pass_bias"]
        and out["pass_cal"] is not False
        and out["pass_zero"] is not False
    )
    return out


# --------------------------------------------------------------------------
# The zero check.


def zero_check():
    """A model whose trials all have p = 1 returns exactly 0 and 0."""
    from pyibs import IBS

    R = np.arange(50, dtype=float)
    results = []
    for vectorized in VECTORIZED:
        for num_reps in NUM_REPS:
            for T in (np.inf, 50 * math.log(2)):
                ibs = IBS(
                    Echo(R),
                    R,
                    vectorized=vectorized,
                    neg_logl_threshold=T,
                    random_seed=0,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    value, variance = ibs(
                        np.zeros(1), num_reps=num_reps, additional_output="var"
                    )
                warned = any(
                    "variance estimate is 0" in str(w.message) for w in caught
                )
                ok = value == 0.0 and variance == 0.0 and warned
                results.append(
                    dict(
                        vectorized=vectorized,
                        num_reps=num_reps,
                        threshold=None if math.isinf(T) else T,
                        value=value,
                        var=variance,
                        warned=warned,
                        ok=ok,
                    )
                )
    return results


# --------------------------------------------------------------------------
# Provenance and output.


def provenance():
    import numpy
    import scipy

    import pyibs

    def git(*args):
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    return dict(
        date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        commit=git("rev-parse", "HEAD"),
        clean=git("status", "--porcelain") == "",
        pyibs=pyibs.__version__,
        pyibs_file=pyibs.__file__,
        python=sys.version.split()[0],
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        platform=platform.platform(),
        processor=platform.processor() or platform.machine(),
        cpu_count=os.cpu_count(),
    )


def _json(x):
    if isinstance(x, dict):
        return {str(k): _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else repr(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return _json(x.tolist())
    return x


def write(path, payload):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(_json(payload), indent=1))
    os.replace(tmp, path)


HEADER = (
    f"{'cell':<34} {'n':>5} {'bias_z':>7} {'msz':>6} {'msz_z':>7} "
    f"{'ref_msz':>7} {'cover':>6} {'sd_rat':>6} {'smp_rat':>7} "
    f"{'smp_z':>6} {'zero':>5} {'zero_z':>6} {'flag1':>5} {'s/est':>8} pass"
)


def line(cell, s, seconds):
    def f(x, fmt):
        return format(x, fmt) if x is not None else "-"

    passed = "ok" if s["passed"] else "FAIL"
    return (
        f"{cell.name:<34} {s['n']:>5} {s['bias_z']:>7.2f} {s['msz']:>6.3f} "
        f"{s['msz_z']:>7.2f} {f(s.get('ref_msz'), '7.3f'):>7} "
        f"{s['coverage']:>6.3f} {f(s.get('sd_ratio'), '6.3f'):>6} "
        f"{s['samples_ratio']:>7.3f} "
        f"{(s['samples_z'] if cell.vectorized is False else float('nan')):>6.2f} "
        f"{s['zero_var']:>5} {s['zero_var_z']:>6.2f} "
        f"{s['frac_flag1']:>5.2f} {seconds / s['n']:>8.4f} {passed}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--estimates", type=int, default=100)
    parser.add_argument(
        "--estimates-for",
        action="append",
        default=[],
        metavar="REGEX=N",
        help="estimates of the cells whose name matches REGEX",
    )
    parser.add_argument("--only", help="run only the cells matching REGEX")
    parser.add_argument("--skip", help="skip the cells matching REGEX")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--project", type=int, default=2000)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--no-raw", action="store_true")
    parser.add_argument("--list", action="store_true", help="list cells")
    args = parser.parse_args(argv)

    cells = all_cells()
    if args.only:
        cells = [c for c in cells if re.search(args.only, c.name)]
    if args.skip:
        cells = [c for c in cells if not re.search(args.skip, c.name)]
    overrides = []
    for item in args.estimates_for:
        pattern, _, count = item.rpartition("=")
        overrides.append((re.compile(pattern), int(count)))

    def n_for(cell):
        n = args.estimates
        for pattern, count in overrides:
            if pattern.search(cell.name):
                n = count
        return n

    if args.list:
        for c in cells:
            print(c.name, n_for(c))
        print(len(cells), "cells")
        return 0

    # One thread per worker, so that the workers do not compete.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    models = models_by_name()
    meta = provenance()
    meta.update(
        argv=sys.argv,
        data_seed=DATA_SEED,
        run_seed=RUN_SEED,
        tolerance=TOL,
        jobs=args.jobs,
    )
    print("PyIBS validation", json.dumps(_json(meta), indent=1), flush=True)
    model_info = {
        m.name: dict(
            trials=int(m.p.size),
            exact_nll=m.exact_nll,
            exact_var_per_repeat=m.exact_var,
            threshold=m.threshold,
            p_min=float(m.p.min()),
            p_max=float(m.p.max()),
            mean_inv_p=float(np.mean(1 / m.p)),
            info=m.info,
        )
        for m in models.values()
    }
    for name, info in model_info.items():
        print(
            f"model {name}: N={info['trials']} exact_nll="
            f"{info['exact_nll']:.4f} var1={info['exact_var_per_repeat']:.4f}"
            f" p in [{info['p_min']:.4g}, {info['p_max']:.4g}] "
            f"mean(1/p)={info['mean_inv_p']:.3f} T={info['threshold']}",
            flush=True,
        )

    zero = zero_check()
    print(
        "zero check:",
        "ok" if all(r["ok"] for r in zero) else "FAIL",
        f"({len(zero)} settings)",
        flush=True,
    )

    payload = dict(
        meta=meta, models=model_info, zero_check=zero, refs={}, cells={}
    )
    raw_dir = args.out.with_suffix("")
    raw_dir = raw_dir.parent / (raw_dir.name + "_raw")
    if not args.no_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
    write(args.out, payload)

    t_start = time.perf_counter()
    ctx = get_context("spawn")
    refs = {}
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
        # The exact references first: each cell's statistics need them.
        names = sorted({c.model for c in cells})
        ref_futures = []
        for i, name in enumerate(names):
            m = models[name]
            k = list(models).index(name)
            if any(c.model == name and c.threshold for c in cells):
                ref_futures.append(pool.submit(reference_threshold, name, k))
            for j, num_reps in enumerate(NUM_REPS):
                if any(
                    c.model == name
                    and c.num_reps == num_reps
                    and not c.threshold
                    for c in cells
                ):
                    ref_futures.append(
                        pool.submit(
                            reference_calibration,
                            name,
                            num_reps,
                            k * len(NUM_REPS) + j,
                        )
                    )
        # Costly cells first: more repeats, and one sample per call.
        order = sorted(
            cells,
            key=lambda c: (-c.num_reps, c.vectorized is not False, c.index),
        )
        cell_futures = [pool.submit(run_cell, c, n_for(c)) for c in order]
        for fut in as_completed(ref_futures):
            key, stats = fut.result()
            refs[key] = stats
        payload["refs"] = {"|".join(map(str, k)): v for k, v in refs.items()}
        print(
            f"references done after {time.perf_counter() - t_start:.1f} s",
            flush=True,
        )
        print(HEADER, flush=True)
        done = {}
        for fut in as_completed(cell_futures):
            cell, raw, seconds = fut.result()
            stats = cell_stats(cell, raw, refs)
            stats["seconds"] = seconds
            stats["settings"] = dict(
                model=cell.model,
                vectorized=cell.vectorized,
                num_reps=cell.num_reps,
                threshold=cell.threshold,
            )
            done[cell.name] = (cell, stats)
            payload["cells"][cell.name] = stats
            if not args.no_raw:
                np.savez_compressed(raw_dir / f"{cell.name}.npz", **raw)
            write(args.out, payload)
            print(line(cell, stats, seconds), flush=True)

    elapsed = time.perf_counter() - t_start
    failing = [n for n, (_, s) in done.items() if not s["passed"]]
    cpu = sum(s["seconds"] for _, s in done.values())
    projected = sum(
        s["seconds"] / s["n"] * args.project for _, s in done.values()
    )
    slowest = sorted(
        done.values(), key=lambda cs: -cs[1]["seconds"] / cs[1]["n"]
    )[:10]
    payload["summary"] = dict(
        cells=len(done),
        failing=failing,
        wall_seconds=elapsed,
        cell_seconds=cpu,
        projected_cell_seconds=projected,
        projected_estimates=args.project,
    )
    write(args.out, payload)
    print(flush=True)
    print(
        f"{len(done)} cells in {elapsed:.1f} s wall, {cpu:.1f} s in cells; "
        f"zero check {'ok' if all(r['ok'] for r in zero) else 'FAIL'}",
        flush=True,
    )
    print(f"failing cells ({len(failing)}): {', '.join(failing) or '-'}")
    print(
        f"projected at {args.project} estimates per cell: {projected:.0f} s "
        f"in cells, about {projected / args.jobs / 3600:.2f} h wall on "
        f"{args.jobs} workers ({projected / 3600:.2f} h on one)",
        flush=True,
    )
    print("slowest cells per estimate:")
    for cell, s in slowest:
        print(
            f"  {cell.name:<34} {s['seconds'] / s['n']:.4f} s/est, "
            f"{s['seconds'] / s['n'] * args.project / 60:.1f} min at "
            f"{args.project}",
            flush=True,
        )
    return 0 if not failing else 1


if __name__ == "__main__":
    sys.exit(main())
