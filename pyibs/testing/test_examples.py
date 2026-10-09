"""The example model of ``pyibs.examples`` against its closed form."""

import math

import numpy as np
import pytest

from pyibs import IBS
from pyibs.examples.psycho_model import psycho_generator, psycho_neg_logl

SEED = 20261009
# The seed of the IBS objects: a child of SEED, whose stream is independent
# of the data's, default_rng(SEED).
SEED_IBS = np.random.SeedSequence(SEED).spawn(1)[0]
N_TRIALS = 600
# ibs_example.m's generating parameters: log(sigma), bias and lapse.
THETA_TRUE = np.array([math.log(1.0), 0.2, 0.03])


@pytest.fixture(scope="module")
def data():
    """``ibs_example.m``'s data set: 600 orientations and their responses."""
    rng = np.random.default_rng(SEED)
    S = 3 * rng.standard_normal((N_TRIALS, 1))
    R = psycho_generator(THETA_TRUE, S, rng)
    return S, R


@pytest.mark.parametrize(
    "theta",
    [
        THETA_TRUE,
        np.array([math.log(0.5), -0.5, 0.1]),
        np.array([math.log(3.0), 1.0, 0.01]),
    ],
)
def test_ibs_agrees_with_the_closed_form(data, theta):
    S, R = data
    ibs = IBS(psycho_generator, R, S, vectorized=True, random_seed=SEED_IBS)
    neg_logl, sd = ibs(theta, num_reps=10, additional_output="std")
    assert abs(neg_logl - psycho_neg_logl(theta, S, R)) < 4.5 * sd


def test_one_dimensional_design_gives_the_same_estimate(data):
    # The simulator draws as many numbers, in the same order, for an
    # orientation column as for a vector of orientations.
    S, R = data
    column = IBS(psycho_generator, R, S, vectorized=True, random_seed=SEED_IBS)
    vector = IBS(
        psycho_generator,
        R[:, 0],
        S[:, 0],
        vectorized=True,
        random_seed=SEED_IBS,
    )
    assert column(THETA_TRUE, additional_output="var") == vector(
        THETA_TRUE, additional_output="var"
    )


def test_closed_form_takes_responses_of_another_shape(data):
    # A column of responses with a vector of orientations, or the reverse,
    # gives the value of matching shapes.
    S, R = data
    expected = psycho_neg_logl(THETA_TRUE, S, R)
    assert psycho_neg_logl(THETA_TRUE, S[:, 0], R) == expected
    assert psycho_neg_logl(THETA_TRUE, S, R[:, 0]) == expected


def test_closed_form_at_chance():
    # With a lapse rate of 1, every response has probability 1/2.
    S = 3 * np.random.default_rng(SEED).standard_normal(N_TRIALS)
    R = np.where(S > 0, 1.0, -1.0)
    neg_logl = psycho_neg_logl(np.array([0.0, 0.0, 1.0]), S, R)
    assert neg_logl == pytest.approx(N_TRIALS * math.log(2), rel=1e-12)


@pytest.mark.parametrize("shape", [(N_TRIALS,), (N_TRIALS, 1), (3, 4)])
def test_psycho_generator(shape):
    S = 3 * np.random.default_rng(SEED).standard_normal(shape)
    first, second, other = (
        psycho_generator(THETA_TRUE, S, np.random.default_rng(seed))
        for seed in (1, 1, 2)
    )
    assert first.shape == shape
    assert np.all(np.isin(first, [-1.0, 1.0]))
    assert np.array_equal(first, second)
    # Among 600 trials, some responses near the bias differ between seeds.
    if first.size == N_TRIALS:
        assert not np.array_equal(first, other)
