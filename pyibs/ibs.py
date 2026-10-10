"""Inverse binomial sampling (IBS) estimates of the log-likelihood.

:class:`IBS` holds a model's simulator, the observed data and the sampling
settings, and estimates the log-likelihood of a parameter vector each time
it is called, with an estimate of the estimate's variance ([1]).

References
----------
.. [1] van Opheusden, B., Acerbi, L. & Ma, W. J. (2020). Unbiased and
   efficient log-likelihood estimation with inverse binomial sampling.
   PLOS Computational Biology 16(12): e1008483.
   https://doi.org/10.1371/journal.pcbi.1008483
"""

import inspect
import math
import time
import warnings

import numpy as np

from pyibs import _sampler

# The FAQ's answer on an SD of zero, which the warning on a zero variance
# names
_FAQ_ZERO_SD = (
    "https://acerbilab.github.io/pyibs/faq.html#faq-why-is-the-sd-of-the-"
    "estimate-zero-and-why-do-pybads-and-pyvbmc-refuse-it"
)

_ZERO_VARIANCE = (
    "The IBS variance estimate is 0, as it is when every trial of positive "
    "weight matched its response at its first sample. PyBADS and PyVBMC "
    f"refuse an SD of 0 for a noisy target: see {_FAQ_ZERO_SD}"
)

# The messages of the exit flags.
_EXIT_MESSAGES = {
    0: (
        "All requested IBS repeats completed without reaching the "
        "likelihood threshold."
    ),
    1: (
        "The negative log-likelihood threshold ended a repeat; the estimate "
        "is biased."
    ),
    2: (
        "The sampling stopped at max_time; the estimate can be arbitrarily "
        "biased."
    ),
}

_ADDITIONAL_OUTPUTS = ("none", "var", "std", "full")


class EstimateResult(dict):
    """The result of an :class:`IBS` call with ``additional_output="full"``.

    A dictionary whose keys can also be read as attributes.

    Attributes
    ----------
    neg_logl : float
        The negative log-likelihood estimate, or the log-likelihood
        estimate with ``return_positive=True``.
    neg_logl_var : float
        Estimated variance of ``neg_logl``.
    neg_logl_std : float
        Estimated standard deviation of ``neg_logl``: the square root of
        ``neg_logl_var``.
    exit_flag : int
        0 if all requested repeats completed without reaching the
        likelihood threshold; 1 if the threshold ended at least one
        repeat; 2 if ``max_time`` stopped the sampling. Flag 2 takes
        precedence when both stopping rules apply. These flags describe
        how the call ended; see the stopping settings of :class:`IBS`
        for their statistical consequences.
    message : str
        The exit flag's meaning.
    elapsed_time : float
        Seconds spent in the call.
    num_samples_per_trial : float
        Total simulated responses divided by the number of trials.
        Includes samples drawn after a trial's last required match.
    fun_count : int
        Number of simulator calls used to produce this estimate.
    neg_logl_trials : ndarray of shape (N,)
        Unweighted negative log-likelihood estimate for each trial,
        averaged over its completed repeats. All entries are NaN if the
        likelihood threshold ended any repeat. Otherwise, their weighted
        sum equals ``neg_logl`` up to rounding, with the opposite sign
        when ``return_positive=True``.
    neg_logl_var_trials : ndarray of shape (N,)
        Estimated variance for each entry of ``neg_logl_trials``. All
        entries are NaN if the threshold ended any repeat. Otherwise,
        summing them with squared trial weights gives ``neg_logl_var``.
    """

    _ORDER = (
        "neg_logl",
        "neg_logl_var",
        "neg_logl_std",
        "exit_flag",
        "message",
        "elapsed_time",
        "num_samples_per_trial",
        "fun_count",
        "neg_logl_trials",
        "neg_logl_var_trials",
    )

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(name) from err

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if not self:
            return self.__class__.__name__ + "()"
        width = max(map(len, self.keys())) + 1
        keys = [k for k in self._ORDER if k in self]
        keys += [k for k in self if k not in self._ORDER]
        # The per-trial arrays show their first and last three entries.
        with np.printoptions(threshold=10, edgeitems=3):
            return "\n".join(
                k.rjust(width) + ": " + repr(self[k]) for k in keys
            )

    def __dir__(self):
        return list(self.keys())


def _rng(random_seed):
    """The generator of ``random_seed``, as PyBADS's ``random_seed``.

    None derives a new generator from NumPy's global random state; an
    integer (a whole-number float taken as one) or a ``SeedSequence``
    seeds a new one; a ``Generator`` is used as given.
    """
    seed = random_seed
    if isinstance(seed, (float, np.floating)) and float(seed).is_integer():
        seed = int(seed)
    if seed is None:
        seed = np.random.randint(0, 2**32, size=4, dtype=np.uint32)
    try:
        return np.random.default_rng(seed)
    except (TypeError, ValueError) as err:
        raise type(err)(
            "random_seed must be None or a value that "
            "numpy.random.default_rng takes, such as a non-negative "
            "integer, a numpy.random.SeedSequence or a "
            f"numpy.random.Generator, got {random_seed!r}."
        ) from err


def _takes_rng(fun):
    """Whether ``fun`` has a parameter named ``rng`` that takes a keyword.

    A callable whose signature :func:`inspect.signature` cannot read has
    none.
    """
    try:
        parameters = inspect.signature(fun).parameters
    except (TypeError, ValueError):
        return False
    rng = parameters.get("rng")
    return rng is not None and rng.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


class _Simulator:
    """The user's simulator, called as the sampler calls a simulator.

    ``_Simulator(fun)(params, design_rows, rng)`` calls
    ``fun(params, design_rows, rng=rng)`` when ``fun`` has a parameter
    named ``rng`` (:func:`_takes_rng`), and ``fun(params, design_rows)``
    otherwise. It pickles when ``fun`` does.
    """

    def __init__(self, fun):
        self.fun = fun
        self.takes_rng = _takes_rng(fun)

    def __call__(self, params, design_rows, rng):
        if self.takes_rng:
            return self.fun(params, design_rows, rng=rng)
        return self.fun(params, design_rows)


class IBS:
    """Estimate a model's log-likelihood by inverse binomial sampling.

    An ``IBS`` object holds a simulator, the observed responses and the
    sampling settings. Call it with a parameter vector to estimate the
    negative log-likelihood of the data and, optionally, its variance.
    IBS simulates each trial until a response matches the observation
    ([1]_). The estimator is unbiased when sampling is allowed to complete;
    a likelihood threshold or time limit can introduce bias.

    Parameters
    ----------
    sample_from_model : callable
        Called as ``sample_from_model(params, design_rows)``. If it has a
        parameter named ``rng`` that accepts a keyword, IBS also passes
        its random generator as ``rng=rng``.

        ``design_rows`` contains the requested rows of ``design_matrix``,
        or their 0-based trial indices when no design is supplied. A trial
        can appear more than once in a call. Return an independent draw
        for every requested row: an array of shape (r,) or (r, 1) for
        single-column responses, or (r, C) for C > 1 response columns,
        where r is the number of requested rows.
    response_matrix : array_like of shape (N,) or (N, C)
        Observed responses, one row per trial; a scalar represents one
        trial. A simulated response matches only if every column agrees.
        Responses must be discrete and comparable with ``==``: numbers,
        booleans, text, bytes, or objects. The simulator must return the
        same kind of response: text and bytes do not match numbers.
        For responses that mix numbers and text, use ``dtype=object`` for
        both observed and simulated arrays. If only one of them is an
        object array, NumPy has made text of the other's numbers:
        simulated text against observed objects raises ``TypeError``, and
        observed text never matches the numbers of simulated objects. If
        both are text, numbers match only when written alike (1 and 1.0
        differ).
    design_matrix : array_like of shape (N, ...), optional
        Experimental conditions or other simulator inputs, one row per
        trial. With None (the default), the simulator receives trial
        indices instead.
    vectorized : bool or None, optional
        The sampling schedule. True requests several samples per open
        trial in each simulator call; ``acceleration`` controls how this
        number grows. False requests one sample per open trial per call.
        None (the default) chooses a schedule on the first call with
        ``num_reps > 1`` by timing one simulation of all trials. It chooses
        False if that simulation takes at least ``vectorized_threshold``
        seconds, and True otherwise. The choice is retained for later
        calls and exposed through the ``vectorized`` attribute.

        This timing cannot distinguish a fixed cost per simulator call
        from a cost per response. It can choose False for a simulator with
        a large fixed cost per call, which True spreads over more samples;
        set True for such a simulator. If its first call compiles code or
        performs other setup, warm it up before IBS times it. With
        ``num_reps=1``, IBS uses the False schedule and warns if True was
        explicitly requested.
    acceleration : float, optional
        Factor by which the requested samples per trial grow between
        simulator calls. Must be finite and >= 1. Default 1.5.
    num_samples_per_call : int, optional
        Initial number of samples per trial per simulator call, subject to
        ``max_samples`` and ``max_mem``. The default, 0, starts at
        ``num_reps`` samples. Unused with ``vectorized=False``.
    max_iter : int, optional
        Sample cap per trial, expressed per repeat. A call requesting
        ``num_reps`` repeats raises :class:`IBSSamplingError` if any trial
        draws more than ``max_iter * num_reps`` samples. Default 10**5.
        A very large value, such as 10**18, effectively removes the cap;
        sampling can then continue indefinitely if an observed response
        has zero probability under the model.
    max_time : float, optional
        Time limit of each call in seconds, > 0. Checked after each
        simulator call. The default, ``np.inf``, imposes no limit. If the
        limit stops sampling, IBS averages each trial's completed repeats,
        returns exit flag 2 and warns. A trial with no completed repeat
        raises :class:`IBSSamplingError`. Repeats ended by the likelihood
        threshold contribute ``neg_logl_threshold`` to the negative
        log-likelihood before averaging.

        A finite limit can bias the returned estimates. Selecting only
        calls that finish in time can also introduce bias, since runs
        requiring fewer samples tend to have higher log-likelihoods.
    max_samples : int, optional
        Maximum samples per trial in one simulator call. Default 10**4.
    acceleration_threshold : float or None, optional
        None (the default) grows the requested samples after every
        simulator call. A positive time in seconds grows them only after
        calls faster than that time. With a finite threshold, wall-clock
        timing affects the draws, so a seed alone cannot reproduce a run.
    vectorized_threshold : float, optional
        Time in seconds at or above which ``vectorized=None`` selects
        False. Must be > 0. Default 0.1.
    max_mem : int or None, optional
        Approximate limit on samples in one simulator call. Each open
        trial receives at most ``ceil(max_mem / n_open)`` samples, where
        ``n_open`` is the number of trials still being sampled. Rounding
        can exceed the limit by fewer than ``n_open`` samples. None (the
        default) sets ``max(min(N, 10**4), 10) * 100``.
    neg_logl_threshold : float, optional
        Threshold T > 0 for the weighted negative log-likelihood. A repeat
        stops once its accumulating estimate exceeds T and contributes
        T to the negative log-likelihood (or -T to the log-likelihood).
        This saves simulations at poor parameter vectors but biases the
        log-likelihood estimate upwards ([1]_, Appendix C.1).

        For optimization, a common choice is the chance-level negative
        log-likelihood: ``sum_i w_i log(k_i)`` for k_i equally likely
        responses and trial weight w_i, or ``N log 2`` for N binary trials
        with unit weights. For Bayesian inference, leave the threshold
        disabled unless its effect on the posterior and model evidence
        has been assessed. The default, ``np.inf``, disables it.
    random_seed : None, int, numpy.random.SeedSequence or \
numpy.random.Generator, optional
        Seed for ``rng``; keyword only. None (the default) derives a new
        generator from NumPy's global random state, so calling
        ``np.random.seed`` before construction fixes its initial state.
        An integer or ``SeedSequence`` seeds a new generator. An existing
        ``Generator`` is used directly.

    Attributes
    ----------
    rng : numpy.random.Generator
        Random generator passed to ``sample_from_model`` when it has a
        parameter named ``rng`` that accepts a keyword.
    vectorized : bool or None
        Selected sampling schedule. With automatic selection, remains
        None until the first call with ``num_reps > 1``.

    Raises
    ------
    TypeError
        If ``sample_from_model`` is not callable or a setting has the
        wrong type, such as a boolean for a count or a time.
    ValueError
        If ``response_matrix`` is empty, has more than two dimensions, or
        contains a NaN, an element not equal to itself, which no simulated
        response matches; if ``design_matrix`` does not have N rows; or if
        a setting is out of range.

    Notes
    -----
    **Settings.** Constructor parameters other than ``random_seed`` are
    available as read-only attributes. The response and design arrays are
    read-only copies. ``max_mem`` contains the resolved sample limit;
    ``vectorized`` contains the selected schedule. Counts take integers or
    whole-number floats, such as ``1e5``, and are stored as integers; other
    numeric settings are stored as floats.

    **Reproducibility.** The simulator must draw from the ``rng`` it
    receives. Two objects created with the same seed then reproduce the
    same sequence of estimates when timing does not affect the sampling:
    set ``vectorized`` explicitly to True or False, and keep
    ``acceleration_threshold`` and ``max_time`` at their defaults.
    Automatic schedule selection also reproduces the draws if it makes
    the same choice in both runs.

    **Cost.** If the observed response has probability p, its trial takes
    ``1 / p`` samples per repeat on average, before any stopping rule.
    The sample cap stops runs with exceptionally rare or impossible
    responses. A likelihood threshold can reduce the cost at poor
    parameter vectors.

    **MATLAB reference.** ``pyibs/README.md`` catalogues deliberate
    differences from MATLAB ``ibslike.m`` and explains their reasons.

    References
    ----------
    .. [1] van Opheusden, B., Acerbi, L. & Ma, W. J. (2020). Unbiased and
       efficient log-likelihood estimation with inverse binomial sampling.
       PLOS Computational Biology 16(12): e1008483.
       https://doi.org/10.1371/journal.pcbi.1008483
    """

    def __init__(
        self,
        sample_from_model,
        response_matrix,
        design_matrix=None,
        vectorized=None,
        acceleration=1.5,
        num_samples_per_call=0,
        max_iter=10**5,
        max_time=np.inf,
        max_samples=10**4,
        acceleration_threshold=None,
        vectorized_threshold=0.1,
        max_mem=None,
        neg_logl_threshold=np.inf,
        *,
        random_seed=None,
    ):
        if not callable(sample_from_model):
            raise TypeError(
                "sample_from_model must be callable, got "
                f"{sample_from_model!r}."
            )
        responses = _sampler._check_responses(
            np.atleast_1d(response_matrix), "response_matrix"
        )
        if design_matrix is not None:
            design_matrix = np.atleast_1d(design_matrix)
        design = _sampler._check_design(
            design_matrix, responses.shape[0], "design_matrix"
        )
        if vectorized is not None and not isinstance(
            vectorized, (bool, np.bool_)
        ):
            raise TypeError(
                f"vectorized must be None, True or False, got {vectorized!r}."
            )
        num_samples_per_call = _sampler._check_count(
            num_samples_per_call, "num_samples_per_call", minimum=0
        )
        max_iter = _sampler._check_count(max_iter, "max_iter")
        vectorized_threshold = _sampler._check_real(
            vectorized_threshold, "vectorized_threshold"
        )
        if not vectorized_threshold > 0:
            raise ValueError(
                "vectorized_threshold must be > 0 seconds, got "
                f"{vectorized_threshold}."
            )
        neg_logl_threshold = _sampler._check_real(
            neg_logl_threshold, "neg_logl_threshold"
        )
        if not neg_logl_threshold > 0:
            raise ValueError(
                "neg_logl_threshold must be > 0, or np.inf for none, got "
                f"{neg_logl_threshold}."
            )
        self._settings = _sampler._Settings(
            _Simulator(sample_from_model),
            responses,
            design,
            initial_samples=num_samples_per_call or None,
            acceleration=acceleration,
            acceleration_threshold=acceleration_threshold,
            max_samples=max_samples,
            max_mem=max_mem,
            max_samples_per_trial=max_iter,
            max_time=max_time,
            neg_loglik_threshold=(
                None if math.isinf(neg_logl_threshold) else neg_logl_threshold
            ),
            names=("max_iter", "num_reps"),
        )
        self._sample_from_model = sample_from_model
        self._vectorized_given = (
            None if vectorized is None else bool(vectorized)
        )
        self._vectorized = self._vectorized_given
        self._num_samples_per_call = num_samples_per_call
        self._vectorized_threshold = vectorized_threshold
        self._neg_logl_threshold = neg_logl_threshold
        self.rng = _rng(random_seed)

    @property
    def sample_from_model(self):
        """The simulator."""
        return self._sample_from_model

    @property
    def response_matrix(self):
        """The observed responses, a read-only array."""
        return self._settings.responses

    @property
    def design_matrix(self):
        """Simulator inputs per trial as a read-only array, or None."""
        return self._settings.design

    @property
    def vectorized(self):
        """Selected sampling schedule. With automatic selection, remains
        None until the first call with ``num_reps > 1``."""
        return self._vectorized

    @property
    def acceleration(self):
        """Growth factor for samples requested per trial per simulator call."""
        return self._settings.acceleration

    @property
    def num_samples_per_call(self):
        """Initial samples per trial per simulator call; 0 uses
        ``num_reps``."""
        return self._num_samples_per_call

    @property
    def max_iter(self):
        """The cap on the samples of one trial per repeat."""
        return self._settings.max_samples_per_trial

    @property
    def max_time(self):
        """The time limit of a call, in seconds."""
        return self._settings.max_time

    @property
    def max_samples(self):
        """Maximum samples per trial in one simulator call."""
        return self._settings.max_samples

    @property
    def acceleration_threshold(self):
        """Simulator-call time in seconds below which sample requests grow.

        None grows the requests after every call.
        """
        return self._settings.acceleration_threshold

    @property
    def vectorized_threshold(self):
        """Time in seconds at or above which automatic scheduling selects
        ``vectorized=False``."""
        return self._vectorized_threshold

    @property
    def max_mem(self):
        """Resolved sample limit per simulator call, subject to rounding."""
        return self._settings.max_mem

    @property
    def neg_logl_threshold(self):
        """Weighted negative log-likelihood threshold; ``inf`` disables it."""
        return self._neg_logl_threshold

    def __call__(
        self,
        params,
        num_reps=10,
        trial_weights=None,
        additional_output=None,
        return_positive=False,
    ):
        """Estimate the negative log-likelihood of ``params``.

        Parameters
        ----------
        params : array_like
            The parameter vector, passed to the simulator as given.
        num_reps : int, optional
            Number of independent IBS repeats to average, at least 1.
            Whole-number floats are accepted as integers. Default 10.
        trial_weights : None, float or array_like of shape (N,), optional
            Non-negative, finite real weights of the trials'
            log-likelihoods. Booleans and strings are rejected. A scalar
            applies the same weight to every trial; None (the default) uses
            unit weights. Zero-weight trials are still sampled and can
            reach the sample cap or time limit. Remove trials from the data
            to exclude them entirely.
        additional_output : None or str, optional
            None or ``"none"`` returns only the estimate. ``"var"`` adds
            its estimated variance; ``"std"`` adds the square root of that
            variance. ``"full"`` returns an :class:`EstimateResult` with
            diagnostics and per-trial estimates.
        return_positive : bool, optional
            Whether to return the log-likelihood rather than the negative
            log-likelihood; the variance estimate and the per-trial arrays
            of ``"full"`` are unchanged. Default False.

        Returns
        -------
        neg_logl : float
            The negative log-likelihood estimate (the log-likelihood with
            ``return_positive=True``), when ``additional_output`` is None.
        (neg_logl, neg_logl_var) : tuple of two floats
            Estimate and estimated variance, with ``"var"``.
        (neg_logl, neg_logl_std) : tuple of two floats
            Estimate and estimated standard deviation, with ``"std"``.
            PyBADS and PyVBMC accept this pair from a noisy target.
        result : EstimateResult
            Estimate and diagnostics, with ``"full"``.

        Raises
        ------
        IBSSamplingError
            If a trial draws more than ``max_iter * num_reps`` samples, or
            ``max_time`` stops the sampling before a trial has completed a
            repeat.
        ValueError
            If ``num_reps``, ``trial_weights`` or ``additional_output`` is
            invalid, or the simulator returns an array with the wrong
            shape.
        TypeError
            If ``num_reps``, ``trial_weights`` or ``return_positive`` has
            the wrong type, or the simulator returns responses of a kind
            that cannot match the observations, such as text for numbers.

        Warns
        -----
        UserWarning
            When ``max_time`` stops the sampling (exit flag 2); when a call
            returns a variance estimate of 0, which gives an SD that
            PyBADS and PyVBMC refuse; or when ``vectorized=True`` is used
            with ``num_reps=1``.
        """
        t0 = time.perf_counter()
        num_reps = _sampler._check_count(num_reps, "num_reps")
        if not isinstance(return_positive, (bool, np.bool_)):
            raise TypeError(
                "return_positive must be True or False, got "
                f"{return_positive!r}."
            )
        if additional_output is not None and not (
            isinstance(additional_output, str)
            and additional_output in _ADDITIONAL_OUTPUTS
        ):
            raise ValueError(
                "additional_output must be None, 'none', 'var', 'std' or "
                f"'full', got {additional_output!r}."
            )
        if additional_output == "none":
            additional_output = None
        settings = self._settings
        if trial_weights is not None:
            settings = settings.with_trial_weights(trial_weights)
        first = None
        vectorized = self._vectorized
        if num_reps == 1:
            if self._vectorized_given:
                warnings.warn(
                    "vectorized=True needs num_reps > 1: this call requests "
                    "one sample per trial per call, as vectorized=False "
                    "does.",
                    UserWarning,
                    stacklevel=2,
                )
            vectorized = False
        elif vectorized is None:
            first = _sampler.first_round(settings, params, self.rng)
            vectorized = first.elapsed < self._vectorized_threshold
            self._vectorized = vectorized
        draw = _sampler.sample(
            settings,
            params,
            num_reps,
            self.rng,
            vectorized=vectorized,
            start=t0,
            first=first,
        )
        if draw.timed_out:
            exit_flag = 2
            message = (
                f"IBS reached max_time = {self.max_time:g} s before every "
                f"trial completed its {num_reps} repeats: each trial's "
                "value averages its completed repeats, and the estimate can "
                "be arbitrarily biased (exit flag 2)."
            )
            if draw.n_thresholded:
                threshold_value = (
                    -self.neg_logl_threshold
                    if return_positive
                    else self.neg_logl_threshold
                )
                message += (
                    f" The likelihood threshold ended {draw.n_thresholded} "
                    f"of the {num_reps} repeats; each contributes "
                    f"{threshold_value:g} "
                    "to the returned estimate before averaging."
                )
            warnings.warn(message, UserWarning, stacklevel=2)
        elif draw.n_thresholded:
            exit_flag = 1
        else:
            exit_flag = 0
        # 0.0 - x turns a log-likelihood of 0 into 0.0 rather than -0.0.
        neg_logl = 0.0 - draw.loglik
        value = draw.loglik if return_positive else neg_logl
        var = draw.loglik_var
        if additional_output is None:
            return value
        if var == 0:
            warnings.warn(_ZERO_VARIANCE, UserWarning, stacklevel=2)
        if additional_output == "var":
            return value, var
        if additional_output == "std":
            return value, math.sqrt(var)
        if draw.n_thresholded:
            neg_logl_trials = np.full(settings.n_trials, np.nan)
            neg_logl_var_trials = np.full(settings.n_trials, np.nan)
        else:
            neg_logl_trials = (0.0 - draw.trial_value_sums) / draw.trial_counts
            neg_logl_var_trials = draw.trial_var_sums / draw.trial_counts**2
        return EstimateResult(
            neg_logl=value,
            neg_logl_var=var,
            neg_logl_std=math.sqrt(var),
            exit_flag=exit_flag,
            message=_EXIT_MESSAGES[exit_flag],
            elapsed_time=time.perf_counter() - t0,
            num_samples_per_trial=draw.samples / settings.n_trials,
            fun_count=draw.calls,
            neg_logl_trials=neg_logl_trials,
            neg_logl_var_trials=neg_logl_var_trials,
        )
