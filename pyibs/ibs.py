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

_EXIT_MESSAGES = {
    0: "Correct termination (the estimate is unbiased).",
    1: (
        "Termination after negative log-likelihood threshold was reached "
        "(the estimate is biased)."
    ),
    2: (
        "Termination after maximum execution time was reached (the "
        "estimate can be arbitrarily biased)."
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
        The variance estimate of the estimate.
    neg_logl_std : float
        Its square root.
    exit_flag : int
        0 when every repeat was sampled to completion (the estimate is
        unbiased when ``max_time`` is infinite: see ``max_time`` of
        :class:`IBS`), 1 when the likelihood threshold ended a repeat (the
        log-likelihood estimate is biased upwards, and the negative
        log-likelihood estimate downwards), 2 when ``max_time`` stopped the
        sampling (the estimate can be arbitrarily biased).
    message : str
        The exit flag's meaning.
    elapsed_time : float
        Seconds spent in the call.
    num_samples_per_trial : float
        Simulated responses per trial: every row that the simulator
        returned in the call, the samples drawn after a trial's last
        match included, divided by the number of trials.
    fun_count : int
        Calls of ``sample_from_model`` in the call.
    neg_logl_trials : ndarray of shape (N,)
        Each trial's unweighted negative log-likelihood estimate, the
        average of its completed repeats; NaN when the likelihood threshold
        ended a repeat. Weighted by the trial weights, they add up to
        ``neg_logl`` (to rounding, and with the opposite sign under
        ``return_positive=True``).
    neg_logl_var_trials : ndarray of shape (N,)
        Their variance estimates; NaN when the threshold ended a repeat.
        Weighted by the squared trial weights, they add up to
        ``neg_logl_var``.
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
        except KeyError:
            raise AttributeError(name)

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
    """Inverse binomial sampling estimates of a model's log-likelihood.

    An ``IBS`` object holds a simulator of the model's responses, the
    observed responses and the sampling settings. Called with a parameter
    vector, it returns an unbiased estimate of the negative log-likelihood
    of the data, and optionally an estimate of the estimate's variance,
    computed by inverse binomial sampling ([1]): for every trial it
    simulates responses until one matches the observed response.

    Parameters
    ----------
    sample_from_model : callable
        The simulator, ``sample_from_model(params, design_rows)``, or
        ``sample_from_model(params, design_rows, rng=rng)`` when it has a
        parameter named ``rng``, which then receives the object's
        generator, ``rng``. It returns one simulated response per row of
        ``design_rows``, which holds the rows of ``design_matrix`` of the
        requested trials, or their 0-based indices when ``design_matrix``
        is None; a trial can be requested several times in one call, and
        every requested row must be an independent draw. For r requested
        rows, it returns an array of shape (r,) or (r, 1) when the
        responses have one column, and of shape (r, C) when they have C > 1
        columns.
    response_matrix : array_like of shape (N,) or (N, C)
        The observed responses of the N trials, one row per trial; a
        scalar is one trial. A simulated response matches a trial's only
        when every column agrees. The responses must be discrete: numbers,
        booleans, text, bytes, or objects compared by ``==``. A simulator
        must return them of the same kind, since NumPy never finds text or
        bytes equal to numbers. Responses that mix numbers and text are
        given as an object array (``dtype=object``), and the simulator
        returns them as one: an array that NumPy makes of such a mix holds
        text, which never equals a number, and raises ``TypeError``.
    design_matrix : array_like of shape (N, ...), optional
        The design of each trial, one row per trial, which the simulator
        receives for the requested trials. None, the default, passes the
        trial indices instead.
    vectorized : bool or None, optional
        The sampling schedule. True requests several samples of every trial
        per simulator call, a number that grows from call to call
        (``acceleration``); False requests one sample of every trial that
        still needs one. None, the default, decides at the object's first
        call with ``num_reps > 1``, by timing one simulation of all trials:
        False if it takes ``vectorized_threshold`` seconds or more, True
        otherwise. The decision is kept for the object's later calls and
        read as the attribute ``vectorized``. A simulator that is slow only
        at its first call, such as one compiled just in time, can make the
        decision False for good: give it True, or call it once before the
        object's first call. A call with ``num_reps=1`` samples as with
        False, with a warning when True was given.
    acceleration : float, optional
        The factor, finite and >= 1, by which the samples requested per
        trial grow from one call to the next. Default 1.5.
    num_samples_per_call : int, optional
        The level at which the samples per trial and simulator call start,
        bounded by ``max_samples`` and ``max_mem``, which ``acceleration``
        then grows; 0, the default, starts at ``num_reps``. It is unused
        with ``vectorized=False``.
    max_iter : int, optional
        The cap on the samples of one trial, per repeat: a call of
        ``num_reps`` repeats raises :class:`IBSSamplingError` once a trial
        has drawn more than ``max_iter * num_reps`` samples. Default 10**5.
        A very large value, such as 10**18, sets a cap that no call
        reaches; a call whose simulator cannot produce an observed response
        then never ends.
    max_time : float, optional
        The time limit of a call, in seconds, > 0, checked after every
        simulator call. Once it is reached, the sampling stops, and each
        trial's value averages its completed repeats, while a repeat that
        the likelihood threshold ended counts -T; the result has exit flag
        2, with a warning, and a trial with no completed repeat raises
        :class:`IBSSamplingError`. The default, ``np.inf``, sets none. A
        finite limit biases the estimates, also those of the calls that
        complete in time: completing in time favours few samples, which
        give high log-likelihoods.
    max_samples : int, optional
        The bound on the samples of one trial in one simulator call.
        Default 10**4.
    acceleration_threshold : float or None, optional
        None, the default, grows the samples per call after every call. A
        time in seconds > 0 grows them only after calls that took less;
        the samples drawn then depend on the wall-clock time, and a seed
        no longer reproduces a run.
    vectorized_threshold : float, optional
        The time in seconds, > 0, of one simulation of all trials at or
        above which ``vectorized=None`` decides False. Default 0.1.
    max_mem : int or None, optional
        The bound on the samples of one simulator call, which a call
        exceeds by less than its number of trials: each trial gets at most
        ``ceil(max_mem / n_open)`` samples, with ``n_open`` trials still
        sampled. None, the default, sets ``max(min(N, 10**4), 10) * 100``.
    neg_logl_threshold : float, optional
        The likelihood threshold T, > 0. A repeat whose sampling shows
        that its negative log-likelihood exceeds T is ended and counts -T,
        which saves the samples of poor parameter vectors at the price of
        an upward bias of the log-likelihood estimate ([1], Appendix C.1).
        T applies to the weighted negative log-likelihood. The usual choice
        is the chance level, the negative log-likelihood of responding at
        random: ``sum_i w_i log(k_i)`` for k_i possible responses and
        weight w_i on trial i, or ``N log 2`` for N binary choices of
        weight 1. The default, ``np.inf``, sets none.
    random_seed : None, int, numpy.random.SeedSequence or \
numpy.random.Generator, optional
        The seed of ``rng``, keyword only. None, the default, derives the
        generator from NumPy's global random state, so that
        ``np.random.seed`` before creating the object fixes it; an integer
        or a ``SeedSequence`` seeds a new generator; a ``Generator`` is
        used as given.

    Attributes
    ----------
    rng : numpy.random.Generator
        The generator of every random draw of the object's calls, passed
        to ``sample_from_model`` as ``rng`` when it has that parameter.
    vectorized : bool or None
        The sampling schedule: as given, or the decision of
        ``vectorized=None``, which is None until the object's first call
        with ``num_reps > 1``.

    Raises
    ------
    TypeError
        If ``sample_from_model`` is not callable, or a setting is of a type
        that it does not take: a count or a time that is a boolean or not a
        number, for instance.
    ValueError
        If ``response_matrix`` is not a non-empty array of shape (N,) or
        (N, C), or holds a NaN, an element not equal to itself, which no
        simulated response equals; if ``design_matrix`` does not have N
        rows; or if a setting is out of range.

    Notes
    -----
    **Settings.** Each parameter but ``random_seed`` is a read-only
    attribute of the same name. ``response_matrix`` and ``design_matrix``
    hold read-only copies, ``max_mem`` the bound in use, ``vectorized`` the
    schedule as described under Attributes, and the other settings the
    values given, the counts as integers.

    **Reproducibility.** Every random draw of a call comes from ``rng``,
    when the simulator draws from the ``rng`` it receives. Two objects
    created with the same ``random_seed`` then give the same estimates
    from the same sequence of calls, provided that no timing decides the
    sampling: ``vectorized`` given as True or False, or decided alike, and
    ``acceleration_threshold`` and ``max_time`` at their defaults.

    **The cost.** A trial whose observed response the simulator produces
    with probability p takes about ``1 / p`` samples per repeat. The cap,
    ``max_iter``, stops a call whose simulator cannot produce an observed
    response at the parameter vector; the likelihood threshold bounds the
    cost at poor parameter vectors.

    **Differences from ibslike.m.** ``pyibs/README.md`` catalogues where
    PyIBS differs from MATLAB ``ibslike.m`` on purpose, with the reasons.
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
        """The design, a read-only array, or None."""
        return self._settings.design

    @property
    def vectorized(self):
        """The sampling schedule: as given, or the decision of
        ``vectorized=None``, which is None until the object's first call
        with ``num_reps > 1``."""
        return self._vectorized

    @property
    def acceleration(self):
        """The growth factor of the samples per call."""
        return self._settings.acceleration

    @property
    def num_samples_per_call(self):
        """The level at which the samples per call start; 0 for
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
        """The bound on the samples of one trial in one call."""
        return self._settings.max_samples

    @property
    def acceleration_threshold(self):
        """The time under which a call grows the samples, or None."""
        return self._settings.acceleration_threshold

    @property
    def vectorized_threshold(self):
        """The time of one simulation at which ``None`` decides False."""
        return self._vectorized_threshold

    @property
    def max_mem(self):
        """The bound in use on the samples of one simulator call."""
        return self._settings.max_mem

    @property
    def neg_logl_threshold(self):
        """The likelihood threshold, ``inf`` for none."""
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
            The number of independent IBS repeats that the estimate
            averages, at least 1; a whole-number float is taken as the
            integer it equals. Default 10.
        trial_weights : None, float or array_like of shape (N,), optional
            Weights of the trials' log-likelihoods, finite and >= 0, real
            numbers and not booleans or strings; a scalar weighs every
            trial alike. None, the default, gives unit weights. A trial of
            weight 0 adds nothing to the estimate but is still sampled, and
            can reach the cap or the time limit: a trial to leave out is
            better removed from the data.
        additional_output : None or str, optional
            What the call returns besides the estimate: None or ``"none"``,
            nothing; ``"var"``, its variance estimate; ``"std"``, the square
            root of the variance estimate; ``"full"``, an
            :class:`EstimateResult`.
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
            With ``"var"``.
        (neg_logl, neg_logl_std) : tuple of two floats
            With ``"std"``, the form that PyBADS and PyVBMC take from a
            noisy target.
        result : EstimateResult
            With ``"full"``.

        Raises
        ------
        IBSSamplingError
            If a trial draws more than ``max_iter * num_reps`` samples, or
            ``max_time`` stops the sampling before a trial has completed a
            repeat.
        ValueError
            If ``num_reps``, ``trial_weights`` or ``additional_output`` is
            invalid, or the simulator returns an array of a shape that the
            responses do not take.
        TypeError
            If ``num_reps`` or ``trial_weights`` is of a type that it does
            not take, or the simulator returns responses of a kind that
            NumPy never finds equal to the observed ones.

        Warns
        -----
        UserWarning
            When ``max_time`` stops the sampling (exit flag 2); when a call
            returns a variance estimate of 0, which PyBADS and PyVBMC
            refuse as an SD; and when ``vectorized=True`` meets
            ``num_reps=1``.
        """
        t0 = time.perf_counter()
        num_reps = _sampler._check_count(num_reps, "num_reps")
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
                message += (
                    f" The likelihood threshold ended {draw.n_thresholded} "
                    f"of the {num_reps} repeats, which count "
                    f"-{self.neg_logl_threshold:g} each."
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
