# PyIBS: Frequently Asked Questions

This FAQ is curated by [Luigi Acerbi](https://lacerbi.github.io/), and in constant expansion.
It is adapted for PyIBS 1.5 from the [MATLAB IBS FAQ](https://github.com/acerbilab/ibs/wiki),
with further questions on the Python package.

For a tutorial with detailed examples, see the [Jupyter notebook examples](examples.rst).

If you have questions not covered here, please feel free to ask in the lab
[Discussions forum](https://github.com/orgs/acerbilab/discussions).

PyIBS is one of the lab's [tools for fitting models to data](https://acerbilab.org/model-fitting/),
which also give the best-fitting parameters (PyBADS) and the posterior and
the model evidence (PyVBMC), both of which take PyIBS's estimates as their
target.

**Acknowledgments:** Many of the questions answered here originated in a
live Q&A session with the [Ma lab](http://www.cns.nyu.edu/malab/), and thanks
to Hsin-Hung Li for taking notes.

In the answers, [1] is the IBS paper: B. van Opheusden, L. Acerbi and
W. J. Ma (2020), "Unbiased and efficient log-likelihood estimation with
inverse binomial sampling", *PLOS Computational Biology* 16(12): e1008483,
<https://doi.org/10.1371/journal.pcbi.1008483>.

The snippets below use `np` for NumPy and `IBS` for the estimator class:

```python
import numpy as np
from pyibs import IBS
```

Supply your simulator `sample_from_model`, the observed responses `R` and the
design `S`, one row per trial (the arguments `response_matrix` and
`design_matrix` of `IBS`), and a parameter vector `theta`, where they appear
in a snippet. The snippets with PyBADS or PyVBMC also use a starting point
`x0` and the bounds `lb`, `ub`, `plb`, `pub` of those packages, each a
one-dimensional NumPy array with one element per parameter.

## Table of contents

- [General](#faq-general)
  - [Which kind of problems is PyIBS suited for?](#faq-which-kind-of-problems-is-pyibs-suited-for)
  - [When should I use IBS rather than amortized simulation-based inference?](#faq-when-should-i-use-ibs-rather-than-amortized-simulation-based-inference)
  - [How does IBS compare with approximate Bayesian computation (ABC) or synthetic likelihood?](#faq-how-does-ibs-compare-with-approximate-bayesian-computation-abc-or-synthetic-likelihood)
  - [What is `ibs_basic` for?](#faq-what-is-ibs_basic-for)
- [Installing PyIBS](#faq-installation)
  - [Where can I download PyIBS?](#faq-where-can-i-download-pyibs)
  - [Which external packages does PyIBS require?](#faq-which-external-packages-does-pyibs-require)
  - [Which version of Python do I need?](#faq-which-version-of-python-do-i-need)
  - [How do I know whether a newer version of PyIBS exists?](#faq-how-do-i-know-whether-a-newer-version-of-pyibs-exists)
  - [How do I check that PyIBS works on my computer?](#faq-how-do-i-check-that-pyibs-works-on-my-computer)
  - [I am having trouble installing PyIBS. Can you help?](#faq-i-am-having-trouble-installing-pyibs-can-you-help)
- [The simulator and the data](#faq-the-simulator-and-the-data)
  - [How do I write the simulator?](#faq-how-do-i-write-the-simulator)
  - [My simulator needs additional data or inputs. How do I pass them to `IBS`?](#faq-my-simulator-needs-additional-data-or-inputs-how-do-i-pass-them-to-ibs)
  - [The responses of my model depend both on the current trial and on responses or stimuli from previous trials. How can I tell this to `IBS`?](#faq-the-responses-of-my-model-depend-both-on-the-current-trial-and-on-responses-or-stimuli-from-previous-trials-how-can-i-tell-this-to-ibs)
  - [My model uses data structures which are not easily converted to numerical arrays. Can I still use `IBS`?](#faq-my-model-uses-data-structures-which-are-not-easily-converted-to-numerical-arrays-can-i-still-use-ibs)
  - [Can I use IBS with continuous responses?](#faq-can-i-use-ibs-with-continuous-responses)
  - [What does a call of `IBS` return?](#faq-what-does-a-call-of-ibs-return)
- [PyBADS and PyVBMC](#faq-pybads-and-pyvbmc)
  - [How do I use PyIBS with PyBADS?](#faq-how-do-i-use-pyibs-with-pybads)
  - [How do I use PyIBS with PyVBMC?](#faq-how-do-i-use-pyibs-with-pyvbmc)
  - [Why does the target return the SD of the estimate, and not its variance?](#faq-why-does-the-target-return-the-sd-of-the-estimate-and-not-its-variance)
  - [Why is the SD of the estimate zero, and why do PyBADS and PyVBMC refuse it?](#faq-why-is-the-sd-of-the-estimate-zero-and-why-do-pybads-and-pyvbmc-refuse-it)
  - [Do I need to give PyBADS or PyVBMC the SD of every estimate?](#faq-do-i-need-to-give-pybads-or-pyvbmc-the-sd-of-every-estimate)
  - [I have several questions about using PyBADS to optimize the log-likelihood. Can you help?](#faq-i-have-several-questions-about-using-pybads-to-optimize-the-log-likelihood-can-you-help)
  - [What if I want to use IBS to perform Bayesian posterior or model inference?](#faq-what-if-i-want-to-use-ibs-to-perform-bayesian-posterior-or-model-inference)
- [Precision and IBS repeats](#faq-precision-and-ibs-repeats)
  - [How do I choose `num_reps`?](#faq-how-do-i-choose-num_reps)
  - [Once the optimizer has found the best parameters, should I evaluate the log-likelihood there with more repeats?](#faq-once-the-optimizer-has-found-the-best-parameters-should-i-evaluate-the-log-likelihood-there-with-more-repeats)
  - [In an ideal world, would you let the number of repeats depend on how close the optimization algorithm thinks it is to the maximum?](#faq-in-an-ideal-world-would-you-let-the-number-of-repeats-depend-on-how-close-the-optimization-algorithm-thinks-it-is-to-the-maximum)
  - [As I increase `num_reps`, the SD of the estimate goes down slowly, but the computational time increases linearly. Is this normal?](#faq-as-i-increase-num_reps-the-sd-of-the-estimate-goes-down-slowly-but-the-computational-time-increases-linearly-is-this-normal)
  - [What are trial-dependent repeats, and can I use them with PyIBS?](#faq-what-are-trial-dependent-repeats-and-can-i-use-them-with-pyibs)
- [Cost, limits and the likelihood threshold](#faq-cost-limits-and-the-likelihood-threshold)
  - [How many samples does an estimate take?](#faq-how-many-samples-does-an-estimate-take)
  - [Should I set `vectorized`?](#faq-should-i-set-vectorized)
  - [What does the likelihood threshold `neg_logl_threshold` do, and how do I choose it?](#faq-what-does-the-likelihood-threshold-neg_logl_threshold-do-and-how-do-i-choose-it)
  - [Is it okay to stop the IBS algorithm for one trial after a fixed number of samples (e.g., 20)?](#faq-is-it-okay-to-stop-the-ibs-algorithm-for-one-trial-after-a-fixed-number-of-samples-eg-20)
  - [What does `max_time` do?](#faq-what-does-max_time-do)
- [Troubleshooting](#faq-troubleshooting)
  - [A call raises `IBSSamplingError`. What do I do?](#faq-a-call-raises-ibssamplingerror-what-do-i-do)
  - [`IBS` calls my simulator with different subsets of trials. What does this mean for my simulator?](#faq-ibs-calls-my-simulator-with-different-subsets-of-trials-what-does-this-mean-for-my-simulator)
  - [A call raises a `ValueError` or a `TypeError` about the simulator's output. What is wrong?](#faq-a-call-raises-a-valueerror-or-a-typeerror-about-the-simulators-output-what-is-wrong)
  - [How do I make a run reproducible?](#faq-how-do-i-make-a-run-reproducible)
  - [How do I check that the estimates are right for my model?](#faq-how-do-i-check-that-the-estimates-are-right-for-my-model)
- [Miscellanea](#faq-miscellanea)
  - [I used `ibslike` in MATLAB. What is different in PyIBS?](#faq-i-used-ibslike-in-matlab-what-is-different-in-pyibs)
  - [I used PyIBS 0.1.0. What do I need to change?](#faq-i-used-pyibs-010-what-do-i-need-to-change)

(faq-general)=
## General

(faq-which-kind-of-problems-is-pyibs-suited-for)=
### Which kind of problems is PyIBS suited for?

PyIBS estimates the log-likelihood of a model that you can simulate but whose
likelihood you cannot compute. We recommend it for problems in which ([1],
Section 2.2):

- the responses are discrete (choices, categories, counts), or can be
  binned (see [below](#faq-can-i-use-ibs-with-continuous-responses));
- the model's simulator can draw a response for any trial given that trial's
  context, such as its stimulus and possibly earlier stimuli and responses
  (*conditional simulation*);
- every observed response has a non-negligible probability under the model
  at the parameters of interest: a trial whose observed response the model
  produces with probability `p` takes about `1 / p` samples
  ([1], Section 4.2).

For each trial, IBS draws responses from the simulator until one matches the
observed response, and turns the number of draws into an estimate of the
trial's log-likelihood. Summed over the trials, these give an unbiased
estimate of the log-likelihood of the data, with a calibrated estimate of its
variance ([1], Sections 2.4, 4.3 and 4.6). That makes a simulator usable with
likelihood-based methods: maximum-likelihood or maximum-a-posteriori
estimation with [PyBADS](https://acerbilab.github.io/pybads/), the posterior
and the model evidence with [PyVBMC](https://acerbilab.org/pyvbmc/),
both among the lab's [tools for fitting models to data](https://acerbilab.org/model-fitting/),
and model comparison.

If you can compute the likelihood, in closed form or with a numerical
approximation, do so: an exact or accurate log-likelihood beats any estimate
by simulation, and IBS then serves to check its implementation ([1],
Section 6.4).

(faq-when-should-i-use-ibs-rather-than-amortized-simulation-based-inference)=
### When should I use IBS rather than amortized simulation-based inference?

Both fit models that can be simulated but whose likelihood cannot be
computed, and each is the better tool for a different kind of problem.

**What IBS gives.** IBS estimates the log-likelihood of one data set at one
parameter vector, by simulating each trial in its own context. The estimate
is unbiased, for every data set and every parameter vector, and comes with a
calibrated estimate of its variance ([1], Sections 2.4, 4.3 and 4.6). It needs
no training and no summary statistics of the data. Any method that takes a
noisy log-likelihood can then use it: maximum-likelihood or
maximum-a-posteriori estimation with PyBADS, the posterior and the model
evidence with PyVBMC, and model comparison.

**When amortized inference is the better choice.** Amortized methods, such as
neural posterior estimation or neural likelihood estimation, train a neural
network on simulations once; the network then gives the posterior of a new
data set in near-instant time, so that the cost of the training is shared
among all the data sets it serves ([Li et al., 2026](https://openreview.net/forum?id=osV7adJlKD),
Section 1). When one model is fitted to many data sets and its simulations
are cheap, this is often the better choice: IBS draws new simulations for
every data set, at every parameter vector that an optimizer or an inference
method evaluates, and a fit takes many such evaluations: in the
[validation of PyIBS](https://github.com/acerbilab/pyibs/blob/main/dev/results/2026-10-09-validation.md),
392 for PyBADS and 115 for PyVBMC on a model of three parameters. Libraries
such as [sbi](https://github.com/sbi-dev/sbi) and
[BayesFlow](https://github.com/bayesflow-org/bayesflow) implement amortized
inference (Li et al., 2026, Section 2.3).

**Where IBS remains the method of choice.**

- *Each trial's context can be unique.* In a model of game play, each move
  is conditioned on its board position, and a position may occur only once
  in the data ([1], Section 5.4; B. van Opheusden et al., 2023, "Expertise
  increases planning depth in human gameplay", *Nature* 618: 1000–1005,
  <https://doi.org/10.1038/s41586-023-06124-2>). An amortized estimator has
  to learn the model's behaviour across all the contexts it may meet; IBS
  only simulates the model in the contexts of the data. In the four-in-a-row
  game of [1], Section 5.4, whose data sets drew their positions from 5,482
  positions of human play, the model's distribution over moves cannot be
  computed, even numerically, and IBS estimates its log-likelihood from
  simulations alone.
- *Guarantees on each data set matter.* "A given pre-trained amortized
  neural estimator may be perfectly suitable for some real datasets while it
  is utterly untrustworthy for others" (C. Li, A. Vehtari, P.-C. Bürkner,
  S. T. Radev, L. Acerbi and M. Schmitt, 2026, "Amortized Bayesian
  Workflow", *Transactions on Machine Learning Research*,
  <https://openreview.net/forum?id=osV7adJlKD>, Section 2.2). Their workflow
  therefore checks the amortized results of each data set with diagnostics,
  and falls back to slower methods with stronger guarantees, importance
  sampling and then MCMC, where the diagnostics fail. IBS's estimates are
  unbiased, with a calibrated variance, on every data set, without training.
  The optimizer or the inference method that uses them still has errors of
  its own, which you check as for any fit, for instance by comparing runs.

**Its costs.** A trial takes about `1 / p` samples for an observed response
of probability `p` ([1], Section 4.2), so responses that the model finds
improbable are expensive, and the cost of an estimate grows exponentially as
the log-likelihood falls ([1], Appendix C.1). A lapse rate in the model
bounds the cost of each trial, and the
[likelihood threshold](#faq-what-does-the-likelihood-threshold-neg_logl_threshold-do-and-how-do-i-choose-it)
the cost at poor parameters ([1], Section 6.4 and Appendix C.1). The
responses must be discrete, or binned ([1], Section 6.3), and the simulator
must simulate each trial given its context: a model with latent dynamics that
can generate whole sequences of responses, but not one response conditioned
on a given sequence of earlier ones, does not qualify ([1], Section 2.2).

(faq-how-does-ibs-compare-with-approximate-bayesian-computation-abc-or-synthetic-likelihood)=
### How does IBS compare with approximate Bayesian computation (ABC) or synthetic likelihood?

These common approaches to likelihood-free inference compare summary
statistics of the data with summary statistics of simulated data ([1],
Section 1, which cites Beaumont et al., 2002, for ABC and Wood, 2010, for
synthetic likelihood). IBS estimates the likelihood of the full data, trial
by trial, without summary statistics; summary statistics need not be
sufficient, and so need not capture all aspects of the data ([1],
Sections 1 and 6.3). ABC also needs dedicated algorithms for estimation and
inference, whereas an IBS estimate is a noisy log-likelihood that any
likelihood-based method taking noisy targets can use ([1], Section 6.3).

A synthetic likelihood also gives a noisy estimate of a log-likelihood, that
of its summary statistics, and the PyVBMC FAQ notes that it could serve as
PyVBMC's target too
([Can I use *any* technique to estimate a noisy log-likelihood?](https://acerbilab.org/pyvbmc/faq.html#faq-can-i-use-any-technique-to-estimate-a-noisy-log-likelihood)).
For continuous responses, [1] proposes approximate IBS, which borrows ABC's
tolerance but compares the full responses (see
[below](#faq-can-i-use-ibs-with-continuous-responses)).

(faq-what-is-ibs_basic-for)=
### What is `ibs_basic` for?

Teaching. [`ibs_basic`](api/functions/ibs_basic.rst) is a bare-bone
implementation of IBS, after `ibs_basic.m` of MATLAB IBS: it samples one
trial at a time, one response per simulator call, and returns one estimate of
the log-likelihood (not its negative), without repeats, a variance estimate,
a cap or a likelihood threshold. Its loop is the algorithm of [1],
Section 4.1. Use `IBS` for real work.

(faq-installation)=
## Installing PyIBS

(faq-where-can-i-download-pyibs)=
### Where can I download PyIBS?

Install or upgrade PyIBS with pip:

```console
python -m pip install --upgrade pyibs
```

Or install with Conda:

```console
conda install --channel=conda-forge pyibs
```

PyIBS 1.5 requires NumPy 2.0 or newer. In an environment that holds NumPy
1.x, or a package that requires it, conda can install PyIBS 0.1.0 instead,
without a warning. Ask it for the latest release:

```console
conda install --channel=conda-forge "pyibs>=1.5"
```

Conda then upgrades NumPy, or says which package holds it back.

See the [installation instructions](installation.rst) for more details, and
the [GitHub repository](https://github.com/acerbilab/pyibs) for the source
code.

(faq-which-external-packages-does-pyibs-require)=
### Which external packages does PyIBS require?

PyIBS requires NumPy 2.0 and SciPy 1.13 or newer, and no other package;
`pip` and `conda` install them together with PyIBS. PyIBS does not require
MATLAB.

To fit models with PyIBS's estimates, install
[PyBADS](https://acerbilab.github.io/pybads/) (1.5.1 or newer) or
[PyVBMC](https://acerbilab.org/pyvbmc/) (1.5 or newer) as well: both
take PyIBS's output as it comes. To run the
[example notebooks](examples.rst) you also need Jupyter (see the
[installation instructions](installation.rst)).

(faq-which-version-of-python-do-i-need)=
### Which version of Python do I need?

PyIBS requires Python 3.10 or newer.

(faq-how-do-i-know-whether-a-newer-version-of-pyibs-exists)=
### How do I know whether a newer version of PyIBS exists?

Run `pyibs.check_for_updates()`. It asks PyPI for the latest release and
prints one message; if your release is older, the message gives the command
that updates it, `python -m pip install --upgrade pyibs`, or
`conda update --channel=conda-forge pyibs` for an installation by conda (the
conda-forge package can follow PyPI by a few days). It also returns the
installed version, the latest release and whether an update is available.

The function is PyIBS's only access to the network, made only when you call
it; PyIBS prints nothing of its own accord. `pyibs.__version__` gives the
installed version. The
[`check_for_updates` API](api/functions/check_for_updates.rst) has the
details.

(faq-how-do-i-check-that-pyibs-works-on-my-computer)=
### How do I check that PyIBS works on my computer?

Run its test suite, which is installed with the package. Install PyIBS with
its `test` extra, which adds pytest, then run the tests:

```console
python -m pip install "pyibs[test]"
python -m pytest --pyargs pyibs
```

When PyBADS 1.5.1 or newer, or PyVBMC 1.5 or newer, is installed, the suite
also fits the example model with each of them, which takes a few minutes;
`python -m pytest --pyargs pyibs -m "not integration"` leaves those fits out.

(faq-i-am-having-trouble-installing-pyibs-can-you-help)=
### I am having trouble installing PyIBS. Can you help?

Sure. The PyIBS installation should be pretty straightforward, so tell us in
detail which problem you are having in the
[Discussions forum](https://github.com/orgs/acerbilab/discussions). Include
your operating system, Python version, installation command and the full
error message.

(faq-the-simulator-and-the-data)=
## The simulator and the data

(faq-how-do-i-write-the-simulator)=
### How do I write the simulator?

`IBS` calls the simulator as `sample_from_model(theta, design_rows)`, or as
`sample_from_model(theta, design_rows, rng=rng)` when the simulator has a
parameter named `rng`, which then receives the generator of the `IBS` object
(see [How do I make a run reproducible?](#faq-how-do-i-make-a-run-reproducible)).
`theta` is the parameter vector, as given to the call. `design_rows` holds
the rows of `design_matrix` of the trials that `IBS` requests, or their
0-based indices when `design_matrix` is None. The simulator returns one
simulated response for each row:

- for `r` rows requested, an array of shape `(r,)` or `(r, 1)` when the
  responses have one column, and of shape `(r, C)` when they have `C`
  columns; a simulated response matches the observed one only when every
  column agrees;
- one independent draw per row: a call can request the same trial several
  times, so draw the randomness of each row separately, and never share a
  random number among rows;
- responses of the kind of the observed ones, since NumPy never finds
  numbers (or booleans), text and bytes equal to one another.

For example, an observer who reports whether a stimulus `S` lies to the
right (1) or to the left (-1) of a bias, from a noisy measurement, and
responds at random on a fraction `lapse` of the trials:

```python
def sample_from_model(theta, S, rng):
    log_sigma, bias, lapse = theta
    x = S + np.exp(log_sigma) * rng.standard_normal(S.shape)
    r = np.where(x >= bias, 1, -1)
    guess = rng.random(S.shape) < lapse
    r[guess] = rng.choice([-1, 1], size=np.count_nonzero(guess))
    return r


ibs = IBS(sample_from_model, R, S, random_seed=1)
neg_logl = ibs(theta)
```

The example model of the [notebooks](examples.rst),
`pyibs.examples.psycho_model.psycho_generator`, is this simulator; PyIBS
installs it together with its exact negative log-likelihood,
`psycho_neg_logl`. The [`IBS` API](api/classes/ibs.rst) describes the
simulator and every setting.

(faq-my-simulator-needs-additional-data-or-inputs-how-do-i-pass-them-to-ibs)=
### My simulator needs additional data or inputs. How do I pass them to `IBS`?

`IBS` passes the simulator only the parameter vector, the design rows and,
if it asks for it, the generator; it has no argument for further inputs,
such as the `varargin` of MATLAB `ibslike`. Bind them to the simulator
instead.
Suppose that your simulator takes them as `simulator(theta, design_rows, rng,
data)`. The first solution consists of defining a new function

```python
def sample_from_model(theta, design_rows, rng):
    return simulator(theta, design_rows, rng, data)
```

where `data` has been defined before in the code. Alternatively, you can use
`functools.partial`:

```python
from functools import partial

ibs = IBS(partial(simulator, data=data), R, S)
```

Either way, the function that `IBS` receives keeps the parameter `rng`, so
it still receives the generator.

(faq-the-responses-of-my-model-depend-both-on-the-current-trial-and-on-responses-or-stimuli-from-previous-trials-how-can-i-tell-this-to-ibs)=
### The responses of my model depend both on the current trial and on responses or stimuli from previous trials. How can I tell this to `IBS`?

IBS needs a simulator that draws the response of any trial given its
context, here the stimuli and the *observed* responses of the trials before
it: IBS estimates the probability of each observed response given what
actually happened before ([1], Section 2.2). If your simulator depends on
data that go beyond the current stimulus and the current response of a
trial, you should:

- leave `design_matrix` as None: `IBS` then calls the simulator with the
  0-based indices of the requested trials;
- give the simulator the full arrays of stimuli and responses, and any
  further data, by binding them to it (see the
  [previous question](#faq-my-simulator-needs-additional-data-or-inputs-how-do-i-pass-them-to-ibs));
- inside the simulator, for each requested trial index, generate a synthetic
  response from the information of that trial and of the trials before it.

For example, a model in which the observed response of the previous trial
pulls the decision towards itself:

```python
def make_simulator(S, R):
    def sample_from_model(theta, idx, rng):
        log_sigma, bias, pull = theta
        r_prev = np.where(idx > 0, R[idx - 1], 0)  # 0 before the first trial
        x = S[idx] + pull * r_prev
        x = x + np.exp(log_sigma) * rng.standard_normal(idx.size)
        return np.where(x >= bias, 1, -1)

    return sample_from_model


ibs = IBS(make_simulator(S, R), R, random_seed=1)  # no design: trial indices
```

When what the simulator needs of each trial fits in a row, it can also be
given as the design: here, with `r_prev` the array of the previous trial's
observed response of every trial,
`IBS(sample_from_model, R, np.column_stack([S, r_prev]))` and a simulator
that reads both columns of its design rows.

(faq-my-model-uses-data-structures-which-are-not-easily-converted-to-numerical-arrays-can-i-still-use-ibs)=
### My model uses data structures which are not easily converted to numerical arrays. Can I still use `IBS`?

Yes, absolutely.

- The *responses* of your model must be discrete, and `IBS` compares them
  with `==`: numbers, booleans, text (such as `"left"` and `"right"`) or
  bytes. Map any other kind of response to a finite set of such values, for
  instance a move in a board game to the index of its square. Responses that
  mix numbers and text are given as an object array (`dtype=object`), and
  the simulator returns them as one too.
- Any other data used to compute such responses need not be a numerical
  array. Bind them to the simulator and call it with trial indices, as
  explained in the
  [question above](#faq-the-responses-of-my-model-depend-both-on-the-current-trial-and-on-responses-or-stimuli-from-previous-trials-how-can-i-tell-this-to-ibs).
  Or give them as the design: `design_matrix` can be any array with one row
  per trial, an object array with one Python object per trial included, such
  as a board position, and the simulator then receives the objects of the
  requested trials. A list of Python objects, such as dictionaries, becomes
  such an array; for objects that NumPy reads as sequences of numbers, such
  as lists of different lengths, create the array with
  `np.empty(N, dtype=object)` and fill it element by element.

(faq-can-i-use-ibs-with-continuous-responses)=
### Can I use IBS with continuous responses?

Not as they are: a simulated continuous response almost never equals the
observed one, so IBS would never end ([1], Section 6.3). There are two ways
to make them discrete.

The simplest is to bin the responses: map the observed and the simulated
responses to bins with the same edges, for example

```python
edges = np.arange(-10, 10.5, 0.5)  # bins 0.5 wide
R = np.digitize(y, edges)  # y, the observed continuous responses


def sample_from_model(theta, S, rng):
    return np.digitize(simulate(theta, S, rng), edges)
```

where `simulate` draws continuous responses of the model.

The other is *approximate IBS* ([1], Section 6.3), inspired by approximate
Bayesian computation: a simulated response matches when it lies within a
tolerance `eps` of the observed one, and the log-likelihood of trial `i` is
estimated as the log of the probability of a match divided by the volume of
responses within `eps` of the observed one, `2 * eps` for one-dimensional
responses. With `IBS`, the simulator reports whether each sample matches,
and every observed "response" is `True`:

```python
eps = 0.25


def within_eps(theta, idx, rng):
    return np.abs(simulate(theta, S[idx], rng) - y[idx]) <= eps


ibs = IBS(within_eps, np.ones(len(y), dtype=bool))
neg_logl, sd = ibs(theta, num_reps=10, additional_output="std")
neg_logl += len(y) * np.log(2 * eps)  # the volume term
```

The volume term is the same at every parameter vector, so it moves neither
the optimum nor the posterior; it matters when the value is compared with a
log-likelihood computed otherwise.

While approximate IBS is a slightly better approach statistically, for most
problems it would not make a big difference if one simply bins the
responses. Approximate IBS, or binning, is roughly equivalent to adding
localized uniform noise to the responses of the model being fit, with radius
equal to `eps` (or to half the width of a bin). So, as a rule of thumb, you
want this added noise to be (much) less than the magnitude of the noise
present in the data. Narrower bins cost more samples: the expected number of
samples grows without bound as `eps` goes to 0 ([1], Section 6.3).

(faq-what-does-a-call-of-ibs-return)=
### What does a call of `IBS` return?

`ibs(theta)` returns an estimate of the *negative* log-likelihood of the data
at `theta`, as a Python float: the average of `num_reps` independent IBS
estimates, 10 by default. `additional_output` adds to it:

- `"var"`: the tuple `(neg_logl, neg_logl_var)`, with the variance estimate;
- `"std"`: the tuple `(neg_logl, neg_logl_std)`, with its square root, the
  form that PyBADS and PyVBMC take from a noisy target;
- `"full"`: an [`EstimateResult`](api/classes/estimate_result.rst), a
  dictionary whose entries you also read as attributes: `neg_logl`,
  `neg_logl_var`, `neg_logl_std`, `exit_flag` and its `message`,
  `elapsed_time`, `num_samples_per_trial`, `fun_count` (the simulator calls),
  and the per-trial estimates `neg_logl_trials` and `neg_logl_var_trials`.

`return_positive=True` returns the log-likelihood instead, the variance
estimate and the per-trial arrays unchanged. `trial_weights` weighs each
trial's log-likelihood.

The exit flag is 0 when every repeat was sampled to completion, and the
estimate is then unbiased (unless a finite
[`max_time`](#faq-what-does-max_time-do) is set); 1 when the
[likelihood threshold](#faq-what-does-the-likelihood-threshold-neg_logl_threshold-do-and-how-do-i-choose-it)
ended a repeat, which biases the estimate, and leaves the per-trial arrays
NaN; and 2 when `max_time` stopped the sampling, with a warning.

(faq-pybads-and-pyvbmc)=
## PyBADS and PyVBMC

(faq-how-do-i-use-pyibs-with-pybads)=
### How do I use PyIBS with PyBADS?

[PyBADS](https://acerbilab.github.io/pybads/), one of the lab's
[tools for fitting models to data](https://acerbilab.org/model-fitting/),
finds the maximum-likelihood or maximum-a-posteriori parameters. It
minimizes, so its target returns the *negative* log-likelihood, and its SD as
the second element of a tuple, with `options={"specify_target_noise": True}`:

```python
from pybads import BADS

ibs = IBS(sample_from_model, R, S, random_seed=1)


def target(theta):
    return ibs(theta, num_reps=100, additional_output="std")


bads = BADS(target, x0, lb, ub, plb, pub, options={"specify_target_noise": True})
optimize_result = bads.optimize()
```

- Return the tuple as `IBS` gives it. PyBADS takes only a Python `tuple` of
  two elements: a list or an array raises `ValueError`.
- For maximum-a-posteriori estimation, return
  `neg_logl - log_prior(theta)` with the same SD, since the log prior adds no
  noise.
- PyBADS calls the target with one parameter vector, one-dimensional, in the
  coordinates of your bounds.
- Choose `num_reps` for an SD of about 1 near the optimum (see
  [How do I choose `num_reps`?](#faq-how-do-i-choose-num_reps)), and
  re-evaluate the solution with more repeats
  ([below](#faq-once-the-optimizer-has-found-the-best-parameters-should-i-evaluate-the-log-likelihood-there-with-more-repeats)).
- Run PyBADS from several starting points, as the
  [PyBADS FAQ](https://acerbilab.github.io/pybads/faq.html#faq-how-do-i-run-pybads-from-several-starting-points)
  explains.

(faq-how-do-i-use-pyibs-with-pyvbmc)=
### How do I use PyIBS with PyVBMC?

[PyVBMC](https://acerbilab.org/pyvbmc/), one of the lab's
[tools for fitting models to data](https://acerbilab.org/model-fitting/),
computes an approximate posterior over the parameters and an estimate of the
model evidence. Its target is a log density, so it returns the log-likelihood
(`return_positive=True`), and its SD as the second element of the pair, with
`options={"specify_target_noise": True}`. Given a prior, PyVBMC adds the log
prior itself:

```python
from pyvbmc import VBMC

ibs = IBS(sample_from_model, R, S, random_seed=1)


def log_likelihood(theta):
    return ibs(theta, num_reps=100, additional_output="std", return_positive=True)


vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, prior=prior, options={"specify_target_noise": True})
vp, results = vbmc.optimize()
```

Here `prior` is a prior of PyVBMC's, such as
`pyvbmc.priors.Trapezoidal(lb, plb, pub, ub)`; `log_prior=` takes a function
instead. Without either, the target returns the log joint,
`log_likelihood + log_prior(theta)`, with the same SD. PyVBMC calls the target
with one parameter vector, one-dimensional, in the coordinates of your
bounds. `results["elbo"]`, a lower bound on the log model evidence, is
PyVBMC's estimate of it. PyVBMC handles
an SD of about 1 best, "and probably not larger than ~3" where the posterior
has most of its mass (the PyVBMC FAQ,
[How large can the noise be?](https://acerbilab.org/pyvbmc/faq.html#faq-how-large-can-the-noise-be-to-perform-successful-inferences-with-vbmc)).

(faq-why-does-the-target-return-the-sd-of-the-estimate-and-not-its-variance)=
### Why does the target return the SD of the estimate, and not its variance?

Because PyBADS and PyVBMC read the second element of the pair as the
standard deviation (SD) of the value. `additional_output="std"` returns it;
`"var"` returns the variance, which they would take as an SD without an
error, reading a variance of 4 (an SD of 2) as an SD of 4, for instance. MATLAB
`ibslike` returns the variance unless `ReturnStd` is set, so check this in
code ported from MATLAB.

(faq-why-is-the-sd-of-the-estimate-zero-and-why-do-pybads-and-pyvbmc-refuse-it)=
### Why is the SD of the estimate zero, and why do PyBADS and PyVBMC refuse it?

**Why it is zero.** IBS estimates a trial's log-likelihood from the number of
samples `K` that the trial took to match, and estimates its variance as
`ψ₁(1) − ψ₁(K)`, where `ψ₁` is the trigamma function ([1], Sections 2.4 and
4.3). A trial that matched at its first sample, `K = 1`, has the estimate 0
and the variance estimate 0. When every trial of positive weight matches at
its first sample in every repeat of a call, the call returns 0 for the
negative log-likelihood and 0 for its variance. That is the correct output of
the estimator, and `IBS` returns it as it is, with a `UserWarning` that links
this answer:

```text
The IBS variance estimate is 0, as it is when every trial of positive weight matched its response at its first sample. PyBADS and PyVBMC refuse an SD of 0 for a noisy target: see https://acerbilab.github.io/pyibs/faq.html#faq-why-is-the-sd-of-the-estimate-zero-and-why-do-pybads-and-pyvbmc-refuse-it
```

It happens where the model predicts the observed responses with probability
near 1. On 100 trials that the model matches each with probability 0.999, a
call returns a variance of 0 with probability 0.37 at `num_reps=10`, and
0.000045 at `num_reps=100`, and the calls of the
[validation of PyIBS](https://github.com/acerbilab/pyibs/blob/main/dev/results/2026-10-09-validation.md)
on such a model agree.

**Why PyBADS and PyVBMC refuse it.** They take the SD as the noise of that
evaluation in their Gaussian-process models of the target, and an SD of 0
would declare the value exact. An IBS estimate is exact only when every trial
matches with probability exactly 1; otherwise counts that are all 1 are a
likely outcome when the true SD of the estimate is small, and no sign that it
is 0. Both raise a
`ValueError` on an SD that is not finite and strictly positive: PyBADS's
message begins `The target function returned the noise SD 0.0`, and PyVBMC's
says that the returned estimated SD `must be a finite, positive real-valued
scalar`.

**How small the true SD is.** With unit trial weights, a call returns a
variance of 0 with probability at most `exp(-(num_reps * sd) ** 2)`, where
`sd` is the true SD of its estimate: 0.37 at `sd = 1 / num_reps`, 0.018 at
`2 / num_reps` and 0.00012 at `3 / num_reps`. (The probability is the product
of the trials' `p ** num_reps`, the true variance is the sum of the trials'
`Li₂(1 − p)` divided by `num_reps` ([1], Section 4.3), and
`Li₂(1 − p) ≤ −log p`.) A variance of 0 is thus unlikely unless the true SD
is below about `2 / num_reps`, 0.2 or less at 10 repeats or more: far below
the SD of about 1 that PyBADS and PyVBMC need, so that the estimate there is
precise enough.

**What to do.**

- *Put a floor on the SD in your target.* In the function that you give to
  PyBADS or PyVBMC, replace an SD of 0 by a small positive value:

  ```python
  num_reps = 100


  def target(theta):
      neg_logl, sd = ibs(theta, num_reps=num_reps, additional_output="std")
      return neg_logl, max(sd, 1 / num_reps)
  ```

  With unit trial weights, `1 / num_reps` is the smallest SD that a call
  returns when its variance estimate is not 0, that of a call in which one
  trial needed a second sample in one repeat (`ψ₁(1) − ψ₁(2) = 1`). The floor
  thus changes no other SD, and it is of the size that a variance of 0 points
  to. With trial weights, the smallest positive weight divided by `num_reps`
  plays that role.

  The floor is your choice, for the optimizer's sake, and not an estimate: it
  states a precision that the call did not measure, and PyIBS applies none
  for that reason. At this size it costs little: the optimizer takes such an
  evaluation as about as precise as the most precise evaluation it can
  otherwise receive, and where the true SD is far smaller, it only trusts the
  value less than it could. A much smaller floor, such as `1e-8`, costs
  more: PyBADS and PyVBMC then take the value as nearly exact, and PyBADS,
  which weights its final evaluations at the solution by the precision that
  the target reports, can let that one evaluation decide its result (the
  PyBADS FAQ, [How is `fval` computed?](https://acerbilab.github.io/pybads/faq.html#faq-how-is-fval-computed)).
  Once the floor is in place, the warning has done its job, and
  `warnings.filterwarnings("ignore", message="The IBS variance estimate is 0")`
  silences it.
- *More repeats* make a zero less likely, exponentially so in `num_reps`
  (0.37 at 10 repeats and 0.000045 at 100, in the example above), but
  never impossible, and a run of PyBADS or PyVBMC makes a hundred
  evaluations or more. Since the estimate is precise where zeros occur, more repeats
  pay for themselves only when the SD elsewhere calls for them (see
  [How do I choose `num_reps`?](#faq-how-do-i-choose-num_reps)).
- *A lapse rate in the model* bounds every trial's probability below 1: with
  a lapse rate of at least `lapse` among `k` equally likely responses, each
  `p ≤ 1 − lapse * (k − 1) / k`, and a call over `N` trials returns a zero
  with probability at most `(1 − lapse * (k − 1) / k) ** (N * num_reps)`,
  below `1e-13` for 600 binary trials, a lapse rate of 0.01 and 10 repeats.
  [1] recommends a lapse rate for IBS in any case (Section 6.4).

(faq-do-i-need-to-give-pybads-or-pyvbmc-the-sd-of-every-estimate)=
### Do I need to give PyBADS or PyVBMC the SD of every estimate?

It depends:

- If you are *optimizing* the log-likelihood (e.g., for maximum-likelihood or
  maximum-a-posteriori estimation) with PyBADS, it helps but it is not
  necessary, because the variance of the IBS estimate, somewhat surprisingly,
  changes little across the parameter space (see
  [How do I choose `num_reps`?](#faq-how-do-i-choose-num_reps)). Without it,
  `options={"uncertainty_handling": True}` tells PyBADS that the target is
  noisy, and PyBADS estimates the noise itself. Since PyIBS gives the SD at
  no extra cost, give it: the PyBADS FAQ says so too
  ([Should I provide an estimate of the noise associated with each evaluation?](https://acerbilab.github.io/pybads/faq.html#faq-should-i-provide-an-estimate-of-the-noise-associated-with-each-evaluation)).
- If you are performing *Bayesian inference* with PyVBMC, it is *necessary*
  to give it. Bayesian inference is very sensitive to noisy estimates of the
  log-likelihood (or of the log posterior), so it is crucial to provide the
  inference algorithm with all available information about the magnitude of
  the noise.

(faq-i-have-several-questions-about-using-pybads-to-optimize-the-log-likelihood-can-you-help)=
### I have several questions about using PyBADS to optimize the log-likelihood. Can you help?

[PyBADS](https://acerbilab.github.io/pybads/), one of the lab's
[tools for fitting models to data](https://acerbilab.org/model-fitting/), is
a robust optimizer that works well with stochastic target functions, and in
particular with the noisy estimates produced by IBS. Many questions on its
use are answered in the [PyBADS FAQ](https://acerbilab.github.io/pybads/faq.html).
In particular, you might want to start with its section on
[noisy objective functions](https://acerbilab.github.io/pybads/faq.html#faq-noisy-objective-function)
(but do not stop there: all sections of the FAQ are relevant).

(faq-what-if-i-want-to-use-ibs-to-perform-bayesian-posterior-or-model-inference)=
### What if I want to use IBS to perform Bayesian posterior or model inference?

If you are interested in Bayesian inference, that is in the posterior
distribution of the model parameters or in the marginal likelihood (model
evidence), we recommend [Variational Bayesian Monte Carlo (PyVBMC)](https://acerbilab.org/pyvbmc/),
one of the lab's [tools for fitting models to data](https://acerbilab.org/model-fitting/),
which supports noisy estimates of the log-likelihood such as those produced by
IBS (see its FAQ on [noisy target functions](https://acerbilab.org/pyvbmc/faq.html#faq-noisy-target-function)).
In a large empirical benchmark, VBMC has been shown to work very well in
combination with IBS ([Acerbi, 2020](https://arxiv.org/abs/2006.08655)).
[How do I use PyIBS with PyVBMC?](#faq-how-do-i-use-pyibs-with-pyvbmc) gives
the settings.

(faq-precision-and-ibs-repeats)=
## Precision and IBS repeats

IBS affords a simple way to change the precision of its estimates, by using
multiple "repeats", each amounting to an independent run of the estimator
([1], Section 4.4). In PyIBS, a call averages `num_reps` repeats, 10 by
default. In this section, we answer questions on the precision of the IBS
estimate, related to the number of repeats.

(faq-how-do-i-choose-num_reps)=
### How do I choose `num_reps`?

Aim for an SD of the estimate of about 1 near the optimum, the noise that
PyBADS and PyVBMC handle best: "a standard deviation of order 1 or less should
work" for PyBADS (the PyBADS FAQ,
[Can PyBADS handle any arbitrary amount of noise in the objective?](https://acerbilab.github.io/pybads/faq.html#faq-can-pybads-handle-any-arbitrary-amount-of-noise-in-the-objective)),
and for PyVBMC "ideally you want the SD of the noise in the log-likelihood to
be around 1, and probably not larger than ~3" (the PyVBMC FAQ,
[How large can the noise be?](https://acerbilab.org/pyvbmc/faq.html#faq-how-large-can-the-noise-be-to-perform-successful-inferences-with-vbmc)).

The variance of the estimate falls as `1 / num_reps`, so the SD falls as
`1 / sqrt(num_reps)`, while the cost grows linearly with `num_reps`. One call
at a parameter vector near where you expect the optimum, such as the result
of a first, rough fit, tells you how many repeats you need:

```python
_, sd = ibs(theta, num_reps=10, additional_output="std")
num_reps = int(np.ceil(10 * sd**2))  # for an SD of about 1
```

The SD of a single call is itself an estimate; a call with more repeats gives
a steadier guide.

The SD changes little across the parameter space, so one choice serves the
whole fit. On the 600 trials of the example model, for instance, 100 repeats
give an SD of 1.1 at the parameters that generated the data, and 1.2 and 1.4
at two other parameter vectors, whose negative log-likelihoods are higher by
49 and 55 (the [validation of PyIBS](https://github.com/acerbilab/pyibs/blob/main/dev/results/2026-10-09-validation.md)).
Farther from the optimum the SD can be larger, which is usually fine (the
PyBADS FAQ says so of noise in general), and it is bounded: each trial
adds at most `π²/6 ≈ 1.64` to the variance of one repeat ([1],
Section 4.3), so with unit trial weights the SD is at most
`sqrt(1.64 * N / num_reps)` for `N` trials.

If you cannot bring the SD down to about 1 within your computational budget,
be as precise as you can afford.

(faq-once-the-optimizer-has-found-the-best-parameters-should-i-evaluate-the-log-likelihood-there-with-more-repeats)=
### Once the optimizer has found the best parameters, should I evaluate the log-likelihood there with more repeats?

Yes, absolutely. It should be considered standard practice, regardless of
IBS. Whenever optimizing a noisy target function, after obtaining a candidate
solution from an optimization method, one should evaluate the target function
at the solution with higher precision. The value that the optimizer reports
at its solution is biased, since the solution was chosen for its low value
among the points evaluated ([1], Section 4.1). With PyBADS:

```python
x_best = optimize_result["x"]
neg_logl, sd = ibs(x_best, num_reps=1000, additional_output="std")
```

Use an `IBS` object without a
[likelihood threshold](#faq-what-does-the-likelihood-threshold-neg_logl_threshold-do-and-how-do-i-choose-it)
for this estimate, and use this estimate, with its SD, to compare models.

(faq-in-an-ideal-world-would-you-let-the-number-of-repeats-depend-on-how-close-the-optimization-algorithm-thinks-it-is-to-the-maximum)=
### In an ideal world, would you let the number of repeats depend on how close the optimization algorithm thinks it is to the maximum?

Yes, this form of adaptive precision is a good idea and a topic of research.
`num_reps` is an argument of each call, so a target can change it from call
to call, as long as it returns the SD of each estimate; but PyBADS and PyVBMC
call the target with the parameters alone and do not choose the precision of
their evaluations.

(faq-as-i-increase-num_reps-the-sd-of-the-estimate-goes-down-slowly-but-the-computational-time-increases-linearly-is-this-normal)=
### As I increase `num_reps`, the SD of the estimate goes down slowly, but the computational time increases linearly. Is this normal?

Well, think about it. `num_reps` is literally the number of times the IBS
estimator is run, so the number of samples, and with it the computational
time, has to be linear in `num_reps`. On the other hand, `num_reps` is the
number of independent estimates you are averaging over, and the standard
error of the mean decreases with the *square root* of the number of
independent estimates: to halve the SD, you need four times as many repeats.

The number of simulator *calls* grows much more slowly: with
`vectorized=True`, a call of `IBS` asks the simulator for more samples per
call as it goes, and in the validation of PyIBS the example model took 9 to
12 simulator calls per estimate at both 10 and 100 repeats. A simulator whose
time is mostly a fixed cost per call thus takes little more time for more
repeats (see [Should I set `vectorized`?](#faq-should-i-set-vectorized)).

(faq-what-are-trial-dependent-repeats-and-can-i-use-them-with-pyibs)=
### What are trial-dependent repeats, and can I use them with PyIBS?

With trial-dependent repeats, each trial gets its own number of repeats,
chosen to minimize the variance of the estimate for a given budget of
samples ([1], Appendix C.2). The estimate stays unbiased whatever the
allocation, as long as every trial gets at least one repeat. The optimal
allocation favours the trials whose probability is near 1/2, and spends
little on those near 0, which cost many samples, and near 1, whose estimates
are already precise. It depends on the trials' probabilities, which are
unknown, so [1] recommends computing it once, from an estimate with many
repeats at a representative parameter vector, and keeping it for the whole
fit.

The main assumption for this to work *well* is roughly that the trial
likelihoods are correlated across (reasonable) regions of parameter space,
so that an allocation computed at one parameter vector remains beneficial at
the others that the optimization or inference algorithm evaluates. This
seems to hold often in practice, since the improbable trials are often those
where something unexpected occurred, such as a lapse; however, more empirical
studies are needed. For trials whose probabilities are uniform between 0 and
1, the optimal allocation gives a median gain in precision of about 1.6 over
equal repeats, for the same budget, on 500 trials ([1], Appendix C.2).

PyIBS does not offer it directly: a call takes one `num_reps` for every
trial. You can build it from groups of trials that share a number of repeats,
one `IBS` object per group, since the groups' estimates are independent and
their sums, and the sums of their variances, are the estimate and the
variance of the whole data set:

```python
# reps: the number of repeats of each trial, from the allocation of [1]
groups = [(np.flatnonzero(reps == k), k) for k in np.unique(reps)]
ibs_groups = [(IBS(sample_from_model, R[g], S[g]), k) for g, k in groups]


def target(theta):
    neg_logl, var = 0.0, 0.0
    for ibs_g, k in ibs_groups:
        value_g, var_g = ibs_g(theta, num_reps=k, additional_output="var")
        neg_logl += value_g
        var += var_g
    return neg_logl, float(np.sqrt(var))
```

A simulator that receives trial indices gets those of its group, from 0;
give it the original indices as the design, `IBS(sample_from_model, R[g], g)`.
A group of trials that nearly always match can return a variance of 0, with
[its warning](#faq-why-is-the-sd-of-the-estimate-zero-and-why-do-pybads-and-pyvbmc-refuse-it),
which is harmless as long as the total is positive.

(faq-cost-limits-and-the-likelihood-threshold)=
## Cost, limits and the likelihood threshold

(faq-how-many-samples-does-an-estimate-take)=
### How many samples does an estimate take?

A trial whose observed response the model produces with probability `p`
takes on average `1 / p` samples per repeat ([1], Section 4.2), so a call
takes about `num_reps * sum(1 / p_i)` samples over the trials. With
`vectorized=True`, a call also draws some samples of a trial after its last
match, which it discards: in the validation of PyIBS, 1.01 to 1.56 times as
many samples in all. `additional_output="full"` reports the cost of a call:
`num_samples_per_trial`, the simulated responses per trial, `fun_count`, the
simulator calls, and `elapsed_time`.

Since `1 / p = exp(-log p)`, the cost grows exponentially as the
log-likelihood falls, so poor parameter vectors are expensive ([1],
Appendix C.1). The
[likelihood threshold](#faq-what-does-the-likelihood-threshold-neg_logl_threshold-do-and-how-do-i-choose-it)
bounds that cost, and a lapse rate in the model bounds `1 / p` on every
trial ([1], Section 6.4). The cap, `max_iter`, stops a call whose simulator
cannot produce an observed response (see
[A call raises `IBSSamplingError`](#faq-a-call-raises-ibssamplingerror-what-do-i-do)).

(faq-should-i-set-vectorized)=
### Should I set `vectorized`?

Often yes. `vectorized` sets how `IBS` calls the simulator:

- `True` asks, in each call, for several samples of every trial that still
  needs a match, a number that grows by `acceleration` (1.5) from call to
  call: few calls, at the price of some samples that a trial draws after its
  last match;
- `False` asks for one sample of every trial that still needs one: no
  surplus, but a call for each sample of the slowest trial;
- `None`, the default, decides once per `IBS` object, at its first call with
  `num_reps > 1`, by timing one simulation of all trials: `False` if it takes
  `vectorized_threshold` (0.1 s) or more, `True` otherwise. The decision is
  kept for the object's later calls, and `ibs.vectorized` reads it. A call
  with `num_reps=1` always takes one sample per trial and call.

The rule of `None` fits a simulator whose time grows with the number of
responses it simulates: there `False` is the right choice, since the
accelerated schedule simulated 1.2 to 1.5 times as many responses per trial in
the validation of PyIBS. It does not fit a simulator whose time is mostly a
fixed cost per call, such as one that starts a process, loads a model or
moves data to a GPU at every call: there a simulation of all trials takes
long, `None` decides `False`, and the many calls are slow. In the
[validation of PyIBS](https://github.com/acerbilab/pyibs/blob/main/dev/results/2026-10-09-validation.md),
on 100 to 10,000 trials of the example model with 0.2 s per call, an
estimate of 100 repeats took 15.5 to 27 minutes with `False`, and about 2 s
with `vectorized=True`, for its 9 to 12 calls.

So give `vectorized=True` for a simulator dominated by a fixed cost per call,
and `vectorized=False` for one whose time is proportional to the responses
it simulates and that takes 0.1 s or more for all trials. A simulator that is
slow only at its first call, such as one compiled just in time, can make the
decision `False` for good: give it `True`, or call it once before the
object's first call. Giving `vectorized` also keeps the timing out of the
sampling (see [How do I make a run reproducible?](#faq-how-do-i-make-a-run-reproducible)).

(faq-what-does-the-likelihood-threshold-neg_logl_threshold-do-and-how-do-i-choose-it)=
### What does the likelihood threshold `neg_logl_threshold` do, and how do I choose it?

It saves the samples of poor parameter vectors. With a threshold `T`, a
repeat is ended as soon as its sampling shows that its negative
log-likelihood exceeds `T`, and counts as exactly `T`; the call then has exit
flag 1 ([1], Appendix C.1). `T` applies to the weighted negative
log-likelihood, the scale of the returned estimate.

The usual choice is the chance level, the negative log-likelihood of
responding at random: `sum_i w_i log(k_i)` for `k_i` possible responses and
weight `w_i` on trial `i`, or `N log(2)` for `N` binary choices of weight 1.
For example:

```python
ibs_fit = IBS(sample_from_model, R, S, neg_logl_threshold=len(R) * np.log(2))
```

The threshold has a price, where it acts:

- it biases the estimate: the log-likelihood upwards, and the negative
  log-likelihood downwards. The bias is negligible at parameter vectors whose
  log-likelihood lies well above `-T` ([1], Appendix C.1), and the threshold
  is meant for those that lie below it, which an optimizer only needs to know
  are poor. In the validation of PyIBS, on models whose exact negative
  log-likelihood lay at or near the chance level, the expected estimate lay
  3.0 to 6.6 below it;
- its variance estimate overstates the variance, about 2 to 3 times in the
  validation, since the variance estimate of an ended repeat describes the
  counts it had when it ended;
- the per-trial arrays of `additional_output="full"` are NaN when the
  threshold ended a repeat: an ended repeat is valued as a whole.

So use the threshold while optimizing, and not for the final estimate:
evaluate the solution with an `IBS` object that has no threshold, the
default `neg_logl_threshold=np.inf`
([see above](#faq-once-the-optimizer-has-found-the-best-parameters-should-i-evaluate-the-log-likelihood-there-with-more-repeats)).
When the number of possible responses is hard to count for each trial, an
average serves: for its four-in-a-row game, whose number of possible moves
depends on the board, [1] took `N log(20)` (Appendix C.1).

(faq-is-it-okay-to-stop-the-ibs-algorithm-for-one-trial-after-a-fixed-number-of-samples-eg-20)=
### Is it okay to stop the IBS algorithm for one trial after a fixed number of samples (e.g., 20)?

No, this is not okay, in the sense that by doing it one would essentially
revert IBS to a fixed-sampling method, with all the associated problems
discussed in the paper ([1], Section 3). A more principled way is to put an
early-stopping threshold on the log-likelihood of the whole data set, as
described in the paper ([1], Appendix C.1), which `IBS` implements as its
[likelihood threshold](#faq-what-does-the-likelihood-threshold-neg_logl_threshold-do-and-how-do-i-choose-it).
For the same reason, the cap `max_iter` of `IBS` raises an error rather than
return an estimate from truncated counts (see
[A call raises `IBSSamplingError`](#faq-a-call-raises-ibssamplingerror-what-do-i-do)).

(faq-what-does-max_time-do)=
### What does `max_time` do?

It sets a time limit on each call, in seconds, checked after every simulator
call. Once the limit is reached, the sampling stops, and each trial's value
averages the repeats it completed; the call has exit flag 2 and issues a
warning, and raises `IBSSamplingError` for a trial without a completed
repeat. A finite limit biases the estimates, also those of the calls that
complete in time, since completing in time favours few samples, which give
high log-likelihoods. Leave it at its default, `np.inf`, for estimates that
you will use; the likelihood threshold and the cap bound the cost without
that bias. A finite limit also makes the sampling depend on timing, so that
a seed no longer reproduces a run.

(faq-troubleshooting)=
## Troubleshooting

(faq-a-call-raises-ibssamplingerror-what-do-i-do)=
### A call raises `IBSSamplingError`. What do I do?

[`IBSSamplingError`](api/classes/ibs_sampling_error.rst) means that a trial
drew more samples than the cap allows, `max_iter * num_reps` in a call of
`num_reps` repeats (`max_iter` is 10**5 by default), and `IBS` returns no
estimate from counts that the cap cut short. The message names the trials,
by their 0-based indices, and their samples:

```text
In a draw of 10 repeats, trial 1 drew 1135 samples, more than max_iter * num_reps = 1000. 1 of 3 trials still need matches, and IBS returns no estimate for an incomplete repeat. Check that the simulator can produce every observed response at this parameter vector, or raise max_iter.
```

The usual cause is an observed response that the simulator never, or almost
never, produces at that parameter vector:

- check that the simulator codes the responses as the data do, such as 1 and
  -1 rather than 1 and 0, and returns them in the order of its design rows;
- add a lapse rate to the model, with a small positive lower bound (e.g.,
  0.005), so that every response has a positive probability ([1],
  Section 6.4);
- set a [likelihood threshold](#faq-what-does-the-likelihood-threshold-neg_logl_threshold-do-and-how-do-i-choose-it),
  which ends the repeats of a poor parameter vector before the cap.

Raise `max_iter` only when the responses are truly that improbable, and the
cost of about `1 / p` samples each is acceptable. `IBSSamplingError` is also
raised when [`max_time`](#faq-what-does-max_time-do) stops a call before a
trial has completed a repeat.

(faq-ibs-calls-my-simulator-with-different-subsets-of-trials-what-does-this-mean-for-my-simulator)=
### `IBS` calls my simulator with different subsets of trials. What does this mean for my simulator?

`IBS` calls the simulator multiple times with different subsets of trials,
that is with different rows of the design (or trial indices), possibly with
a trial repeated. This means that your simulator should work "row-wise", and
not depend on the order of the rows, nor on any information that depends on
the other rows. For example, assuming that `S[:, 0]` contains for each trial
the contrast of the stimulus presented in the trial, be very careful that
`np.min(design_rows[:, 0])` inside the simulator will *not* be the minimum
contrast across all trials. Instead, it will be the minimum contrast of the
trials that IBS requests in that call, which changes from call to call (likely
*not* what your model needs). If your simulator needs global information,
give it separately, by binding it to the simulator (see
[this question](#faq-my-simulator-needs-additional-data-or-inputs-how-do-i-pass-them-to-ibs)).

Likewise, each row must be an independent draw: a simulator that draws one
random number per call and uses it for all the rows of that call makes the
samples of a call dependent, which breaks the assumptions of IBS, so that its
estimate or its variance estimate can be wrong.

(faq-a-call-raises-a-valueerror-or-a-typeerror-about-the-simulators-output-what-is-wrong)=
### A call raises a `ValueError` or a `TypeError` about the simulator's output. What is wrong?

- `ValueError: The simulator was asked for ... and returned an array of shape
  ...`: the simulator must return one response per requested row, of shape
  `(r,)` or `(r, 1)` for `r` rows when the responses have one column, and
  `(r, C)` when they have `C` columns. A common cause is a simulator that
  returns one response per trial of the data set rather than per requested
  row, or that ignores the requested rows.
- `TypeError: The simulator returned responses of dtype ..., which NumPy
  never finds equal to the ... among the observed responses`: the simulator
  returns text where the observed responses are numbers, or the reverse, so
  no sample could ever match. Return responses of the observed kind; for
  responses that mix numbers and text, use object arrays (`dtype=object`) for
  both.

When `IBS` is created, `ValueError: response_matrix holds a NaN` names the
trials whose observed response is NaN, which no simulated response equals:
recode such a response as a value that the simulator returns, or remove the
trial.

(faq-how-do-i-make-a-run-reproducible)=
### How do I make a run reproducible?

Pass a seed when you create the `IBS` object, and have the simulator draw
from the generator it receives:

```python
def sample_from_model(theta, S, rng):
    ...  # every random draw from rng


ibs = IBS(sample_from_model, R, S, vectorized=True, random_seed=1)
```

`random_seed` works as PyBADS's does: an integer or a
`numpy.random.SeedSequence` seeds a new generator, a `numpy.random.Generator`
is used as given, and None, the default, derives the generator from NumPy's
global random state, so that `np.random.seed` before creating the object also
fixes it. The object keeps the generator as `ibs.rng`, and passes it to a
simulator that has a parameter named `rng`.

Two objects created with the same seed then give the same estimates from the
same sequence of calls, provided that no timing decides the sampling:

- `vectorized` is given as `True` or `False`, or `None` decides alike in both
  (it decides by timing the simulator, see
  [Should I set `vectorized`?](#faq-should-i-set-vectorized));
- `acceleration_threshold` and `max_time` keep their defaults, `None` and
  `np.inf`, under which the sampling does not depend on how long the
  simulator calls take.

A simulator that draws from NumPy's global functions, such as
`np.random.normal`, escapes the seed of `IBS`. On another computer, or with
other versions of Python, NumPy or SciPy, the same seed can give different
results.

With PyBADS or PyVBMC, seed both, each with a seed of its own:

```python
ibs = IBS(sample_from_model, R, S, vectorized=True, random_seed=1)
bads = BADS(target, x0, lb, ub, plb, pub, options={"specify_target_noise": True, "random_seed": 2})
vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, prior=prior, options={"specify_target_noise": True}, seed=2)
```

Seed the `IBS` object once, for the whole run. Creating a new `IBS` object
with the same seed at every evaluation would *freeze* the noise rather than
remove it, which can bias a fit (the PyBADS FAQ,
[Can I make a noisy objective function deterministic by fixing the noise process?](https://acerbilab.github.io/pybads/faq.html#faq-can-i-make-a-noisy-objective-function-deterministic-by-fixing-the-noise-process)).

(faq-how-do-i-check-that-the-estimates-are-right-for-my-model)=
### How do I check that the estimates are right for my model?

PyIBS's own estimates are validated against exact log-likelihoods across
models and settings (the [validation of PyIBS](https://github.com/acerbilab/pyibs/blob/main/dev/results/2026-10-09-validation.md));
what remains to check is your simulator. [1] recommends comparing IBS
estimates with a log-likelihood computed otherwise, on well-chosen test
trials and parameters (Section 6.4): where your model, or a simplified
version of it, has a closed-form or numerical likelihood, compare the two.
If the estimates are
unbiased and their variance estimates calibrated, the z-scores
`(neg_logl - exact) / sd` of many calls, of 10 repeats or more each, have a
mean close to 0 and an SD close to 1. The example model,
`pyibs.examples.psycho_model`, has both a simulator and its exact negative
log-likelihood, `psycho_neg_logl`, for such comparisons, and the
[first example notebook](examples.rst) makes one.

Then check the fit as for any fit: compare several runs of PyBADS or PyVBMC
from different starting points.

(faq-miscellanea)=
## Miscellanea

(faq-i-used-ibslike-in-matlab-what-is-different-in-pyibs)=
### I used `ibslike` in MATLAB. What is different in PyIBS?

PyIBS implements IBS as `ibslike.m` 0.96 of
[MATLAB IBS](https://github.com/acerbilab/ibs) does, with a Python interface:

- You create an `IBS` object with the simulator, the responses and the design,
  `ibs = IBS(sample_from_model, response_matrix, design_matrix, ...)`, and call
  it, `ibs(params, num_reps, trial_weights, additional_output, return_positive)`,
  where MATLAB calls `ibslike(fun, params, respMat, designMat, options,
  varargin)`. Extra arguments of the simulator are
  [bound to it](#faq-my-simulator-needs-additional-data-or-inputs-how-do-i-pass-them-to-ibs)
  rather than passed as `varargin`.
- `Nreps`, `TrialWeights` and `ReturnPositive` are arguments of the call,
  `num_reps`, `trial_weights` and `return_positive`. `ReturnStd` is
  `additional_output="std"`, and `additional_output="full"` returns the
  exit flag and the output structure of `ibslike` together (`funcCount` is
  `fun_count`, `NsamplesPerTrial` is `num_samples_per_trial`, `nlogL_trials`
  and `nlogLvar_trials` are `neg_logl_trials` and `neg_logl_var_trials`).
- The other options are settings of the object, with snake_case names:
  `Vectorized` (`'auto'` is `None`), `Acceleration`, `NsamplesPerCall`
  (`num_samples_per_call`), `MaxIter`, `MaxTime` and `NegLogLikeThreshold`
  (`neg_logl_threshold`, `np.inf` for none). `ibslike`'s hard-coded
  `MaxSamples`, `AccelerationThreshold`, `VectorizedThreshold` and `MaxMem`
  are settings too, `max_samples`, `acceleration_threshold`,
  `vectorized_threshold` and `max_mem`.
- With no design, the simulator receives 0-based trial indices.
- The simulator can take the object's random generator, `rng`, for
  [reproducible runs](#faq-how-do-i-make-a-run-reproducible).

Some behaviours differ on purpose. Among them: `vectorized=None` decides once
per object; the samples per call grow after every call unless
`acceleration_threshold=0.1` restores `ibslike`'s time rule; the likelihood
threshold checks every repeat and values an ended repeat at exactly the
threshold, as in [1], Appendix C.1; the cap counts the samples of each trial
and raises `IBSSamplingError`; `max_time` averages each trial's completed
repeats, with a warning; and `ibslike('test')` is the test suite,
`pytest --pyargs pyibs`. The
[catalogue of deliberate differences](https://github.com/acerbilab/pyibs/blob/main/pyibs/README.md)
lists each difference between PyIBS and `ibslike.m`, and the didactic
`ibs_basic.m`, with its reason. Runs of PyIBS and of `ibslike` do not match
draw for draw, even with the same simulator: they draw their random numbers
differently.

(faq-i-used-pyibs-010-what-do-i-need-to-change)=
### I used PyIBS 0.1.0. What do I need to change?

Usually little: PyIBS 1.5 keeps the calling convention of 0.1.0, `IBS(...)`
and `ibs(params, num_reps, ...)` with the same names and outputs. Its results
differ from 0.1.0's, also after `np.random.seed`: PyIBS 1.5 is rebuilt on a
new sampler, and fixes defects of 0.1.0, among them its default `max_iter`,
15, which biased the estimates of improbable responses. PyIBS 1.5 checks its settings and raises
for values that 0.1.0 accepted, raises `IBSSamplingError` where 0.1.0
returned exit flag 3, and moves the example model to
`pyibs.examples.psycho_model`. The list "Upgrading from 0.1.0" of the
[changelog](https://github.com/acerbilab/pyibs/blob/main/CHANGELOG.md) says
what to check in an existing script.
