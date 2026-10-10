# PyIBS: Frequently Asked Questions

This FAQ covers PyIBS 1.5 and is curated by
[Luigi Acerbi](https://lacerbi.github.io/). It adapts the
[MATLAB IBS FAQ](https://github.com/acerbilab/ibs/wiki) and adds questions
about the Python package.

For a tutorial with detailed examples, see the [Jupyter notebook examples](examples.rst).

For questions not covered here, ask in the lab's
[Discussions forum](https://github.com/orgs/acerbilab/discussions).

PyIBS estimates log-likelihoods from a model's simulator. PyBADS can use
these estimates to fit the parameters, and PyVBMC can use them to infer
the posterior and model evidence. They are among the lab's
[tools for fitting models to data](https://acerbilab.org/model-fitting/).

**Acknowledgments:** Many of the questions answered here originated in a
live Q&A session with the [Ma lab](http://www.cns.nyu.edu/malab/). We thank
Hsin-Hung Li for taking notes.

In the answers, [1] is the IBS paper: B. van Opheusden, L. Acerbi and
W. J. Ma (2020), "Unbiased and efficient log-likelihood estimation with
inverse binomial sampling", *PLOS Computational Biology* 16(12): e1008483,
<https://doi.org/10.1371/journal.pcbi.1008483>.

The snippets below use `np` for NumPy and `IBS` for the estimator class:

```python
import numpy as np
from pyibs import IBS
```

The snippets use `sample_from_model` for your simulator, `R` for the
observed responses, `S` for the design, and `theta` for a parameter vector.
Both `R` and `S` have one row per trial; they are the `response_matrix`
and `design_matrix` arguments of `IBS`. Define these inputs for your model
before running a snippet.

The fitting snippets also use a starting point `x0`, hard bounds `lb` and
`ub`, and plausible bounds `plb` and `pub`. Each is a one-dimensional NumPy
array with one element per parameter. The PyVBMC snippet additionally
requires a prior. The [example notebooks](examples.rst) provide complete
worked setups.

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

Use PyIBS when you can simulate your model but cannot evaluate its
likelihood. The problem must meet these requirements ([1], Section 2.2):

- the responses are discrete (choices, categories, counts), or you define
  a matching event by binning or an approximate matching rule (see
  [below](#faq-can-i-use-ibs-with-continuous-responses));
- the simulator can generate a response for any trial given its context,
  such as the stimulus and any relevant earlier observed stimuli and
  responses (*conditional simulation*);
- every observed response, or the matching event you define, has a positive
  probability under the model.
  For IBS to be practical, these probabilities must also be large enough
  at the parameters of interest: a response of probability `p` takes
  `1 / p` samples on average ([1], Section 4.2).

For each trial, IBS simulates responses until one matches the observation.
For discrete responses, the number of draws gives an unbiased estimate of
the trial's log-likelihood. Summing these estimates gives an unbiased
estimate of the data's log-likelihood. IBS also gives an unbiased estimate
of its variance ([1], Sections 2.4, 4.3 and 4.6). With continuous responses,
the same guarantee applies to the log-probability of the matching event,
with approximation error from discretization. The
[answer on checking estimates](#faq-how-do-i-check-that-the-estimates-are-right-for-my-model)
explains how to assess the uncertainty and its limitations.

These estimates let you use likelihood-based methods for fitting and model
comparison. [PyBADS](https://acerbilab.github.io/pybads/) supports
maximum-likelihood and maximum-a-posteriori estimation;
[PyVBMC](https://acerbilab.org/pyvbmc/) approximates the posterior and
model evidence. Both accept noisy targets and are among the lab's
[tools for fitting models to data](https://acerbilab.org/model-fitting/).

Use a closed-form likelihood or an accurate analytical or numerical
approximation when one is tractable. IBS can then help check its
implementation and accuracy ([1], Section 6.4).

(faq-when-should-i-use-ibs-rather-than-amortized-simulation-based-inference)=
### When should I use IBS rather than amortized simulation-based inference?

Both approaches can fit models with an intractable likelihood. The better
choice depends on the simulator, the trial contexts, how many datasets you
will fit, and the accuracy you need.

**What IBS gives.** IBS estimates the log-likelihood of one dataset at one
parameter vector by simulating each trial in its own context. For discrete
responses, the estimator is unbiased for each dataset and parameter vector,
provided the observed responses have positive probabilities and the draws
are neither truncated nor selected. It also estimates the variance ([1], Sections 2.4, 4.3 and
4.6). IBS requires no training or summary statistics. Its estimates can
serve as a target for methods that support noisy log-likelihoods, including
PyBADS for optimization and PyVBMC for posterior and evidence inference,
both among the lab's
[tools for fitting models to data](https://acerbilab.org/model-fitting/).

**When amortized inference is the better choice.** Amortized methods train
a neural network on simulations, then reuse it across datasets. In neural
posterior estimation, a trained network can produce a posterior
approximation for a new dataset almost immediately
([Li et al., 2026](https://openreview.net/forum?id=osV7adJlKD), Section 1).
The training cost is shared across all the datasets it serves. This is
often the better choice when you fit one model to many datasets and its
simulations are cheap.

IBS, by contrast, needs new simulations for every dataset and every
parameter vector evaluated during a fit. A fit can require hundreds of
evaluations: in the
[validation of PyIBS](https://github.com/acerbilab/pyibs/blob/main/dev/results/2026-10-09-validation.md),
the three-parameter example required 392 for PyBADS and 115 for PyVBMC. Libraries
such as [sbi](https://github.com/sbi-dev/sbi) and
[BayesFlow](https://github.com/bayesflow-org/bayesflow) implement amortized
inference (Li et al., 2026, Section 2.3).

**Where IBS remains the method of choice.**

- *The trials' contexts are many and richly structured.* An amortized
  estimator must learn the model's behavior across the contexts it may
  encounter. IBS simulates only the contexts in the observed data. For
  example, a model of human board-game play chooses a move given the
  current board position, which may occur only once in the dataset ([1],
  Section 5.4; B. van Opheusden et al., 2023, "Expertise increases planning
  depth in human gameplay", *Nature* 618: 1000–1005,
  <https://doi.org/10.1038/s41586-023-06124-2>). In the four-in-a-row
  game studied in [1], Section 5.4, the datasets used positions drawn from
  5,482 positions recorded in human play. The model's distribution over
  moves could not be computed even numerically, so IBS estimated its
  log-likelihood from simulations.
- *Guarantees on each data set matter.* "A given pre-trained amortized
  neural estimator may be perfectly suitable for some real datasets while it
  is utterly untrustworthy for others" (C. Li, A. Vehtari, P.-C. Bürkner,
  S. T. Radev, L. Acerbi and M. Schmitt, 2026, "Amortized Bayesian
  Workflow", *Transactions on Machine Learning Research*,
  <https://openreview.net/forum?id=osV7adJlKD>, Section 2.2). Their workflow
  checks each dataset's results with diagnostics, then uses importance
  sampling and MCMC when the checks fail. IBS provides unbiased
  log-likelihood estimates for each dataset without training, together with
  variance estimates. These guarantees concern the likelihood estimator.
  The optimizer or inference method can still make errors, so assess the
  fit itself, for example by comparing independent runs.

**Its costs and requirements.** A response of probability `p` takes
`1 / p` samples on average ([1], Section 4.2). Since this is
`exp(-log p)`, an improbable response can be very expensive to match.
A lapse rate can put a lower bound on response probabilities, and a
[likelihood threshold](#faq-what-does-the-likelihood-threshold-neg_logl_threshold-do-and-how-do-i-choose-it)
can stop sampling at poor parameters ([1], Section 6.4 and Appendix C.1).

Responses must be discrete or have a matching event defined by binning or
an approximate matching rule ([1], Section 6.3). The
simulator must also generate an individual trial's response conditioned on
its observed context. A model with latent dynamics that can generate whole
sequences, but cannot simulate one response conditional on an observed
history, does not meet this requirement ([1], Section 2.2).

(faq-how-does-ibs-compare-with-approximate-bayesian-computation-abc-or-synthetic-likelihood)=
### How does IBS compare with approximate Bayesian computation (ABC) or synthetic likelihood?

ABC and synthetic likelihood commonly compare summary statistics of
observed and simulated data ([1], Section 1, citing Beaumont et al., 2002,
for ABC and Wood, 2010, for synthetic likelihood). Such statistics may
discard information relevant to inference. IBS instead estimates the
log-likelihood of the full data, trial by trial ([1], Sections 1 and 6.3).
Its output can be used by likelihood-based methods that support noisy
targets; ABC uses dedicated inference algorithms ([1], Section 6.3).

A synthetic likelihood gives a noisy estimate of the log-likelihood of
the summary statistics. It can also serve as a target for PyVBMC, one of
the lab's [tools for fitting models to data](https://acerbilab.org/model-fitting/),
as its FAQ explains
([Can I use *any* technique to estimate a noisy log-likelihood?](https://acerbilab.org/pyvbmc/faq.html#faq-can-i-use-any-technique-to-estimate-a-noisy-log-likelihood)).
For continuous responses, [1] proposes approximate IBS, which borrows ABC's
tolerance but compares the full responses (see
[below](#faq-can-i-use-ibs-with-continuous-responses)).

(faq-what-is-ibs_basic-for)=
### What is `ibs_basic` for?

[`ibs_basic`](api/functions/ibs_basic.rst) is a minimal implementation for
teaching and reading the algorithm. It follows `ibs_basic.m` of MATLAB IBS
and the loop in [1], Section 4.1: sample one trial at a time, one response
per simulator call, until it matches. The function returns one
log-likelihood estimate. It has no averaging over repeats, variance
estimate, sampling cap or likelihood threshold. Use `IBS` for analyses.

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

PyIBS 1.5 requires NumPy 2.0 or newer. If your environment contains NumPy
1.x or a package that requires it, conda may select PyIBS 0.1.0 without
warning. Require version 1.5 or newer explicitly:

```console
conda install --channel=conda-forge "pyibs>=1.5"
```

Conda will then upgrade NumPy or report a dependency conflict.

See the [installation instructions](installation.rst) for more details, and
the [GitHub repository](https://github.com/acerbilab/pyibs) for the source
code.

(faq-which-external-packages-does-pyibs-require)=
### Which external packages does PyIBS require?

PyIBS requires NumPy 2.0 or newer and SciPy 1.13 or newer. These are its
only runtime dependencies, and `pip` or `conda` installs them with PyIBS.
MATLAB is not required.

For fitting, install [PyBADS](https://acerbilab.github.io/pybads/) 1.5.1 or
newer, or [PyVBMC](https://acerbilab.org/pyvbmc/) 1.5 or newer. Both accept
PyIBS's estimate and estimated SD as a noisy target, and are among the
lab's [tools for fitting models to data](https://acerbilab.org/model-fitting/).
Running the [example notebooks](examples.rst) also requires Jupyter; see
the [installation instructions](installation.rst).

(faq-which-version-of-python-do-i-need)=
### Which version of Python do I need?

PyIBS requires Python 3.10 or newer.

(faq-how-do-i-know-whether-a-newer-version-of-pyibs-exists)=
### How do I know whether a newer version of PyIBS exists?

Run `pyibs.check_for_updates()`. It queries PyPI and prints a message. If a
newer release is available, the message includes an update command:
`python -m pip install --upgrade pyibs` for pip, or
`conda update --channel=conda-forge pyibs` for conda. The conda-forge
package can appear a few days after the PyPI release. The function also
returns the installed version, the latest release and whether an update
is available.

This function is PyIBS's only network access, and it runs only when you
call it. PyIBS prints no automatic messages. Use `pyibs.__version__` to
read the installed version
without a network request. See the
[`check_for_updates` API](api/functions/check_for_updates.rst) for details.

(faq-how-do-i-check-that-pyibs-works-on-my-computer)=
### How do I check that PyIBS works on my computer?

The test suite ships with PyIBS. Install the `test` extra to add pytest,
then run the tests:

```console
python -m pip install --upgrade "pyibs[test]"
python -m pytest --pyargs pyibs
```

If PyBADS 1.5.1 or newer or PyVBMC 1.5 or newer is installed, the suite
also fits the example model with the available package. These integration
tests take a few minutes. To omit them, run
`python -m pytest --pyargs pyibs -m "not integration"`.

(faq-i-am-having-trouble-installing-pyibs-can-you-help)=
### I am having trouble installing PyIBS. Can you help?

Ask in the [Discussions forum](https://github.com/orgs/acerbilab/discussions).
Include your operating system, Python version, installation command and
the full error message so that we can reproduce the problem.

(faq-the-simulator-and-the-data)=
## The simulator and the data

(faq-how-do-i-write-the-simulator)=
### How do I write the simulator?

Write a function `sample_from_model(theta, design_rows)`. If it also has a
parameter named `rng` that accepts a keyword, `IBS` calls it as
`sample_from_model(theta, design_rows, rng=rng)` and supplies the object's
random number generator. Use that generator for
[reproducible runs](#faq-how-do-i-make-a-run-reproducible).

`theta` is the parameter vector passed to `ibs`. `design_rows` contains
the design rows for the trials requested in this simulator call. If
`design_matrix=None`, it contains their 0-based trial indices instead.
Return one simulated response per requested row:

- For `r` requested rows, return an array of shape `(r,)` or `(r, 1)` for
  single-column responses, or `(r, C)` for `C` response columns. A response
  matches only when all its columns match the observation.
- Generate each row independently. A trial may be requested more than
  once, so each occurrence needs a fresh draw. Sharing random draws
  across rows violates the sampling assumptions.
- Use compatible response types. Numbers and booleans can match each
  other, but NumPy does not equate them with text or bytes.

For example, this observer makes a noisy measurement of stimulus `S`,
reports right (1) when the measurement is at least a bias and left (-1)
otherwise, and responds at random on a fraction `lapse` of trials:

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

This is the model used in the [notebooks](examples.rst). PyIBS installs
its simulator as `pyibs.examples.psycho_model.psycho_generator` and its
closed-form negative log-likelihood as `psycho_neg_logl`. The
[`IBS` API](api/classes/ibs.rst) describes the simulator interface and all
settings.

(faq-my-simulator-needs-additional-data-or-inputs-how-do-i-pass-them-to-ibs)=
### My simulator needs additional data or inputs. How do I pass them to `IBS`?

`IBS` passes the parameter vector, the requested design rows and, if the
simulator accepts it, the random generator. Bind additional inputs to a
wrapper function or use `functools.partial`. Unlike MATLAB `ibslike`,
`IBS` has no `varargin` argument.

For a simulator with signature `simulator(theta, design_rows, rng, data)`,
you can write:

```python
def sample_from_model(theta, design_rows, rng):
    return simulator(theta, design_rows, rng, data)
```

Define `data` before calling this wrapper. Alternatively, bind it with
`functools.partial`:

```python
from functools import partial

ibs = IBS(partial(simulator, data=data), R, S)
```

Both approaches retain a parameter named `rng`, so `IBS` still passes its
generator to the simulator.

(faq-the-responses-of-my-model-depend-both-on-the-current-trial-and-on-responses-or-stimuli-from-previous-trials-how-can-i-tell-this-to-ibs)=
### The responses of my model depend both on the current trial and on responses or stimuli from previous trials. How can I tell this to `IBS`?

IBS estimates each response's probability conditional on the observed
history ([1], Section 2.2). Your simulator must therefore generate the
requested trial's response given the earlier stimuli and *observed*
responses. It must not replace that history with newly simulated trials.

To give the simulator access to the full history:

- Leave `design_matrix=None`: `IBS` then calls the simulator with the
  0-based indices of the requested trials;
- Bind the full stimulus and response arrays, and any
  other data, to the simulator (see the
  [previous question](#faq-my-simulator-needs-additional-data-or-inputs-how-do-i-pass-them-to-ibs));
- For each requested index, generate a response using that trial's
  stimulus and the relevant observed history.

For example, this model shifts the decision towards the previous observed
response. Here `S` and `R` are one-dimensional arrays:

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

You can also include the history in the design when each trial's inputs
fit in one row. If `r_prev` contains the previous observed response for
each trial, use
`IBS(sample_from_model, R, np.column_stack([S, r_prev]))` and write the
simulator to read the two columns.

(faq-my-model-uses-data-structures-which-are-not-easily-converted-to-numerical-arrays-can-i-still-use-ibs)=
### My model uses data structures which are not easily converted to numerical arrays. Can I still use `IBS`?

Yes. The responses must support comparison, but the simulator's other
inputs can be arbitrary Python objects.

- Encode discrete *responses* as numbers, booleans, text (such as
  `"left"` and `"right"`) or bytes; `IBS` compares them with `==`. For
  example, encode a board-game move as the index of its square. If a
  response combines numbers and text, give both the observed and simulated
  responses as object arrays (`dtype=object`).
- Other inputs need not be numerical arrays. Bind them to the simulator
  and request trials by index, as
  explained in the
  [question above](#faq-the-responses-of-my-model-depend-both-on-the-current-trial-and-on-responses-or-stimuli-from-previous-trials-how-can-i-tell-this-to-ibs).
  Alternatively, pass them as `design_matrix`, which accepts an array with
  one row per trial, including an object array of board positions or other
  Python objects. The simulator receives the requested entries. A list of
  dictionaries becomes an object array directly. For objects that NumPy
  may interpret as sequences, such as lists of different lengths, create
  an array with `np.empty(N, dtype=object)` and fill it element by element.

(faq-can-i-use-ibs-with-continuous-responses)=
### Can I use IBS with continuous responses?

A simulated continuous response has probability zero of matching the
observed value exactly, so ordinary IBS cannot estimate its density
([1], Section 6.3). You can instead define discrete matching events in
either of two ways.

The simplest approach is binning. Map both observed and simulated
responses to bins with the same edges:

```python
edges = np.arange(-10, 10.5, 0.5)  # bins 0.5 wide
R = np.digitize(y, edges)  # y, the observed continuous responses


def sample_from_model(theta, S, rng):
    return np.digitize(simulate(theta, S, rng), edges)
```

where `simulate` draws continuous responses of the model.

The second approach is *approximate IBS* ([1], Section 6.3), inspired by
approximate Bayesian computation. Define a match as a simulated response
within `eps` of the observation. For a one-dimensional response, the
approximate density is `p_match / (2 * eps)`, where `p_match` is the
probability of this event. IBS estimates `log(p_match)`; subtract
`log(2 * eps)` to estimate the log density. In more dimensions, replace
`2 * eps` by the volume of the matching region.

Implement this by returning a boolean for each requested trial, with
every observed "response" set to `True`:

```python
eps = 0.25


def within_eps(theta, idx, rng):
    return np.abs(simulate(theta, S[idx], rng) - y[idx]) <= eps


ibs = IBS(within_eps, np.ones(len(y), dtype=bool))
neg_logl, sd = ibs(theta, num_reps=10, additional_output="std")
neg_logl += len(y) * np.log(2 * eps)  # the volume term
```

For a fixed tolerance, the volume correction is constant across parameter
vectors. It therefore changes neither the optimum nor the posterior, but
it matters for model evidence or a comparison with another calculation of
the log-likelihood.

Approximate IBS centers the matching region on each observation and avoids
the choice of bin boundaries. Binning is often adequate if the bins are
narrow compared with the noise in the data. Both approaches introduce a
discretization approximation; roughly, they smooth the model's responses
on the scale of `eps` or half the bin width. Make this scale much smaller
than the noise you wish to model. Narrower regions take more samples to
match, and the expected sample count diverges as `eps` tends to zero
([1], Section 6.3).

(faq-what-does-a-call-of-ibs-return)=
### What does a call of `IBS` return?

`ibs(theta)` returns a Python float estimating the *negative*
log-likelihood at `theta`. It averages `num_reps` independent IBS repeats,
10 by default. Set `additional_output` to request more information:

- `"var"`: the tuple `(neg_logl, neg_logl_var)`, containing the estimate
  and its estimated variance;
- `"std"`: the tuple `(neg_logl, neg_logl_std)`, containing the estimate
  and the square root of the variance estimate;
- `"full"`: an [`EstimateResult`](api/classes/estimate_result.rst), a
  dictionary whose entries can also be read as attributes: `neg_logl`,
  `neg_logl_var`, `neg_logl_std`, `exit_flag` and its `message`,
  `elapsed_time`, `num_samples_per_trial` (the average sample count per
  trial), `fun_count` (the number of simulator calls),
  and the per-trial estimates `neg_logl_trials` and `neg_logl_var_trials`.

PyBADS and PyVBMC use the `"std"` tuple for their noisy targets; both are
among the lab's [tools for fitting models to data](https://acerbilab.org/model-fitting/).
`return_positive=True` changes the returned value to the log-likelihood.
The variance estimate and the per-trial arrays keep their values and signs.
`trial_weights` weights the trials' log-likelihoods. A zero-weight trial is
still sampled; remove a trial from the data if you want to exclude it from
sampling.

With `"full"`, `exit_flag` reports how sampling ended:

- **0:** every repeat completed without reaching the likelihood threshold;
- **1:** the
  [likelihood threshold](#faq-what-does-the-likelihood-threshold-neg_logl_threshold-do-and-how-do-i-choose-it)
  ended at least one repeat. The estimate is clipped and biased, and all
  per-trial estimates are NaN;
- **2:** [`max_time`](#faq-what-does-max_time-do) stopped sampling. The call
  returns a potentially biased estimate and issues a warning.

Flag 0 describes the work completed by this call. It does not establish
unbiasedness after filtering calls by their exit status or discarding
sampling errors. A finite time limit also affects which draws finish in
time. See the answers on
[sampling errors](#faq-a-call-raises-ibssamplingerror-what-do-i-do) and
[`max_time`](#faq-what-does-max_time-do).

(faq-pybads-and-pyvbmc)=
## PyBADS and PyVBMC

(faq-how-do-i-use-pyibs-with-pybads)=
### How do I use PyIBS with PyBADS?

[PyBADS](https://acerbilab.github.io/pybads/) minimizes the target function,
so return the *negative* log-likelihood estimate and its estimated SD as
a tuple. Set `options={"specify_target_noise": True}` to tell PyBADS that
the target supplies its noise level. PyBADS is among the lab's
[tools for fitting models to data](https://acerbilab.org/model-fitting/).

```python
from pybads import BADS

ibs = IBS(sample_from_model, R, S, random_seed=1)


def target(theta):
    return ibs(theta, num_reps=100, additional_output="std")


bads = BADS(target, x0, lb, ub, plb, pub, options={"specify_target_noise": True})
optimize_result = bads.optimize()
```

- Return the tuple directly from `IBS`. PyBADS requires a Python `tuple`
  of length two; returning a list or array raises `ValueError`.
- For maximum-a-posteriori estimation, return
  `neg_logl - log_prior(theta)` with the same SD, since the log prior adds no
  noise.
- PyBADS calls the target with one one-dimensional parameter vector in
  the original parameter space, using the same coordinates as your bounds.
- Choose `num_reps` for an SD of about 1 near the optimum (see
  [How do I choose `num_reps`?](#faq-how-do-i-choose-num_reps)), and
  re-evaluate the solution with more repeats
  ([below](#faq-once-the-optimizer-has-found-the-best-parameters-should-i-evaluate-the-log-likelihood-there-with-more-repeats)).
- Run PyBADS from several starting points, as the
  [PyBADS FAQ](https://acerbilab.github.io/pybads/faq.html#faq-how-do-i-run-pybads-from-several-starting-points)
  explains.

(faq-how-do-i-use-pyibs-with-pyvbmc)=
### How do I use PyIBS with PyVBMC?

[PyVBMC](https://acerbilab.org/pyvbmc/) approximates the parameter
posterior and estimates the model evidence. When you give it a prior, its
target should return the log-likelihood estimate (`return_positive=True`)
and its estimated SD. PyVBMC adds the log prior itself. Set
`options={"specify_target_noise": True}` so it uses the supplied noise
level. PyVBMC is among the lab's
[tools for fitting models to data](https://acerbilab.org/model-fitting/).

```python
from pyvbmc import VBMC

ibs = IBS(sample_from_model, R, S, random_seed=1)


def log_likelihood(theta):
    return ibs(theta, num_reps=100, additional_output="std", return_positive=True)


vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, prior=prior, options={"specify_target_noise": True})
vp, results = vbmc.optimize()
```

Here `prior` is a PyVBMC prior object, such as
`pyvbmc.priors.Trapezoidal(lb, plb, pub, ub)`. You can instead pass a
function with `log_prior=`. Without either argument, return the log joint
yourself: add `log_prior(theta)` to the log-likelihood estimate and keep
the same SD. PyVBMC calls the target with one one-dimensional parameter
vector in the original parameter space, using your bounds' coordinates.

`results["elbo"]` estimates the evidence lower bound (ELBO), which PyVBMC
uses as an approximation to the log model evidence. `results["elbo_sd"]`
reports uncertainty in that estimate. Aim for target noise with an SD of
about 1, "and probably not larger than ~3" in the region containing most
posterior mass (the PyVBMC FAQ,
[How large can the noise be?](https://acerbilab.org/pyvbmc/faq.html#faq-how-large-can-the-noise-be-to-perform-successful-inferences-with-vbmc)).

(faq-why-does-the-target-return-the-sd-of-the-estimate-and-not-its-variance)=
### Why does the target return the SD of the estimate, and not its variance?

PyBADS and PyVBMC interpret the second element as a standard deviation
(SD). Use `additional_output="std"`. With `"var"`, they would silently
interpret the variance as an SD: a variance of 4 means an SD of 2, but
they would read it as an SD of 4. MATLAB `ibslike` returns the variance
unless `ReturnStd` is set, so check this when porting code.

(faq-why-is-the-sd-of-the-estimate-zero-and-why-do-pybads-and-pyvbmc-refuse-it)=
### Why is the SD of the estimate zero, and why do PyBADS and PyVBMC refuse it?

**Why it is zero.** For a trial that first matches at sample `K`, IBS
estimates the variance as `ψ₁(1) − ψ₁(K)`, where `ψ₁` is the trigamma
function ([1], Sections 2.4 and 4.3). At `K = 1`, both the log-likelihood
estimate and the variance estimate are zero. If every positive-weight
trial matches immediately in every repeat, the whole call returns zero
for both quantities.

This is a valid outcome of the estimator. When you request a variance,
SD or full result, `IBS` returns it and issues a `UserWarning` linking
this answer:

```text
The IBS variance estimate is 0, as it is when every trial of positive weight matched its response at its first sample. PyBADS and PyVBMC refuse an SD of 0 for a noisy target: see https://acerbilab.github.io/pyibs/faq.html#faq-why-is-the-sd-of-the-estimate-zero-and-why-do-pybads-and-pyvbmc-refuse-it
```

It is common when the model predicts every observed response with
probability close to one. For 100 trials with matching probability 0.999,
the probability of a zero variance estimate is 0.37 at `num_reps=10` and
0.000045 at `num_reps=100`. The
[validation of PyIBS](https://github.com/acerbilab/pyibs/blob/main/dev/results/2026-10-09-validation.md)
confirms these probabilities.

**Why PyBADS and PyVBMC refuse it.** Their Gaussian-process models use
the supplied SD as the noise level of the evaluation. Zero would declare
the value exact, although immediate matches do not establish that it is.
With positive trial weights, IBS has zero true variance only when every
observed response has probability one. Otherwise a zero *estimated*
variance is a possible sampling outcome.

Both packages require a finite, strictly positive SD for a noisy target.
PyBADS raises `ValueError` with a message beginning
`The target function returned the noise SD 0.0`. PyVBMC's `ValueError`
says the estimated SD `must be a finite, positive real-valued scalar`.

**What to do.**

- *More repeats* make zero variance estimates exponentially less likely
  as `num_reps` increases. They do not rule them out. A fit can make
  hundreds of evaluations, so account for the chance of encountering a
  zero over the whole run (see
  [How do I choose `num_reps`?](#faq-how-do-i-choose-num_reps)).
- *A lapse rate in the model* puts an upper bound below one on matching
  probabilities. If the lapse rate is at least `lapse` and a lapse chooses
  uniformly among `k` responses, then
  `p ≤ 1 − lapse * (k − 1) / k`. For `N` positive-weight trials, the
  probability of zero estimated variance is at most
  `(1 − lapse * (k − 1) / k) ** (N * num_reps)`. This is below `1e-13`
  for 600 binary trials, a lapse rate of 0.01 and ten repeats. The IBS
  paper also recommends a positive lapse rate to control sampling cost
  ([1], Section 6.4).

(faq-do-i-need-to-give-pybads-or-pyvbmc-the-sd-of-every-estimate)=
### Do I need to give PyBADS or PyVBMC the SD of every estimate?

Supplying the SD is recommended for both packages, but neither strictly
requires it:

- For *optimization* with PyBADS, you can return the value alone and set
  `options={"uncertainty_handling": True}`. PyBADS then estimates the
  noise. IBS variance often changes moderately over the useful part of
  parameter space, so this can work (see
  [How do I choose `num_reps`?](#faq-how-do-i-choose-num_reps)). Still,
  PyIBS computes its SD without additional simulations, and supplying it
  gives PyBADS more information. The PyBADS FAQ recommends this too
  ([Should I provide an estimate of the noise associated with each evaluation?](https://acerbilab.github.io/pybads/faq.html#faq-should-i-provide-an-estimate-of-the-noise-associated-with-each-evaluation)).
- For *Bayesian inference* with PyVBMC, the evaluation noise affects both
  the posterior approximation and the evidence estimate. PyVBMC can
  infer a noise level from a scalar-valued target with
  `options={"uncertainty_handling": True}`, but supplying the SD of each
  evaluation is preferable. Its FAQ recommends this
  ([Does VBMC automatically detect that the target function is noisy?](https://acerbilab.org/pyvbmc/faq.html#faq-does-vbmc-automatically-detect-that-the-target-function-is-noisy)),
  and PyIBS provides it without additional simulations.

(faq-i-have-several-questions-about-using-pybads-to-optimize-the-log-likelihood-can-you-help)=
### I have several questions about using PyBADS to optimize the log-likelihood. Can you help?

The [PyBADS FAQ](https://acerbilab.github.io/pybads/faq.html) covers
optimization settings, bounds, starting points and interpretation of
results. Start with its section on
[noisy objective functions](https://acerbilab.github.io/pybads/faq.html#faq-noisy-objective-function)
for IBS targets. PyBADS is among the lab's
[tools for fitting models to data](https://acerbilab.org/model-fitting/).

(faq-what-if-i-want-to-use-ibs-to-perform-bayesian-posterior-or-model-inference)=
### What if I want to use IBS to perform Bayesian posterior or model inference?

We recommend [Variational Bayesian Monte Carlo (PyVBMC)](https://acerbilab.org/pyvbmc/)
for approximating the posterior over model parameters and estimating the
marginal likelihood, or model evidence. It supports noisy log-likelihood
estimates such as IBS; see its FAQ on
[noisy target functions](https://acerbilab.org/pyvbmc/faq.html#faq-noisy-target-function).
VBMC performed well with IBS in the empirical benchmark of
[Acerbi, 2020](https://arxiv.org/abs/2006.08655).

The [integration answer](#faq-how-do-i-use-pyibs-with-pyvbmc) gives the
target and settings. PyVBMC is among the lab's
[tools for fitting models to data](https://acerbilab.org/model-fitting/).

(faq-precision-and-ibs-repeats)=
## Precision and IBS repeats

A *repeat* is an independent run of IBS ([1], Section 4.4). A PyIBS call
averages `num_reps` repeats, ten by default. Increasing this number makes
the estimate more precise at the cost of more simulations.

(faq-how-do-i-choose-num_reps)=
### How do I choose `num_reps`?

For fitting, aim for an estimated SD of about 1 in the region that matters:
near the optimum for PyBADS, or where most posterior mass lies for PyVBMC.
The PyBADS FAQ says "a standard deviation of order 1 or less should work"
([Can PyBADS handle any arbitrary amount of noise in the objective?](https://acerbilab.github.io/pybads/faq.html#faq-can-pybads-handle-any-arbitrary-amount-of-noise-in-the-objective)).
The PyVBMC FAQ says "ideally you want the SD of the noise in the
log-likelihood to be around 1, and probably not larger than ~3"
([How large can the noise be?](https://acerbilab.org/pyvbmc/faq.html#faq-how-large-can-the-noise-be-to-perform-successful-inferences-with-vbmc)).

For ordinary IBS, variance falls as `1 / num_reps` and SD as
`1 / sqrt(num_reps)`. The expected sample count grows in proportion to
`num_reps`. A pilot evaluation near the expected solution, perhaps from
a preliminary fit, gives a starting choice:

```python
_, sd = ibs(theta, num_reps=10, additional_output="std")
num_reps = max(1, int(np.ceil(10 * sd**2)))  # for an SD of about 1
```

This uses the pilot's variance estimate to target an SD of about 1. That
estimate is noisy: use more pilot repeats, or average variance estimates
from several independent calls, for a steadier guide. A pilot SD of zero
does not establish that the target is deterministic; see the
[zero-SD answer](#faq-why-is-the-sd-of-the-estimate-zero-and-why-do-pybads-and-pyvbmc-refuse-it).

One fixed `num_reps` often works across a fit because IBS variance changes
moderately over the relevant parameter region. In the
[validation of PyIBS](https://github.com/acerbilab/pyibs/blob/main/dev/results/2026-10-09-validation.md),
100 repeats on the example model's 600 trials gave an SD of 1.1 at the
generating parameters and 1.2 and 1.4 at two other vectors. Their negative
log-likelihoods were higher by 49 and 55. Noise can be larger farther from
the solution, which is usually acceptable during optimization.

The true variance contributed by one trial to one repeat is bounded by
`π²/6 ≈ 1.645` ([1], Section 4.3). With unit weights, the true SD is
therefore at most `sqrt(π²/6 * N / num_reps)` for `N` trials.

If you cannot bring the SD down to about 1 within your computational budget,
be as precise as you can afford.

(faq-once-the-optimizer-has-found-the-best-parameters-should-i-evaluate-the-log-likelihood-there-with-more-repeats)=
### Once the optimizer has found the best parameters, should I evaluate the log-likelihood there with more repeats?

Yes. Re-evaluate the candidate solution with an independent, more precise
estimate. Selecting a point because its noisy value was low can make the
reported minimum too optimistic ([1], Section 4.1). With PyBADS:

```python
x_best = optimize_result["x"]
neg_logl, sd = ibs(x_best, num_reps=1000, additional_output="std")
```

Use an `IBS` object without a
[likelihood threshold](#faq-what-does-the-likelihood-threshold-neg_logl_threshold-do-and-how-do-i-choose-it)
for this evaluation. Compare models using these independent estimates
and their SDs.

(faq-in-an-ideal-world-would-you-let-the-number-of-repeats-depend-on-how-close-the-optimization-algorithm-thinks-it-is-to-the-maximum)=
### In an ideal world, would you let the number of repeats depend on how close the optimization algorithm thinks it is to the maximum?

Adaptive precision can be useful and is a topic of research. PyIBS accepts
`num_reps` separately for each call, so your target can choose a precision
and return the corresponding estimated SD. PyBADS and PyVBMC call the
target with the parameters alone; they do not choose its repeat count.
Choose the count before drawing that evaluation's samples. Changing the
count in response to those same samples can introduce selection bias.

(faq-as-i-increase-num_reps-the-sd-of-the-estimate-goes-down-slowly-but-the-computational-time-increases-linearly-is-this-normal)=
### As I increase `num_reps`, the SD of the estimate goes down slowly, but the computational time increases linearly. Is this normal?

Yes. The expected sample count grows in proportion to `num_reps`, while
the SD falls as `1 / sqrt(num_reps)`. Halving the SD therefore takes four
times as many repeats.

Elapsed time also depends on batching and simulator overhead. With
`vectorized=True`, IBS requests larger batches as sampling proceeds, so
the number of simulator calls can grow much more slowly than the sample
count. In the validation, the example model took 9 to 12 calls per
estimate at both ten and 100 repeats. If most time is spent on a fixed
cost per call, additional repeats may therefore cost little extra time
(see [Should I set `vectorized`?](#faq-should-i-set-vectorized)).

(faq-what-are-trial-dependent-repeats-and-can-i-use-them-with-pyibs)=
### What are trial-dependent repeats, and can I use them with PyIBS?

Trial-dependent repeats assign a separate repeat count to each trial to
minimize variance for a fixed expected sample budget ([1], Appendix C.2).
A fixed allocation preserves unbiasedness as long as every trial gets at
least one repeat. The optimal allocation favors probabilities near 1/2.
It allocates fewer repeats to responses near probability zero, which
are expensive to match, and near probability one, whose estimates are
already precise.

The allocation depends on unknown trial probabilities. The paper
recommends computing it from a pilot estimate with many repeats at a
representative parameter vector, and retaining it for the fit.

For this to improve efficiency across a fit, the relatively improbable
trials should remain so across the relevant parameter region. Unexpected
responses, such as lapses, can make this plausible, but the benefit needs
assessment for the model. In the paper's example of 500 trials with
probabilities drawn uniformly between zero and one, the optimal allocation
gave a median precision gain of about 1.6 over equal repeats at the same
budget ([1], Appendix C.2).

PyIBS takes one `num_reps` for all trials in a call. To use different
counts, group trials with the same count and create an `IBS` object for
each group. Sum the independent groups' estimates and variance estimates:

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

Without a design, each group's simulator receives indices local to that
group, starting at zero. To preserve the original trial indices, pass them
as the design: `IBS(sample_from_model, R[g], g)`.

A group whose responses nearly always match can return zero estimated
variance, with the
[zero-SD warning](#faq-why-is-the-sd-of-the-estimate-zero-and-why-do-pybads-and-pyvbmc-refuse-it).
An individual group's zero is harmless for the fitting interface when the
sum of the groups' variances is positive.

(faq-cost-limits-and-the-likelihood-threshold)=
## Cost, limits and the likelihood threshold

(faq-how-many-samples-does-an-estimate-take)=
### How many samples does an estimate take?

If trial `i` has matching probability `p_i`, it takes `1 / p_i` samples
per repeat on average ([1], Section 4.2). With no early stopping, a call
therefore needs an expected `num_reps * sum(1 / p_i)` samples. Batching
with `vectorized=True` also generates surplus samples after a trial's
last needed match. These are discarded; the validation observed total
sample counts 1.01 to 1.56 times the theoretical requirement.

Use `additional_output="full"` to inspect the cost:
`num_samples_per_trial` is the total number of simulated responses divided
by the number of trials, including surplus samples; `fun_count` counts
simulator calls; and `elapsed_time` gives the call's duration.

Since `1 / p = exp(-log p)`, the cost of matching an individual response
can grow rapidly as its log-probability falls ([1], Appendix C.1).
A positive lapse rate can bound this expected cost, and a
[likelihood threshold](#faq-what-does-the-likelihood-threshold-neg_logl_threshold-do-and-how-do-i-choose-it)
can stop sampling at poor parameters ([1], Section 6.4).
The sampling cap `max_iter` raises an error when a trial takes too many
samples, including when its response is impossible (see
[A call raises `IBSSamplingError`](#faq-a-call-raises-ibssamplingerror-what-do-i-do)).

(faq-should-i-set-vectorized)=
### Should I set `vectorized`?

An explicit choice can improve speed and makes seeded runs more reliable.
`vectorized` controls how `IBS` batches simulator requests:

- `True` requests several samples of each trial that still needs matches.
  The batch size grows by the `acceleration` factor, 1.5 by default,
  subject to the batch limits. This reduces simulator calls but can
  generate samples beyond the last match needed.
- `False` requests one sample per unfinished trial per call. It generates
  no surplus, but can require many calls while waiting for rare responses.
- `None`, the default, times one simulation of all trials at the object's
  first call with `num_reps > 1`. It selects `False` if that simulation
  takes at least `vectorized_threshold` seconds (0.1 by default), and
  `True` otherwise. The object retains the choice, readable as
  `ibs.vectorized`.

A call with `num_reps=1` always requests one sample per trial per simulator
call. It warns if you explicitly set `vectorized=True`.

To choose explicitly, consider where your simulator spends its time:

- **Expensive individual responses:** `False` avoids surplus samples and
  is often preferable when runtime is proportional to sample count. In
  the validation, batching generated 1.2 to 1.5 times as many responses
  for the example model.
- **Expensive calls:** choose `True` when most cost is incurred once per
  call, for example when starting a process, loading a model or moving
  data to a GPU. A slow initial simulation would make `None` choose
  `False`, although fewer, larger calls would be faster.

The difference can be large. For the example model with 100 to 10,000
trials and a fixed cost of 0.2 s per call, the
[validation of PyIBS](https://github.com/acerbilab/pyibs/blob/main/dev/results/2026-10-09-validation.md)
found that 100 repeats took 15.5 to 27 minutes with `False`. Requesting
batches with `True` required only 9 to 12 calls, corresponding to about
2 s of simulator overhead.

For a cheap simulator, batching can also save Python and sampler overhead.
If both call overhead and per-response cost matter, time both schedules
at representative parameters.

A simulator that is slow only on its first call, such as one compiled
just in time, can distort the automatic decision. Warm it up before the
first IBS call, or set `vectorized` explicitly. An explicit choice also
removes this timing decision from
[reproducibility](#faq-how-do-i-make-a-run-reproducible).

(faq-what-does-the-likelihood-threshold-neg_logl_threshold-do-and-how-do-i-choose-it)=
### What does the likelihood threshold `neg_logl_threshold` do, and how do I choose it?

The threshold saves simulations at poor parameter vectors. During each
repeat, IBS maintains a running lower bound on the repeat's final negative
log-likelihood *estimate*. Once the bound exceeds `T`, the repeat stops
and contributes exactly `T` to the negative log-likelihood estimate. If
any repeat stops this way, the call reports exit flag 1, unless a time
limit subsequently stops the call with exit flag 2 ([1], Appendix C.1).

`T` applies to the weighted negative log-likelihood. With
`return_positive=True`, the same rule applies, but the stopped repeat
contributes `-T` to the returned log-likelihood estimate.

A usual choice during optimization is the chance level: the negative
log-likelihood of choosing uniformly among each trial's possible
responses. It is `sum_i w_i log(k_i)` for `k_i` responses and weight `w_i`
on trial `i`, or `N log(2)` for `N` binary trials with unit weights:

```python
ibs_fit = IBS(sample_from_model, R, S, neg_logl_threshold=len(R) * np.log(2))
```

Clipping changes the estimator:

- It biases log-likelihood estimates upwards and negative log-likelihood
  estimates downwards. The bias becomes negligible when the true
  log-likelihood is sufficiently above `-T` ([1], Appendix C.1). The
  threshold is intended to identify poor parameters during optimization.
  In the validation, models at or near chance level had expected negative
  log-likelihood estimates 3.0 to 6.6 below the true value.
- The variance estimate for a stopped repeat uses its counts at stopping,
  rather than estimating the variance of the clipped value. In the tested
  cases where the threshold acted, reported variances were about two to
  three times the actual variance.
- All per-trial arrays returned by `additional_output="full"` are NaN if
  any repeat was thresholded. The clipped contribution belongs to the
  repeat as a whole and has no per-trial decomposition.

You can use the threshold during optimization, but evaluate the final
solution with an object that has no threshold:
`neg_logl_threshold=np.inf`, the default
([see above](#faq-once-the-optimizer-has-found-the-best-parameters-should-i-evaluate-the-log-likelihood-there-with-more-repeats)).

For Bayesian posterior or evidence inference, leave the threshold off
unless you have assessed its effect on the regions contributing to the
posterior and evidence. A high maximum likelihood alone does not establish
that clipping elsewhere is harmless: those regions' prior mass and volume
also matter. Compare results with a higher `neg_logl_threshold` or no
threshold if you intend to use one.

If each trial's number of possible responses is hard to count, an average
can provide a practical optimization threshold. For the four-in-a-row
game, where the number of legal moves depends on the board, [1] used
`N log(20)` (Appendix C.1).

(faq-is-it-okay-to-stop-the-ibs-algorithm-for-one-trial-after-a-fixed-number-of-samples-eg-20)=
### Is it okay to stop the IBS algorithm for one trial after a fixed number of samples (e.g., 20)?

No. Returning a value from a trial that has not matched truncates its
sampling distribution and introduces bias, losing the benefit of IBS
over fixed sampling ([1], Section 3). For optimization, the paper instead
proposes an early-stopping threshold on the whole dataset's estimate
([1], Appendix C.1), implemented by the
[likelihood threshold](#faq-what-does-the-likelihood-threshold-neg_logl_threshold-do-and-how-do-i-choose-it).
The sampling cap `max_iter` therefore raises an error instead of
returning an estimate from incomplete counts (see
[A call raises `IBSSamplingError`](#faq-a-call-raises-ibssamplingerror-what-do-i-do)).

(faq-what-does-max_time-do)=
### What does `max_time` do?

`max_time` limits each call's duration in seconds. The limit is checked
after every simulator call, so a simulator call can run past it. If the
limit is reached while sampling is unfinished, each trial's estimate
averages the repeats it completed. The result has exit flag 2 and a
warning. A trial without a completed repeat causes `IBSSamplingError`.
If some repeats were thresholded, they retain their clipped contributions.

A finite time limit can bias estimates because draws needing fewer
samples are more likely to finish. Even the distribution of calls that
finish in time can be biased. Keep `max_time=np.inf`, its default, for
ordinary estimation and fitting. Use the sampling cap to raise an error
on excessive cost, or consider the likelihood threshold for optimization;
see their respective answers for the statistical consequences. A finite
time limit also makes the result depend on timing, so a seed alone no
longer reproduces the run.

(faq-troubleshooting)=
## Troubleshooting

(faq-a-call-raises-ibssamplingerror-what-do-i-do)=
### A call raises `IBSSamplingError`. What do I do?

[`IBSSamplingError`](api/classes/ibs_sampling_error.rst) usually means a
trial exceeded the sampling cap. A call allows `max_iter * num_reps`
samples for each trial, with `max_iter=10**5` by default. It returns no
estimate from an incomplete draw. The message gives the affected trials'
0-based indices and sample counts:

```text
In a draw of 10 repeats, trial 1 drew 1135 samples, more than max_iter * num_reps = 1000. 1 of 3 trials still need matches, and IBS returns no estimate for an incomplete repeat. Check that the simulator can produce every observed response at this parameter vector, or raise max_iter.
```

The simulator may be unable to produce an observed response, or that
response may be extremely improbable at the current parameters:

- Check the response coding and order. For example, a simulator returning
  0 and 1 cannot match data encoded as -1 and 1. Return responses in the
  order of the requested design rows.
- A lapse component with a small positive lower bound, such as 0.005,
  can make every response possible and bound the expected cost ([1],
  Section 6.4). It should be part of the model you intend to fit.
- For optimization, consider a
  [likelihood threshold](#faq-what-does-the-likelihood-threshold-neg_logl_threshold-do-and-how-do-i-choose-it)
  to stop sampling poor parameter vectors before they reach the cap.

Raise `max_iter` when the response probabilities justify the sample cost
and that cost is acceptable. Do not discard failed draws and retry until
one succeeds: finishing below a finite cap favors shorter draws, which
give higher log-likelihood estimates. Keeping only successful calls can
therefore introduce selection bias, even though each retained call has
exit flag 0. Filtering calls by their exit flags can likewise select a
biased subset. Investigate failures and revise the simulator or sampling
settings instead.

`IBSSamplingError` is also raised if
[`max_time`](#faq-what-does-max_time-do) stops a call before a trial has
completed a repeat.

(faq-ibs-calls-my-simulator-with-different-subsets-of-trials-what-does-this-mean-for-my-simulator)=
### `IBS` calls my simulator with different subsets of trials. What does this mean for my simulator?

Each requested row must have the same response distribution regardless
of which other rows are requested or their order. Requests may contain a
subset of trials and may repeat a trial several times.

For example, if `S[:, 0]` holds stimulus contrast, then
`np.min(design_rows[:, 0])` is the minimum over the current request. It is
not the dataset's minimum and may change on every call. If the model needs
the dataset's minimum or other global information, compute it from the
full data and
[bind it to the simulator](#faq-my-simulator-needs-additional-data-or-inputs-how-do-i-pass-them-to-ibs).

Generate each requested response independently. Drawing one random
number and sharing it across all rows makes samples dependent and can
invalidate the likelihood estimate or its estimated variance.

(faq-a-call-raises-a-valueerror-or-a-typeerror-about-the-simulators-output-what-is-wrong)=
### A call raises a `ValueError` or a `TypeError` about the simulator's output. What is wrong?

- `ValueError: The simulator was asked for ... and returned an array of shape
  ...`: return one response per requested row. For `r` rows, the array
  must have shape `(r,)` or `(r, 1)` for one response column, or `(r, C)`
  for `C` columns. Check that you use the requested rows, rather than
  returning a response for every trial in the full dataset.
- `TypeError: The simulator returned responses of dtype ..., which NumPy
  never finds equal to the ... among the observed responses`: the simulator
  returned a type that cannot match the observed responses, such as text
  for numerical observations. Use compatible types. For responses mixing
  numbers and text, use object arrays (`dtype=object`) for both the data
  and simulator output.

`ValueError: response_matrix holds a NaN` occurs during construction and
names the affected trials. NaN is unequal to itself, so it cannot match.
If it denotes a response category, encode that category with a value the
simulator can return. If it denotes missing data, handle the missingness
in your analysis or remove the trial.

(faq-how-do-i-make-a-run-reproducible)=
### How do I make a run reproducible?

Seed the `IBS` object and draw all simulator randomness from its supplied
generator:

```python
def sample_from_model(theta, S, rng):
    ...  # every random draw from rng


ibs = IBS(sample_from_model, R, S, vectorized=True, random_seed=1)
```

`random_seed` follows PyBADS's convention:

- An integer or `numpy.random.SeedSequence` seeds a new generator.
- A `numpy.random.Generator` is used directly.
- `None`, the default, derives a new generator from NumPy's global random
  state. Calling `np.random.seed` before constructing the object therefore
  also fixes its seed.

The object stores its generator as `ibs.rng` and passes it to simulators
with a keyword-compatible parameter named `rng`.

Two objects with the same seed reproduce the same sequence of estimates
when given the same calls and simulator, provided timing does not change
the sampling:

- Set `vectorized=True` or `False`. Automatic `None` also reproduces draws
  if it makes the same decision in both runs, but that decision depends on
  timing (see [Should I set `vectorized`?](#faq-should-i-set-vectorized)).
- Keep `acceleration_threshold=None` and `max_time=np.inf`, their
  defaults. Neither then uses simulator runtime to alter sampling.

A simulator using global functions such as `np.random.normal` is not
controlled by the seed passed to `IBS`. Reproducibility across computers
or different Python, NumPy and SciPy versions is also not guaranteed.

For fitting, give separate seeds to PyIBS and the optimizer or inference
method:

```python
ibs = IBS(sample_from_model, R, S, vectorized=True, random_seed=1)
bads = BADS(target, x0, lb, ub, plb, pub, options={"specify_target_noise": True, "random_seed": 2})
vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, prior=prior, options={"specify_target_noise": True}, seed=2)
```

Create and seed the `IBS` object once for the whole fit. Recreating it with
the same seed at every evaluation reuses the random stream and freezes
the noise, which can bias fitting (the PyBADS FAQ,
[Can I make a noisy objective function deterministic by fixing the noise process?](https://acerbilab.github.io/pybads/faq.html#faq-can-i-make-a-noisy-objective-function-deterministic-by-fixing-the-noise-process)).

(faq-how-do-i-check-that-the-estimates-are-right-for-my-model)=
### How do I check that the estimates are right for my model?

Check the simulator against a likelihood computed independently wherever
possible ([1], Section 6.4). This may be a closed-form or accurate numerical
likelihood for the full model, a simplified model, or a few carefully
chosen trials and parameter vectors. PyIBS itself has been
[validated across models and settings](https://github.com/acerbilab/pyibs/blob/main/dev/results/2026-10-09-validation.md),
but that does not check your simulator.

Draw many independent IBS estimates at a fixed parameter vector, without
a likelihood threshold or time limit. Compare their mean with the
reference value, allowing for the mean's sampling error. Also compare
their empirical variance with the mean of the reported variance
estimates. Ordinary IBS gives unbiased estimates of both log-likelihood
and variance when its draws are completed and not selected.

If the estimates are approximately normal and the reported SDs are stable
and positive, inspect the z-scores `(neg_logl - exact) / sd`. They should
be approximately centered on zero with SD near one. This is an
approximation, not a guarantee from unbiasedness alone or from using ten
repeats. If nearly all trials match on their first samples, estimated
variances can be zero and z-scores can behave poorly even with many
repeats; see the
[zero-SD answer](#faq-why-is-the-sd-of-the-estimate-zero-and-why-do-pybads-and-pyvbmc-refuse-it).

The installed example model, `pyibs.examples.psycho_model`, provides a
simulator and the closed-form negative log-likelihood `psycho_neg_logl`.
The [first example notebook](examples.rst) shows this comparison on 600
trials, where the standardized errors are close to normal.

Check the fit separately, for example by comparing independent runs of
PyBADS or PyVBMC from different starting points. These are among the lab's
[tools for fitting models to data](https://acerbilab.org/model-fitting/).

(faq-miscellanea)=
## Miscellanea

(faq-i-used-ibslike-in-matlab-what-is-different-in-pyibs)=
### I used `ibslike` in MATLAB. What is different in PyIBS?

PyIBS implements the same IBS estimator as `ibslike.m` 0.96 of
[MATLAB IBS](https://github.com/acerbilab/ibs), with a Python interface:

- Create an `IBS` object with the simulator, responses and design:
  `ibs = IBS(sample_from_model, response_matrix, design_matrix, ...)`.
  Evaluate it with
  `ibs(params, num_reps, trial_weights, additional_output, return_positive)`.
  The corresponding MATLAB call is
  `ibslike(fun, params, respMat, designMat, options, varargin)`.
  Extra simulator arguments are
  [bound to it](#faq-my-simulator-needs-additional-data-or-inputs-how-do-i-pass-them-to-ibs)
  rather than passed as `varargin`.
- MATLAB's `Nreps`, `TrialWeights` and `ReturnPositive` become the call
  arguments `num_reps`, `trial_weights` and `return_positive`.
  `ReturnStd` becomes `additional_output="std"`.
  `additional_output="full"` combines the exit flag and output structure
  in one result. Its fields use Python names: `funcCount` becomes
  `fun_count`, `NsamplesPerTrial` becomes `num_samples_per_trial`, and
  `nlogL_trials` and `nlogLvar_trials` become `neg_logl_trials` and
  `neg_logl_var_trials`.
- Other options become object settings: `Vectorized` becomes
  `vectorized` (`'auto'` becomes `None`), `Acceleration` becomes
  `acceleration`, `NsamplesPerCall` becomes `num_samples_per_call`,
  `MaxIter` becomes `max_iter`, `MaxTime` becomes `max_time`, and
  `NegLogLikeThreshold` becomes `neg_logl_threshold` (`np.inf` turns it
  off). MATLAB's hard-coded `MaxSamples`, `AccelerationThreshold`,
  `VectorizedThreshold` and `MaxMem` are also settings:
  `max_samples`, `acceleration_threshold`, `vectorized_threshold` and
  `max_mem`.
- With no design, the simulator receives 0-based trial indices.
- The simulator can take the object's random generator, `rng`, for
  [reproducible runs](#faq-how-do-i-make-a-run-reproducible).

The sampling and stopping rules also have deliberate differences:

- `vectorized=None` makes its decision once per object.
- The number of samples per call grows after every call by default.
  Set `acceleration_threshold=0.1` to use `ibslike`'s timing rule.
- The likelihood threshold checks each repeat separately and assigns
  a stopped repeat exactly the threshold value ([1], Appendix C.1).
- The sample cap counts samples separately for each trial and raises
  `IBSSamplingError` when exceeded.
- A time-limited estimate averages each trial's completed repeats and
  issues a warning.

For the tests corresponding to `ibslike('test')`, run
`pytest --pyargs pyibs`. The
[catalogue of deliberate differences](https://github.com/acerbilab/pyibs/blob/main/pyibs/README.md)
lists all differences from `ibslike.m` and the didactic `ibs_basic.m`,
with their reasons. The two implementations draw random numbers
differently, so runs do not match draw for draw even with the same
simulator. MATLAB IBS is among the lab's
[tools for fitting models to data](https://acerbilab.org/model-fitting/).

(faq-i-used-pyibs-010-what-do-i-need-to-change)=
### I used PyIBS 0.1.0. What do I need to change?

PyIBS 1.5 keeps the main calling convention: construct `IBS(...)` and
evaluate it with `ibs(params, num_reps, ...)`. Check an existing script
for the following changes:

- Results differ because of the new sampler and fixes to 0.1.0, including
  its default `max_iter=15`, which biased estimates for improbable
  responses. Calling `np.random.seed` does not reproduce 0.1.0's results.
- Settings are checked more strictly. Values accepted by 0.1.0 may raise
  an exception.
- Exceeding the sample cap raises `IBSSamplingError` instead of returning
  exit flag 3.
- The example model is imported from `pyibs.examples.psycho_model`.

The "Upgrading from 0.1.0" list in the
[changelog](https://github.com/acerbilab/pyibs/blob/main/CHANGELOG.md)
gives the full set of checks, including changes to returned values and
object settings.
