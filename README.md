# PyIBS: Inverse Binomial Sampling in Python
![Version](https://img.shields.io/badge/dynamic/json?label=python&query=info.requires_python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpyibs%2Fjson)
[![Conda](https://img.shields.io/conda/v/conda-forge/pyibs)](https://anaconda.org/conda-forge/pyibs)
[![PyPI](https://img.shields.io/pypi/v/pyibs)](https://pypi.org/project/pyibs/)
<br />
[![Discussion](https://img.shields.io/badge/-discussion-blue?logo=github)](https://github.com/orgs/acerbilab/discussions)
[![tests](https://img.shields.io/github/actions/workflow/status/acerbilab/pyibs/tests.yml?branch=main&label=tests)](https://github.com/acerbilab/pyibs/actions/workflows/tests.yml)
[![docs](https://img.shields.io/github/actions/workflow/status/acerbilab/pyibs/docs.yml?branch=main&label=docs)](https://github.com/acerbilab/pyibs/actions/workflows/docs.yml)
[![build](https://img.shields.io/github/actions/workflow/status/acerbilab/pyibs/build.yml?branch=main&label=build)](https://github.com/acerbilab/pyibs/actions/workflows/build.yml)

PyIBS is one of the open-source [tools for fitting models to data](https://acerbilab.org/model-fitting/) developed by [Luigi Acerbi's group](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence) at the University of Helsinki. It works with [PyBADS](https://github.com/acerbilab/pybads) for point estimates and [PyVBMC](https://github.com/acerbilab/pyvbmc) for posterior and model-evidence estimates.

## What is it?

PyIBS estimates the log-likelihood of a model that you can simulate but whose likelihood you cannot compute. It implements inverse binomial sampling (IBS) for data with discrete responses [[1](#references-and-citation)]. For each trial, IBS simulates responses until one matches the observed response, then uses the number of draws to estimate that trial's log-likelihood. Without early stopping, the estimate is exactly unbiased. PyIBS also estimates its variance, which measures the noise introduced by simulation. The reference implementation is `ibslike.m` in [MATLAB IBS](https://github.com/acerbilab/ibs).

Use the estimates with a method that handles noisy objectives: [PyBADS](https://github.com/acerbilab/pybads) for maximum-likelihood or maximum-a-posteriori estimation, or [PyVBMC](https://github.com/acerbilab/pyvbmc) for posterior and model-evidence estimation. PyIBS can return the estimate and its estimated standard deviation in the format both methods accept. These packages are among the lab's [tools for fitting models to data](https://acerbilab.org/model-fitting/).

## What's new in PyIBS 1.5

- **Rebuilt and checked against MATLAB IBS.** The sampling code follows `ibslike.m` 0.96 and has been compared with it line by line. The [catalogue of deliberate differences](https://github.com/acerbilab/pyibs/blob/main/pyibs/README.md) explains the choices made for the Python implementation.
- **Validated against exact likelihoods.** Tests across 16 models found no detectable bias without early stopping and reproduced the expected values when a likelihood threshold was used. The [validation record](https://github.com/acerbilab/pyibs/blob/main/dev/results/2026-10-09-validation.md) describes the test coverage and the limits of the uncertainty estimates.
- **Ready for PyBADS and PyVBMC.** `additional_output="std"` returns a tuple of Python floats: the estimate and its estimated standard deviation. PyBADS 1.5 and PyVBMC 1.5 accept this format for noisy targets. PyIBS warns when the standard deviation is 0, which both methods refuse, and links to [guidance in the FAQ](https://acerbilab.github.io/pyibs/faq.html#faq-why-is-the-sd-of-the-estimate-zero-and-why-do-pybads-and-pyvbmc-refuse-it).
- **Reproducible runs.** `IBS(..., random_seed=...)` seeds a random number generator that the simulator receives through its `rng` parameter. Set the sampling schedule explicitly to avoid a decision based on timing; see [Quick start](#quick-start).
- **Fixes.** PyIBS 0.1.0's default `max_iter` was 15, which biased estimates for improbable responses. It also failed on responses with several columns. Version 1.5 fixes these and other defects.
- **Faster.** In [timings of the example model](https://github.com/acerbilab/pyibs/blob/main/dev/results/2026-10-09-validation.md), PyIBS 0.1.0 took 1.3 to 2.1 times as long per estimate with a fast simulator, and up to 3.8 times as long when a fixed cost per simulator call dominated the runtime.
- **Requirements.** PyIBS requires Python 3.10 or later. Its only runtime dependencies are NumPy 2.0 or later and SciPy 1.13 or later.
- **Documentation and examples.** The [documentation site](https://acerbilab.github.io/pyibs/) includes three example notebooks covering basic use, PyBADS, and PyVBMC, plus a [FAQ](https://acerbilab.github.io/pyibs/faq.html). The [PyIBS skill](https://github.com/acerbilab/pyibs/blob/main/skills/pyibs/SKILL.md) helps coding agents find the relevant documentation (see [Documentation](#documentation)).

Results differ from PyIBS 0.1.0 even with a fixed seed, and `IBS` rejects some settings that the earlier version accepted. Before upgrading an existing analysis, read "Upgrading from 0.1.0" in the [changelog](https://github.com/acerbilab/pyibs/blob/main/CHANGELOG.md), which also lists the changes in detail.

## Documentation

Read the [full documentation](https://acerbilab.github.io/pyibs/) for tutorials, practical guidance, and the API reference.

To use the [PyIBS skill](https://github.com/acerbilab/pyibs/blob/main/skills/pyibs/SKILL.md), give the file to your coding agent or copy the `skills/pyibs` folder into its skill directory. Update a copied skill from the PyIBS version you use.

## When should I use PyIBS?

Use PyIBS when you can simulate a model's responses but cannot compute its likelihood. The responses must be discrete, such as choices, categories, or counts, and the simulator must be able to draw each trial's response given its context: its stimulus and, when relevant, earlier stimuli and responses [[1](#references-and-citation), Section 2.2]. IBS then supplies the noisy log-likelihood and uncertainty estimates needed for maximum-likelihood or maximum-a-posteriori fitting with PyBADS, posterior and model-evidence estimation with PyVBMC, and model comparison.

- **Use a tractable likelihood when available.** The IBS paper recommends a closed form or an analytical or numerical approximation whenever one is practical. IBS estimates can help check that implementation [[1](#references-and-citation), Section 6.4].
- **Amortized simulation-based inference is often better when fitting one model to many datasets with cheap simulations.** A neural network trained once on simulations can share that training cost across datasets. In neural posterior estimation, it then estimates the posterior for each new dataset almost immediately [[3](#references-and-citation)].
- **IBS remains the method of choice when trial contexts are numerous and richly structured.** An amortized estimator must learn the model's behaviour across the contexts it may encounter; IBS only simulates the contexts present in the data. For example, a model of human board-game play chooses each move from the current board position, which may occur only once in the dataset [[1](#references-and-citation), Section 5.4; [2](#references-and-citation)].
- **IBS also helps when guarantees for each dataset matter.** An amortized estimator can be accurate on some datasets and unreliable on others, so each dataset needs diagnostics and a fallback if they fail [[3](#references-and-citation)]. IBS requires no training and gives unbiased log-likelihood and variance estimates for any dataset when all observed responses have positive probability and sampling is allowed to complete. These guarantees concern the likelihood estimates: optimization and posterior approximation still introduce errors that need their own checks. An estimated standard deviation also need not give accurate normal confidence intervals; see [How does it work?](#how-does-it-work).
- **Account for the cost of improbable responses.** A trial whose observed response has model probability p takes 1/p samples on average. A lapse component that gives every possible response positive probability limits this cost. The likelihood threshold can save work at poor parameter vectors, at the cost of bias [[1](#references-and-citation), Sections 2.2 and 6.4, Appendix C.1]. Continuous responses require binning or an approximate matching rule [[1](#references-and-citation), Section 6.3]; see the [FAQ](https://acerbilab.github.io/pyibs/faq.html#faq-can-i-use-ibs-with-continuous-responses).

The FAQ discusses [IBS and amortized simulation-based inference](https://acerbilab.github.io/pyibs/faq.html#faq-when-should-i-use-ibs-rather-than-amortized-simulation-based-inference) in more detail.

## Installation

Install PyIBS from PyPI or conda-forge. It requires Python 3.10 or later, NumPy 2.0 or later, and SciPy 1.13 or later.

1. Install or upgrade with pip:
    ```console
    python -m pip install --upgrade pyibs
    ```
    Or use Conda:
    ```console
    conda install --channel=conda-forge "pyibs>=1.5"
    ```
    The minimum version in the Conda command prevents it from silently selecting PyIBS 0.1.0 in an environment with NumPy 1.x.

2. To run the example notebooks, also install [Jupyter Notebook](https://jupyter.org/install), [PyBADS](https://github.com/acerbilab/pybads), and [PyVBMC](https://github.com/acerbilab/pyvbmc). The latter two are the lab's [tools for fitting models to data](https://acerbilab.org/model-fitting/) used in the fitting examples:
   ```console
   python -m pip install --upgrade "pybads>=1.5.1" "pyvbmc>=1.5" notebook
   ```
   The notebooks are included with PyIBS. Find their installation folder with:
   ```console
   python -c "import os, pyibs.examples; print(os.path.dirname(pyibs.examples.__file__))"
   ```

Run `pyibs.check_for_updates()` to check for a newer release. To install from source, follow the [instructions for developers and contributors](https://acerbilab.github.io/pyibs/development.html).

## Quick start

To estimate a model's log-likelihood:

1. Write a simulator that draws one response for each requested trial.
2. Supply the observed responses and each trial's design, such as its stimulus.
3. Create an `IBS` object and call it with a parameter vector. For fitting, wrap this call as the target of an optimizer or inference method.
4. Check the estimates and the fit.

This complete example generates 600 trials from the included orientation-discrimination model and compares an IBS estimate with its exact negative log-likelihood:

```python
import numpy as np

from pyibs import IBS
from pyibs.examples.psycho_model import psycho_generator, psycho_neg_logl

# The parameters are (log(sigma), bias, lapse).
rng = np.random.default_rng(20261009)
S = 3 * rng.standard_normal(600)  # the orientation of each trial's stimulus
theta = np.array([0.0, 0.2, 0.03])
R = psycho_generator(theta, S, rng)  # the responses, 1 or -1

ibs = IBS(psycho_generator, R, S, vectorized=True, random_seed=1)
neg_logl, sd = ibs(theta, num_reps=100, additional_output="std")
exact = psycho_neg_logl(theta, S, R)
print(f"IBS: {neg_logl:.1f} +/- {sd:.1f}; exact: {exact:.1f}")
```

To use your own model, pass these inputs to `IBS`:

- `sample_from_model`: a simulator called as `sample_from_model(params, design_rows)`. If it has an `rng` parameter that accepts a keyword, PyIBS also passes its random number generator as `rng=rng`. Return an independent simulated response for each requested row; a trial may be requested several times in one call.
- `response_matrix`: the observed responses, one row per trial.
- `design_matrix` (optional): the design of each trial, one row per trial. The simulator receives the rows requested by IBS. When no design is supplied, it receives the trials' 0-based indices instead.

`ibs(params)` returns the *negative* log-likelihood estimate, averaging 10 independent IBS repeats by default. Set `num_reps` to change this number. With `additional_output="std"`, the call also returns an estimated standard deviation, which measures variation between IBS calls at the same parameters; it is not uncertainty in the parameters. With `"full"`, it returns an [`EstimateResult`](https://acerbilab.github.io/pyibs/api/classes/estimate_result.html) containing per-trial estimates and the cost of the call. The [`IBS` reference](https://acerbilab.github.io/pyibs/api/classes/ibs.html) describes all settings, including the likelihood threshold `neg_logl_threshold`.

**Choose the sampling schedule for your simulator.** The example sets `vectorized=True` to request batches of independent responses. The default, `vectorized=None`, times one response per trial at the first call with more than one repeat: a simulation faster than 0.1 seconds selects batching; otherwise, it selects one sample per active trial per simulator call. The object keeps that choice. This timing cannot distinguish a fixed overhead per simulator call from a cost per response. If fixed overhead dominates, `vectorized=True` can be much faster even when the automatic choice is False. Compilation at the first simulator call can also mislead the choice; warm up the simulator or set `vectorized` explicitly. See [the FAQ](https://acerbilab.github.io/pyibs/faq.html#faq-should-i-set-vectorized).

The fitting snippets below are templates. Supply a starting parameter vector `x0`, hard bounds `lb` and `ub`, and plausible bounds `plb` and `pub`; PyVBMC also needs a prior. [Example 2](https://acerbilab.github.io/pyibs/_examples/pyibs_example_2_maximum_likelihood_with_pybads.html) and [Example 3](https://acerbilab.github.io/pyibs/_examples/pyibs_example_3_posterior_with_pyvbmc.html) define these inputs for the orientation-discrimination model.

For [PyBADS](https://github.com/acerbilab/pybads), which minimizes its target, return the negative log-likelihood and its estimated standard deviation:

```python
from pybads import BADS

def target(theta):
    return ibs(theta, num_reps=100, additional_output="std")

bads = BADS(
    target, x0, lb, ub, plb, pub,
    options={"specify_target_noise": True, "random_seed": 2},
)
result = bads.optimize()
```

For [PyVBMC](https://github.com/acerbilab/pyvbmc), return the log-likelihood (`return_positive=True`) and its estimated standard deviation. When you supply `prior`, PyVBMC adds the log prior itself:

```python
from pyvbmc import VBMC

def log_likelihood(theta):
    return ibs(
        theta, num_reps=100, additional_output="std", return_positive=True
    )

vbmc = VBMC(
    log_likelihood, x0, lb, ub, plb, pub,
    prior=prior, options={"specify_target_noise": True}, seed=2,
)
vp, results = vbmc.optimize()
```

Choose `num_reps` for a standard deviation of about 1 near the optimum. PyBADS works best with noise of 1 or less there; PyVBMC works best with about 1, and not much more than 3, in the region containing most posterior mass. For this model, 100 repeats give an SD of about 1.1 on 600 trials.

For reproducible runs, seed both PyIBS and the fitting method, and have the simulator draw from the `rng` it receives. Set `vectorized` explicitly and leave the timing-dependent settings `max_time` and `acceleration_threshold` at their defaults. The examples above use separate seeds for IBS and the fit; the FAQ explains [when a seed reproduces a run](https://acerbilab.github.io/pyibs/faq.html#faq-how-do-i-make-a-run-reproducible).

## Next steps

The [example notebooks](https://acerbilab.github.io/pyibs/examples.html) form a short tutorial: basic use and checks of the uncertainty estimates, maximum-likelihood fitting with PyBADS, and posterior and model-evidence estimation with PyVBMC. Read their saved outputs on the [documentation site](https://acerbilab.github.io/pyibs/index.html), or run the notebooks installed in `pyibs/examples`.

The [FAQ](https://acerbilab.github.io/pyibs/faq.html) explains how to write a simulator, choose `num_reps` and the likelihood threshold, and troubleshoot a failed call.

## How does it work?

Suppose a trial's observed response has probability p under the model. IBS does not need to know p: it simulates responses until one matches, taking K draws, and estimates log p as

    -(1 + 1/2 + 1/3 + ... + 1/(K - 1)),

The estimate is 0 when K = 1. For every p > 0, it is exactly unbiased and has the least variance among unbiased estimators based on this sampling procedure. Its variance is bounded by π²/6 even as p approaches 0 [[1](#references-and-citation), Sections 2.4 and 4.3]. The same count K gives an unbiased variance estimate, ψ₁(1) − ψ₁(K), where ψ₁ is the trigamma function [[1](#references-and-citation), Section 4.3].

Summing over trials and averaging independent repeats often makes the estimated standard deviation useful for normal confidence intervals [[1](#references-and-citation), Section 4.6]. This approximation is not reliable for every model or number of repeats. For example, if every response matches on its first draw, the variance estimate is 0 even when repeated IBS calls can vary. The [validation record](https://github.com/acerbilab/pyibs/blob/main/dev/results/2026-10-09-validation.md) examines these limitations.

**Fig 1: the cost and variance of IBS.** A trial with matching probability p takes 1/p samples on average (left). The variance of its log p estimate, Li₂(1 − p), stays below π²/6 even as p approaches 0 (right) [[1](#references-and-citation), Sections 4.2 and 4.3]. ![The expected number of samples, 1/p, and the variance of the IBS estimate, Li2(1 - p), against p](https://raw.githubusercontent.com/acerbilab/pyibs/main/docsrc/source/_static/ibs-cost-and-variance.png)

The data's log-likelihood is the sum of the trial log-likelihoods. IBS therefore adds their estimates and variance estimates. An `IBS` call averages `num_reps` independent repeats, reducing the variance by that factor. Each simulator call samples all trials that still need a match. With `vectorized=True`, it requests several samples per trial, increasing the batch size between calls as `ibslike.m` does.

The optional likelihood threshold, `neg_logl_threshold`, stops a repeat once its accumulating negative log-likelihood estimate exceeds the threshold. That repeat contributes the threshold value to the negative log-likelihood estimate. This saves simulation at poor parameters but biases the estimate [[1](#references-and-citation), Appendix C.1]. A finite `max_time` also biases estimates; use neither limit when unbiasedness is essential.

See the IBS paper for more details ([van Opheusden, Acerbi and Ma, 2020](#references-and-citation)).

## Troubleshooting and contact

PyIBS is under active development. Its estimates have been checked against exact likelihoods across models and settings, but you should also check the simulator, estimates, and fits for your own model.

For questions, unexpected behavior, or bugs:

- Ask in the lab's [Discussions forum](https://github.com/orgs/acerbilab/discussions), including questions about your model-fitting application.
- [Open a GitHub issue](https://github.com/acerbilab/pyibs/issues/new).
- Email the project lead at <luigi.acerbi@helsinki.fi> with 'PyIBS' in the subject.

## References and citation

1. van Opheusden, B.\*, Acerbi, L.\* & Ma, W. J. (2020). Unbiased and efficient log-likelihood estimation with inverse binomial sampling. *PLOS Computational Biology* 16(12): e1008483. (\* equal contribution) [https://doi.org/10.1371/journal.pcbi.1008483](https://doi.org/10.1371/journal.pcbi.1008483)

2. van Opheusden, B., Kuperwajs, I., Galbiati, G., Bnaya, Z., Li, Y. & Ma, W. J. (2023). Expertise increases planning depth in human gameplay. *Nature* 618: 1000-1005. [https://doi.org/10.1038/s41586-023-06124-2](https://doi.org/10.1038/s41586-023-06124-2)

3. Li, C., Vehtari, A., Bürkner, P.-C., Radev, S. T., Acerbi, L. & Schmitt, M. (2026). Amortized Bayesian workflow. *Transactions on Machine Learning Research*. [https://openreview.net/forum?id=osV7adJlKD](https://openreview.net/forum?id=osV7adJlKD)

Please cite reference 1 when using PyIBS. For example:

> We estimated the log-likelihood of our models with inverse binomial sampling (IBS; van Opheusden, Acerbi and Ma, 2020), implemented in PyIBS. IBS simulates responses until each observed response is matched, giving unbiased log-likelihood and variance estimates when sampling is allowed to complete.

To support the project and keep in touch:

- *Star :star:* the PyIBS repository on GitHub;
- Follow Luigi Acerbi on [X](https://x.com/AcerbiLuigi) or [Bluesky](https://bsky.app/profile/lacerbi.bsky.social) for updates about IBS/PyIBS and other projects;
- Tell us about your model-fitting problem and your experience with PyIBS (positive or negative) in the lab's [Discussions forum](https://github.com/orgs/acerbilab/discussions).

Explore [PyBADS](https://github.com/acerbilab/pybads), [PyVBMC](https://github.com/acerbilab/pyvbmc), and the lab's other [tools for fitting models to data](https://acerbilab.org/model-fitting/) for ways to fit models using these estimates.

### BibTeX

```BibTeX
@article{vanopheusden2020unbiased,
  title={Unbiased and efficient log-likelihood estimation with inverse binomial sampling},
  author={van Opheusden, Bas and Acerbi, Luigi and Ma, Wei Ji},
  journal={PLOS Computational Biology},
  volume={16},
  number={12},
  pages={e1008483},
  year={2020},
  publisher={Public Library of Science},
  doi={10.1371/journal.pcbi.1008483}
}
```

### License

PyIBS is released under the terms of the [BSD 3-Clause License](https://github.com/acerbilab/pyibs/blob/main/LICENSE).

### Acknowledgments

PyIBS is developed by members (past and current) of the [Machine and Human Intelligence Group](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence) at the University of Helsinki and [ELLIS Institute Finland](https://www.ellisinstitute.fi/). We thank Julia Maria Perathoner for her work on PyIBS 0.1.0, an earlier Python port of IBS. Development of PyIBS 1.5 was assisted by coding agents, including Anthropic's [Claude](https://www.anthropic.com/claude).
Work on the PyIBS package is supported by the Research Council of Finland (grants 356498 and 358980 to Luigi Acerbi) and its Flagship programme: [Finnish Center for Artificial Intelligence FCAI](https://fcai.fi/).
