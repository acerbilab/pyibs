# PyIBS: Inverse Binomial Sampling in Python
![Version](https://img.shields.io/badge/dynamic/json?label=python&query=info.requires_python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpyibs%2Fjson)
[![Conda](https://img.shields.io/conda/v/conda-forge/pyibs)](https://anaconda.org/conda-forge/pyibs)
[![PyPI](https://img.shields.io/pypi/v/pyibs)](https://pypi.org/project/pyibs/)
<br />
[![Discussion](https://img.shields.io/badge/-discussion-blue?logo=github)](https://github.com/orgs/acerbilab/discussions)
[![tests](https://img.shields.io/github/actions/workflow/status/acerbilab/pyibs/tests.yml?branch=main&label=tests)](https://github.com/acerbilab/pyibs/actions/workflows/tests.yml)
[![docs](https://img.shields.io/github/actions/workflow/status/acerbilab/pyibs/docs.yml?branch=main&label=docs)](https://github.com/acerbilab/pyibs/actions/workflows/docs.yml)
[![build](https://img.shields.io/github/actions/workflow/status/acerbilab/pyibs/build.yml?branch=main&label=build)](https://github.com/acerbilab/pyibs/actions/workflows/build.yml)

PyIBS is one of the open-source [tools for fitting models to data](https://acerbilab.org/model-fitting/) from [Luigi Acerbi's group](https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence) at the University of Helsinki. Check out our other tools, such as [PyBADS](https://github.com/acerbilab/pybads) for point estimates and [PyVBMC](https://github.com/acerbilab/pyvbmc) for the posterior and the model evidence.

## What is it?

PyIBS is a Python implementation of inverse binomial sampling (IBS), a method that estimates the log-likelihood of a model that can be simulated but whose likelihood cannot be computed, for data with discrete responses [[1](#references-and-citation)]. For each trial, IBS draws responses from the model's simulator until one matches the observed response. The number of draws gives an estimate of the trial's log-likelihood that is exactly unbiased, and IBS also returns an estimate of the estimate's variance, which is calibrated. PyIBS follows `ibslike.m` of [MATLAB IBS](https://github.com/acerbilab/ibs), the reference implementation.

An IBS estimate is noisy, so it serves as the target of an optimizer or an inference method that takes noisy targets: [PyBADS](https://github.com/acerbilab/pybads) for maximum-likelihood or maximum-a-posteriori estimation, and [PyVBMC](https://github.com/acerbilab/pyvbmc) for the posterior and the model evidence. Both take PyIBS's estimate and its standard deviation as they come. PyIBS, PyBADS and PyVBMC are among the lab's [tools for fitting models to data](https://acerbilab.org/model-fitting/).

## What's new in PyIBS 1.5

- **Rebuilt and checked against MATLAB IBS.** PyIBS is rebuilt on a tested sampler that follows `ibslike.m` 0.96 and was checked against it line by line. A [catalogue](https://github.com/acerbilab/pyibs/blob/main/pyibs/README.md) lists where PyIBS differs from it on purpose, and why.
- **Validated.** On 16 models with an exact log-likelihood, under every setting of `vectorized` and `num_reps` that was tested, 2,000 estimates each, no bias was detected, and the variance estimates are calibrated as those of exact IBS draws are; under the likelihood threshold, the estimates agree with the expected value of a thresholded estimate ([the record](https://github.com/acerbilab/pyibs/blob/main/dev/results/2026-10-09-validation.md)).
- **Ready for PyBADS and PyVBMC.** `additional_output="std"` returns the tuple of the estimate and its standard deviation, as Python floats, which PyBADS 1.5 and PyVBMC 1.5 take from a noisy target. A standard deviation of 0, which both refuse, comes with a warning that links the [FAQ](https://acerbilab.github.io/pyibs/faq.html#faq-why-is-the-sd-of-the-estimate-zero-and-why-do-pybads-and-pyvbmc-refuse-it).
- **Reproducible runs.** `IBS(..., random_seed=...)` seeds the object's random number generator, which a simulator with a parameter named `rng` receives.
- **Fixes.** PyIBS 0.1.0's default `max_iter` was 15, which biased the estimates of trials with improbable responses, and responses of several columns raised an error. These and other defects are fixed.
- **Faster.** In timings of the example model, PyIBS 0.1.0 took 1.3 to 2.1 times as long per estimate with a fast simulator, and up to 3.8 times as long with a simulator dominated by a fixed cost per call.
- **Requirements.** PyIBS needs Python 3.10 or later, NumPy 2.0 or later and SciPy 1.13 or later, and no other package.
- **Documentation, examples, FAQ and a coding-agent skill.** PyIBS has a [documentation site](https://acerbilab.github.io/pyibs/) with three example notebooks, one of them with PyBADS and one with PyVBMC, a [page of frequently asked questions](https://acerbilab.github.io/pyibs/faq.html), and the [PyIBS skill](https://github.com/acerbilab/pyibs/blob/main/skills/pyibs/SKILL.md), which points a coding agent to the documentation relevant to its task (see [Documentation](#documentation)).

The [changelog](https://github.com/acerbilab/pyibs/blob/main/CHANGELOG.md) lists what changed since PyIBS 0.1.0. Results differ from 0.1.0, also with a fixed seed, and `IBS` refuses some settings that 0.1.0 accepted: the changelog's list "Upgrading from 0.1.0" says what to check in an existing script.

## Documentation

The full documentation is available at: https://acerbilab.github.io/pyibs/

For coding agents, the [PyIBS skill](https://github.com/acerbilab/pyibs/blob/main/skills/pyibs/SKILL.md) points to the
documentation relevant to each task. Give your agent that file, or copy the
`skills/pyibs` folder into its skill directory. To update a copied skill,
copy the folder again from the PyIBS version you use.

## When should I use PyIBS?

PyIBS suits a model that you can simulate but whose likelihood you cannot compute, on data with discrete responses (choices, categories, counts), each trial's response conditioned on its own context, such as the stimulus and possibly earlier stimuli and responses [[1](#references-and-citation), Section 2.2]. IBS turns such a simulator into unbiased estimates of the log-likelihood, with a calibrated estimate of their variance [[1](#references-and-citation)]: the bridge from a simulator to likelihood-based methods, such as maximum-likelihood or maximum-a-posteriori estimation with PyBADS, posteriors and model evidence with PyVBMC, and model comparison.

- **When the likelihood can be computed, compute it.** The IBS paper recommends a closed form, or an analytical or numerical approximation, wherever one is tractable, with IBS estimates to check its implementation [[1](#references-and-citation), Section 6.4].
- **Amortized simulation-based inference is often the better choice when one model is fitted to many datasets and its simulations are cheap.** A neural network trained once on simulations, as in neural posterior estimation, then gives the posterior of each new dataset almost at once [[3](#references-and-citation)].
- **IBS remains the method of choice where amortization is hard, because the trials' contexts are many and richly structured.** An amortized estimator has to learn the model's behaviour across every context it may meet, while IBS only simulates the model in the contexts of the data. For example, a model of how people play a board game chooses each move from the current position on the board, and a position may occur only once in the data [[1](#references-and-citation), Section 5.4; [2](#references-and-citation)].
- **IBS also serves where guarantees on each dataset matter.** An amortized estimator can be accurate on some datasets and untrustworthy on others, so its results need diagnostics on each dataset and a fallback [[3](#references-and-citation)]. IBS's estimates are unbiased on every dataset, without training, and come with an estimate of their variance; the optimizer or the inference method that uses them still has errors of its own, which you check as for any fit.
- **Its cost grows with improbable responses.** A trial whose observed response the model produces with probability p takes 1/p samples on average, so improbable responses are expensive; a lapse rate in the model and the likelihood threshold of `IBS` bound the cost at poor parameters [[1](#references-and-citation), Sections 2.2 and 6.4, Appendix C.1]. Continuous responses must be binned, since a simulated continuous response never matches exactly [[1](#references-and-citation), Section 6.3].

The FAQ answers [when to use IBS rather than amortized simulation-based inference](https://acerbilab.github.io/pyibs/faq.html#faq-when-should-i-use-ibs-rather-than-amortized-simulation-based-inference) at more length.

## Installation

PyIBS is available via `pip` and `conda-forge`.

1. Install or upgrade with pip:
    ```console
    python -m pip install --upgrade pyibs
    ```
    Or install with Conda:
    ```console
    conda install --channel=conda-forge pyibs
    ```
    PyIBS requires Python version 3.10 or newer, NumPy 2.0 or newer and SciPy 1.13 or newer. In an environment that holds NumPy 1.x, `conda` can install PyIBS 0.1.0 instead, without a warning: ask it for `"pyibs>=1.5"`.

2. (Optional): Install [PyBADS](https://github.com/acerbilab/pybads) and [PyVBMC](https://github.com/acerbilab/pyvbmc), among the lab's [tools for fitting models to data](https://acerbilab.org/model-fitting/), to fit models with PyIBS's estimates as the examples do, and [Jupyter Notebook](https://jupyter.org/install), to run the examples:
   ```console
   python -m pip install --upgrade "pybads>=1.5.1" "pyvbmc>=1.5" notebook
   ```
   The example notebooks are installed with PyIBS, in the folder that this command prints:
   ```console
   python -c "import os, pyibs.examples; print(os.path.dirname(pyibs.examples.__file__))"
   ```

`pyibs.check_for_updates()` tells whether a newer release of PyIBS exists. If you wish to install directly from latest source code, please see the [instructions for developers and contributors](https://acerbilab.github.io/pyibs/development.html).

## Quick start

The typical workflow of PyIBS follows four steps:

1. Define the simulator, a function that draws a response of the model for each trial it is given;
2. Set up the data: the observed responses and the design of each trial, such as its stimulus;
3. Create an `IBS` object and call it with a parameter vector, or give it as the target of an optimizer or an inference method;
4. Examine the results.

Here with the example model that PyIBS installs, a model of an orientation discrimination task:

```python
import numpy as np

from pyibs import IBS
from pyibs.examples.psycho_model import psycho_generator, psycho_neg_logl

# A data set of the example model: 600 trials of an orientation
# discrimination task, at the parameters (log(sigma), bias, lapse)
rng = np.random.default_rng(20261009)
S = 3 * rng.standard_normal(600)  # the orientation of each trial's stimulus
theta = np.array([0.0, 0.2, 0.03])
R = psycho_generator(theta, S, rng)  # the responses, 1 or -1

ibs = IBS(psycho_generator, R, S, random_seed=1)
neg_logl, sd = ibs(theta, num_reps=100, additional_output="std")
exact = psycho_neg_logl(theta, S, R)
print(f"IBS: {neg_logl:.1f} +/- {sd:.1f}; exact: {exact:.1f}")
```

`IBS(sample_from_model, response_matrix, design_matrix)` takes:

- ``sample_from_model``: the simulator, called as ``sample_from_model(params, design_rows)``, or with ``rng=`` too when it has a parameter named ``rng``; it returns one simulated response for each row of ``design_rows``, the rows of the design of the trials requested;
- ``response_matrix``: the observed responses, one row per trial;
- ``design_matrix`` (optional): the design of each trial, one row per trial; without it, the simulator receives the trial indices.

Calling the object, ``ibs(params, num_reps=10, additional_output=None)``, returns an estimate of the *negative* log-likelihood of the data at ``params``, the average of ``num_reps`` independent IBS estimates; ``additional_output="std"`` returns its standard deviation too, and ``"full"`` an [`EstimateResult`](https://acerbilab.github.io/pyibs/api/classes/estimate_result.html) with per-trial estimates and the cost of the call. The [`IBS` reference](https://acerbilab.github.io/pyibs/api/classes/ibs.html) describes every setting, such as the likelihood threshold `neg_logl_threshold`.

With [PyBADS](https://github.com/acerbilab/pybads), which minimizes, the target returns the negative log-likelihood and its standard deviation:

```python
from pybads import BADS

def target(theta):
    return ibs(theta, num_reps=100, additional_output="std")

bads = BADS(target, x0, lb, ub, plb, pub, options={"specify_target_noise": True})
result = bads.optimize()
```

With [PyVBMC](https://github.com/acerbilab/pyvbmc), given a prior, the target returns the log-likelihood (``return_positive=True``) and its standard deviation:

```python
from pyvbmc import VBMC

def log_likelihood(theta):
    return ibs(theta, num_reps=100, additional_output="std", return_positive=True)

vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, prior=prior, options={"specify_target_noise": True})
vp, results = vbmc.optimize()
```

Choose `num_reps` so that the standard deviation is about 1 near the optimum: PyBADS works best with noise of 1 or less there, and PyVBMC with about 1, and not much more than 3, where the posterior has its mass. Here 100 repeats give about 1.1 on 600 trials. For a reproducible run, seed both PyIBS and the method it serves, such as `IBS(..., vectorized=True, random_seed=1)` and `BADS(..., options={"specify_target_noise": True, "random_seed": 2})`, and have the simulator draw from the `rng` it receives; the FAQ says [when a seed reproduces a run](https://acerbilab.github.io/pyibs/faq.html#faq-how-do-i-make-a-run-reproducible).

## Next steps

Once installed, the example notebooks can be found in the `pyibs/examples` directory. They can also be [viewed statically](https://acerbilab.github.io/pyibs/examples.html) on the [main documentation pages](https://acerbilab.github.io/pyibs/index.html). They make a short tutorial: the basic use of PyIBS and the calibration of its estimates; maximum-likelihood estimation with PyBADS; and the posterior and the model evidence with PyVBMC.

For practical recommendations, such as how to choose `num_reps` and the likelihood threshold, how to write a simulator, and what to do when a call fails, check out the [PyIBS FAQ](https://acerbilab.github.io/pyibs/faq.html).

## How does it work?

Suppose the model's simulator produces the observed response of a trial with probability p, which is unknown. IBS draws responses from the simulator until one matches, which takes K draws, and estimates log p as

    -(1 + 1/2 + 1/3 + ... + 1/(K - 1)),

which is 0 for K = 1. The estimate is exactly unbiased for every p, and among the unbiased estimates from such sampling it has the least variance; its variance is bounded, by π²/6, however small p is [[1](#references-and-citation), Sections 2.4 and 4.3]. With K known, ψ₁(1) − ψ₁(K), where ψ₁ is the trigamma function, estimates the variance; summed over the trials of a data set, these variance estimates are calibrated [[1](#references-and-citation), Sections 4.3 and 4.6].

**Fig 1: the cost and the variance of IBS.** For a trial whose observed response the simulator produces with probability p, IBS takes 1/p samples on average (left), while the variance of its estimate of log p, Li₂(1 − p), stays below π²/6 however small p is (right) [[1](#references-and-citation), Sections 4.2 and 4.3]. ![The expected number of samples, 1/p, and the variance of the IBS estimate, Li2(1 - p), against p](https://raw.githubusercontent.com/acerbilab/pyibs/main/docsrc/source/_static/ibs-cost-and-variance.png)

The log-likelihood of a data set is the sum of its trials' log-likelihoods, so IBS sums the trials' estimates, and their variance estimates. An `IBS` call averages `num_reps` such estimates, independent repeats, which divides the variance by `num_reps`. Rather than sampling one trial at a time, PyIBS asks the simulator, in each call, for samples of every trial that still needs one, and with `vectorized=True` for several samples of each, a number that grows from call to call, as `ibslike.m` does. A likelihood threshold, `neg_logl_threshold`, ends a repeat once its negative log-likelihood is known to exceed the threshold, which saves the samples of poor parameters at the price of a bias there [[1](#references-and-citation), Appendix C.1].

See the IBS paper for more details ([van Opheusden, Acerbi and Ma, 2020](#references-and-citation)).

## Troubleshooting and contact

PyIBS is under active development. Its estimates have been validated against exact log-likelihoods across models and settings. However, as with any method, you should double-check your results.

If you have trouble doing something with PyIBS, spot bugs or strange behavior, or you simply have some questions, please feel free to:
- Post in the lab's [Discussions forum](https://github.com/orgs/acerbilab/discussions) with questions or comments about PyIBS, your problems & applications;
- [Open an issue](https://github.com/acerbilab/pyibs/issues/new) on GitHub;
- Contact the project lead at <luigi.acerbi@helsinki.fi>, putting 'PyIBS' in the subject of the email.

## References and citation

1. van Opheusden, B.\*, Acerbi, L.\* & Ma, W. J. (2020). Unbiased and efficient log-likelihood estimation with inverse binomial sampling. *PLOS Computational Biology* 16(12): e1008483. (\* equal contribution) [https://doi.org/10.1371/journal.pcbi.1008483](https://doi.org/10.1371/journal.pcbi.1008483)

2. van Opheusden, B., Kuperwajs, I., Galbiati, G., Bnaya, Z., Li, Y. & Ma, W. J. (2023). Expertise increases planning depth in human gameplay. *Nature* 618: 1000-1005. [https://doi.org/10.1038/s41586-023-06124-2](https://doi.org/10.1038/s41586-023-06124-2)

3. Li, C., Vehtari, A., Bürkner, P.-C., Radev, S. T., Acerbi, L. & Schmitt, M. (2026). Amortized Bayesian workflow. *Transactions on Machine Learning Research*. [https://openreview.net/forum?id=osV7adJlKD](https://openreview.net/forum?id=osV7adJlKD)

Please cite reference 1 if you use PyIBS in your work. You can cite PyIBS in your work with something along the lines of

> We estimated the log-likelihood of our models by inverse binomial sampling (IBS; van Opheusden, Acerbi and Ma, 2020), via the PyIBS software. IBS draws samples from the model's simulator until one matches each observed response, which gives unbiased estimates of the log-likelihood with calibrated estimates of their variance.

Besides formal citations, you can demonstrate your appreciation for PyIBS in the following ways:

- *Star :star:* the PyIBS repository on GitHub;
- Follow Luigi Acerbi on [X](https://x.com/AcerbiLuigi) or [Bluesky](https://bsky.app/profile/lacerbi.bsky.social) for updates about IBS/PyIBS and other projects;
- Tell us about your model-fitting problem and your experience with PyIBS (positive or negative) in the lab's [Discussions forum](https://github.com/orgs/acerbilab/discussions).

You may also want to check out [PyBADS](https://github.com/acerbilab/pybads) and [PyVBMC](https://github.com/acerbilab/pyvbmc), which take PyIBS's estimates as their target, and the lab's other [tools for fitting models to data](https://acerbilab.org/model-fitting/).

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
