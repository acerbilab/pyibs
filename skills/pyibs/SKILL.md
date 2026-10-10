---
name: pyibs
description: Find and apply PyIBS documentation when estimating the log-likelihood of a model that can only be simulated, on data with discrete responses, fitting it with PyBADS or PyVBMC, troubleshooting an estimate, or deciding whether inverse binomial sampling fits the problem.
---

# PyIBS

Use the documentation below to answer the user's question or work on their
analysis. Read the sections relevant to the task; follow links for details
as needed.

Check the installed PyIBS version (`importlib.metadata.version("pyibs")`)
before using version-specific features: the
[changelog](https://github.com/acerbilab/pyibs/blob/main/CHANGELOG.md) says
what changed in each release, and its "Upgrading from" list says what to
check in a script written for an earlier release. This skill accompanies
PyIBS 1.5. Prefer documentation from the user's checkout when available.
The source links below point to the
[`main` branch](https://github.com/acerbilab/pyibs/tree/main), from which
the published [documentation](https://acerbilab.github.io/pyibs/) is built.
For another version, use the corresponding Git tag and check API signatures
and docstrings in that version.

When the user reports a problem with PyIBS 1.5 or later, run
`pyibs.check_for_updates()`, which asks PyPI for the latest release: the fix
may already be available. For an earlier version, check the changelog for
fixes and upgrade instructions. See
the [`check_for_updates` API](https://github.com/acerbilab/pyibs/blob/main/docsrc/source/api/functions/check_for_updates.rst).

## When PyIBS fits

IBS estimates the log-likelihood of a model from simulated discrete
responses. The simulator generates each trial in its own context, which
may include earlier observed responses. When every observed response has
positive model probability and sampling runs to completion, both the
log-likelihood estimate and its variance estimate are unbiased. Normal
confidence intervals based on the estimated SD are an approximation whose
reliability depends on the model and number of repeats. PyBADS and PyVBMC
use IBS estimates as a noisy target.

When choosing a method for a new analysis, discuss the alternatives with
the user, using the README's and FAQ's explanations below. For an existing
analysis, inspect its setup and saved results and follow the user's task;
revisit the method choice when the user asks or the analysis shows a reason
to question it.

- Use a closed-form likelihood or an accurate numerical likelihood when
  one is tractable. IBS can help check an approximation to the likelihood.
- Amortized simulation-based inference, such as neural posterior
  estimation, is often the better choice when one model is fitted to many
  datasets and its simulations are cheap.
- IBS remains the method of choice when trial contexts are numerous and
  richly structured, such as the board positions in a model of human
  gameplay. It also helps when guarantees on each dataset matter. IBS needs
  no training and has the unbiasedness guarantees stated above for each
  dataset; an amortized
  estimator can be inaccurate on a particular dataset. Optimization and
  posterior inference using IBS still have errors of their own.
- Its cost is about `1 / p` samples for a trial whose observed response has
  probability `p`. A lapse component covering every possible response can
  limit that cost. A likelihood threshold saves work at poor parameter
  vectors but introduces bias. Continuous responses require binning or an
  approximate matching rule; see the FAQ's "Can I use IBS with continuous
  responses?".

## What to read

Paths are relative to the PyIBS repository root. The links also work when
this skill folder has been copied elsewhere.

| Task | Read |
| --- | --- |
| Decide whether IBS fits the problem, or whether a closed-form likelihood or amortized simulation-based inference fits it better; install PyIBS | [README.md](https://github.com/acerbilab/pyibs/blob/main/README.md): “When should I use PyIBS?” and “Installation”; the FAQ's “General” section, above all “When should I use IBS rather than amortized simulation-based inference?”. |
| Write the simulator; set up the responses and the design | [docsrc/source/quickstart.rst](https://github.com/acerbilab/pyibs/blob/main/docsrc/source/quickstart.rst), [Example 1](https://github.com/acerbilab/pyibs/blob/main/examples/pyibs_example_1_basic_usage.ipynb), the FAQ's “The simulator and the data” section (extra inputs, earlier trials, continuous responses), and the docstring of `IBS` in [pyibs/ibs.py](https://github.com/acerbilab/pyibs/blob/main/pyibs/ibs.py), which gives what each argument takes, the shapes the simulator returns, and what `IBS` refuses. |
| Fit a model with PyBADS (maximum likelihood or maximum a posteriori) or with PyVBMC (posterior and model evidence) | The FAQ's “PyBADS and PyVBMC” section, above all “How do I use PyIBS with PyBADS?” and “How do I use PyIBS with PyVBMC?”; [Example 2](https://github.com/acerbilab/pyibs/blob/main/examples/pyibs_example_2_maximum_likelihood_with_pybads.ipynb) and [Example 3](https://github.com/acerbilab/pyibs/blob/main/examples/pyibs_example_3_posterior_with_pyvbmc.ipynb); the lab's page of [tools for fitting models to data](https://acerbilab.org/model-fitting/), whose PyBADS and PyVBMC have skills of their own. |
| Choose `num_reps`, `vectorized` and the likelihood threshold; estimate simulation cost | The FAQ's “Precision and IBS repeats” and “Cost, limits and the likelihood threshold” sections: “How do I choose `num_reps`?”, “Should I set `vectorized`?”, “What does the likelihood threshold `neg_logl_threshold` do, and how do I choose it?” and “How many samples does an estimate take?”. |
| Reproduce a run | The FAQ's “How do I make a run reproducible?”, and the Notes of the docstring of `IBS`. |
| Interpret a result; troubleshoot a call (an SD of 0, `IBSSamplingError`, the time limit, an error about the simulator's output) | The FAQ's “What does a call of `IBS` return?”, “Why is the SD of the estimate zero, and why do PyBADS and PyVBMC refuse it?” and “Troubleshooting” section; the docstrings of `EstimateResult` (`pyibs/ibs.py`, exit flags and per-trial estimates) and `IBSSamplingError` (`pyibs/_sampler.py`). |
| Check that the estimates are right for the user's model | The FAQ's “How do I check that the estimates are right for my model?”; [Example 1](https://github.com/acerbilab/pyibs/blob/main/examples/pyibs_example_1_basic_usage.ipynb) checks them against a closed form. |
| Port a script or an analysis from MATLAB `ibslike.m` | The FAQ's “I used `ibslike` in MATLAB. What is different in PyIBS?”, and the [catalogue of deliberate differences](https://github.com/acerbilab/pyibs/blob/main/pyibs/README.md). |
| Compare with PyIBS 0.1.0 | The FAQ's “I used PyIBS 0.1.0. What do I need to change?”; the [changelog](https://github.com/acerbilab/pyibs/blob/main/CHANGELOG.md): its “Upgrading from 0.1.0” list, then the entries it points to. |
| Look up exact arguments, outputs or errors | The [API reference](https://acerbilab.github.io/pyibs/documentation.html), with sources under `docsrc/source/api/` and the docstrings under `pyibs/`. |

The FAQ is [docsrc/source/faq.md](https://github.com/acerbilab/pyibs/blob/main/docsrc/source/faq.md),
published at <https://acerbilab.github.io/pyibs/faq.html>.

Before running estimates in bulk or fitting a model, estimate the cost. A
call takes about `num_reps` times the sum of `1 / p_i` over trials, where
`p_i` is the probability of trial i's observed response. Parameter vectors
that make some responses improbable can therefore be expensive, and an
optimizer or inference method can call the target hundreds of times. Time
a call at plausible parameters and include diagnostic calls in the user's
budget. Check the sampling schedule: the automatic decision can be
misleading when fixed overhead per simulator call or compilation dominates
the first timing; the FAQ explains when to set `vectorized` explicitly.
Inspect an existing analysis's saved results before deciding whether
another run is needed. Link the relevant documentation when explaining
your conclusions.
