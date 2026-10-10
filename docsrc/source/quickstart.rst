***************
Getting started
***************

Start with
:doc:`PyIBS Example 1: Basic usage and calibration <_examples/pyibs_example_1_basic_usage>`
for a worked introduction. The summary below shows how to create an estimate
and use it for model fitting.

Estimate a log-likelihood
=========================

To estimate a model's log-likelihood:

1. Write a simulator that draws one response for each requested trial.
2. Supply the observed responses and each trial's design, such as its
   stimulus.
3. Create an ``IBS`` object and call it with a parameter vector. For fitting,
   wrap this call as the target of an optimizer or inference method.
4. Check the estimates and the fit.

This complete example generates 600 trials from the included
orientation-discrimination model and compares an IBS estimate with its
exact negative log-likelihood:

.. code-block:: python

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

To use your own model, pass these inputs to ``IBS``:

- ``sample_from_model``: a simulator called as
  ``sample_from_model(params, design_rows)``. If it has an ``rng`` parameter
  that accepts a keyword, PyIBS also passes its random number generator as
  ``rng=rng``. Return an independent simulated response for each requested
  row; a trial may be requested several times in one call.
- ``response_matrix``: the observed responses, one row per trial.
- ``design_matrix`` (optional): the design of each trial, one row per trial.
  The simulator receives the rows requested by IBS. When no design is
  supplied, it receives the trials' 0-based indices instead.

``ibs(params)`` returns the *negative* log-likelihood estimate, averaging 10
independent IBS repeats by default. Set ``num_reps`` to change this number.
With ``additional_output="std"``, the call also returns an estimated
standard deviation, which measures variation between IBS calls at the same
parameters; it is not uncertainty in the parameters. With ``"full"``, it
returns an :doc:`EstimateResult <api/classes/estimate_result>` containing
per-trial estimates and the cost of the call. The
:doc:`IBS reference <api/classes/ibs>` describes all settings, including the
likelihood threshold ``neg_logl_threshold``.

Choose a sampling schedule
==========================

The example sets ``vectorized=True`` to request batches of independent
responses. The default, ``vectorized=None``, times one response per trial at
the first call with more than one repeat: a simulation faster than 0.1
seconds selects batching; otherwise, it selects one sample per active trial
per simulator call. The object keeps that choice.

.. important::

   This timing cannot distinguish a fixed overhead per simulator call from
   a cost per response. If fixed overhead dominates, ``vectorized=True`` can
   be much faster even when the automatic choice is False. Compilation at
   the first simulator call can also mislead the choice; warm up the
   simulator or set ``vectorized`` explicitly. See
   :ref:`Should I set vectorized? <faq-should-i-set-vectorized>`.

With PyBADS and PyVBMC
======================

The fitting snippets below are templates. Supply a starting parameter
vector ``x0``, hard bounds ``lb`` and ``ub``, and plausible bounds ``plb``
and ``pub``; PyVBMC also needs a prior.
:doc:`Example 2 <_examples/pyibs_example_2_maximum_likelihood_with_pybads>`
and :doc:`Example 3 <_examples/pyibs_example_3_posterior_with_pyvbmc>` define
these inputs for the orientation-discrimination model. PyBADS and PyVBMC
are among the lab's
`tools for fitting models to data <https://acerbilab.org/model-fitting/>`__.

For `PyBADS <https://acerbilab.github.io/pybads/>`__, which minimizes its
target, return the negative log-likelihood and its estimated standard
deviation:

.. code-block:: python

  from pybads import BADS

  def target(theta):
      return ibs(theta, num_reps=100, additional_output="std")

  bads = BADS(
      target, x0, lb, ub, plb, pub,
      options={"specify_target_noise": True, "random_seed": 2},
  )
  result = bads.optimize()

For `PyVBMC <https://acerbilab.org/pyvbmc/>`__, return the log-likelihood
(``return_positive=True``) and its estimated standard deviation. When you
supply ``prior``, PyVBMC adds the log prior itself:

.. code-block:: python

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

Choose ``num_reps`` for a standard deviation of about 1 near the optimum.
PyBADS works best with noise of 1 or less there; PyVBMC works best with
about 1, and not much more than 3, in the region containing most posterior
mass. For this model, 100 repeats give an SD of about 1.1 on 600 trials.

For reproducible runs, seed both PyIBS and the fitting method, and have the
simulator draw from the ``rng`` it receives. Set ``vectorized`` explicitly
and leave the timing-dependent settings ``max_time`` and
``acceleration_threshold`` at their defaults. The examples above use
separate seeds for IBS and the fit; the FAQ explains
:ref:`when a seed reproduces a run <faq-how-do-i-make-a-run-reproducible>`.

Examples and FAQ
================

The :doc:`example notebooks <examples>` form a short tutorial: basic use
and checks of the uncertainty estimates, maximum-likelihood fitting with
PyBADS, and posterior and model-evidence estimation with PyVBMC.

The :doc:`FAQ <faq>` explains how to write a simulator, choose ``num_reps``
and the likelihood threshold, and troubleshoot a failed call.
