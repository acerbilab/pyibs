***************
Getting started
***************

The best way to get started with PyIBS is via the tutorials and worked
examples. In particular, start with
:doc:`PyIBS Example 1: Basic usage and calibration <_examples/pyibs_example_1_basic_usage>`
and continue from there.

If you are already familiar with inverse binomial sampling, you can find a
summary usage below.

Summary usage
=============

The typical workflow of PyIBS follows four steps:

1. Define the simulator, a function that draws a response of the model for
   each trial it is given;
2. Set up the data: the observed responses and the design of each trial,
   such as its stimulus;
3. Create an ``IBS`` object and call it with a parameter vector, or give it
   as the target of an optimizer or an inference method;
4. Examine the results.

Here with the example model that PyIBS installs, a model of an orientation
discrimination task:

.. code-block:: python

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

``IBS(sample_from_model, response_matrix, design_matrix)`` takes:

- ``sample_from_model``: the simulator, called as
  ``sample_from_model(params, design_rows)``, or with ``rng=`` too when it
  has a parameter named ``rng``; it returns one simulated response for each
  row of ``design_rows``, the rows of the design of the trials requested;
- ``response_matrix``: the observed responses, one row per trial;
- ``design_matrix`` (optional): the design of each trial, one row per trial;
  without it, the simulator receives the trial indices.

Calling the object, ``ibs(params, num_reps=10, additional_output=None)``,
returns an estimate of the *negative* log-likelihood of the data at
``params``, the average of ``num_reps`` independent IBS estimates;
``additional_output="std"`` returns its standard deviation too, and
``"full"`` an :doc:`EstimateResult <api/classes/estimate_result>` with
per-trial estimates and the cost of the call. The
:doc:`IBS reference <api/classes/ibs>` describes every setting, such as the
likelihood threshold ``neg_logl_threshold``.

With PyBADS and PyVBMC
======================

With `PyBADS <https://acerbilab.github.io/pybads/>`__, which minimizes, the
target returns the negative log-likelihood and its standard deviation:

.. code-block:: python

  from pybads import BADS

  def target(theta):
      return ibs(theta, num_reps=100, additional_output="std")

  bads = BADS(target, x0, lb, ub, plb, pub, options={"specify_target_noise": True})
  result = bads.optimize()

With `PyVBMC <https://acerbilab.github.io/pyvbmc/>`__, given a prior, the
target returns the log-likelihood (``return_positive=True``) and its
standard deviation:

.. code-block:: python

  from pyvbmc import VBMC

  def log_likelihood(theta):
      return ibs(theta, num_reps=100, additional_output="std", return_positive=True)

  vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, prior=prior, options={"specify_target_noise": True})
  vp, results = vbmc.optimize()

Choose ``num_reps`` so that the standard deviation is about 1 near the
optimum: the noise that PyBADS and PyVBMC handle best. Here 100 repeats give
about 1.1 on 600 trials. For a reproducible run, seed both PyIBS and the
method it serves, such as ``IBS(..., random_seed=1)`` and
``BADS(..., options={"specify_target_noise": True, "random_seed": 2})``, and
have the simulator draw from the ``rng`` it receives; the FAQ says
:ref:`when a seed reproduces a run <faq-how-do-i-make-a-run-reproducible>`.

Examples & FAQ
==============

See the :doc:`examples` for more detailed information: the basic use of
PyIBS and the calibration of its estimates, maximum-likelihood estimation
with PyBADS, and the posterior and the model evidence with PyVBMC.

In addition, check out the :doc:`FAQ <faq>` for practical recommendations,
such as how to choose ``num_reps`` and the likelihood threshold, how to
write a simulator, and what to do when a call fails.
