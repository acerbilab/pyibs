:tocdepth: 2

=======
``IBS``
=======

.. note::

  The ``IBS`` class uses inverse binomial sampling to estimate a model's
  log-likelihood from simulated discrete responses.

  Create an ``IBS`` object with the model's simulator, the observed
  responses and, optionally, the design of each trial; then call it with a
  parameter vector. Each call draws a new estimate of the
  *negative* log-likelihood (the log-likelihood with
  ``return_positive=True``). Request its estimated variance with
  ``additional_output="var"``, its estimated standard deviation with
  ``"std"``, or an :doc:`EstimateResult <estimate_result>` with ``"full"``.
  Exceeding the sample cap raises
  :doc:`IBSSamplingError <ibs_sampling_error>`. A time limit can return a
  partial estimate with a warning; it raises the error if any trial has no
  completed repeat.

  With ``additional_output="std"``, a call returns the pair that PyBADS and
  PyVBMC take from a noisy target: the :doc:`quick start <../../quickstart>`
  shows both. The
  :mainbranch:`catalogue of deliberate differences <pyibs/README.md>` lists
  where ``IBS`` differs from ``ibslike.m`` of MATLAB IBS, the reference
  implementation, and why.

.. autoclass:: pyibs.IBS
   :members:
   :special-members: __call__
   :member-order: groupwise
