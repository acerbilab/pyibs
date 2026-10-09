:tocdepth: 2

=======
``IBS``
=======

.. note::

  The ``IBS`` class estimates the log-likelihood of a model that can be
  simulated but whose likelihood cannot be computed, for data with discrete
  responses, by inverse binomial sampling.

  Create an ``IBS`` object with the model's simulator, the observed
  responses and, optionally, the design of each trial; then call it with a
  parameter vector. Each call returns a new, independent estimate of the
  *negative* log-likelihood (the log-likelihood with
  ``return_positive=True``), with its variance (``additional_output="var"``)
  or its standard deviation (``"std"``) on request, or an
  :doc:`EstimateResult <estimate_result>` with ``"full"``. A call that cannot
  complete raises :doc:`IBSSamplingError <ibs_sampling_error>`.

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
