==================
``EstimateResult``
==================

.. note::

  A call of an :doc:`IBS <ibs>` object with ``additional_output="full"``
  returns an ``EstimateResult``: the estimate, its variance and standard
  deviation, the exit flag and its message, the cost of the call, and the
  estimate of each trial, with its variance.

  It is a dictionary whose keys can also be read as attributes:
  ``result["neg_logl"]`` and ``result.neg_logl`` are the same value.

.. autoclass:: pyibs.EstimateResult
