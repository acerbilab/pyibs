=============
``ibs_basic``
=============

.. note::

  ``ibs_basic`` is a minimal implementation of inverse binomial sampling,
  for teaching: for each trial in turn, it calls the simulator once per
  sample until a simulated response matches the observed one, and adds up
  the trials' estimates of the log-likelihood. It follows ``ibs_basic.m`` of
  :labrepos:`MATLAB IBS <ibs>`.

  It returns a log-likelihood estimate without a variance estimate or any
  stopping limits. Use :doc:`IBS <../classes/ibs>` for fitting models.

.. autofunction:: pyibs.ibs_basic
