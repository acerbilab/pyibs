=============
``ibs_basic``
=============

.. note::

  ``ibs_basic`` is a bare-bone implementation of inverse binomial sampling,
  for teaching: for each trial in turn, it calls the simulator once per
  sample until a simulated response matches the observed one, and adds up
  the trials' estimates of the log-likelihood. It follows ``ibs_basic.m`` of
  :labrepos:`MATLAB IBS <ibs>`.

  It returns one estimate, without its variance and with none of the
  settings of :doc:`IBS <../classes/ibs>`, which is the implementation to use.

.. autofunction:: pyibs.ibs_basic
