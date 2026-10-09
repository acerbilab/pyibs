====================
``IBSSamplingError``
====================

.. note::

  A call of an :doc:`IBS <ibs>` object raises ``IBSSamplingError`` when it
  cannot return an estimate: when a trial draws more than
  ``max_iter * num_reps`` samples, or when ``max_time`` stops the sampling
  before a trial has completed a repeat. Its message names the trials
  concerned and the setting to raise.

  It is a subclass of ``RuntimeError``.

.. autoexception:: pyibs.IBSSamplingError
