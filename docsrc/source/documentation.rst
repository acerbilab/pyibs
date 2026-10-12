*************
Documentation
*************

Use ``IBS`` to create a log-likelihood estimator from your simulator and
data. Its calls return a negative log-likelihood estimate by default; with
``additional_output="full"``, they return an ``EstimateResult`` containing
the estimate, uncertainty estimates, and sampling diagnostics. The pages
below document these entry points and all other public classes and
functions.

.. toctree::
   :maxdepth: 1

   api/classes/ibs
   api/classes/estimate_result
   api/classes/classes
   api/functions/functions
