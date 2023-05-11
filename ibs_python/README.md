# Inverse binomial sampling (IBS) for Python

This folder contains Python implementations and examples of IBS.

## Code
- `ibs_basic.py` is a bare-bone implementation of IBS for didactic purposes.
- `psycho_gen.py` and `psycho_nll.py` are functions implementing, respectively, the generative model (simulator) and the negative log-likelihood function for the orientation discrimination model used in the example notebooks.
- `ibs.py` is an advanced vectorized implementation of IBS, which supports several advanced features: it allows for repeated sampling, early stopping through a log-likelihood threshold and to return variance or standard deviation of the estimation.
  - Initialize an IBS object by passing it a generative model, a response matrix and a design matrix. Call the object with a parameter to return an estimate of the *negative* log-likelihood.
  - Note that by default it returns the *negative* log-likelihood as it is meant to be used with an optimization method such as [PyBADS](https://github.com/acerbilab/pybads). Set `return_positive = true` to return the *positive* log-likelihood.
  - If you want to run  with [PyVBMC](https://github.com/acerbilab/pyvbmc), note that you need to pass the following arguments when calling the IBS object
    - `return_positive = true` to return the *positive* log-likelihood;
    - `additinal_output = std` to return as second output the standard deviation of the estimate.
- `ibs_simple_example.ipynb` is an example notebook for running `ibs_basic.py`. It is only for didactic purposes.
- `ibs_example_1_basic_use.ipynb` is an example notebook for running `ibs.py`. It contains an example on how to run the estimations and how to obtain different output types. It contains examples using the orientation discrimination model and one using a binomial model. The unbiasedness of the estimator is checked; this notebook is only for didactic purposes.
- `ibs_example_2_parameter_estimation.ipynb` is a full working example usage of IBS. It requires the installation of [PyBADS](https://github.com/acerbilab/pybads) and [PyVBMC](https://github.com/acerbilab/pyvbmc).
