*****
PyIBS
*****

PyIBS is one of the open-source `tools for fitting models to data <https://acerbilab.org/model-fitting/>`__ from `Luigi Acerbi's group <https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence>`__ at the University of Helsinki. Check out our other tools, such as `PyBADS <https://acerbilab.github.io/pybads/>`__ for point estimates and `PyVBMC <https://acerbilab.org/pyvbmc/>`__ for the posterior and the model evidence.

What is it?
###########

PyIBS is a Python implementation of inverse binomial sampling (IBS), a method that estimates the log-likelihood of a model that can be simulated but whose likelihood cannot be computed, for data with discrete responses [`1 <#references>`__]. For each trial, IBS draws responses from the model's simulator until one matches the observed response. The number of draws gives an estimate of the trial's log-likelihood that is exactly unbiased, and IBS also returns an estimate of the estimate's variance, which is calibrated. PyIBS follows ``ibslike.m`` of :labrepos:`MATLAB IBS <ibs>`, the reference implementation.

An IBS estimate is noisy, so it serves as the target of an optimizer or an inference method that takes noisy targets: `PyBADS <https://acerbilab.github.io/pybads/>`__ for maximum-likelihood or maximum-a-posteriori estimation, and `PyVBMC <https://acerbilab.org/pyvbmc/>`__ for the posterior and the model evidence. Both take PyIBS's estimate and its standard deviation as they come. PyIBS, PyBADS and PyVBMC are among the lab's `tools for fitting models to data <https://acerbilab.org/model-fitting/>`__.

What's new in PyIBS 1.5
-----------------------

- **Rebuilt and checked against MATLAB IBS.** PyIBS is rebuilt on one
  tested sampler that follows ``ibslike.m`` 0.96 and was checked against it
  line by line. A :mainbranch:`catalogue <pyibs/README.md>` lists where PyIBS
  differs from it on purpose, and why.
- **Validated.** On 16 models with an exact log-likelihood, under every
  setting of ``vectorized``, ``num_reps`` and the likelihood threshold that
  was tested, 2,000 estimates each, no bias was detected; without the
  threshold, the variance estimates are calibrated as those of exact IBS
  draws are (:mainbranch:`the record <dev/results/2026-10-09-validation.md>`).
- **Ready for PyBADS and PyVBMC.** ``additional_output="std"`` returns the
  tuple of the estimate and its standard deviation, as Python floats, which
  PyBADS 1.5 and PyVBMC 1.5 take from a noisy target. A standard deviation
  of 0, which both refuse, comes with a warning that links the
  :ref:`FAQ <faq-why-is-the-sd-of-the-estimate-zero-and-why-do-pybads-and-pyvbmc-refuse-it>`.
- **Reproducible runs.** ``IBS(..., random_seed=...)`` seeds the object's
  random number generator, which a simulator with a parameter named ``rng``
  receives.
- **Fixes.** PyIBS 0.1.0's default ``max_iter`` was 15, which biased the
  estimates of trials with improbable responses, and responses of several
  columns raised an error. These and other defects are fixed.
- **Faster.** In timings of the example model, PyIBS 0.1.0 took 1.3 to 2.1
  times as long per estimate with a fast simulator, and up to 3.8 times as
  long with a simulator dominated by a fixed cost per call.
- **Requirements.** PyIBS needs Python 3.10 or later, NumPy 2.0 or later and
  SciPy 1.13 or later, and no other package.
- **Examples, FAQ and a coding-agent skill.** The documentation has three
  :doc:`example notebooks <examples>`, one of them with PyBADS and one with
  PyVBMC, and a :doc:`page of frequently asked questions <faq>`; the
  :mainbranch:`PyIBS skill <skills/pyibs/SKILL.md>` points a coding agent to
  the documentation relevant to its task: give the agent that file, or copy
  the ``skills/pyibs`` folder into its skill directory.

The :mainbranch:`changelog <CHANGELOG.md>` lists what changed since PyIBS
0.1.0. Results differ from 0.1.0, also with a fixed seed, and ``IBS``
refuses some settings that 0.1.0 accepted: the changelog's list "Upgrading
from 0.1.0" says what to check in an existing script.

How does it work?
-----------------

Suppose the model's simulator produces the observed response of a trial with probability :math:`p`, which is unknown. IBS draws responses from the simulator until one matches, which takes :math:`K` draws, and estimates :math:`\log p` as

.. math::

   \hat{L} = -\sum_{k=1}^{K-1} \frac{1}{k},

which is 0 for :math:`K = 1`. The estimate is exactly unbiased for every :math:`p`, and among the unbiased estimates from such sampling it has the least variance; its variance is bounded, by :math:`\pi^2/6`, however small :math:`p` is [`1 <#references>`__, Sections 2.4 and 4.3]. With :math:`K` known, :math:`\psi_1(1) - \psi_1(K)`, where :math:`\psi_1` is the trigamma function, estimates the variance, and the estimate is calibrated [`1 <#references>`__, Sections 4.3 and 4.6].

.. image:: _static/ibs-cost-and-variance.png
    :align: center
    :alt: Fig 1: the expected number of samples, 1/p, and the variance of the IBS estimate, Li2(1 - p), against p

Fig 1: For a trial whose observed response the simulator produces with probability :math:`p`, IBS takes :math:`1/p` samples on average (left), while the variance of its estimate of :math:`\log p`, :math:`\operatorname{Li}_2(1 - p)`, stays below :math:`\pi^2/6` however small :math:`p` is (right) [`1 <#references>`__, Sections 4.2 and 4.3].

The log-likelihood of a data set is the sum of its trials' log-likelihoods, so IBS sums the trials' estimates, and their variance estimates. An ``IBS`` call averages ``num_reps`` such estimates, independent repeats, which divides the variance by ``num_reps``. Rather than sampling one trial at a time, PyIBS asks the simulator, in each call, for samples of every trial that still needs one, and with ``vectorized=True`` for several samples of each, a number that grows from call to call, as ``ibslike.m`` does. A likelihood threshold, ``neg_logl_threshold``, ends a repeat once its negative log-likelihood is known to exceed the threshold, which saves the samples of poor parameters at the price of a bias there [`1 <#references>`__, Appendix C.1].

See the IBS paper for more details (`van Opheusden, Acerbi and Ma, 2020 <#references>`__).

When should I use PyIBS?
------------------------

PyIBS suits a model that you can simulate but whose likelihood you cannot compute, on data with discrete responses (choices, categories, counts), each trial's response conditioned on its own context, such as the stimulus and possibly earlier stimuli and responses [`1 <#references>`__, Section 2.2]. IBS turns such a simulator into unbiased estimates of the log-likelihood, with a calibrated estimate of their variance [`1 <#references>`__]: the bridge from a simulator to likelihood-based methods, such as maximum-likelihood or maximum-a-posteriori estimation with PyBADS, posteriors and model evidence with PyVBMC, and model comparison.

- **When the likelihood can be computed, compute it.** The IBS paper
  recommends a closed form, or an analytical or numerical approximation,
  wherever one is tractable, with IBS estimates to check its implementation
  [`1 <#references>`__, Section 6.4].
- **Amortized simulation-based inference is often the better choice when
  one model is fitted to many datasets and its simulations are cheap.** A
  neural network trained once on simulations, as in neural posterior
  estimation, then gives the posterior of each new dataset almost at once
  [`3 <#references>`__].
- **IBS remains the method of choice where amortization is hard because each
  trial's context can be unique.** In a model of game play, each move is
  conditioned on its board position, and a position may occur only once in
  the data [`1 <#references>`__, Section 5.4; `2 <#references>`__].
- **IBS also serves where guarantees on each dataset matter.** An amortized
  estimator can be accurate on some datasets and untrustworthy on others, so
  its results need diagnostics on each dataset and a fallback
  [`3 <#references>`__]. IBS's estimates are unbiased, with a calibrated
  variance, on every dataset, without training.
- **Its cost grows with improbable responses.** A trial whose observed
  response the model produces with probability :math:`p` takes :math:`1/p`
  samples on average, so improbable responses are expensive; a lapse rate in
  the model and the likelihood threshold of ``IBS`` bound the cost at poor
  parameters [`1 <#references>`__, Sections 2.2 and 6.4, Appendix C.1].
  Continuous responses must be binned, since a simulated continuous response
  never matches exactly [`1 <#references>`__, Section 6.3].

The FAQ answers
:ref:`when to use IBS rather than amortized simulation-based inference <faq-when-should-i-use-ibs-rather-than-amortized-simulation-based-inference>`
at more length.

How-to
######

.. toctree::
   :maxdepth: 2
   :titlesonly:

   installation
   quickstart
   faq
   examples
   documentation

Contributing
############

.. toctree::
   :maxdepth: 1
   :titlesonly:

   development

References
##########

1. van Opheusden, B.\*, Acerbi, L.\* & Ma, W. J. (2020). Unbiased and efficient log-likelihood estimation with inverse binomial sampling. *PLOS Computational Biology* 16(12): e1008483. (\* equal contribution) (`paper <https://doi.org/10.1371/journal.pcbi.1008483>`__)

2. van Opheusden, B., Kuperwajs, I., Galbiati, G., Bnaya, Z., Li, Y. & Ma, W. J. (2023). Expertise increases planning depth in human gameplay. *Nature* 618: 1000-1005. (`paper <https://doi.org/10.1038/s41586-023-06124-2>`__)

3. Li, C., Vehtari, A., Bürkner, P.-C., Radev, S. T., Acerbi, L. & Schmitt, M. (2026). Amortized Bayesian workflow. *Transactions on Machine Learning Research*. (`paper on OpenReview <https://openreview.net/forum?id=osV7adJlKD>`__)

Please cite reference 1 if you use PyIBS in your work. You can cite PyIBS in your work with something along the lines of

    We estimated the log-likelihood of our models by inverse binomial sampling (IBS; van Opheusden, Acerbi and Ma, 2020), via the PyIBS software. IBS draws samples from the model's simulator until one matches each observed response, which gives unbiased estimates of the log-likelihood with calibrated estimates of their variance.

BibTeX
------
::

  @article{vanopheusden2020unbiased,
    title={Unbiased and efficient log-likelihood estimation with inverse binomial sampling},
    author={van Opheusden, Bas and Acerbi, Luigi and Ma, Wei Ji},
    journal={PLOS Computational Biology},
    volume={16},
    number={12},
    pages={e1008483},
    year={2020},
    publisher={Public Library of Science},
    doi={10.1371/journal.pcbi.1008483}
  }

License and source
------------------

PyIBS is released under the terms of the :mainbranch:`BSD 3-Clause License <LICENSE>`.
The Python source code is on :labrepos:`GitHub <pyibs>`.
You may also want to check out the original :labrepos:`MATLAB toolbox <ibs>`, `PyBADS <https://acerbilab.github.io/pybads/>`__ and `PyVBMC <https://acerbilab.org/pyvbmc/>`__, which take PyIBS's estimates as their target, and the lab's other `tools for fitting models to data <https://acerbilab.org/model-fitting/>`__.

Acknowledgments
###############

PyIBS is developed by members (past and current) of the `Machine and Human Intelligence Group <https://www.helsinki.fi/en/researchgroups/machine-and-human-intelligence>`__ at the University of Helsinki and `ELLIS Institute Finland <https://www.ellisinstitute.fi/>`__. Julia Maria Perathoner wrote PyIBS 0.1.0, its first release. Development of PyIBS 1.5 was assisted by coding agents, including Anthropic's `Claude <https://www.anthropic.com/claude>`__. Work on the PyIBS package is supported by the Research Council of Finland (grants 356498 and 358980 to Luigi Acerbi) and its Flagship programme: `Finnish Center for Artificial Intelligence FCAI <https://fcai.fi/>`__.

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   about_us
