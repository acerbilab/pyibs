************
Installation
************

PyIBS is available via ``pip`` and ``conda-forge``.

1. Install or upgrade with pip::

     python -m pip install --upgrade pyibs

   Or install with Conda::

     conda install --channel=conda-forge pyibs

   PyIBS requires Python version 3.10 or newer, NumPy 2.0 or newer and SciPy
   1.13 or newer, and no other package. In an environment that holds NumPy
   1.x, ``conda`` can install PyIBS 0.1.0 instead, without a warning: ask it
   for ``"pyibs>=1.5"``.

   To learn whether a newer release exists and how to update, see the
   :ref:`FAQ <faq-how-do-i-know-whether-a-newer-version-of-pyibs-exists>`.

2. (Optional): Install `PyBADS <https://acerbilab.github.io/pybads/>`__ and
   `PyVBMC <https://acerbilab.github.io/pyvbmc/>`__, to fit models with
   PyIBS's estimates as the examples do, and
   `Jupyter Notebook <https://jupyter.org/install>`__, to run the examples::

     python -m pip install pybads pyvbmc notebook

   The example notebooks are installed with PyIBS, in the folder that this
   command prints::

     python -c "import os, pyibs.examples; print(os.path.dirname(pyibs.examples.__file__))"

To install or upgrade PyIBS with its test dependencies and run the tests::

  python -m pip install --upgrade "pyibs[test]"
  pytest --pyargs pyibs

When PyBADS 1.5.1 or later, or PyVBMC 1.5 or later, is installed, the tests
also fit the example model with it, which takes a few minutes;
``pytest --pyargs pyibs -m "not integration"`` leaves those tests out.

If you wish to install directly from the latest source code, please see the
:doc:`instructions for developers and contributors <development>`.
