=====================
``check_for_updates``
=====================

Check for a newer release
-------------------------

Ask PyPI whether a newer release of PyIBS is available:

.. code-block:: python

   import pyibs

   check = pyibs.check_for_updates()

The function prints one line. When PyPI has a newer release, the line names
it and gives the command that installs it, for example:

.. code-block:: text

   PyIBS 1.6.0 is available; you have 1.5.0. Update with: python -m pip install --upgrade pyibs

The command follows the installer recorded with your installation:
``python -m pip install --upgrade pyibs`` for pip,
``conda update --channel=conda-forge pyibs`` for conda (the conda-forge
package can follow PyPI by a few days), and both when the installer is
another or unknown. The other messages are listed below, with the returned
named tuple, which gives a script the same answer. A network failure raises
no error.

Network access
--------------

PyIBS contacts PyPI only when you call this function, and makes no other
network request. The request names the installed version of PyIBS and
nothing else about your installation, and the call writes nothing to disk.

.. autofunction:: pyibs.check_for_updates
