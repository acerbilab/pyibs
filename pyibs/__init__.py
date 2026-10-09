"""PyIBS: inverse binomial sampling in Python.

Unbiased estimates of the log-likelihood of a model that can be simulated,
for data with discrete responses, with an estimate of their variance.
"""

from importlib.metadata import PackageNotFoundError, version

from pyibs._sampler import IBSSamplingError
from pyibs._update_check import check_for_updates
from pyibs.ibs import IBS, EstimateResult
from pyibs.ibs_basic import ibs_basic

try:
    __version__ = version("pyibs")
except PackageNotFoundError:  # not installed, run from a source tree
    __version__ = "unknown"

__all__ = [
    "IBS",
    "EstimateResult",
    "IBSSamplingError",
    "ibs_basic",
    "check_for_updates",
    "__version__",
]
