"""
MonteCarlo
Introduction to the Monte Carlo method
"""

# Add imports here
from .montecarlo import *
from .metropolis import *


# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
