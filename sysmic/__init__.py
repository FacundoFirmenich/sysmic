"""
Sysmic v8.0.0
=============
Fractal Tomography and the Fisher Information Barrier of Seismicity.

Companion software to Firmenich et al. (2026), JGR Solid Earth.
DOI: 10.5281/zenodo.18480821
License: GPLv3
"""

__version__ = "8.0.0"
__doi__     = "10.5281/zenodo.18480821"
__author__  = "Facundo Firmenich, Pau Firmenich, León Firmenich"

from sysmic.core        import *  # noqa: F401, F403
from sysmic.bayesian_d3 import *  # noqa: F401, F403
from sysmic.statistics  import *  # noqa: F401, F403
from sysmic.geometry    import *  # noqa: F401, F403

__all__ = [
    "FractalDimensionEstimator",
    "TripleValidator",
    "ZaccagninoStabilityScore",
    "bayesian_d3_inference",
    "log_prior_d3",
    "PRIOR_LOWER",
    "PRIOR_UPPER",
]
