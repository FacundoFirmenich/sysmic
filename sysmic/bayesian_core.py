"""
SFA Bayesian Core - Dynesty Complementary Backend
==================================================

ADDS dynesty nested sampling as backend choice for D₂/D₃.
PRESERVES all existing methods (GP bootstrap, emcee, M-H).
"""

import numpy as np
from scipy import spatial
from scipy.stats import beta
from typing import Dict, Optional, Literal
import warnings

# Import likelihood from bayesian_d3 (reuse calculation)
from sysmic.bayesian_d3 import log_likelihood_d3

# Dynesty
try:
    import dynesty
    from dynesty import DynamicNestedSampler
    from dynesty.utils import resample_equal
    HAS_DYNESTY = True
except ImportError:
    HAS_DYNESTY = False

# Priors (D₂ and D₃)
PRIOR_D2_ALPHA = 5.0
PRIOR_D2_BETA = 2.0
PRIOR_D2_LOWER = 0.5
PRIOR_D2_UPPER = 3.0

PRIOR_D3_ALPHA = 7.5  # From bayesian_d3.py but defined here for standalone
PRIOR_D3_BETA = 2.5
PRIOR_D3_LOWER = 1.5
PRIOR_D3_UPPER = 4.0


def prior_transform_d2(u: float) -> float:
    return PRIOR_D2_LOWER + (PRIOR_D2_UPPER - PRIOR_D2_LOWER) * beta.ppf(u, PRIOR_D2_ALPHA, PRIOR_D2_BETA)


def prior_transform_d3(u: float) -> float:
    return PRIOR_D3_LOWER + (PRIOR_D3_UPPER - PRIOR_D3_LOWER) * beta.ppf(u, PRIOR_D3_ALPHA, PRIOR_D3_BETA)


def bayesian_dimension_dynesty(
    coordinates: np.ndarray,
    dimension_type: Literal[2, 3] = 3,
    nlive: int = 500,
    dlogz: float = 0.5,
    verbose: bool = True,
    **kwargs
) -> Dict:
    """Dynesty nested sampling for D2 or D3 - COMPLEMENTARY method."""
    if not HAS_DYNESTY:
        raise ImportError("dynesty not installed. Use bootstrap or emcee instead.")
    
    # Select prior transform based on dimension type
    if dimension_type == 2:
        prior_tfm = prior_transform_d2
        dim_name = "D2"
    elif dimension_type == 3:
        prior_tfm = prior_transform_d3
        dim_name = "D3"
    else:
        raise ValueError(f"dimension_type must be 2 or 3, got {dimension_type}")
    
    # Filter kwargs: remove dynesty-specific parameters not accepted by likelihood
    likelihood_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ['nlive', 'dlogz', 'dynamic', 'random_state', 'verbose', 'sampler']}
    
    # Pre-calculate distances for maximum speed
    n_reference = likelihood_kwargs.get('n_reference', 1000)
    n_ref = min(n_reference, len(coordinates))
    ref_indices = np.random.choice(len(coordinates), n_ref, replace=False)
    ref_points = coordinates[ref_indices]
    
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(ref_points, coordinates)
    precomputed_distances = np.sort(dist_matrix.flatten())
    
    # Pre-calculate counts for each r (Tier-4 optimization)
    # Since radii are fixed for a given coordinate set, we can count once
    extent = np.max(np.ptp(coordinates, axis=0))
    r_min = likelihood_kwargs.get('r_min_factor', 0.01) * extent
    r_max = likelihood_kwargs.get('r_max_factor', 0.3) * extent
    radii = np.logspace(np.log10(r_min), np.log10(r_max), likelihood_kwargs.get('n_radii', 20))
    
    precomputed_counts = []
    for r in radii:
        count = np.searchsorted(precomputed_distances, r)
        precomputed_counts.append(max(count - n_ref, 0))
    precomputed_counts = np.array(precomputed_counts)
    
    # Likelihood wrapper
    def loglike(d):
        return log_likelihood_d3(d[0], coordinates, precomputed_counts=precomputed_counts, radii=radii, **likelihood_kwargs)
    
    if verbose:
        print(f"Dynesty D{dimension_type}: nlive={nlive}, dlogz={dlogz}")
    
    sampler = DynamicNestedSampler(loglike, prior_tfm, ndim=1, nlive=nlive)
    sampler.run_nested(dlogz_init=dlogz, print_progress=verbose)
    
    results = sampler.results
    samples_weighted = results.samples.flatten()
    weights = np.exp(results.logwt - results.logz[-1])
    samples_equal = resample_equal(samples_weighted, weights)
    
    d_mean = np.average(samples_weighted, weights=weights)
    d_var = np.average((samples_weighted - d_mean)**2, weights=weights)
    
    if verbose:
        print(f"✓ D{dimension_type} = {d_mean:.3f} ± {np.sqrt(d_var):.3f}")
        print(f"  logZ = {results.logz[-1]:.2f} ± {results.logzerr[-1]:.2f}")
    
    return {
        'samples': samples_equal,
        'd_mean': d_mean,
        'd_std': np.sqrt(d_var),
        'd_credible_interval': np.percentile(samples_equal, [2.5, 97.5]),
        'logz': results.logz[-1],
        'sampler': 'dynesty'
    }


def bayesian_d2_inference(coordinates: np.ndarray, sampler: str = 'dynesty', **kwargs) -> Dict:
    """D2 Bayesian - COMPLEMENTARY (does not replace GP bootstrap)."""
    if sampler == 'dynesty':
        return bayesian_dimension_dynesty(coordinates, dimension_type=2, **kwargs)
    else:
        raise NotImplementedError(f"D2 sampler '{sampler}' not implemented. Use 'dynesty'.")


def bayesian_d3_inference(coordinates: np.ndarray, sampler: str = 'dynesty', **kwargs) -> Dict:
    """D3 Bayesian - ADDS dynesty option (emcee PRESERVED in bayesian_d3.py)."""
    if sampler == 'dynesty':
        result = bayesian_dimension_dynesty(coordinates, dimension_type=3, **kwargs)
        # Compatibility aliases for bayesian_d3.py naming convention
        # (scripts expect 'd3_mean', 'd3_std' but dynesty returns 'd_mean', 'd_std')
        result['d3_mean'] = result['d_mean']
        result['d3_std'] = result['d_std']
        result['d3_credible_interval'] = result['d_credible_interval']
        return result
    elif sampler == 'emcee':
        from sysmic.bayesian_d3 import bayesian_d3_inference as emcee_inference
        return emcee_inference(coordinates, **kwargs)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")


if __name__ == "__main__":
    print(r"""
  _____           _                       _   _             _   _             
 |  __ \         | |                     | | | |           | | | |            
 | |  | |   ___  | |_    ___    _ __     | |_| |__     ___ | |_| |_   _ __ ___
 | |  | |  / _ \ | __|  / _ \  | '_ \    | __| '_ \   / _ \| __| __| | '__/ _ \
 | |__| | | (_) || |_  | (_) | | | | |   | |_| | | | | (_) | |_| |_  | | |  __/
 |_____/   \___/  \__|  \___/  |_| |_|    \__|_| |_|  \___/ \__|\__| |_|  \___|
                                                                               
    """)
    np.random.seed(42)
    coords_2d = np.column_stack([np.random.rand(1000), np.random.rand(1000), np.zeros(1000)])
    
    result = bayesian_d2_inference(coords_2d, nlive=300, dlogz=0.5, verbose=True)
    print(f"D2 test: {result['d_mean']:.3f} (expected ~2.0)")
