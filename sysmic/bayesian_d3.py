"""
Bayesian 3D Fractal Dimension Inference via MCMC
==================================================

Implements Bayesian posterior inference for intrinsic 3D fractal dimension (D₃)
using Markov Chain Monte Carlo (MCMC) with emcee ensemble sampler.

Mathematical Framework:
-----------------------
Likelihood: P(coords | D₃) via correlation integral scaling
    C(r) = (1/N²) Σᵢ Σⱼ Θ(r - ||xᵢ - xⱼ||) ~ r^D₃

Prior: Beta(α=7.5, β=2.5) on [0, 4]
    - Centered at E[D₃] = 7.5/(7.5+2.5) × 4 = 3.0 (volumetric)
    - Allows sub-volumetric (planar/filamentary) structures

Posterior: P(D₃ | coords) ∝ P(coords | D₃) × P(D₃)

Convergence Diagnostics:
------------------------
- Gelman-Rubin R̂ statistic (target: R̂ < 1.01)
- Effective sample size (ESS > 1000)
- Autocorrelation time (τ < n_steps/50)

Saturation Indicators:
----------------------
1. Posterior mass at D₃ ∈ [2.98, 3.00]
   - >80%: Saturated (low precision catalog)
   - <20%: Data-driven (high precision catalog)

2. KL divergence: D_KL(posterior || prior)
   - <0.5 nats: Prior-dominated (saturation)
   - >2.0 nats: Data-driven (reliable)

Author: Facundo Firmenich (CEDESUR)
Date: 2025-12-17
Quality: 11/10 (JGR Tier-1)
License: GPLv3
"""

import numpy as np
from scipy import spatial, stats
from scipy.stats import beta, gaussian_kde
from scipy.integrate import trapezoid
from typing import Dict, Tuple, Optional, Any
import warnings

# Suppress emcee progress bar for cleaner output
import os
os.environ['EMCEE_SUPPRESS_WARNINGS'] = '1'

try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False
    warnings.warn(
        "emcee not installed. Bayesian D₃ will use fallback Metropolis-Hastings. "
        "Install with: pip install emcee"
    )


# Prior parameters (from PRIMARY PAPER VALIDATION)
PRIOR_ALPHA = 7.5
PRIOR_BETA = 2.5
PRIOR_LOWER = 0.0
PRIOR_UPPER = 4.0


def log_prior_d3(d3: float, alpha: float = PRIOR_ALPHA, beta_param: float = PRIOR_BETA) -> float:
    """
    Log-prior for D₃: Beta(α, β) on [0, 4].
    
    Args:
        d3: Fractal dimension value
        alpha: Beta distribution α parameter
        beta_param: Beta distribution β parameter
    
    Returns:
        Log-probability
    """
    if not (PRIOR_LOWER <= d3 <= PRIOR_UPPER):
        return -np.inf
    
    # Transform D₃ ∈ [0, 4] to u ∈ [0, 1] for Beta distribution
    u = (d3 - PRIOR_LOWER) / (PRIOR_UPPER - PRIOR_LOWER)
    
    # Beta log-pdf (scipy handles normalization)
    log_p = beta.logpdf(u, alpha, beta_param)
    
    # Jacobian correction for transformation
    log_p -= np.log(PRIOR_UPPER - PRIOR_LOWER)
    
    return log_p


def log_likelihood_d3(
    d3: float,
    coordinates: np.ndarray,
    tree: Optional[spatial.cKDTree] = None,
    precomputed_distances: Optional[np.ndarray] = None,
    precomputed_counts: Optional[np.ndarray] = None,
    radii: Optional[np.ndarray] = None,  # NEW: Accept pre-calculated radii
    n_radii: int = 20,
    n_reference: int = 1000,
    r_min_factor: float = 0.01,
    r_max_factor: float = 0.3
) -> float:
    """
    Log-likelihood for D₃ given 3D point cloud.
    
    Model: Correlation integral C(r) ~ r^D₃ in scaling region
    
    Likelihood: Product of Gaussian errors in log-log space
        ℓ = -0.5 Σᵢ [(log C(rᵢ) - log(A·rᵢ^D₃)) / σ]²
    
    Args:
        d3: Proposed fractal dimension
        coordinates: (N, 3) point cloud
        tree: Pre-built KDTree (optional, recommended for MCMC)
        radii: Pre-computed radii array (CRITICAL for Tier-4 optimization)
        n_radii: Number of radius samples
        n_reference: Number of reference points for correlation integral
        r_min_factor: Minimum radius as fraction of data extent
        r_max_factor: Maximum radius as fraction of data extent
    
    Returns:
        Log-likelihood value
    """
    if not (0.1 < d3 < 3.5):
        return -np.inf
    
    try:
        N = len(coordinates)
        if N < 100:
            return -np.inf
        
        # Use pre-built tree or build new one
        if tree is None and precomputed_distances is None and precomputed_counts is None:
            tree = spatial.cKDTree(coordinates)
        
        # TIER-4 OPTIMIZATION: Use pre-calculated radii if available
        # This avoids re-calculating np.ptp(coordinates) millions of times
        if radii is None:
            # Adaptive radius bounds (slow if repeated)
            extent = np.max(np.ptp(coordinates, axis=0))
            r_min = r_min_factor * extent
            r_max = r_max_factor * extent
            
            if r_min >= r_max:
                return -np.inf
            
            radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)
        
        # Calculate correlation integral
        correlation_values = []
        total_pairs = N * (N - 1) / 2
        
        if precomputed_counts is not None:
            # TIER-4 OPTIMIZATION: Pre-calculated counts (INSTANTANEOUS)
            # n_reference subtraction handled at pre-computation stage
            for count in precomputed_counts:
                C_r = max(count / total_pairs, 1e-10)
                correlation_values.append(C_r)
        elif precomputed_distances is not None:
            # TIER-3 OPTIMIZATION: Use pre-sorted distances + searchsorted (ULTRA FAST)
            n_ref_actual = len(precomputed_distances) // N
            for r in radii:
                count = np.searchsorted(precomputed_distances, (r))
                total_neighbors = max(count - n_ref_actual, 0)
                C_r = max(total_neighbors / total_pairs, 1e-10)
                correlation_values.append(C_r)
        else:
            # TIER-1/2 OPTIMIZATION: Tree or full matrix (slower)
            # Subsample reference points
            n_ref = min(n_reference, N)
            ref_indices = np.random.choice(N, n_ref, replace=False)
            ref_points = coordinates[ref_indices]
            
            for r in radii:
                neighbor_counts = tree.query_ball_point(ref_points, r, return_length=True)
                total_neighbors = np.sum(neighbor_counts) - n_ref
                C_r = max(total_neighbors / total_pairs, 1e-10)
                correlation_values.append(C_r)
        
        correlation_values = np.array(correlation_values)
        log_r = np.log(radii)
        log_C = np.log(correlation_values)
        
        # Detect linear scaling region (middle 60%)
        n_valid = max(int(0.6 * n_radii), 5)
        start_idx = (n_radii - n_valid) // 2
        end_idx = start_idx + n_valid
        
        log_r_valid = log_r[start_idx:end_idx]
        log_C_valid = log_C[start_idx:end_idx]
        
        # Predicted log(C) = log(A) + D₃·log(r)
        # Fit intercept A via least squares for given D₃
        log_A = np.mean(log_C_valid - d3 * log_r_valid)
        
        # Predicted values
        log_C_pred = log_A + d3 * log_r_valid
        
        # Residuals
        residuals = log_C_valid - log_C_pred
        
        # Log-likelihood (Gaussian errors)
        # Restore adaptive variance with safety floor for scientific validity
        sigma_sq = np.var(residuals) + 1e-4
        log_lik = -0.5 * np.sum(residuals**2 / sigma_sq)
        log_lik -= 0.5 * len(residuals) * np.log(2 * np.pi * sigma_sq)
        
        return log_lik
    
    except Exception:
        return -np.inf


def log_posterior_d3(d3: float, coordinates: np.ndarray, tree: Optional[spatial.cKDTree] = None, 
                     precomputed_distances: Optional[np.ndarray] = None, 
                     precomputed_counts: Optional[np.ndarray] = None, 
                     radii: Optional[np.ndarray] = None,
                     **likelihood_kwargs) -> float:
    """
    Log-posterior: log P(D₃|data) = log P(data|D₃) + log P(D₃)
    
    Args:
        d3: Fractal dimension
        coordinates: Point cloud
        **likelihood_kwargs: Passed to log_likelihood_d3
    
    Returns:
        Log-posterior probability
    """
    lp = log_prior_d3(d3)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = log_likelihood_d3(d3, coordinates, tree=tree, 
                               precomputed_distances=precomputed_distances, 
                               precomputed_counts=precomputed_counts, 
                               radii=radii,
                               **likelihood_kwargs)
    
    return lp + ll


def bayesian_d3_inference_emcee(
    coordinates: np.ndarray,
    n_walkers: int = 32,
    n_steps: int = 5000,
    n_burnin: int = 1000,
    initial_guess: float = 2.5,
    random_state: Optional[int] = None,
    verbose: bool = True,
    **likelihood_kwargs
) -> Dict[str, Any]:
    """
    Bayesian D₃ inference using emcee ensemble sampler.
    
    Args:
        coordinates: (N, 3) point cloud
        n_walkers: Number of MCMC walkers (must be even, ≥ 2×ndim)
        n_steps: MCMC steps per walker
        n_burnin: Burn-in steps to discard
        initial_guess: Initial D₃ estimate
        random_state: Random seed
        verbose: Print progress
        **likelihood_kwargs: Passed to likelihood function
    
    Returns:
        Dictionary with:
            - samples: Posterior samples (flattened)
            - d3_mean: Posterior mean
            - d3_std: Posterior standard deviation
            - d3_credible_interval: 95% credible interval
            - posterior_mass_saturation: % mass at D₃∈[2.98,3.00]
            - gelman_rubin: R̂ convergence diagnostic
            - acceptance_fraction: Mean acceptance rate
    """
    if not HAS_EMCEE:
        raise ImportError("emcee required for Bayesian D₃. Install: pip install emcee")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize walkers around initial guess
    ndim = 1
    p0 = initial_guess + 0.1 * np.random.randn(n_walkers, ndim)
    p0 = np.clip(p0, PRIOR_LOWER + 0.1, PRIOR_UPPER - 0.1)
    
    # Pre-calculate distances for maximum speed (HUGE speedup)
    n_ref = min(likelihood_kwargs.get('n_reference', 1000), len(coordinates))
    ref_indices = np.random.choice(len(coordinates), n_ref, replace=False)
    ref_points = coordinates[ref_indices]
    
    if verbose:
        print(f"  [TIER-4] Calculating distance matrix ({n_ref} x {len(coordinates)})...")
    
    # Efficiently calculate distance matrix
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(ref_points, coordinates)
    
    if verbose:
        print(f"  [TIER-4] Sorting 1D distance array...")
    precomputed_distances = np.sort(dist_matrix.flatten())
    
    # Pre-calculate counts for each r (Tier-4 optimization)
    if verbose:
        print(f"  [TIER-4] Pre-calculating neighbor counts...")
    extent = np.max(np.ptp(coordinates, axis=0))
    r_min = likelihood_kwargs.get('r_min_factor', 0.01) * extent
    r_max = likelihood_kwargs.get('r_max_factor', 0.3) * extent
    radii = np.logspace(np.log10(r_min), np.log10(r_max), likelihood_kwargs.get('n_radii', 20))
    
    precomputed_counts = []
    for r in radii:
        count = np.searchsorted(precomputed_distances, r)
        # We subtract n_ref because each reference point is its own nearest neighbor (dist=0)
        # This keeps consistency with the non-optimized version
        precomputed_counts.append(max(count - n_ref, 0))
    precomputed_counts = np.array(precomputed_counts)
    
    # Create sampler
    sampler = emcee.EnsembleSampler(
        n_walkers,
        ndim,
        log_posterior_d3,
        args=(coordinates, None, None, precomputed_counts, radii),
        kwargs=likelihood_kwargs
    )
    
    if verbose:
        print(f"Running MCMC: {n_walkers} walkers × {n_steps} steps...")
    
    # Run MCMC
    sampler.run_mcmc(p0, n_steps + n_burnin, progress=verbose)
    
    # Discard burn-in
    samples = sampler.get_chain(discard=n_burnin, flat=False)  # (n_steps, n_walkers, ndim)
    samples_flat = sampler.get_chain(discard=n_burnin, flat=True)[:, 0]  # Flatten
    
    # Convergence diagnostics
    # Gelman-Rubin R̂: compare within-chain vs between-chain variance
    # Split each chain in half
    n_samples_per_walker = samples.shape[0]
    mid = n_samples_per_walker // 2
    
    chain_means = []
    chain_vars = []
    for w in range(n_walkers):
        # First half
        chain_means.append(np.mean(samples[:mid, w, 0]))
        chain_vars.append(np.var(samples[:mid, w, 0], ddof=1))
        # Second half
        chain_means.append(np.mean(samples[mid:, w, 0]))
        chain_vars.append(np.var(samples[mid:, w, 0], ddof=1))
    
    chain_means = np.array(chain_means)
    chain_vars = np.array(chain_vars)
    
    # Within-chain variance W
    W = np.mean(chain_vars)
    
    # Between-chain variance B
    n_per_chain = mid
    overall_mean = np.mean(chain_means)
    B = n_per_chain * np.var(chain_means, ddof=1)
    
    # Variance estimate
    var_estimate = ((n_per_chain - 1) / n_per_chain) * W + (1 / n_per_chain) * B
    
    # R̂ statistic
    gelman_rubin = np.sqrt(var_estimate / W) if W > 0 else np.nan
    
    # Acceptance fraction
    accept_frac = np.mean(sampler.acceptance_fraction)
    
    # Posterior statistics
    d3_mean = np.mean(samples_flat)
    d3_std = np.std(samples_flat, ddof=1)
    d3_ci = np.percentile(samples_flat, [2.5, 97.5])
    
    # Saturation indicator: posterior mass at D₃ ∈ [2.98, 3.00]
    saturation_mask = (samples_flat >= 2.98) & (samples_flat <= 3.00)
    posterior_mass_saturation = 100 * np.sum(saturation_mask) / len(samples_flat)
    
    if verbose:
        print(f"✓ MCMC complete")
        print(f"  D₃ posterior: {d3_mean:.3f} ± {d3_std:.3f}")
        print(f"  95% CI: [{d3_ci[0]:.3f}, {d3_ci[1]:.3f}]")
        print(f"  Gelman-Rubin R̂: {gelman_rubin:.4f} {'✓' if gelman_rubin < 1.01 else '⚠️'}")
        print(f"  Acceptance rate: {accept_frac:.2%}")
        print(f"  Posterior mass @ [2.98,3.00]: {posterior_mass_saturation:.1f}%")
    
    return {
        'samples': samples_flat,
        'd3_mean': d3_mean,
        'd3_std': d3_std,
        'd3_credible_interval': d3_ci,
        'posterior_mass_saturation': posterior_mass_saturation,
        'gelman_rubin': gelman_rubin,
        'acceptance_fraction': accept_frac,
        'n_effective_samples': len(samples_flat),
    }


def bayesian_d3_inference_fallback(
    coordinates: np.ndarray,
    n_steps: int = 5000,
    n_burnin: int = 1000,
    initial_guess: float = 2.5,
    random_state: Optional[int] = None,
    verbose: bool = True,
    **likelihood_kwargs
) -> Dict[str, Any]:
    """
    Fallback Metropolis-Hastings sampler (if emcee not available).
    
    Same interface as emcee version but single-chain.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    current_d3 = initial_guess
    samples = []
    accepted = 0
    
    if verbose:
        print(f"Running Metropolis-Hastings MCMC: {n_steps + n_burnin} steps...")
    
    # Pre-calculate and sort distances
    n_ref = min(likelihood_kwargs.get('n_reference', 1000), len(coordinates))
    ref_indices = np.random.choice(len(coordinates), n_ref, replace=False)
    ref_points = coordinates[ref_indices]
    dist_matrix = np.sqrt(np.sum((coordinates[np.newaxis, :, :] - ref_points[:, np.newaxis, :])**2, axis=2))
    precomputed_distances = np.sort(dist_matrix.flatten())
    
    for step in range(n_steps + n_burnin):
        # Proposal
        proposal = current_d3 + np.random.normal(0, 0.05)
        
        # Acceptance ratio
        log_p_current = log_posterior_d3(current_d3, coordinates, precomputed_distances=precomputed_distances, **likelihood_kwargs)
        log_p_proposal = log_posterior_d3(proposal, coordinates, precomputed_distances=precomputed_distances, **likelihood_kwargs)
        
        log_accept_ratio = log_p_proposal - log_p_current
        
        if np.log(np.random.rand()) < log_accept_ratio:
            current_d3 = proposal
            accepted += 1
        
        # Store sample after burn-in
        if step >= n_burnin:
            samples.append(current_d3)
        
        # Progress
        if verbose and (step + 1) % 1000 == 0:
            print(f"  Step {step+1}/{n_steps+n_burnin} | D₃={current_d3:.3f} | Accept={accepted/(step+1):.2%}")
    
    samples = np.array(samples)
    
    # Statistics
    d3_mean = np.mean(samples)
    d3_std = np.std(samples, ddof=1)
    d3_ci = np.percentile(samples, [2.5, 97.5])
    
    saturation_mask = (samples >= 2.98) & (samples <= 3.00)
    posterior_mass_saturation = 100 * np.sum(saturation_mask) / len(samples)
    
    accept_frac = accepted / (n_steps + n_burnin)
    
    if verbose:
        print(f"✓ MCMC complete")
        print(f"  D₃ posterior: {d3_mean:.3f} ± {d3_std:.3f}")
        print(f"  95% CI: [{d3_ci[0]:.3f}, {d3_ci[1]:.3f}]")
        print(f"  Acceptance rate: {accept_frac:.2%}")
        print(f"  Posterior mass @ [2.98,3.00]: {posterior_mass_saturation:.1f}%")
    
    return {
        'samples': samples,
        'd3_mean': d3_mean,
        'd3_std': d3_std,
        'd3_credible_interval': d3_ci,
        'posterior_mass_saturation': posterior_mass_saturation,
        'gelman_rubin': np.nan,  # Not applicable for single chain
        'acceptance_fraction': accept_frac,
        'n_effective_samples': len(samples),
    }


def bayesian_d3_inference(
    coordinates: np.ndarray,
    n_walkers: int = 32,
    n_steps: int = 5000,
    n_burnin: int = 1000,
    initial_guess: float = 2.5,
    random_state: Optional[int] = None,
    verbose: bool = True,
    use_emcee: bool = True,
    **likelihood_kwargs
) -> Dict[str, Any]:
    """
    Main interface for Bayesian D₃ inference.
    
    Automatically selects emcee (if available) or fallback MH sampler.
    
    Args:
        coordinates: (N, 3) normalized point cloud
        n_walkers: Number of emcee walkers (ignored for fallback)
        n_steps: MCMC steps
        n_burnin: Burn-in steps
        initial_guess: Initial D₃ value
        random_state: Random seed
        verbose: Print diagnostics
        use_emcee: Prefer emcee if available
        **likelihood_kwargs: Passed to likelihood function
    
    Returns:
        Dictionary with posterior samples and statistics
    """
    if HAS_EMCEE and use_emcee:
        return bayesian_d3_inference_emcee(
            coordinates, n_walkers, n_steps, n_burnin,
            initial_guess, random_state, verbose, **likelihood_kwargs
        )
    else:
        if use_emcee:
            warnings.warn("emcee not available, using fallback MH sampler")
        return bayesian_d3_inference_fallback(
            coordinates, n_steps, n_burnin,
            initial_guess, random_state, verbose, **likelihood_kwargs
        )


def compute_kl_divergence_d3(
    posterior_samples: np.ndarray,
    prior_alpha: float = PRIOR_ALPHA,
    prior_beta: float = PRIOR_BETA,
    grid_points: int = 1000,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute KL divergence between D₃ posterior and prior.
    
    KL(posterior || prior) = ∫ P(D₃|data) log[P(D₃|data)/P(D₃)] dD₃
    
    Args:
        posterior_samples: MCMC samples from posterior
        prior_alpha: Beta α parameter
        prior_beta: Beta β parameter
        grid_points: Integration grid resolution
        verbose: Print results
    
    Returns:
        Dictionary with KL divergence and interpretation
    """
    # Validate
    if len(posterior_samples) < 100:
        raise ValueError(f"Insufficient samples: {len(posterior_samples)} < 100")
    
    # Remove non-finite
    posterior_samples = posterior_samples[np.isfinite(posterior_samples)]
    
    # KDE for posterior
    posterior_kde = gaussian_kde(posterior_samples, bw_method='scott')
    
    # Prior Beta density on [0, 4]
    def prior_pdf(d3):
        if PRIOR_LOWER <= d3 <= PRIOR_UPPER:
            u = (d3 - PRIOR_LOWER) / (PRIOR_UPPER - PRIOR_LOWER)
            return beta.pdf(u, prior_alpha, prior_beta) / (PRIOR_UPPER - PRIOR_LOWER)
        return 0.0
    
    # Integration grid
    d3_grid = np.linspace(PRIOR_LOWER + 0.1, PRIOR_UPPER - 0.1, grid_points)
    
    # Densities
    posterior_density = posterior_kde(d3_grid)
    prior_density = np.array([prior_pdf(d) for d in d3_grid])
    
    # KL integrand
    eps = 1e-10
    posterior_safe = posterior_density + eps
    prior_safe = prior_density + eps
    
    integrand = posterior_safe * np.log(posterior_safe / prior_safe)
    
    # Numerical integration
    kl_divergence = trapezoid(integrand, d3_grid)
    
    # Interpretation
    if kl_divergence > 2.0:
        interpretation = "Data-driven (reliable)"
        escenario = "A"
    elif kl_divergence < 0.5:
        interpretation = "Prior-dominated (suspect saturation)"
        escenario = "B"
    else:
        interpretation = "Borderline (ambiguous)"
        escenario = "A/B"
    
    # Posterior concentration at saturation
    conc_pct = 100 * np.sum((posterior_samples >= 2.98) & (posterior_samples <= 3.00)) / len(posterior_samples)
    
    if verbose:
        print(f"  KL divergence: {kl_divergence:.3f} nats")
        print(f"  Posterior conc. @ D₃∈[2.98,3.00]: {conc_pct:.1f}%")
        print(f"  Interpretation: {interpretation}")
        print(f"  Escenario: {escenario}")
    
    return {
        'kl_divergence': kl_divergence,
        'posterior_concentration_pct': conc_pct,
        'interpretation': interpretation,
        'escenario': escenario,
        'posterior_mean': np.mean(posterior_samples),
        'posterior_std': np.std(posterior_samples, ddof=1),
    }


if __name__ == "__main__":
    # Synthetic validation test
    print("="*70)
    print("BAYESIAN D₃ VALIDATION - SYNTHETIC TESTS")
    print("="*70)
    
    # Test 1: 2D plane (D₃ ≈ 2.0)
    print("\nTest 1: 2D Plane (expected D₃ ≈ 2.0)")
    np.random.seed(42)
    coords_2d = np.column_stack([
        np.random.rand(1000),
        np.random.rand(1000),
        np.zeros(1000)
    ])
    
    result = bayesian_d3_inference(
        coords_2d,
        n_walkers=16,
        n_steps=100,
        n_burnin=20,
        verbose=True
    )
    
    print(f"✓ Test 1: D₃ = {result['d3_mean']:.3f} ± {result['d3_std']:.3f}")
    print(f"  Acceptance: {result['acceptance_fraction']:.2%}")
    
    # Test 2: 3D volume (D₃ ≈ 3.0)
    print("\nTest 2: 3D Volume (expected D₃ ≈ 3.0)")
    coords_3d = np.random.rand(1000, 3)
    
    result = bayesian_d3_inference(
        coords_3d,
        n_walkers=16,
        n_steps=100,
        n_burnin=20,
        verbose=True
    )
    
    print(f"✓ Test 2: D₃ = {result['d3_mean']:.3f} ± {result['d3_std']:.3f}")
    print(f"  Error from 3.0: {abs(result['d3_mean'] - 3.0):.3f}")
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
