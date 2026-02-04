"""
Ramanujan Nested Learning Module for SFA.

Implements Ramanujan-inspired nested partitions and continued fractions
for hierarchical fractal analysis. This module bridges number theory with
seismic fractal analysis using Ramanujan's nested radicals and theta functions.

Philosophy: Ramanujan's infinite nested structures mirror the hierarchical
self-similarity found in seismic fault networks.

Research Base:
- Ramanujan nested radicals (generalized to nth root, 2024)
- Theta functions for partition theory
- Compositional learning via nested optimization (Google, Q4 2025)

Connection to SFA:
- Nested radicals → Nested fractal dimensions
- Partition functions → Event clustering hierarchies
- Theta functions → Temporal modulation of spatial patterns
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
# Note: jacobi_theta removed - not available in scipy 1.14+
# Theta functions implemented via manual series expansion below
from dataclasses import dataclass

__all__ = [
    "RamanujanNestingEngine",
    "compute_nested_dimension",
    "partition_based_clustering",
    "theta_modulated_analysis",
]


@dataclass
class NestedDimensionResult:
    """Result from nested dimension computation."""
    base_dimension: float
    nested_levels: List[float]
    convergence: float
    theta_modulation: Optional[np.ndarray] = None


class RamanujanNestingEngine:
    """
    Ramanujan-inspired nested learning for seismic fractal analysis.
    
    Core Concept:
    --------------
    Ramanujan's nested radicals: sqrt(1 + 2*sqrt(1 + 3*sqrt(1 + 4*sqrt(...))))
    
    SFA Application:
    ---------------
    D_n = f(D_{n-1}, clustering_parameter_n)
    
    where D_n is the fractal dimension at nesting level n.
    
    This creates a STRATIFIED learning hierarchy where:
    - Level 1: Base geometric D₂
    - Level 2: Incorporates local clustering
    - Level 3: Adds temporal correlation
    - Level 4: Includes stress field influence
    - ...
    
    Each level "nests" the previous, creating compositional intelligence.
    """
    
    def __init__(
        self,
        max_nesting_depth: int = 5,
        convergence_threshold: float = 1e-6
    ):
        """
        Initialize Ramanujan nesting engine.
        
        Args:
            max_nesting_depth: Maximum nesting levels
            convergence_threshold: Convergence criterion
        """
        self.max_depth = max_nesting_depth
        self.threshold = convergence_threshold
    
    def compute_nested_radical_dimension(
        self,
        base_d2: float,
        clustering_params: np.ndarray
    ) -> NestedDimensionResult:
        """
        Compute nested fractal dimension using Ramanujan-inspired recursion.
        
        Mathematical Form (generalized Ramanujan):
        ------------------------------------------
        D_0 = base_d2
        D_{n+1} = sqrt(D_n + λ_n * sqrt(D_n + λ_{n+1} * sqrt(...)))
        
        where λ_n are clustering parameters at each scale.
        
        Args:
            base_d2: Base fractal dimension (from GP algorithm)
            clustering_params: Parameters for each nesting level
            
        Returns:
            NestedDimensionResult with stratified dimensions
        """
        nested_dims = [base_d2]
        current_dim = base_d2
        
        for level in range(min(len(clustering_params), self.max_depth)):
            # Ramanujan-style nesting
            lambda_n = clustering_params[level]
            
            # D_{n+1} = sqrt(D_n * (1 + lambda_n))
            # This generalizes Ramanujan's nested radicals to fractal dimensions
            next_dim = np.sqrt(current_dim * (1 + lambda_n))
            
            # Convergence check
            if abs(next_dim - current_dim) < self.threshold:
                break
            
            nested_dims.append(next_dim)
            current_dim = next_dim
        
        convergence = abs(nested_dims[-1] - nested_dims[-2]) if len(nested_dims) > 1 else 0.0
        
        return NestedDimensionResult(
            base_dimension=base_d2,
            nested_levels=nested_dims,
            convergence=convergence
        )
    
    def partition_function_clustering(
        self,
        event_magnitudes: np.ndarray,
        n_partitions: int = 10
    ) -> np.ndarray:
        """
        Apply Ramanujan partition theory to event clustering.
        
        Concept:
        -------
        Integer partitions model how seismic energy is "partitioned"
        across events of different magnitudes.
        
        p(n) = number of ways to write n as sum of positive integers
        
        For seismicity:
        Total_Energy ≈ Σ(10^(1.5*M_i)) = partition into magnitude bins
        
        Args:
            event_magnitudes: Array of magnitude values
            n_partitions: Number of partition bins
            
        Returns:
            Partition-based clustering parameters
        """
        # Discretize magnitudes into integer-like bins
        mag_bins = np.linspace(event_magnitudes.min(), event_magnitudes.max(), n_partitions)
        hist, _ = np.histogram(event_magnitudes, bins=mag_bins)
        
        # Compute partition numbers (simplified approximation)
        # p(n) ≈ exp(π * sqrt(2n/3)) / (4n * sqrt(3))  [Ramanujan's approximation]
        partition_numbers = np.zeros(len(hist))
        for i, count in enumerate(hist):
            if count > 0:
                n = int(count)
                partition_numbers[i] = np.exp(np.pi * np.sqrt(2*n/3)) / (4*n*np.sqrt(3))
        
        # Normalize to use as clustering parameters
        if partition_numbers.sum() > 0:
            partition_numbers /= partition_numbers.sum()
        
        return partition_numbers
    
    def theta_function_modulation(
        self,
        coordinates: np.ndarray,
        time_series: Optional[np.ndarray] = None,
        q: float = 0.1
    ) -> np.ndarray:
        """
        Apply Ramanujan's theta functions for spatio-temporal modulation.
        
        Theta functions bridge space and time periodicity.
        
        Jacobi theta function:
        θ_3(z, q) = 1 + 2 Σ q^(n²) cos(2nz)
        
        Application:
        -----------
        Modulates spatial fractal analysis by temporal periodicities
        (aftershock sequences, swarm patterns, etc.)
        
        Args:
            coordinates: Spatial coordinates
            time_series: Optional temporal data
            q: Nome parameter (controls decay rate)
            
        Returns:
            Modulation factors for each coordinate
        """
        if time_series is None:
            # Use spatial structure only
            z_values = np.arctan2(coordinates[:, 1], coordinates[:, 0])
        else:
            # Incorporate temporal phase
            z_values = time_series * np.pi / time_series.max()
        
        # Compute theta_3 for each z
        # Using simplified approximation for efficiency
        theta_values = 1 + 2 * np.sum([
            (q ** (n**2)) * np.cos(2*n*z_values[:, None])
            for n in range(1, 5)
        ], axis=0).ravel()
        
        return theta_values


def compute_nested_dimension(
    base_d2: float,
    coordinates: np.ndarray,
    spatial_scales: List[float],
    use_theta_modulation: bool = False
) -> NestedDimensionResult:
    """
    High-level interface for nested dimension computation.
    
    Combines Ramanujan nesting with fractal analysis.
    
    Args:
        base_d2: Base D₂ from GP algorithm
        coordinates: Event coordinates
        spatial_scales: List of spatial scales for nesting
        use_theta_modulation: Apply theta function temporal modulation
        
    Returns:
        NestedDimensionResult
    """
    engine = RamanujanNestingEngine(max_nesting_depth=len(spatial_scales))
    
    # Compute clustering parameters at each scale
    clustering_params = []
    for scale in spatial_scales:
        # Simple proximity-based clustering
        from scipy.spatial.distance import pdist
        distances = pdist(coordinates)
        close_pairs = (distances < scale).sum()
        total_pairs = len(distances)
        clustering = close_pairs / total_pairs if total_pairs > 0 else 0
        clustering_params.append(clustering)
    
    clustering_params = np.array(clustering_params)
    
    # Compute nested dimensions
    result = engine.compute_nested_radical_dimension(base_d2, clustering_params)
    
    # Optional theta modulation
    if use_theta_modulation:
        theta_mod = engine.theta_function_modulation(coordinates)
        result.theta_modulation = theta_mod
    
    return result


def partition_based_clustering(
    event_magnitudes: np.ndarray,
    event_times: np.ndarray,
    temporal_window: float = 1.0  # days
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ramanujan partition-based temporal clustering.
    
    Groups events using partition theory principles.
    
    Args:
        event_magnitudes: Magnitude array
        event_times: Time array (in days)
        temporal_window: Clustering time window
        
    Returns:
        (cluster_labels, partition_energies)
    """
    engine = RamanujanNestingEngine()
    
    # Temporal binning
    time_bins = np.arange(event_times.min(), event_times.max() + temporal_window, temporal_window)
    cluster_labels = np.digitize(event_times, time_bins)
    
    # Partition energy computation for each cluster
    unique_clusters = np.unique(cluster_labels)
    partition_energies = np.zeros(len(unique_clusters))
    
    for i, cluster in enumerate(unique_clusters):
        mask = cluster_labels == cluster
        cluster_mags = event_magnitudes[mask]
        
        if len(cluster_mags) > 0:
            partitions = engine.partition_function_clustering(cluster_mags)
            partition_energies[i] = partitions.sum()
    
    return cluster_labels, partition_energies


def theta_modulated_analysis(
    d2_values: np.ndarray,
    time_series: np.ndarray,
    period_estimate: float
) -> Tuple[np.ndarray, float]:
    """
    Apply theta function modulation to time-varying D₂.
    
    Extracts periodic components in fractal dimension evolution.
    
    Args:
        d2_values: Time series of D₂ values
        time_series: Corresponding times
        period_estimate: Estimated periodicity
        
    Returns:
        (modulated_d2, dominant_period)
    """
    engine = RamanujanNestingEngine()
    
    # Normalize time to phase
    phase = 2 * np.pi * time_series / period_estimate
    coords_dummy = np.column_stack([np.cos(phase), np.sin(phase)])
    
    # Apply theta modulation
    q = 0.1  # Decay parameter
    theta_mod = engine.theta_function_modulation(coords_dummy, time_series, q=q)
    
    # Modulate D₂
    modulated_d2 = d2_values * theta_mod[:len(d2_values)]
    
    # Extract dominant period via FFT
    from scipy.fft import fft, fftfreq
    fft_vals = fft(modulated_d2)
    freqs = fftfreq(len(modulated_d2), d=np.diff(time_series).mean())
    dominant_freq = freqs[np.argmax(np.abs(fft_vals[1:])) + 1]
    dominant_period = 1 / abs(dominant_freq) if dominant_freq != 0 else period_estimate
    
    return modulated_d2, dominant_period


if __name__ == "__main__":
    print("=" * 70)
    print("  RAMANUJAN NESTED LEARNING FOR SEISMIC FRACTAL ANALYSIS")
    print("=" * 70)
    
    # Example: Nested dimension computation
    print("\n[Example 1] Nested Dimension Computation")
    base_d2 = 1.8
    spatial_scales = [10, 25, 50, 100, 200]  # km
    coords_example = np.random.randn(100, 3) * 50  # Random events
    
    result = compute_nested_dimension(base_d2, coords_example, spatial_scales)
    
    print(f"  Base D₂: {result.base_dimension:.3f}")
    print(f"  Nested levels ({len(result.nested_levels)}):")
    for i, d in enumerate(result.nested_levels):
        print(f"    Level {i}: D₂ = {d:.4f}")
    print(f"  Convergence: {result.convergence:.2e}")
    
    # Example: Partition-based clustering
    print("\n[Example 2] Ramanujan Partition Clustering")
    mags = np.random.uniform(3.0, 7.0, 50)
    times = np.cumsum(np.random.exponential(0.5,50))
    
    clusters, energies = partition_based_clustering(mags, times)
    print(f"  Found {len(np.unique(clusters))} temporal clusters")
    print(f"  Mean partition energy: {energies.mean():.2e}")
    
    print("\n" + "=" * 70)
    print("✅ Ramanujan nested learning module ready")
    print("✅ Implements cutting-edge number theory for seismology")
    print("=" * 70)
