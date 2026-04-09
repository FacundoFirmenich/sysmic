# PART 1: Core SFA Statistics
"""
Statistical analysis module for Sysmic Advanced Spatial Tomography.
Includes:
- Seismicity Analysis (Gutenberg-Richter)
- Spatial Statistics (Clark-Evans)
- Statistical Inference (Hypothesis testing)
- Bayesian Robustness (Hierarchical modeling)
"""

import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.cluster import DBSCAN
from typing import Dict, Any, Tuple, Optional


class SeismicityAnalysis:
    """Magnitude-frequency and completeness analysis."""

    @staticmethod
    def estimate_completeness_magnitude(
        magnitudes: np.ndarray,
    ) -> Tuple[float, float]:
        """Maximum curvature method for Mc estimation."""
        if len(magnitudes) == 0:
            return 0.0, 0.0

        bins = np.arange(
            np.floor(np.min(magnitudes)),
            np.ceil(np.max(magnitudes)) + 0.1,
            0.1,
        )
        hist, bin_edges = np.histogram(magnitudes, bins=bins)

        mc_index = np.argmax(hist)
        mc = bin_edges[mc_index] + 0.05
        completeness_fraction = np.sum(magnitudes >= mc) / len(magnitudes)

        return mc, completeness_fraction

    def compute_b_value(
        self,
        magnitudes: np.ndarray,
        completeness_magnitude: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Aki-Utsu MLE b-value estimation."""
        if completeness_magnitude is None:
            completeness_magnitude, _ = self.estimate_completeness_magnitude(magnitudes)

        complete_magnitudes = magnitudes[magnitudes >= completeness_magnitude]
        if len(complete_magnitudes) < 50:
            return np.nan, np.nan, completeness_magnitude

        mean_magnitude = np.mean(complete_magnitudes)
        # b = log10(e) / (mean - (Mc - delta/2))
        b_value = np.log10(np.exp(1)) / (
            mean_magnitude - (completeness_magnitude - 0.05)
        )
        b_uncertainty = b_value / np.sqrt(len(complete_magnitudes))

        return b_value, b_uncertainty, completeness_magnitude


class SpatialStatisticalAnalysis:
    """Clark-Evans nearest neighbor analysis."""

    @staticmethod
    def clark_evans_3d(coordinates: np.ndarray) -> float:
        """Three-Dimensional Clark-Evans clustering index."""
        if len(coordinates) < 10:
            return np.nan

        # Subsample for performance if needed
        if len(coordinates) > 5000:
            sample_indices = np.random.choice(len(coordinates), 5000, replace=False)
            coords_sample = coordinates[sample_indices]
        else:
            coords_sample = coordinates

        tree = spatial.cKDTree(coords_sample)
        distances, _ = tree.query(coords_sample, k=2)
        mean_nearest_neighbor = np.mean(distances[:, 1])

        minima = coords_sample.min(axis=0)
        maxima = coords_sample.max(axis=0)
        volume = np.prod(maxima - minima)

        if volume <= 0:
            return np.nan

        density = len(coords_sample) / volume
        expected_distance = 0.554 / (density ** (1 / 3))  # Diggle (2003)

        return mean_nearest_neighbor / expected_distance

    @staticmethod
    def analyze_planar_clustering(
        coordinates: np.ndarray, eps: float = 0.05, min_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Identify planar clusters using DBSCAN to validate multi-planar
        hypothesis.

        Args:
            coordinates: Normalized coordinates (N, 3).
            eps: The maximum distance between two samples for one to be
                 considered as in the neighborhood of the other.
            min_samples: The number of samples (or total weight) in a
                         neighborhood for a point to be considered as a core
                         point.

        Returns:
            Dictionary with cluster statistics.
        """
        if len(coordinates) < min_samples:
            return {"n_clusters": 0, "noise_ratio": 1.0}

        # Run DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        return {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": n_noise / len(coordinates),
            "labels": labels,
        }


class BayesianRobustness:
    """Hierarchical Bayesian inference for D2 estimation."""

    @staticmethod
    def hierarchical_bayesian_d2(
        bootstrap_estimates_by_region: Dict[str, np.ndarray],
        n_samples: int = 5000,
        n_warmup: int = 1000,
    ) -> Dict[str, Any]:
        """
        Hierarchical Bayesian model with Metropolis-Hastings sampling.
        Estimates global and regional means/variances.
        """
        region_names = list(bootstrap_estimates_by_region.keys())
        # n_regions = len(region_names) # Unused

        # Initialize parameters
        mu_global = 1.5
        tau_global = 0.5
        mu_regions = np.array(
            [np.mean(bootstrap_estimates_by_region[r]) for r in region_names]
        )
        sigma_regions = np.array(
            [np.std(bootstrap_estimates_by_region[r], ddof=1) for r in region_names]
        )

        # Storage
        posterior_mu_global = []
        posterior_mu_regions = {r: [] for r in region_names}

        # Metropolis-Hastings loop (Simplified for brevity/performance in
        # Python)
        # Note: In a full production system, we might use PyMC or Stan.
        # Here we implement a basic MH sampler as per the original validated
        # code.

        for iteration in range(n_warmup + n_samples):
            # 1. Update Global Mean
            proposal_mu_global = mu_global + np.random.normal(0, 0.1)

            # Likelihood of regions given global
            ll_current = -0.5 * np.sum((mu_regions - mu_global) ** 2 / tau_global**2)
            ll_proposal = -0.5 * np.sum(
                (mu_regions - proposal_mu_global) ** 2 / tau_global**2
            )

            # Prior on global mean (Normal(1.5, 1.0))
            prior_current = -0.5 * ((mu_global - 1.5) ** 2)
            prior_proposal = -0.5 * ((proposal_mu_global - 1.5) ** 2)

            if np.log(np.random.rand()) < (
                ll_proposal + prior_proposal - ll_current - prior_current
            ):
                mu_global = proposal_mu_global

            # 2. Update Region Means
            for i, region in enumerate(region_names):
                data = bootstrap_estimates_by_region[region]
                proposal_mu = mu_regions[i] + np.random.normal(0, 0.05)

                # Likelihood of data given region mean
                ll_data_curr = -0.5 * np.sum(
                    (data - mu_regions[i]) ** 2 / sigma_regions[i] ** 2
                )
                ll_data_prop = -0.5 * np.sum(
                    (data - proposal_mu) ** 2 / sigma_regions[i] ** 2
                )

                # Prior of region mean given global
                prior_reg_curr = -0.5 * (
                    (mu_regions[i] - mu_global) ** 2 / tau_global**2
                )
                prior_reg_prop = -0.5 * ((proposal_mu - mu_global) ** 2 / tau_global**2)

                if np.log(np.random.rand()) < (
                    ll_data_prop + prior_reg_prop - ll_data_curr - prior_reg_curr
                ):
                    mu_regions[i] = proposal_mu

            # Store samples
            if iteration >= n_warmup:
                posterior_mu_global.append(mu_global)
                for i, region in enumerate(region_names):
                    posterior_mu_regions[region].append(mu_regions[i])

        # Compute summary statistics
        credible_intervals = {}
        for region in region_names:
            samples = np.array(posterior_mu_regions[region])
            lower = np.percentile(samples, 2.5)
            upper = np.percentile(samples, 97.5)
            credible_intervals[region] = (lower, upper)

        return {
            "posterior_mu_global": np.array(posterior_mu_global),
            "posterior_mu_regions": {
                r: np.array(v) for r, v in posterior_mu_regions.items()
            },
            "credible_intervals_95": credible_intervals,
        }

    @staticmethod
    def morans_i_depth(coordinates: np.ndarray, k: int = 10) -> Tuple[float, float]:
        """
        Compute Moran's I spatial autocorrelation index for depth.
        Tests whether earthquake depths are spatially clustered.

        Args:
            coordinates: (N, 3) array with [x, y, depth]
            k: Number of nearest neighbors to consider

        Returns:
            Tuple of (Moran's_I, p_value)
        """
        from scipy import stats
        from sklearn.neighbors import NearestNeighbors

        coords = np.asarray(coordinates, float)
        n = len(coords)
        if n < k + 2:
            return np.nan, 1.0

        # Subsample for performance if dataset is large
        if n > 2500:
            idx = np.random.choice(n, 2500, replace=False)
            coords = coords[idx]
            n = len(coords)

        depths = coords[:, 2]

        # Build k-NN spatial weights
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
        dists, indices = nbrs.kneighbors(coords)

        # Create weighted adjacency matrix
        W = np.zeros((n, n), float)
        for i in range(n):
            neigh = indices[i, 1:]  # Exclude self
            w = 1.0 / (dists[i, 1:] + 1e-6)  # Inverse distance weights
            W[i, neigh] = w

        # Row-normalize weights
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        W = W / row_sums[:, None]

        # Compute Moran's I
        z = depths - depths.mean()
        z2 = np.sum(z**2)
        if z2 == 0:
            return np.nan, 1.0

        moran_i = n * np.sum(W * np.outer(z, z)) / (W.sum() * z2)

        # Z-score under null hypothesis of no spatial autocorrelation
        # Using exact variance from Cliff & Ord (1981), eq. 5.7
        EI = -1.0 / (n - 1)

        # Compute weight moments for exact variance
        W_sum = W.sum()
        S1 = 0.5 * np.sum((W + W.T) ** 2)
        W_rowsum = W.sum(axis=1)
        W_colsum = W.sum(axis=0)
        S2 = np.sum((W_rowsum + W_colsum) ** 2)

        # Exact variance under randomization assumption
        # VI = [n*S1 - 2*n*W² + 6*W_sum²] / [(n-1)(n+1)W_sum²] - EI²
        numerator = n * S1 - 2 * n * np.sum(W**2) + 6 * W_sum**2
        denominator = (n - 1) * (n + 1) * W_sum**2
        VI = (numerator / denominator) - EI**2

        z_score = (moran_i - EI) / np.sqrt(max(VI, 1e-10))
        p = 2 * (1.0 - stats.norm.cdf(abs(z_score)))

        return float(moran_i), float(p)


class StatisticalInference:
    """
    Effect size estimation and hypothesis testing for regional comparisons.
    """

    @staticmethod
    def hedges_g(
        mean1: float,
        std1: float,
        n1: int,
        mean2: float,
        std2: float,
        n2: int,
    ) -> float:
        """
        Compute Hedges' g effect size (bias-corrected Cohen's d).

        Args:
            mean1, std1, n1: Mean, std, and sample size for group 1
            mean2, std2, n2: Mean, std, and sample size for group 2

        Returns:
            Hedges' g effect size
        """
        # Pooled standard deviation
        pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        if pooled_var <= 0:
            return 0.0
        pooled_sd = np.sqrt(pooled_var)
        if pooled_sd == 0:
            return 0.0

        # Cohen's d
        d = (mean2 - mean1) / pooled_sd

        # Bias correction factor J
        df = n1 + n2 - 2
        if df > 1:
            J = 1.0 - 3.0 / (4.0 * df - 1.0)
        else:
            J = 1.0

        # Hedges' g
        g = d * J

        # Cap extreme values
        if abs(g) > 3.0:
            g = np.copysign(3.0, g)

        return float(g)

    @staticmethod
    def interpret_effect(g: float) -> str:
        """
        Interpret Hedges' g effect size magnitude.

        Args:
            g: Hedges' g value

        Returns:
            Verbal interpretation
        """
        a = abs(g)
        if a < 0.2:
            return "negligible"
        if a < 0.5:
            return "small"
        if a < 0.8:
            return "medium"
        return "large"

    @staticmethod
    def pairwise_comparisons(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Perform pairwise comparisons of D2 between all regions.

        Args:
            results: Dictionary mapping region name to results dict
                     (must contain 'd2_mean', 'd2_std', 'event_count')

        Returns:
            DataFrame with pairwise comparison results
        """
        regions = list(results.keys())
        rows = []

        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                r1_name = regions[i]
                r2_name = regions[j]

                r1 = results[r1_name]
                r2 = results[r2_name]

                # Skip if D2 not available
                if not (
                    np.isfinite(r1.get("d2_mean", np.nan))
                    and np.isfinite(r2.get("d2_mean", np.nan))
                ):
                    continue

                # Compute Hedges' g
                g = StatisticalInference.hedges_g(
                    r1["d2_mean"],
                    r1.get("d2_std", 0.1),
                    r1.get("event_count", 1000),
                    r2["d2_mean"],
                    r2.get("d2_std", 0.1),
                    r2.get("event_count", 1000),
                )

                rows.append(
                    {
                        "Comparison": f"{r2_name} vs {r1_name}",
                        "Region_1": r1_name,
                        "Region_2": r2_name,
                        "D2_Region_1": r1["d2_mean"],
                        "D2_Region_2": r2["d2_mean"],
                        "Delta_D2": r2["d2_mean"] - r1["d2_mean"],
                        "Hedges_g": g,
                        "Magnitude": StatisticalInference.interpret_effect(g),
                    }
                )

        return pd.DataFrame(rows)


# PART 2: Advanced Sysmic Statistics
"""
Advanced statistical analysis module for seismicity.
- Temporal Mc(t) analysis with sliding windows
- Vincenty geodetic distances (accurate for >200 km)
- ROC analysis for earthquake precursor skill scoring
- Uncertainty propagation Mc → b-value
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TemporalMcResult:
    """Results from temporal Mc analysis."""
    times: np.ndarray
    mc_values: np.ndarray
    mc_uncertainties: np.ndarray
    n_events: np.ndarray
    
    def to_dict(self) -> Dict:
        return {
            'times': self.times,
            'mc_values': self.mc_values,
            'mc_uncertainties': self.mc_uncertainties,
            'n_events': self.n_events
        }


class TemporalMcAnalysis:
    """
    Temporal analysis of magnitude of completeness.
    
    Implements sliding window estimation to detect temporal
    variations in network detection capability.
    """
    
    @staticmethod
    def sliding_window_mc(
        times: np.ndarray,
        magnitudes: np.ndarray,
        window_days: float = 1826.25,  # 5 years
        step_days: float = 365.25,  # 1 year
        method: str = 'maxc'  # Maximum curvature
    ) -> TemporalMcResult:
        """
        Compute Mc in sliding temporal windows.
        
        Args:
            times: Event times (decimal years or datetime)
            magnitudes: Event magnitudes
            window_days: Window size in days
            step_days: Step size in days
            method: 'maxc' (maximum curvature) or 'gft' (goodness-of-fit)
        
        Returns:
            TemporalMcResult with time series
        """
        # Convert to days if needed
        if times.max() < 3000:  # Likely decimal years
            times_days = times * 365.25
        else:
            times_days = times
        
        t_min = times_days.min()
        t_max = times_days.max()
        
        # Window centers
        window_centers = np.arange(
            t_min + window_days/2,
            t_max - window_days/2,
            step_days
        )
        
        mc_values = []
        mc_uncertainties = []
        n_events_list = []
        
        for t_center in window_centers:
            # Select events in window
            mask = (times_days >= t_center - window_days/2) & \
                   (times_days < t_center + window_days/2)
            
            window_mags = magnitudes[mask]
            n_events = len(window_mags)
            
            if n_events < 50:
                mc_values.append(np.nan)
                mc_uncertainties.append(np.nan)
                n_events_list.append(n_events)
                continue
            
            # Estimate Mc
            if method == 'maxc':
                mc, mc_unc = _estimate_mc_maxc(window_mags)
            else:  # GFT method
                mc, mc_unc = _estimate_mc_gft(window_mags)
            
            mc_values.append(mc)
            mc_uncertainties.append(mc_unc)
            n_events_list.append(n_events)
        
        return TemporalMcResult(
            times=window_centers / 365.25,  # Back to years
            mc_values=np.array(mc_values),
            mc_uncertainties=np.array(mc_uncertainties),
            n_events=np.array(n_events_list)
        )
    
    @staticmethod
    def detect_mc_shifts(
        mc_result: TemporalMcResult,
        threshold_sigma: float = 2.0
    ) -> List[Dict]:
        """
        Detect significant shifts in Mc time series.
        
        Uses change point detection via cumulative sum (CUSUM).
        
        Args:
            mc_result: Results from sliding_window_mc
            threshold_sigma: Detection threshold in standard deviations
        
        Returns:
            List of detected shifts with times and magnitudes
        """
        mc = mc_result.mc_values
        valid = np.isfinite(mc)
        
        if np.sum(valid) < 10:
            return []
        
        mc_valid = mc[valid]
        times_valid = mc_result.times[valid]
        
        # Standardize
        mc_mean = np.mean(mc_valid)
        mc_std = np.std(mc_valid)
        mc_z = (mc_valid - mc_mean) / (mc_std + 1e-6)
        
        # CUSUM
        cusum_pos = np.maximum.accumulate(
            np.maximum(0, mc_z - 0.5)
        )
        cusum_neg = np.maximum.accumulate(
            np.maximum(0, -mc_z - 0.5)
        )
        
        # Detect exceedances
        shifts = []
        exceedances = (cusum_pos > threshold_sigma) | \
                     (cusum_neg > threshold_sigma)
        
        if np.any(exceedances):
            change_indices = np.where(np.diff(exceedances.astype(int)) != 0)[0]
            
            for idx in change_indices:
                if idx > 0 and idx < len(mc_valid) - 1:
                    shifts.append({
                        'time': times_valid[idx],
                        'mc_before': mc_valid[idx-1],
                        'mc_after': mc_valid[idx+1],
                        'delta_mc': mc_valid[idx+1] - mc_valid[idx-1]
                    })
        
        return shifts


class VincentyDistance:
    """
    Accurate geodetic distances using Vincenty's formulae.
    
    More accurate than Haversine for distances >200 km.
    Reference: Vincenty, T. (1975). Survey Review, 23(176), 88-93.
    """
    
    # WGS84 ellipsoid parameters
    a = 6378137.0  # Semi-major axis (m)
    f = 1 / 298.257223563  # Flattening
    b = a * (1 - f)  # Semi-minor axis
    
    @classmethod
    def distance(
        cls,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        max_iterations: int = 200,
        tolerance: float = 1e-12
    ) -> float:
        """
        Calculate distance between two points on WGS84 ellipsoid.
        
        Args:
            lat1, lon1: First point (degrees)
            lat2, lon2: Second point (degrees)
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence criterion
        
        Returns:
            Distance in meters
        """
        # Convert to radians
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        # Reduced latitudes
        U1 = np.arctan((1 - cls.f) * np.tan(lat1_rad))
        U2 = np.arctan((1 - cls.f) * np.tan(lat2_rad))
        
        L = lon2_rad - lon1_rad
        lambda_prev = L
        
        sin_U1, cos_U1 = np.sin(U1), np.cos(U1)
        sin_U2, cos_U2 = np.sin(U2), np.cos(U2)
        
        for iteration in range(max_iterations):
            sin_lambda = np.sin(lambda_prev)
            cos_lambda = np.cos(lambda_prev)
            
            sin_sigma = np.sqrt(
                (cos_U2 * sin_lambda)**2 +
                (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lambda)**2
            )
            
            if sin_sigma == 0:
                return 0.0  # Coincident points
            
            cos_sigma = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lambda
            sigma = np.arctan2(sin_sigma, cos_sigma)
            
            sin_alpha = cos_U1 * cos_U2 * sin_lambda / sin_sigma
            cos_sq_alpha = 1 - sin_alpha**2
            
            if cos_sq_alpha != 0:
                cos_2sigma_m = cos_sigma - 2 * sin_U1 * sin_U2 / cos_sq_alpha
            else:
                cos_2sigma_m = 0
            
            C = cls.f / 16 * cos_sq_alpha * (4 + cls.f * (4 - 3 * cos_sq_alpha))
            
            lambda_new = L + (1 - C) * cls.f * sin_alpha * (
                sigma + C * sin_sigma * (
                    cos_2sigma_m + C * cos_sigma * (
                        -1 + 2 * cos_2sigma_m**2
                    )
                )
            )
            
            if abs(lambda_new - lambda_prev) < tolerance:
                break
            
            lambda_prev = lambda_new
        
        # Calculate distance
        u_sq = cos_sq_alpha * (cls.a**2 - cls.b**2) / cls.b**2
        A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
        B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
        
        delta_sigma = B * sin_sigma * (
            cos_2sigma_m + B / 4 * (
                cos_sigma * (-1 + 2 * cos_2sigma_m**2) -
                B / 6 * cos_2sigma_m * (-3 + 4 * sin_sigma**2) *
                (-3 + 4 * cos_2sigma_m**2)
            )
        )
        
        distance = cls.b * A * (sigma - delta_sigma)
        
        return distance
    
    @classmethod
    def distance_array(
        cls,
        lats1: np.ndarray,
        lons1: np.ndarray,
        lats2: np.ndarray,
        lons2: np.ndarray
    ) -> np.ndarray:
        """Vectorized distance calculation."""
        n = len(lats1)
        distances = np.zeros(n)
        
        for i in range(n):
            distances[i] = cls.distance(
                lats1[i], lons1[i], lats2[i], lons2[i]
            )
        
        return distances


class ROCPrecursorAnalysis:
    """
    ROC (Receiver Operating Characteristic) analysis for precursor skill.
    
    Evaluates whether a precursor signal (e.g., D₂ < threshold) has
    predictive skill above random chance.
    """
    
    @staticmethod
    def compute_roc_curve(
        precursor_values: np.ndarray,
        earthquake_occurred: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute ROC curve for precursor signal.
        
        Args:
            precursor_values: Precursor measurements (e.g., D₂ values)
            earthquake_occurred: Binary (1=earthquake, 0=no earthquake)
            thresholds: Custom thresholds (if None, auto-generate)
        
        Returns:
            Dict with TPR, FPR, thresholds, AUC
        """
        if thresholds is None:
            thresholds = np.linspace(
                precursor_values.min(),
                precursor_values.max(),
                100
            )
        
        tpr_list = []
        fpr_list = []
        
        n_positives = np.sum(earthquake_occurred == 1)
        n_negatives = np.sum(earthquake_occurred == 0)
        
        for thresh in thresholds:
            # Alarm if precursor < threshold (assuming lower is worse)
            alarm = precursor_values < thresh
            
            # True positives: alarm AND earthquake
            tp = np.sum(alarm & (earthquake_occurred == 1))
            
            # False positives: alarm but NO earthquake
            fp = np.sum(alarm & (earthquake_occurred == 0))
            
            # Rates
            tpr = tp / n_positives if n_positives > 0 else 0
            fpr = fp / n_negatives if n_negatives > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        tpr_array = np.array(tpr_list)
        fpr_array = np.array(fpr_list)
        
        # Compute AUC via trapezoidal rule
        # Sort by FPR
        sort_idx = np.argsort(fpr_array)
        fpr_sorted = fpr_array[sort_idx]
        tpr_sorted = tpr_array[sort_idx]
        
        auc = np.trapz(tpr_sorted, fpr_sorted)
        
        return {
            'tpr': tpr_array,
            'fpr': fpr_array,
            'thresholds': thresholds,
            'auc': auc
        }
    
    @staticmethod
    def optimal_threshold(
        roc_result: Dict,
        criterion: str = 'youden'
    ) -> Tuple[float, Dict]:
        """
        Find optimal threshold from ROC curve.
        
        Args:
            roc_result: Output from compute_roc_curve
            criterion: 'youden' (J=TPR-FPR) or 'closest' (to top-left)
        
        Returns:
            (optimal_threshold, metrics_at_threshold)
        """
        tpr = roc_result['tpr']
        fpr = roc_result['fpr']
        thresholds = roc_result['thresholds']
        
        if criterion == 'youden':
            # Youden's J statistic = TPR - FPR
            j_stats = tpr - fpr
            optimal_idx = np.argmax(j_stats)
        
        else:  # closest to (0, 1)
            distances = np.sqrt((1 - tpr)**2 + fpr**2)
            optimal_idx = np.argmin(distances)
        
        optimal_thresh = thresholds[optimal_idx]
        
        metrics = {
            'threshold': optimal_thresh,
            'tpr': tpr[optimal_idx],
            'fpr': fpr[optimal_idx],
            'tnr': 1 - fpr[optimal_idx],  # Specificity
            'precision': tpr[optimal_idx] / (tpr[optimal_idx] + fpr[optimal_idx])
                         if (tpr[optimal_idx] + fpr[optimal_idx]) > 0 else 0
        }
        
        return optimal_thresh, metrics
    
    @staticmethod
    def skill_score(auc: float) -> Dict:
        """
        Compute skill scores from AUC.
        
        Args:
            auc: Area under ROC curve
        
        Returns:
            Dict with skill metrics and interpretation
        """
        # Skill score relative to random (AUC=0.5)
        skill = 2 * (auc - 0.5)
        
        # Interpretation
        if auc < 0.55:
            interpretation = "No skill (random)"
        elif auc < 0.65:
            interpretation = "Poor"
        elif auc < 0.75:
            interpretation = "Fair"
        elif auc < 0.85:
            interpretation = "Good"
        elif auc < 0.95:
            interpretation = "Excellent"
        else:
            interpretation = "Outstanding"
        
        return {
            'auc': auc,
            'skill_score': skill,
            'interpretation': interpretation,
            'above_random': auc > 0.5
        }


class UncertaintyPropagation:
    """
    Propagate uncertainties from Mc to b-value estimation.
    
    Uses delta method and bootstrap for rigorous error bars.
    """
    
    @staticmethod
    def mc_to_bvalue_uncertainty(
        magnitudes: np.ndarray,
        mc: float,
        mc_uncertainty: float,
        method: str = 'delta'
    ) -> Tuple[float, float]:
        """
        Propagate Mc uncertainty to b-value.
        
        Args:
            magnitudes: Magnitude catalog
            mc: Estimated magnitude of completeness
            mc_uncertainty: Uncertainty in Mc
            method: 'delta' (analytical) or 'bootstrap'
        
        Returns:
            (b_value, b_uncertainty)
        """
        # Filter by Mc
        mags_complete = magnitudes[magnitudes >= mc]
        
        if len(mags_complete) < 50:
            return np.nan, np.nan
        
        # Aki-Utsu MLE
        mean_mag = np.mean(mags_complete)
        b_value = 1.0 / (np.log(10) * (mean_mag - mc))
        
        if method == 'delta':
            # Delta method: Var(b) ≈ (db/dMc)² Var(Mc) + (db/dM)² Var(M)
            n = len(mags_complete)
            
            # Variance of mean magnitude
            var_mean = np.var(mags_complete, ddof=1) / n
            
            # Partial derivatives
            db_dmc = b_value**2 * np.log(10)
            db_dmean = -1.0 / (np.log(10) * (mean_mag - mc)**2)
            
            # Total variance
            var_b = (db_dmc**2 * mc_uncertainty**2 + 
                    db_dmean**2 * var_mean)
            
            b_uncertainty = np.sqrt(var_b)
        
        else:  # bootstrap
            n_boot = 200
            b_boot = np.zeros(n_boot)
            
            for i in range(n_boot):
                # Resample magnitudes
                boot_mags = np.random.choice(
                    mags_complete, 
                    size=len(mags_complete),
                    replace=True
                )
                
                # Perturb Mc
                mc_boot = mc + np.random.normal(0, mc_uncertainty)
                
                # Recompute b
                boot_complete = boot_mags[boot_mags >= mc_boot]
                if len(boot_complete) > 10:
                    mean_boot = np.mean(boot_complete)
                    b_boot[i] = 1.0 / (np.log(10) * (mean_boot - mc_boot))
                else:
                    b_boot[i] = np.nan
            
            b_uncertainty = np.nanstd(b_boot, ddof=1)
        
        return b_value, b_uncertainty


# Helper functions

def _estimate_mc_maxc(magnitudes: np.ndarray) -> Tuple[float, float]:
    """Maximum curvature method for Mc estimation."""
    # Bin magnitudes
    bins = np.arange(
        magnitudes.min(),
        magnitudes.max() + 0.1,
        0.1
    )
    
    hist, bin_edges = np.histogram(magnitudes, bins=bins)
    
    # Find maximum
    max_idx = np.argmax(hist)
    mc = bin_edges[max_idx]
    
    # Uncertainty: bootstrap
    n_boot = 50
    mc_boot = np.zeros(n_boot)
    
    for i in range(n_boot):
        boot_mags = np.random.choice(
            magnitudes,
            size=len(magnitudes),
            replace=True
        )
        hist_boot, _ = np.histogram(boot_mags, bins=bins)
        mc_boot[i] = bin_edges[np.argmax(hist_boot)]
    
    mc_unc = np.std(mc_boot, ddof=1)
    
    return mc, mc_unc


def _estimate_mc_gft(magnitudes: np.ndarray) -> Tuple[float, float]:
    """Goodness-of-fit test method for Mc."""
    # Simplified GFT method
    # In practice, would need full Wiemer & Wyss (2000) implementation
    
    mc_candidates = np.arange(
        magnitudes.min(),
        np.percentile(magnitudes, 90),
        0.1
    )
    
    best_mc = mc_candidates[len(mc_candidates)//2]
    best_r = -1
    
    for mc_test in mc_candidates:
        complete = magnitudes[magnitudes >= mc_test]
        
        if len(complete) < 50:
            continue
        
        # Fit exponential
        mean_mag = np.mean(complete - mc_test)
        b_test = 1.0 / (np.log(10) * mean_mag)
        
        # Theoretical distribution
        theo_cdf = 1 - 10**(-b_test * (complete - mc_test))
        
        # Empirical CDF
        emp_cdf = np.arange(1, len(complete) + 1) / len(complete)
        
        # R correlation
        r = np.corrcoef(theo_cdf, emp_cdf)[0, 1]
        
        if r > best_r:
            best_r = r
            best_mc = mc_test
    
    # Uncertainty: ±0.2 typical
    mc_unc = 0.2
    
    return best_mc, mc_unc
