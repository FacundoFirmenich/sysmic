"""
Heurística del Loco (Mad Heuristic) Module
==========================================
Subversive algorithm for asymmetric tail distribution exploration.

Philosophy:
-----------
Conventional statistics focus on central tendencies and bulk distributions.
The Mad Heuristic explicitly seeks the margins, the outliers, the extremes—
the 20% of data in distribution tails where seismic anomalies hide.

Key Principles:
---------------
1. **Tail Focus**: Exclusive exploration of distribution extremes (default 20%)
2. **Asymmetric Weighting**: Variable cola proportions (e.g., 70% right, 30% left)
3. **Inverse Transformation**: 1/x perspective shift for non-linear insight
4. **Subversive Sampling**: Rejection of gaussian assumptions and median bias

Applications in Seismology:
----------------------------
- Detect rare megaquakes (magnitude tail)
- Identify anomalous depth regimes (depth distribution extremes)
- Find spatial outliers (geographic clustering edges)
- Discover extreme D2 values (fractal dimension anomalies)

Author: Sysmic Framework v2.0
License: GPL-3.0
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
from scipy import stats
import warnings


class MadHeuristic:
    """
    The Mad Heuristic: Asymmetric Tail Distribution Explorer.

    Subverts conventional statistical paradigms by focusing exclusively on
    distribution extremes, using inverse transformations and asymmetric weighting.
    """

    def __init__(
        self,
        tail_proportion: float = 0.20,
        left_weight: float = 0.5,
        right_weight: float = 0.5,
        inverse_transform: bool = True,
        subversion_mode: str = "extreme",
    ):
        """
        Initialize the Mad Heuristic.

        Args:
            tail_proportion: Total proportion of distribution to explore (0.0-0.5)
                            Default 0.20 = 20% (10% each tail if symmetric)
            left_weight: Weight for left tail (0.0-1.0, must sum to 1.0 with right_weight)
            right_weight: Weight for right tail (0.0-1.0)
            inverse_transform: Apply 1/x transformation for perspective shift
            subversion_mode: 'extreme' (outliers only), 'marginal' (edges),
                            'chaotic' (random tail sampling)
        """
        # Validate inputs
        if not 0.0 < tail_proportion <= 0.5:
            raise ValueError("tail_proportion must be in (0, 0.5]")

        if not np.isclose(left_weight + right_weight, 1.0):
            raise ValueError(
                f"Weights must sum to 1.0, got {left_weight + right_weight}"
            )

        self.tail_proportion = tail_proportion
        self.left_weight = left_weight
        self.right_weight = right_weight
        self.inverse_transform = inverse_transform
        self.subversion_mode = subversion_mode

        # Compute actual tail sizes
        self.left_tail_size = tail_proportion * left_weight
        self.right_tail_size = tail_proportion * right_weight

    def explore_tails(
        self, data: np.ndarray, return_indices: bool = False
    ) -> Dict[str, Any]:
        """
        Execute mad heuristic exploration on data distribution.

        Args:
            data: 1D array of values to analyze
            return_indices: If True, return indices of tail samples

        Returns:
            Dictionary containing:
                - 'left_tail': Left tail values
                - 'right_tail': Right tail values
                - 'left_transformed': Inverse-transformed left tail (if enabled)
                - 'right_transformed': Inverse-transformed right tail (if enabled)
                - 'mad_score': Heuristic anomaly score
                - 'indices_left': Indices of left tail (if return_indices=True)
                - 'indices_right': Indices of right tail (if return_indices=True)
        """
        data = np.asarray(data).flatten()

        if len(data) < 10:
            raise ValueError("Insufficient data for tail analysis (need ≥10 points)")

        # Sort data and get percentiles
        sorted_data = np.sort(data)
        n = len(data)

        # Calculate tail boundaries
        left_cutoff_idx = int(n * self.left_tail_size)
        right_cutoff_idx = int(n * (1.0 - self.right_tail_size))

        # Extract tails
        left_tail = (
            sorted_data[:left_cutoff_idx] if left_cutoff_idx > 0 else np.array([])
        )
        right_tail = (
            sorted_data[right_cutoff_idx:] if right_cutoff_idx < n else np.array([])
        )

        results = {
            "left_tail": left_tail,
            "right_tail": right_tail,
            "left_boundary": sorted_data[left_cutoff_idx]
            if left_cutoff_idx < n
            else np.nan,
            "right_boundary": sorted_data[right_cutoff_idx]
            if right_cutoff_idx < n
            else np.nan,
        }

        # Apply inverse transformation if enabled
        if self.inverse_transform:
            results["left_transformed"] = self._inverse_transform(left_tail)
            results["right_transformed"] = self._inverse_transform(right_tail)

        # Compute Mad Score (anomaly intensity)
        results["mad_score"] = self._compute_mad_score(left_tail, right_tail, data)

        # Get indices if requested
        if return_indices:
            indices_left = (
                np.where(data <= results["left_boundary"])[0]
                if len(left_tail) > 0
                else np.array([], dtype=int)
            )
            indices_right = (
                np.where(data >= results["right_boundary"])[0]
                if len(right_tail) > 0
                else np.array([], dtype=int)
            )
            results["indices_left"] = indices_left
            results["indices_right"] = indices_right

        return results

    def _inverse_transform(self, tail: np.ndarray) -> np.ndarray:
        """
        Apply inverse transformation: f(x) = 1 / (x - x_min + ε)

        Shifts perspective to emphasize relative differences at extremes.
        """
        if len(tail) == 0:
            return np.array([])

        # Shift to avoid division by zero
        x_min = np.min(tail)
        epsilon = 1e-10
        shifted = tail - x_min + epsilon

        # Apply inverse
        transformed = 1.0 / shifted

        return transformed

    def _compute_mad_score(
        self, left_tail: np.ndarray, right_tail: np.ndarray, full_data: np.ndarray
    ) -> float:
        """
        Compute Mad Score: measure of tail extremity vs. bulk.

        Score > 1.0 indicates significant tail deviation.
        Score → 0.0 indicates gaussian-like distribution (boring).
        """
        # Median Absolute Deviation of full data
        median = np.median(full_data)
        mad_full = np.median(np.abs(full_data - median))

        if mad_full == 0:
            return 0.0

        # Tail deviations
        tail_vals = (
            np.concatenate([left_tail, right_tail])
            if len(left_tail) > 0 and len(right_tail) > 0
            else left_tail
            if len(left_tail) > 0
            else right_tail
        )

        if len(tail_vals) == 0:
            return 0.0

        tail_deviation = np.median(np.abs(tail_vals - median))

        # Mad Score: ratio of tail deviation to full MAD
        mad_score = tail_deviation / mad_full

        return float(mad_score)

    def subversive_filter(
        self, data: np.ndarray, threshold_sigma: float = 2.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subversive filtering: KEEP only tail outliers, REJECT bulk.

        This inverts the conventional outlier removal paradigm.

        Args:
            data: Input data
            threshold_sigma: Sigma threshold for "boring" (non-tail) detection

        Returns:
            (outliers, outlier_indices)
        """
        results = self.explore_tails(data, return_indices=True)

        # Combine tail indices
        left_idx = results.get("indices_left", np.array([], dtype=int))
        right_idx = results.get("indices_right", np.array([], dtype=int))
        outlier_indices = np.concatenate([left_idx, right_idx])

        outliers = data[outlier_indices]

        return outliers, outlier_indices

    def asymmetric_reweight(
        self, data: np.ndarray, emphasize: str = "right"
    ) -> np.ndarray:
        """
        Reweight data to emphasize one tail over the other.

        Args:
            data: Input data
            emphasize: 'left' or 'right' tail to emphasize

        Returns:
            Reweighted data array
        """
        results = self.explore_tails(data, return_indices=True)

        weights = np.ones_like(data)

        if emphasize == "right":
            right_idx = results.get("indices_right", np.array([], dtype=int))
            if len(right_idx) > 0:
                weights[right_idx] *= 1.0 / self.right_weight  # Amplify
        elif emphasize == "left":
            left_idx = results.get("indices_left", np.array([], dtype=int))
            if len(left_idx) > 0:
                weights[left_idx] *= 1.0 / self.left_weight  # Amplify
        else:
            raise ValueError("emphasize must be 'left' or 'right'")

        return data * weights


class SeismicMadAnalysis:
    """
    Apply Mad Heuristic to seismic catalogs for extreme event detection.
    """

    @staticmethod
    def detect_megaquake_regime(
        magnitudes: np.ndarray, tail_proportion: float = 0.10
    ) -> Dict[str, Any]:
        """
        Detect megaquake tail regime (right tail of magnitude distribution).

        Args:
            magnitudes: Array of earthquake magnitudes
            tail_proportion: Proportion to consider as "mega" (default 10%)

        Returns:
            Analysis results including megaquake threshold and statistics
        """
        mad = MadHeuristic(
            tail_proportion=tail_proportion,
            left_weight=0.0,  # Ignore left tail
            right_weight=1.0,  # Focus entirely on right tail
            inverse_transform=True,
        )

        results = mad.explore_tails(magnitudes, return_indices=True)

        megaquakes = results["right_tail"]
        megaquake_threshold = results["right_boundary"]

        return {
            "megaquake_threshold": megaquake_threshold,
            "megaquake_count": len(megaquakes),
            "megaquake_magnitudes": megaquakes,
            "megaquake_indices": results["indices_right"],
            "mad_score": results["mad_score"],
            "inverse_perspective": results.get("right_transformed", None),
        }

    @staticmethod
    def detect_depth_anomalies(
        depths: np.ndarray, asymmetry: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect anomalous shallow AND deep seismicity (both tails).

        Args:
            depths: Array of earthquake depths (km)
            asymmetry: Left/right weight (0.5 = symmetric, <0.5 = emphasize shallow)

        Returns:
            Analysis of shallow and deep anomalies
        """
        mad = MadHeuristic(
            tail_proportion=0.20,
            left_weight=asymmetry,
            right_weight=1.0 - asymmetry,
            inverse_transform=True,
        )

        results = mad.explore_tails(depths, return_indices=True)

        return {
            "shallow_anomalies": results["left_tail"],
            "deep_anomalies": results["right_tail"],
            "shallow_threshold": results["left_boundary"],
            "deep_threshold": results["right_boundary"],
            "shallow_indices": results.get("indices_left", []),
            "deep_indices": results.get("indices_right", []),
            "mad_score": results["mad_score"],
        }

    @staticmethod
    def detect_fractal_extremes(
        d2_values: np.ndarray, target: str = "both"
    ) -> Dict[str, Any]:
        """
        Detect extreme D2 values (anomalously low or high fractal dimensions).

        Args:
            d2_values: Array of fractal dimension estimates
            target: 'low' (planar), 'high' (volumetric), or 'both'

        Returns:
            Extreme D2 regimes
        """
        if target == "low":
            left_w, right_w = 1.0, 0.0
        elif target == "high":
            left_w, right_w = 0.0, 1.0
        else:  # both
            left_w, right_w = 0.5, 0.5

        mad = MadHeuristic(
            tail_proportion=0.15,
            left_weight=left_w,
            right_weight=right_w,
            inverse_transform=True,
        )

        results = mad.explore_tails(d2_values, return_indices=True)

        return {
            "planar_extreme": results["left_tail"],  # Low D2 (planar)
            "volumetric_extreme": results["right_tail"],  # High D2 (3D filling)
            "planar_threshold": results["left_boundary"],
            "volumetric_threshold": results["right_boundary"],
            "planar_indices": results.get("indices_left", []),
            "volumetric_indices": results.get("indices_right", []),
            "mad_score": results["mad_score"],
        }


# =============================================================================
# CHAOS MODE: Maximum Subversion
# =============================================================================


class ChaoticMadHeuristic(MadHeuristic):
    """
    Extreme variant: Chaotic tail sampling with non-deterministic selection.

    For when you want maximum subversion and don't trust ANY patterns.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subversion_mode = "chaotic"

    def chaotic_sample(
        self, data: np.ndarray, n_samples: int = 100, randomness: float = 0.3
    ) -> np.ndarray:
        """
        Chaotically sample from tails with intentional noise injection.

        Args:
            data: Input data
            n_samples: Number of samples to draw
            randomness: Noise injection level (0.0-1.0)

        Returns:
            Chaotic tail samples
        """
        results = self.explore_tails(data, return_indices=True)

        # Get tail indices
        left_idx = results.get("indices_left", np.array([], dtype=int))
        right_idx = results.get("indices_right", np.array([], dtype=int))
        tail_idx = np.concatenate([left_idx, right_idx])

        if len(tail_idx) == 0:
            return np.array([])

        # Chaotic sampling: mix deterministic + random
        n_deterministic = int(n_samples * (1.0 - randomness))
        n_random = n_samples - n_deterministic

        # Deterministic: most extreme
        deterministic_samples = (
            data[tail_idx][-n_deterministic:] if n_deterministic > 0 else np.array([])
        )

        # Random: totally unpredictable
        random_samples = (
            np.random.choice(data[tail_idx], n_random, replace=True)
            if n_random > 0
            else np.array([])
        )

        chaotic = np.concatenate([deterministic_samples, random_samples])

        return chaotic
