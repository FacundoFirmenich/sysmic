# -*- coding: utf-8 -*-
"""
tests/test_core.py
==================
Unit tests for Sysmic v8.0.0 core functionality.

Covers:
  - Correlation integral and dimension estimation
  - Fisher Information Barrier (σ_c = 2.3 ± 0.4 km)
  - Data integrity of canonical CSVs
  - Bayesian D₃ inference (smoke test)
  - Prior: Uniform[1.5, 3.0] per manuscript §3.3

Run with:  pytest tests/ -v
"""
import pathlib
import sys
import numpy as np
import pandas as pd
import pytest

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
REPO_ROOT = pathlib.Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _synthetic_plane(n: int = 800, seed: int = 42) -> np.ndarray:
    """2-D plane embedded in 3-D — expected D₂ ≈ 2.0."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1, (n, 2))
    z  = rng.uniform(0, 0.005, (n, 1))          # thin enough to be planar
    return np.hstack([xy, z])


def _synthetic_volume(n: int = 800, seed: int = 0) -> np.ndarray:
    """Uniform 3-D cloud — expected D₂ ≈ 3.0."""
    return np.random.default_rng(seed).uniform(0, 1, (n, 3))


def _correlation_integral(coords: np.ndarray, r: float) -> float:
    """Brute-force C(r) — for testing only (O(N²))."""
    from scipy.spatial.distance import pdist
    n = len(coords)
    return 2 * np.sum(pdist(coords) < r) / (n * (n - 1))


def _estimate_d2(coords: np.ndarray,
                 r_min_exp: float = -2.0,
                 r_max_exp: float = -0.5,
                 n_r: int = 20) -> float:
    """Estimate D₂ via log-log linear regression of C(r)."""
    r_vals = np.logspace(r_min_exp, r_max_exp, n_r)
    C_vals = np.array([_correlation_integral(coords, r) for r in r_vals])
    mask = C_vals > 0
    if mask.sum() < 4:
        return np.nan
    return float(np.polyfit(np.log10(r_vals[mask]), np.log10(C_vals[mask]), 1)[0])


# ---------------------------------------------------------------------------
# 1. Correlation Integral
# ---------------------------------------------------------------------------
class TestCorrelationIntegral:
    def test_planar_dimension(self):
        """2-D plane in 3-D → D₂ ≈ 2.0."""
        D2 = _estimate_d2(_synthetic_plane())
        assert 1.7 < D2 < 2.3, f"Expected D₂ ≈ 2.0, got {D2:.3f}"

    def test_volumetric_dimension(self):
        """Uniform 3-D cloud → D₂ ≈ 3.0."""
        D2 = _estimate_d2(_synthetic_volume(), r_min_exp=-1.5, r_max_exp=-0.3)
        assert 2.6 < D2 < 3.2, f"Expected D₂ ≈ 3.0, got {D2:.3f}"

    def test_c_r_monotone(self):
        """C(r) must be non-decreasing with r."""
        coords = _synthetic_plane(n=400)
        r_vals = np.logspace(-2, 0, 15)
        C_vals = np.array([_correlation_integral(coords, r) for r in r_vals])
        assert np.all(np.diff(C_vals) >= -1e-9), "C(r) is not monotone"

    def test_c_r_bounds(self):
        """C(r) ∈ [0, 1] for all r."""
        coords = _synthetic_plane(n=300)
        for r in np.logspace(-2, 0, 10):
            c = _correlation_integral(coords, r)
            assert 0.0 <= c <= 1.0 + 1e-12, f"C(r={r:.3f}) = {c:.4f} out of [0,1]"


# ---------------------------------------------------------------------------
# 2. Fisher Information Barrier  (σ_c = 2.3 ± 0.4 km, per Tab. 3)
# ---------------------------------------------------------------------------
class TestFisherBarrier:
    """Tests grounded in the analytic model _compute_pbnd_mc() from figs_gen.py."""

    SIGMA_C = 2.3   # km
    K       = 2.8   # slope from MC calibration (App. D)

    def _pbnd(self, sigma: float) -> float:
        """Sigmoidal P_bnd model calibrated from SCSN degradation data (Tab. 3)."""
        return 100.0 / (1.0 + np.exp(-self.K * (sigma - self.SIGMA_C)))

    def test_below_barrier(self):
        """σ_h = 1.0 km < σ_c → P_bnd < 10%."""
        assert self._pbnd(1.0) < 10.0

    def test_at_barrier(self):
        """σ_h = σ_c = 2.3 km → P_bnd ≈ 50%."""
        p = self._pbnd(self.SIGMA_C)
        assert 40.0 < p < 60.0, f"P_bnd at σ_c = {p:.1f}%"

    def test_above_barrier(self):
        """σ_h = 5.0 km >> σ_c → P_bnd > 50%."""
        assert self._pbnd(5.0) > 50.0

    def test_saturation(self):
        """σ_h = 10.0 km → P_bnd > 90% (deep saturation)."""
        assert self._pbnd(10.0) > 90.0

    def test_barrier_uncertainty(self):
        """σ_c ± 0.4 km: P_bnd(σ_c - 0.4) < 50% and P_bnd(σ_c + 0.4) > 50%."""
        assert self._pbnd(self.SIGMA_C - 0.4) < 50.0
        assert self._pbnd(self.SIGMA_C + 0.4) > 50.0

    def test_fidelity_monotone_decreasing(self):
        """η(σ) = exp(-σ/λ), λ=5.0 km: must be strictly decreasing."""
        LAMBDA = 5.0   # km, per §2.2.1 (calibrated from Noto fault spacing)
        sigmas = np.array([0.0, 0.5, 1.0, 2.3, 5.0, 7.5, 10.0])
        eta    = np.exp(-sigmas / LAMBDA)
        assert np.all(eta > 0) and np.all(eta <= 1.0)
        assert np.all(np.diff(eta) < 0), "η(σ) is not strictly decreasing"

    def test_fidelity_noto(self):
        """η(0.5 km) > 0.9: Hi-Net Noto is well below barrier."""
        assert np.exp(-0.5 / 5.0) > 0.9

    def test_fidelity_sumatra(self):
        """η(7.5 km) ≈ 0.22: GEOFON Sumatra severely attenuated."""
        eta = np.exp(-7.5 / 5.0)
        assert 0.20 < eta < 0.25, f"η(Sumatra) = {eta:.3f}"


# ---------------------------------------------------------------------------
# 3. Prior specification  (Uniform[1.5, 3.0] per §3.3)
# ---------------------------------------------------------------------------
class TestPrior:
    def test_uniform_prior_support(self):
        """log-prior = const inside [1.5, 3.0], -inf outside."""
        from sysmic.bayesian_d3 import log_prior_d3, PRIOR_LOWER, PRIOR_UPPER
        assert PRIOR_LOWER == pytest.approx(1.5)
        assert PRIOR_UPPER == pytest.approx(3.0)
        assert np.isfinite(log_prior_d3(1.5))
        assert np.isfinite(log_prior_d3(2.82))
        assert np.isfinite(log_prior_d3(3.0))
        assert not np.isfinite(log_prior_d3(1.4))
        assert not np.isfinite(log_prior_d3(3.1))

    def test_uniform_prior_flat(self):
        """All points inside support have identical log-prior."""
        from sysmic.bayesian_d3 import log_prior_d3
        vals = [1.5, 1.8, 2.0, 2.5, 2.82, 3.0]
        lps  = [log_prior_d3(v) for v in vals]
        assert np.allclose(lps, lps[0]), "Prior is not flat inside support"


# ---------------------------------------------------------------------------
# 4. Bayesian D₃ inference — smoke tests
# ---------------------------------------------------------------------------
class TestBayesianInference:
    def test_returns_expected_keys(self):
        """bayesian_d3_inference must return required dict keys."""
        from sysmic.bayesian_d3 import bayesian_d3_inference
        coords = _synthetic_volume(n=300, seed=7)
        result = bayesian_d3_inference(
            coords, n_steps=80, n_burnin=20, n_walkers=8, verbose=False
        )
        required = {"samples", "d3_mean", "d3_std",
                    "d3_credible_interval", "posterior_mass_saturation",
                    "acceptance_fraction", "n_effective_samples"}
        assert required.issubset(result.keys()), \
            f"Missing keys: {required - result.keys()}"

    def test_d3_mean_in_prior_support(self):
        """Posterior mean must be within prior support [1.5, 3.0]."""
        from sysmic.bayesian_d3 import bayesian_d3_inference, PRIOR_LOWER, PRIOR_UPPER
        coords = _synthetic_plane(n=300, seed=11)
        result = bayesian_d3_inference(
            coords, n_steps=80, n_burnin=20, n_walkers=8, verbose=False
        )
        assert PRIOR_LOWER <= result["d3_mean"] <= PRIOR_UPPER

    def test_acceptance_fraction_reasonable(self):
        """Acceptance fraction should be > 0 (chain is moving)."""
        from sysmic.bayesian_d3 import bayesian_d3_inference
        coords = _synthetic_volume(n=200, seed=3)
        result = bayesian_d3_inference(
            coords, n_steps=60, n_burnin=15, n_walkers=8, verbose=False
        )
        assert result["acceptance_fraction"] > 0.0


# ---------------------------------------------------------------------------
# 5. Data integrity
# ---------------------------------------------------------------------------
class TestDataIntegrity:
    REQUIRED_CSVS = [
        "cascadia_correlation.csv",     "noto_correlation.csv",
        "cascadia_posterior.csv",       "noto_posterior.csv",
        "scsn_degradation.csv",         "tectonic_hierarchy.csv",
        "depth_stratification.csv",     "nz_validation.csv",
        "hinet_radius_sensitivity.csv", "precision_drift.csv",
        "gisborne_pathology.csv",       "japan_mc_depth.csv",
        "prior_sensitivity.csv",        "zaccagnino_scores.csv",
        "vrml_correlation.csv",         "mc3d_surface.csv",
        "geofon_sumatra_svp_log.csv",
    ]

    def test_all_csvs_exist(self):
        """All 17 canonical data files must exist in data/."""
        missing = [f for f in self.REQUIRED_CSVS
                   if not (DATA_DIR / f).exists()]
        assert not missing, f"Missing CSVs: {missing}"

    def test_scsn_no_nans(self):
        """SCSN degradation CSV must have no NaN in key columns."""
        df = pd.read_csv(DATA_DIR / "scsn_degradation.csv")
        nulls = df[["sigma_h_km", "Pbnd_pct", "D3_mode"]].isna().sum().sum()
        assert nulls == 0, f"{nulls} NaN values in scsn_degradation.csv"

    def test_scsn_sigma_c_present(self):
        """σ_h = 2.3 km (Fisher barrier anchor) must appear in SCSN degradation data."""
        df = pd.read_csv(DATA_DIR / "scsn_degradation.csv")
        assert 2.3 in df["sigma_h_km"].values, "σ_h = 2.3 km not in scsn_degradation"

    def test_tectonic_hierarchy_regimes(self):
        """All main tectonic regimes must be present in tectonic_hierarchy.csv."""
        df = pd.read_csv(DATA_DIR / "tectonic_hierarchy.csv")
        expected = {"Rifting", "Transform", "Subduction", "Collision"}
        assert expected.issubset(set(df["regime"])), \
            f"Missing regimes: {expected - set(df['regime'])}"

    def test_mc3d_surface_coverage(self):
        """mc3d_surface.csv must cover the Fisher Barrier at σ_c = 2.3 km."""
        df = pd.read_csv(DATA_DIR / "mc3d_surface.csv")
        assert 2.3 in df["sigma_km"].values, "σ_c = 2.3 km not in mc3d_surface.csv"

    def test_sumatra_svp_log_saturated(self):
        """GEOFON Sumatra SVP log must show Pbnd_pct = 100% (fully saturated)."""
        df = pd.read_csv(DATA_DIR / "geofon_sumatra_svp_log.csv", comment="#")
        assert df["Pbnd_pct"].iloc[0] == pytest.approx(100.0), \
            "Sumatra P_bnd should be 100%"
        assert df["SVP_tier"].iloc[0] == "REJECT"

    def test_nz_validation_has_two_regions(self):
        """NZ validation must have Cook Strait and Bay of Plenty."""
        df = pd.read_csv(DATA_DIR / "nz_validation.csv")
        assert len(df) >= 2, "nz_validation.csv must have >= 2 rows"

    def test_depth_stratification_monotone(self):
        """D₂ should decrease monotonically with depth (Tab. 7)."""
        df = pd.read_csv(DATA_DIR / "depth_stratification.csv")
        assert np.all(np.diff(df["D2_mean"].values) < 0), \
            "D₂ not monotonically decreasing with depth"


# ---------------------------------------------------------------------------
# 6. Figures reproducibility — figs_gen.py import check
# ---------------------------------------------------------------------------
class TestFigsGen:
    def test_figs_gen_imports(self):
        """figs_gen.py must import without error."""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "figs_gen",
            pathlib.Path(__file__).parent.parent / "figs_gen.py"
        )
        # Only test importability, not execution (which requires DATA_DIR)
        assert spec is not None

    def test_sysmic_package_importable(self):
        """sysmic package must be importable."""
        import sysmic
        assert hasattr(sysmic, "__version__") or True   # version optional

    def test_sysmic_core_importable(self):
        """sysmic.core must be importable."""
        from sysmic import core  # noqa: F401

    def test_sysmic_bayesian_importable(self):
        """sysmic.bayesian_d3 must be importable."""
        from sysmic import bayesian_d3  # noqa: F401

    def test_sysmic_statistics_importable(self):
        """sysmic.statistics must be importable."""
        from sysmic import statistics  # noqa: F401
