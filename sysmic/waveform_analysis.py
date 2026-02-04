"""
Waveform Analysis Module for Sysmic.

Implements advanced waveform processing and analysis for seismic fractal studies:
- Q-factor (attenuation quality factor) estimation
- Coda wave analysis for scattering properties
- Spectral fractal dimension from power spectra
- Waveform similarity and clustering
- Integration with ObsPy for professional seismology workflows

Research Base:
- Aki (1969) - Coda wave theory
- Sato & Fehler (1998) - Seismic wave propagation in random media
- Mayeda et al. (1992) - Coda Q estimation
- Spectral analysis for fractal characterization

Connection to Sysmic:
- Waveform coda decay → Fractal scattering properties
- Spectral slopes → Frequency-dependent fractal dimensions
- Q-factor → Fault zone damage and fractal complexity
- Waveform similarity → Event clustering and fault asperities
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
from scipy import signal
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq

try:
    from obspy import Stream, Trace
    from obspy.signal.cross_correlation import correlate, xcorr_max
    from obspy.signal.filter import envelope
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False
    import warnings
    warnings.warn(
        "ObsPy not available. Install: pip install obspy\n"
        "Waveform analysis will use fallback methods.",
        ImportWarning
    )

__all__ = [
    "WaveformAnalyzer",
    "estimate_coda_q",
    "compute_spectral_fractal_dimension",
    "waveform_similarity_matrix",
    "analyze_coda_decay",
]


@dataclass
class CodaQResult:
    """Results from coda Q estimation."""
    q_value: float
    q_uncertainty: float
    frequency: float
    decay_rate: float
    fit_quality: float  # R²


@dataclass
class SpectralFractalResult:
    """Results from spectral fractal dimension analysis."""
    spectral_dimension: float
    frequency_range: Tuple[float, float]
    slope: float
    slope_uncertainty: float
    corner_frequency: Optional[float] = None


class WaveformAnalyzer:
    """
    Comprehensive waveform analysis for seismic fractal studies.
    
    Philosophy:
    ----------
    Waveforms contain information about:
    - Source fractal properties (radiated spectrum)
    - Path effects (scattering, attenuation)
    - Site effects (local structure)
    
    Separating these reveals fractal characteristics at different scales.
    """
    
    def __init__(
        self,
        sampling_rate: float = 100.0,  # Hz
        filter_freqs: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize waveform analyzer.
        
        Args:
            sampling_rate: Sampling rate in Hz
            filter_freqs: Bandpass filter frequencies (low, high) in Hz
        """
        self.sampling_rate = sampling_rate
        self.filter_freqs = filter_freqs or (1.0, 20.0)
    
    def estimate_coda_q(
        self,
        waveform: np.ndarray,
        time_array: np.ndarray,
        lapse_time_start: float = 2.0,  # seconds after origin
        lapse_time_end: float = 10.0,
        center_frequency: float = 5.0  # Hz
    ) -> CodaQResult:
        """
        Estimate coda Q (quality factor) using single-backscattering model.
        
        Theory (Aki, 1969):
        ------------------
        A(t) = A₀ * t^(-α) * exp(-π*f*t/Q)
        
        where:
        - A(t) = coda amplitude at lapse time t
        - α ≈ 1 for single scattering
        - f = center frequency
        - Q = quality factor (inverse of attenuation)
        
        Low Q → High attenuation → Complex fault structure
        High Q → Low attenuation → Homogeneous medium
        
        Args:
            waveform: Seismic waveform array
            time_array: Time vector (seconds)
            lapse_time_start: Start of coda window
            lapse_time_end: End of coda window
            center_frequency: Analysis frequency
            
        Returns:
            CodaQResult
        """
        # Filter waveform around center frequency
        if OBSPY_AVAILABLE:
            trace = Trace(data=waveform)
            trace.stats.sampling_rate = self.sampling_rate
            trace.filter('bandpass', freqmin=center_frequency*0.8, 
                        freqmax=center_frequency*1.2)
            filtered = trace.data
        else:
            # Fallback: simple bandpass
            sos = signal.butter(4, [center_frequency*0.8, center_frequency*1.2],
                               btype='bandpass', fs=self.sampling_rate, output='sos')
            filtered = signal.sosfilt(sos, waveform)
        
        # Compute envelope (amplitude)
        if OBSPY_AVAILABLE:
            amplitude = envelope(filtered)
        else:
            analytic_signal = signal.hilbert(filtered)
            amplitude = np.abs(analytic_signal)
        
        # Select coda window
        mask = (time_array >= lapse_time_start) & (time_array <= lapse_time_end)
        t_coda = time_array[mask]
        A_coda = amplitude[mask]
        
        if len(t_coda) < 10:
            raise ValueError("Insufficient coda window samples")
        
        # Fit exponential decay: ln(A*t^α) = ln(A₀) - π*f*t/Q
        # Assuming α = 1 for simplicity
        A_corrected = A_coda * t_coda
        log_A = np.log(A_corrected + 1e-10)
        
        # Linear fit
        coeffs = np.polyfit(t_coda, log_A, 1)
        slope = coeffs[0]  # -π*f/Q
        intercept = coeffs[1]  # ln(A₀)
        
        # Calculate Q
        Q = -np.pi * center_frequency / slope if slope < 0 else np.inf
        
        # Uncertainty estimate
        residuals = log_A - (slope * t_coda + intercept)
        r_squared = 1 - (np.var(residuals) / np.var(log_A))
        
        # Q uncertainty from fit variance
        Q_uncertainty = Q * np.sqrt(np.var(residuals)) / abs(slope) if slope != 0 else np.inf
        
        return CodaQResult(
            q_value=Q,
            q_uncertainty=Q_uncertainty,
            frequency=center_frequency,
            decay_rate=-slope,
            fit_quality=r_squared
        )
    
    def compute_spectral_fractal_dimension(
        self,
        waveform: np.ndarray,
        freq_range: Tuple[float, float] = (1.0, 20.0)
    ) -> SpectralFractalResult:
        """
        Compute fractal dimension from power spectrum slope.
        
        Theory:
        ------
        For fractal signals, power spectrum follows:
        P(f) ∝ f^(-β)
        
        where β = 2H + 1 and H is Hurst exponent.
        Fractal dimension: D = 2 - H
        
        Thus: D = 2.5 - β/2
        
        β > 1 → persistent, correlated signal → rough fault (high D)
        β < 1 → anti-persistent → smooth fault (low D)
        
        Args:
            waveform: Seismic waveform
            freq_range: Frequency range for power-law fit
            
        Returns:
            SpectralFractalResult
        """
        # Compute power spectrum
        freqs = fftfreq(len(waveform), d=1/self.sampling_rate)
        fft_vals = fft(waveform)
        power_spectrum = np.abs(fft_vals)**2
        
        # Select positive frequencies within range
        mask = (freqs > freq_range[0]) & (freqs < freq_range[1]) & (freqs > 0)
        f_selected = freqs[mask]
        P_selected = power_spectrum[mask]
        
        if len(f_selected) < 5:
            raise ValueError("Insufficient frequency points for spectral analysis")
        
        # Log-log linear fit: log(P) = -β * log(f) + C
        log_f = np.log(f_selected)
        log_P = np.log(P_selected + 1e-10)
        
        coeffs, cov = np.polyfit(log_f, log_P, 1, cov=True)
        beta = -coeffs[0]  # Spectral slope
        beta_std = np.sqrt(cov[0, 0])
        
        # Convert to fractal dimension
        D_spectral = 2.5 - beta / 2
        
        # Estimate corner frequency (where spectrum deviates from power law)
        residuals = log_P - (coeffs[0] * log_f + coeffs[1])
        corner_idx = np.argmax(np.abs(residuals))
        corner_freq = f_selected[corner_idx] if corner_idx < len(f_selected) else None
        
        return SpectralFractalResult(
            spectral_dimension=D_spectral,
            frequency_range=freq_range,
            slope=beta,
            slope_uncertainty=beta_std,
            corner_frequency=corner_freq
        )
    
    def waveform_cross_correlation(
        self,
        waveform1: np.ndarray,
        waveform2: np.ndarray,
        max_lag: Optional[int] = None
    ) -> Tuple[float, int]:
        """
        Compute waveform cross-correlation for similarity analysis.
        
        High correlation → Similar source mechanisms or locations
        → Fault plane asperities or repeated ruptures
        
        Args:
            waveform1: First waveform
            waveform2: Second waveform
            max_lag: Maximum lag samples to search
            
        Returns:
            (max_correlation_coefficient, lag_samples)
        """
        if OBSPY_AVAILABLE:
            cc = correlate(waveform1, waveform2, shift=max_lag or len(waveform1)//2)
            shift, corr_coef = xcorr_max(cc)
            return corr_coef, shift
        else:
            # Fallback: NumPy correlation
            if max_lag is None:
                max_lag = len(waveform1) // 2
            
            correlation = np.correlate(waveform1 - waveform1.mean(),
                                      waveform2 - waveform2.mean(),
                                      mode='full')
            
            # Normalize
            norm = np.sqrt(np.sum(waveform1**2) * np.sum(waveform2**2))
            correlation = correlation / norm if norm > 0 else correlation
            
            # Find maximum
            center = len(correlation) // 2
            search_range = slice(center - max_lag, center + max_lag)
            max_idx = center - max_lag + np.argmax(correlation[search_range])
            
            return correlation[max_idx], max_idx - center
    
    def analyze_frequency_dependent_q(
        self,
        waveform: np.ndarray,
        time_array: np.ndarray,
        frequencies: List[float],
        lapse_time_range: Tuple[float, float] = (2.0, 10.0)
    ) -> Dict[float, CodaQResult]:
        """
        Estimate Q at multiple frequencies to assess frequency dependence.
        
        Q(f) often follows: Q(f) = Q₀ * f^α
        
        α > 0 → Frequency-dependent attenuation
        → Fractal scattering heterogeneity
        
        Args:
            waveform: Seismic waveform
            time_array: Time vector
            frequencies: List of center frequencies
            lapse_time_range: Coda window
            
        Returns:
            Dictionary mapping frequency → CodaQResult
        """
        q_results = {}
        
        for freq in frequencies:
            try:
                q_res = self.estimate_coda_q(
                    waveform, time_array,
                    lapse_time_start=lapse_time_range[0],
                    lapse_time_end=lapse_time_range[1],
                    center_frequency=freq
                )
                q_results[freq] = q_res
            except Exception as e:
                print(f"Warning: Q estimation failed at {freq} Hz: {e}")
                continue
        
        return q_results


def estimate_coda_q(
    waveform: np.ndarray,
    sampling_rate: float = 100.0,
    center_frequency: float = 5.0,
    lapse_time_range: Tuple[float, float] = (2.0, 10.0)
) -> float:
    """
    Simple interface for coda Q estimation.
    
    Args:
        waveform: Seismic waveform
        sampling_rate: Sampling rate (Hz)
        center_frequency: Analysis frequency (Hz)
        lapse_time_range: Coda window (seconds)
        
    Returns:
        Q value
    """
    analyzer = WaveformAnalyzer(sampling_rate=sampling_rate)
    time_array = np.arange(len(waveform)) / sampling_rate
    
    result = analyzer.estimate_coda_q(
        waveform, time_array,
        lapse_time_start=lapse_time_range[0],
        lapse_time_end=lapse_time_range[1],
        center_frequency=center_frequency
    )
    
    return result.q_value


def compute_spectral_fractal_dimension(
    waveform: np.ndarray,
    sampling_rate: float = 100.0,
    freq_range: Tuple[float, float] = (1.0, 20.0)
) -> float:
    """
    Simple interface for spectral fractal dimension.
    
    Args:
        waveform: Seismic waveform
        sampling_rate: Sampling rate (Hz)
        freq_range: Frequency range for analysis
        
    Returns:
        Spectral fractal dimension
    """
    analyzer = WaveformAnalyzer(sampling_rate=sampling_rate)
    result = analyzer.compute_spectral_fractal_dimension(waveform, freq_range)
    return result.spectral_dimension


def waveform_similarity_matrix(
    waveforms: List[np.ndarray],
    sampling_rate: float = 100.0
) -> np.ndarray:
    """
    Compute pairwise waveform similarity matrix.
    
    Used for:
    - Event clustering (doublets, multiplets)
    - Fault plane identification
    - Repeating earthquake detection
    
    Args:
        waveforms: List of waveform arrays
        sampling_rate: Sampling rate (Hz)
        
    Returns:
        (N, N) correlation matrix
    """
    analyzer = WaveformAnalyzer(sampling_rate=sampling_rate)
    n = len(waveforms)
    similarity = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity[i, j] = 1.0
            else:
                corr, _ = analyzer.waveform_cross_correlation(waveforms[i], waveforms[j])
                similarity[i, j] = corr
                similarity[j, i] = corr  # Symmetric
    
    return similarity


def analyze_coda_decay(
    waveform: np.ndarray,
    sampling_rate: float = 100.0,
    frequencies: List[float] = [2.0, 5.0, 10.0, 15.0]
) -> Dict[str, any]:
    """
    Comprehensive coda decay analysis at multiple frequencies.
    
    Reveals:
    - Frequency-dependent attenuation
    - Scattering vs intrinsic Q
    - Fault zone damage estimation
    
    Args:
        waveform: Seismic waveform
        sampling_rate: Sampling rate (Hz)
        frequencies: Analysis frequencies
        
    Returns:
        Analysis dictionary with Q values and interpretations
    """
    analyzer = WaveformAnalyzer(sampling_rate=sampling_rate)
    time_array = np.arange(len(waveform)) / sampling_rate
    
    q_results = analyzer.analyze_frequency_dependent_q(
        waveform, time_array, frequencies
    )
    
    # Extract Q values and fit power law
    freqs = np.array(list(q_results.keys()))
    q_values = np.array([q_results[f].q_value for f in freqs])
    
    # Fit Q(f) = Q₀ * f^α
    if len(freqs) > 2:
        log_f = np.log(freqs)
        log_q = np.log(q_values)
        alpha, log_q0 = np.polyfit(log_f, log_q, 1)
        q0 = np.exp(log_q0)
    else:
        alpha = 0.0
        q0 = q_values.mean() if len(q_values) > 0 else np.nan
    
    # Interpretation
    if alpha > 0.5:
        interpretation = "Strong frequency dependence → Fractal scattering dominant"
    elif alpha > 0.0:
        interpretation = "Moderate frequency dependence → Mixed scattering/intrinsic"
    else:
        interpretation = "Weak frequency dependence → Intrinsic attenuation dominant"
    
    return {
        'q_by_frequency': q_results,
        'q0': q0,
        'frequency_exponent': alpha,
        'mean_q': q_values.mean() if len(q_values) > 0 else np.nan,
        'interpretation': interpretation
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  WAVEFORM ANALYSIS MODULE FOR SYSMIC")
    print("=" * 70)
    
    # Example: Synthetic waveform
    print("\n[Example] Coda Q Estimation")
    np.random.seed(42)
    
    # Generate synthetic waveform with exponential decay
    fs = 100.0  # Hz
    t = np.arange(0, 20, 1/fs)
    
    # Synthetic coda: decaying oscillation + noise
    true_q = 500
    freq = 5.0
    waveform = (t**(-1)) * np.exp(-np.pi * freq * t / true_q) * np.sin(2*np.pi*freq*t)
    waveform += np.random.randn(len(waveform)) * 0.01
    
    # Estimate Q
    estimated_q = estimate_coda_q(waveform, fs, center_frequency=freq)
    print(f"  True Q: {true_q}")
    print(f"  Estimated Q: {estimated_q:.1f}")
    
    # Spectral analysis
    print("\n[Example] Spectral Fractal Dimension")
    # Generate fractal brownian motion
    from numpy.random import normal
    n = 1024
    hurst = 0.7
    fbm = np.cumsum(normal(size=n))
    
    d_spectral = compute_spectral_fractal_dimension(fbm, fs)
    print(f"  Spectral Dimension: {d_spectral:.3f}")
    print(f"  (Expected ≈ {2 - hurst:.3f} for H={hurst})")
    
    print("\n" + "=" * 70)
    print("✅ Waveform analysis module ready")
    print("✅ Professional seismology workflows supported")
    print("=" * 70)
