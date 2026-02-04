"""
================================================================================
FOURIER META-ANALYSIS - Dual Transform for Hypersystemic Integration
================================================================================
Part of Framework Hypersistémico 3A+\CAHTPhase

Implements dual Fourier transform with logic/counter-logic segmentation:
1. Direct FFT with dual segmentation
2. Consistent aggregation of resultants
3. Inverse transform [Σ(FFT)]^-1
4. Dual logic segmentation on reconstruction

Generates volumetric multifractal unified hypersurface.

Author: SFA Framework
Status: Core Component (Permanent)
================================================================================
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import fft
import warnings

__all__ = [
    'FourierDualResult',
    'dual_fourier_transform',
    'apply_logic_segmentation',
    'apply_counter_logic_segmentation',
    'volumetric_multifractal_synthesis'
]


@dataclass
class FourierDualResult:
    """
    Result from dual Fourier transform.
    
    Attributes:
        unified_surface: Reconstructed volumetric multifractal surface
        logic_component: Logic segmentation component
        counter_logic_component: Counter-logic segmentation component
        frequency_components: Frequency domain representation
        reconstruction_quality: Quality metric (0-1)
    """
    unified_surface: np.ndarray
    logic_component: np.ndarray
    counter_logic_component: np.ndarray
    frequency_components: np.ndarray
    reconstruction_quality: float


def apply_logic_segmentation(data: np.ndarray, 
                             threshold: Optional[float] = None) -> np.ndarray:
    """
    Apply logic segmentation to data.
    
    Logic segmentation: Separates coherent/structured components.
    
    Args:
        data: Input data (spatial or frequency domain)
        threshold: Optional threshold for segmentation
        
    Returns:
        Logic component
    """
    if threshold is None:
        # Auto-threshold: median + std
        threshold = np.nanmedian(np.abs(data)) + np.nanstd(np.abs(data))
    
    # Logic: Above threshold (structured/coherent)
    logic_mask = np.abs(data) >= threshold
    logic_component = np.where(logic_mask, data, 0)
    
    return logic_component


def apply_counter_logic_segmentation(data: np.ndarray,
                                     threshold: Optional[float] = None) -> np.ndarray:
    """
    Apply counter-logic segmentation to data.
    
    Counter-logic segmentation: Separates incoherent/noise components.
    
    Args:
        data: Input data (spatial or frequency domain)
        threshold: Optional threshold for segmentation
        
    Returns:
        Counter-logic component
    """
    if threshold is None:
        threshold = np.nanmedian(np.abs(data)) + np.nanstd(np.abs(data))
    
    # Counter-logic: Below threshold (noise/incoherent)
    counter_logic_mask = np.abs(data) < threshold
    counter_logic_component = np.where(counter_logic_mask, data, 0)
    
    return counter_logic_component


def dual_fourier_transform(subsets: List[Dict],
                           n_dimensions: int = 3) -> FourierDualResult:
    """
    Dual Fourier transform with logic/counter-logic segmentation.
    
    Process:
    1. Direct FFT on each subset (frequency domain)
    2. Apply logic/counter-logic segmentation
    3. Aggregate FFT resultants consistently
    4. Apply inverse transform [Σ]^-1
    5. Dual logic segmentation on reconstruction
    
    Args:
        subsets: Strategic subsets with data
        n_dimensions: Dimensionality for FFT (1, 2, or 3)
        
    Returns:
        FourierDualResult with volumetric multifractal unified hypersurface
    """
    print(f"\n[FOURIER DUAL] Processing {len(subsets)} subsets...")
    print(f"  Dimensions: {n_dimensions}D")
    
    # Step 1: Direct FFT with dual logic on each subset
    fft_results_logic = []
    fft_results_counter = []
    
    for i, subset in enumerate(subsets):
        data = subset.get('data')
        
        # Convert to array if needed
        if isinstance(data, pd.DataFrame):
            # Filter only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            data_array = numeric_data.values if len(numeric_data.columns) > 0 else np.array([[]])
        elif isinstance(data, np.ndarray):
            data_array = data
        else:
            warnings.warn(f"Subset {i}: Unknown data type, skipping")
            continue
        
        # Ensure correct dimensionality
        if data_array.ndim < n_dimensions:
            # Pad dimensions
            while data_array.ndim < n_dimensions:
                data_array = data_array[..., np.newaxis]
        elif data_array.ndim > n_dimensions:
            # Reduce dimensions (take first columns)
            data_array = data_array[:, :n_dimensions]
        
        # Apply FFT (N-dimensional)
        if n_dimensions == 1:
            fft_direct = fft.fft(data_array.flatten())
        elif n_dimensions == 2:
            fft_direct = fft.fft2(data_array[:, :2])
        else:  # 3D
            fft_direct = fft.fftn(data_array[:, :3])
        
        # Dual segmentation in frequency domain
        logic_seg = apply_logic_segmentation(fft_direct)
        counter_seg = apply_counter_logic_segmentation(fft_direct)
        
        fft_results_logic.append(logic_seg)
        fft_results_counter.append(counter_seg)
    
    print(f"  ✓ Direct FFT complete ({len(fft_results_logic)} subsets)")
    
    # Step 2: Consistent aggregation
    # Pad to common shape
    max_shape = max([arr.shape for arr in fft_results_logic], key=lambda s: np.prod(s))
    
    def pad_to_shape(arr, target_shape):
        """Pad array to target shape."""
        pad_width = [(0, max(0, t - s)) for t, s in zip(target_shape, arr.shape)]
        return np.pad(arr, pad_width, mode='constant', constant_values=0)
    
    # Pad and aggregate
    logic_padded = [pad_to_shape(arr, max_shape) for arr in fft_results_logic]
    counter_padded = [pad_to_shape(arr, max_shape) for arr in fft_results_counter]
    
    # Sum (consistent aggregation)
    aggregated_logic = np.sum(logic_padded, axis=0)
    aggregated_counter = np.sum(counter_padded, axis=0)
    aggregated_total = aggregated_logic + aggregated_counter
    
    print(f"  ✓ Aggregation complete (shape: {aggregated_total.shape})")
    
    # Step 3: Inverse transform [Σ]^-1
    if n_dimensions == 1:
        reconstructed = fft.ifft(aggregated_total)
    elif n_dimensions == 2:
        reconstructed = fft.ifft2(aggregated_total)
    else:
        reconstructed = fft.ifftn(aggregated_total)
    
    # Take real part (imaginary should be ~0 for real input)
    reconstructed_real = np.real(reconstructed)
    
    print(f"  ✓ Inverse transform complete")
    
    # Step 4: Dual segmentation on reconstructed signal
    final_logic = apply_logic_segmentation(reconstructed_real)
    final_counter = apply_counter_logic_segmentation(reconstructed_real)
    
    # Step 5: Quality metric
    # Reconstruction quality: correlation with frequency components
    freq_magnitude = np.abs(aggregated_total)
    recon_magnitude = np.abs(reconstructed)
    
    if freq_magnitude.size > 0 and recon_magnitude.size > 0:
        quality = np.corrcoef(freq_magnitude.flatten(), recon_magnitude.flatten())[0, 1]
        quality = np.abs(quality)  # Absolute correlation
    else:
        quality = 0.0
    
    print(f"  ✓ Dual segmentation complete")
    print(f"  Reconstruction quality: {quality:.4f}")
    
    result = FourierDualResult(
        unified_surface=reconstructed_real,
        logic_component=final_logic,
        counter_logic_component=final_counter,
        frequency_components=aggregated_total,
        reconstruction_quality=quality
    )
    
    return result


def volumetric_multifractal_synthesis(fourier_result: FourierDualResult,
                                      grid_resolution: int = 100) -> Dict[str, np.ndarray]:
    """
    Synthesize volumetric multifractal representation.
    
    Converts Fourier dual result into 3D volumetric grid for visualization
    and analysis.
    
    Args:
        fourier_result: Result from dual_fourier_transform
        grid_resolution: Resolution of 3D grid
        
    Returns:
        Dictionary with volumetric representations
    """
    print(f"\n[VOLUMETRIC SYNTHESIS] Grid resolution: {grid_resolution}³")
    
    # Create 3D grid
    unified = fourier_result.unified_surface
    
    # Reshape to 3D if needed
    if unified.ndim == 1:
        # 1D → 3D: Create cubic volume
        side = int(np.ceil(len(unified) ** (1/3)))
        padded = np.pad(unified, (0, side**3 - len(unified)), mode='constant')
        volume_unified = padded.reshape((side, side, side))
    elif unified.ndim == 2:
        # 2D → 3D: Extrude
        volume_unified = np.repeat(unified[:, :, np.newaxis], unified.shape[0], axis=2)
    else:
        # Already 3D
        volume_unified = unified
    
    # Interpolate to target resolution
    from scipy.ndimage import zoom
    
    current_shape = volume_unified.shape
    zoom_factors = [grid_resolution / s for s in current_shape]
    volume_resampled = zoom(volume_unified, zoom_factors, order=3)
    
    # Same for logic/counter-logic
    logic_vol = fourier_result.logic_component
    if logic_vol.ndim < 3:
        if logic_vol.ndim == 1:
            side = int(np.ceil(len(logic_vol) ** (1/3)))
            logic_vol = np.pad(logic_vol, (0, side**3 - len(logic_vol))).reshape((side, side, side))
        else:
            logic_vol = np.repeat(logic_vol[:, :, np.newaxis], logic_vol.shape[0], axis=2)
    
    logic_resampled = zoom(logic_vol, [grid_resolution / s for s in logic_vol.shape], order=3)
    
    print(f"  ✓ Volumetric synthesis complete")
    print(f"    Unified volume: {volume_resampled.shape}")
    print(f"    Value range: [{volume_resampled.min():.2e}, {volume_resampled.max():.2e}]")
    
    return {
        'unified_volume': volume_resampled,
        'logic_volume': logic_resampled,
        'grid_resolution': grid_resolution,
        'quality': fourier_result.reconstruction_quality
    }


if __name__ == "__main__":
    print("="*80)
    print("  FOURIER META-ANALYSIS - Framework Hypersistémico {3A+\\CAHTPhase}")
    print("  Dual Transform for Volumetric Multifractal Synthesis")
    print("="*80)
    
    # Example usage
    print("\n[Example] Dual Fourier transform on synthetic subsets...\n")
    
    # Create synthetic subsets
    subsets = []
    for i in range(3):
        data = pd.DataFrame(np.random.randn(50, 3) + i, columns=['x', 'y', 'z'])
        subsets.append({'data': data, 'id': f'subset_{i}'})
    
    # Apply dual Fourier transform
    fourier_result = dual_fourier_transform(subsets, n_dimensions=3)
    
    # Volumetric synthesis
    volumes = volumetric_multifractal_synthesis(fourier_result, grid_resolution=50)
    
    print(f"\n✅ Fourier Dual Transform Demonstrated")
    print(f"  Unified surface shape: {fourier_result.unified_surface.shape}")
    print(f"  Reconstruction quality: {fourier_result.reconstruction_quality:.4f}")
    print(f"  Volumetric grid: {volumes['unified_volume'].shape}")
    print("\n" + "="*80)
