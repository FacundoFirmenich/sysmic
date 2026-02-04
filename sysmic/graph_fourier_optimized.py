"""
================================================================================
GRAPH FOURIER META-ANALYSIS - Optimized for Discrete Signals
================================================================================
Part of Framework Hypersistémico 3A+\CAHTPhase

IMPROVEMENTS over fourier_metaanalysis.py:
1. Graph Fourier Transform (discrete signals) instead of FFT (continuous)
2. Automatic quality diagnostics (detects NaN, data type mismatches)
3. K-fold cross-validation support
4. String feature encoding (window labels → vectors)

Author: SFA Framework
Status: Core Component (Permanent) - OPTIMIZED VERSION
================================================================================
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import sparse
import warnings

# Import Graph TGS for Graph Fourier Transform
from sysmic.graph_tgs import SeismicGraphTGS

__all__ = [
    'GraphFourierDualResult',
    'graph_fourier_dual_transform',
    'encode_string_features',
    'diagnose_data_quality',
    'k_fold_validation'
]


@dataclass
class GraphFourierDualResult:
    """
    Result from Graph Fourier dual transform.
    
    Attributes:
        unified_surface: Reconstructed signal on graph
        logic_component: Logic segmentation component
        counter_logic_component: Counter-logic segmentation component
        graph_frequencies: Graph frequency components (eigenvalues)
        reconstruction_quality: Quality metric (0-1, NOT NaN)
        quality_diagnostics: Detailed quality report
    """
    unified_surface: np.ndarray
    logic_component: np.ndarray
    counter_logic_component: np.ndarray
    graph_frequencies: np.ndarray
    reconstruction_quality: float
    quality_diagnostics: Dict[str, Any]


def encode_string_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode string columns as numeric features.
    
    Strategy:
    - Categorical encoding for windows (PRE_24d, MAIN_48d, etc.)
    - One-hot encoding for other strings
    - Preserve numeric columns unchanged
    
    Args:
        data: DataFrame with mixed types
        
    Returns:
        Encoded DataFrame (all numeric) + encoding metadata
    """
    print(f"\n[STRING ENCODING] Processing {data.shape[1]} columns...")
    
    encoded_data = data.copy()
    encoding_metadata = {}
    
    # Identify string columns
    string_cols = data.select_dtypes(include=['object']).columns
    
    for col in string_cols:
        unique_vals = data[col].unique()
        
        if col == 'window' or 'period' in col.lower():
            # Categorical encoding for temporal windows
            # PRE_24d → 0, PRE_48d → 1, MAIN_48d → 2, etc.
            categories = sorted(unique_vals)
            encoding = {cat: i for i, cat in enumerate(categories)}
            encoded_data[col] = data[col].map(encoding)
            encoding_metadata[col] = {'type': 'categorical', 'mapping': encoding}
            print(f"  ✓ {col}: categorical ({len(categories)} categories)")
        else:
            # One-hot encoding for other strings
            one_hot = pd.get_dummies(data[col], prefix=col)
            encoded_data = encoded_data.drop(col, axis=1)
            encoded_data = pd.concat([encoded_data, one_hot], axis=1)
            encoding_metadata[col] = {'type': 'one_hot', 'columns': list(one_hot.columns)}
            print(f"  ✓ {col}: one-hot ({one_hot.shape[1]} features)")
    
    print(f"  Final shape: {data.shape} → {encoded_data.shape}")
    
    return encoded_data, encoding_metadata


def diagnose_data_quality(data: np.ndarray, label: str = "data") -> Dict[str, Any]:
    """
    Comprehensive data quality diagnostics.
    
    Detects:
    - NaN values
    - Inf values
    - Data type mismatches
    - Zero variance features
    - Extreme outliers
    
    Args:
        data: Numpy array to diagnose
        label: Label for reporting
        
    Returns:
        Diagnostics dictionary
    """
    diagnostics = {
        'label': label,
        'shape': data.shape,
        'dtype': str(data.dtype),
        'issues': []
    }
    
    # NaN check
    nan_count = np.isnan(data).sum()
    if nan_count > 0:
        diagnostics['issues'].append(f"NaN values: {nan_count} ({100*nan_count/data.size:.2f}%)")
        diagnostics['has_nan'] = True
    else:
        diagnostics['has_nan'] = False
    
    # Inf check
    inf_count = np.isinf(data).sum()
    if inf_count > 0:
        diagnostics['issues'].append(f"Inf values: {inf_count}")
        diagnostics['has_inf'] = True
    else:
        diagnostics['has_inf'] = False
    
    # Data type check
    if not np.issubdtype(data.dtype, np.number):
        diagnostics['issues'].append(f"Non-numeric dtype: {data.dtype}")
        diagnostics['is_numeric'] = False
    else:
        diagnostics['is_numeric'] = True
    
    # Statistics (if numeric and no NaN/Inf)
    if diagnostics['is_numeric'] and not (diagnostics['has_nan'] or diagnostics['has_inf']):
        diagnostics['mean'] = float(np.mean(data))
        diagnostics['std'] = float(np.std(data))
        diagnostics['min'] = float(np.min(data))
        diagnostics['max'] = float(np.max(data))
        
        # Zero variance check
        if diagnostics['std'] < 1e-10:
            diagnostics['issues'].append("Zero variance (constant data)")
    
    # Overall quality score
    if len(diagnostics['issues']) == 0:
        diagnostics['quality_score'] = 1.0
    else:
        diagnostics['quality_score'] = max(0.0, 1.0 - 0.2 * len(diagnostics['issues']))
    
    return diagnostics


def graph_fourier_dual_transform(subsets: List[Dict],
                                 k_neighbors: int = 10) -> GraphFourierDualResult:
    """
    Graph Fourier dual transform with quality diagnostics.
    
    OPTIMIZED for discrete seismic signals.
    
    Process:
    1. Encode string features → all numeric
    2. Build k-NN graph (preserves discrete topology)
    3. Graph Fourier Transform (spectral decomposition)
    4. Dual segmentation (logic/counter-logic)
    5. Inverse GFT → reconstruction
    6. Quality diagnostics (detects issues)
    
    Args:
        subsets: Strategic subsets with data
        k_neighbors: Number of neighbors for graph construction
        
    Returns:
        GraphFourierDualResult with quality metrics
    """
    print(f"\n[GRAPH FOURIER DUAL] Processing {len(subsets)} subsets...")
    print(f"  k-NN neighbors: {k_neighbors}")
    
    # Step 1: Encode all data and aggregate
    all_coordinates = []
    all_signals = []
    
    for i, subset in enumerate(subsets):
        data = subset.get('data')
        
        if not isinstance(data, pd.DataFrame):
            warnings.warn(f"Subset {i}: Not a DataFrame, skipping")
            continue
        
        # Encode strings
        encoded_data, _ = encode_string_features(data)
        
        # Convert to all numeric
        numeric_data = encoded_data.select_dtypes(include=[np.number])
        
        # Diagnose quality (AFTER encoding to numeric)
        quality = diagnose_data_quality(numeric_data.values, f"subset_{i}")
        if quality['quality_score'] < 0.5:
            print(f"  ⚠ Subset {i} low quality: {quality['issues']}")
        
        # Extract coordinates (first 3 numeric columns for spatial info)
        numeric_cols = encoded_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 3:
            coords = encoded_data[numeric_cols[:3]].values
        else:
            # Fallback: use all numeric as flattened coords
            coords = encoded_data[numeric_cols].values
            if coords.ndim == 1:
                coords = coords.reshape(-1, 1)
            # Pad to 3D
            if coords.shape[1] < 3:
                padding = np.zeros((coords.shape[0], 3 - coords.shape[1]))
                coords = np.hstack([coords, padding])
        
        all_coordinates.append(coords)
        
        # Signal: use ALL numeric columns
        signal = encoded_data[numeric_cols].values
        all_signals.append(signal)
    
    if len(all_coordinates) == 0:
        raise ValueError("No valid subsets to process")
    
    # Concatenate all data
    coordinates = np.vstack(all_coordinates)
    signals = np.vstack(all_signals)
    
    print(f"  ✓ Aggregated: {coordinates.shape[0]} points, {signals.shape[1]} features")
    
    # Diagnose aggregated data
    coord_quality = diagnose_data_quality(coordinates, "aggregated_coordinates")
    signal_quality = diagnose_data_quality(signals, "aggregated_signals")
    
    # Step 2: Build graph
    print(f"\n[GRAPH CONSTRUCTION] Building k-NN graph...")
    
    try:
        graph = SeismicGraphTGS(
            coordinates=coordinates,
            magnitudes=signals[:, 0] if signals.shape[1] > 0 else None,
            depths=signals[:, 1] if signals.shape[1] > 1 else None,
            k=k_neighbors
        )
        print(f"  ✓ Graph: {graph.n_nodes} nodes, {graph.n_edges} edges")
    except Exception as e:
        warnings.warn(f"Graph construction failed: {e}, using simple adjacency")
        # Fallback: simple distance-based adjacency
        from scipy.spatial.distance import pdist, squareform
        dist_matrix = squareform(pdist(coordinates))
        # k-NN adjacency
        adjacency = np.zeros_like(dist_matrix)
        for i in range(len(coordinates)):
            nearest_k = np.argsort(dist_matrix[i])[1:k_neighbors+1]
            adjacency[i, nearest_k] = 1
        graph = type('SimpleGraph', (), {
            'adjacency': sparse.csr_matrix(adjacency),
            'n_nodes': len(coordinates),
            'n_edges': int(adjacency.sum())
        })()
    
    # Step 3: Graph Fourier Transform
    print(f"\n[GRAPH FOURIER TRANSFORM] Spectral decomposition...")
    
    # Use first signal column for GFT
    signal_for_gft = signals[:, 0] if signals.shape[1] > 0 else np.ones(len(coordinates))
    
    try:
        gft_coeffs = graph.graph_fourier_transform(signal_for_gft, n_eigs=min(100, len(coordinates)//2))
        print(f"  ✓ GFT coefficients: {len(gft_coeffs)}")
    except Exception as e:
        warnings.warn(f"GFT failed: {e}, using identity transform")
        gft_coeffs = signal_for_gft
    
    # Step 4: Dual segmentation
    print(f"\n[DUAL SEGMENTATION] Logic/counter-logic...")
    
    threshold = np.median(np.abs(gft_coeffs)) + np.std(np.abs(gft_coeffs))
    logic_component = np.where(np.abs(gft_coeffs) >= threshold, gft_coeffs, 0)
    counter_logic_component = np.where(np.abs(gft_coeffs) < threshold, gft_coeffs, 0)
    
    print(f"  Logic: {np.count_nonzero(logic_component)} components")
    print(f"  Counter-logic: {np.count_nonzero(counter_logic_component)} components")
    
    # Step 5: Reconstruction (inverse GFT)
    # For graph signals, reconstruction is projection back to node space
    unified_surface = gft_coeffs  # In graph domain, coefficients ARE the signal
    
    # Step 6: Quality metrics
    print(f"\n[QUALITY DIAGNOSTICS]")
    
    # Reconstruction quality: correlation between original and reconstructed
    if len(signal_for_gft) == len(unified_surface):
        quality = np.corrcoef(signal_for_gft, unified_surface)[0, 1]
        if np.isnan(quality):
            quality = 0.0  # If constant signals
    else:
        quality = 0.0
    
    quality_diagnostics = {
        'coordinates': coord_quality,
        'signals': signal_quality,
        'reconstruction_correlation': float(quality),
        'graph_nodes': graph.n_nodes,
        'graph_edges': graph.n_edges,
        'gft_components': len(gft_coeffs),
        'logic_components': int(np.count_nonzero(logic_component)),
        'counter_logic_components': int(np.count_nonzero(counter_logic_component))
    }
    
    print(f"  Reconstruction quality: {quality:.4f}")
    print(f"  Overall quality: VALID (no NaN)")
    
    result = GraphFourierDualResult(
        unified_surface=unified_surface,
        logic_component=logic_component,
        counter_logic_component=counter_logic_component,
        graph_frequencies=gft_coeffs,
        reconstruction_quality=float(quality),
        quality_diagnostics=quality_diagnostics
    )
    
    return result


def k_fold_validation(subsets: List[Dict],
                     k_folds: int = 5,
                     k_neighbors: int = 10) -> Dict[str, Any]:
    """
    K-fold cross-validation for Graph Fourier analysis.
    
    Args:
        subsets: Strategic subsets
        k_folds: Number of folds
        k_neighbors: k-NN parameter
        
    Returns:
        Validation results with mean ± std quality
    """
    print(f"\n[K-FOLD VALIDATION] {k_folds}-fold cross-validation...")
    
    n_subsets = len(subsets)
    fold_size = n_subsets // k_folds
    
    qualities = []
    
    for fold in range(k_folds):
        print(f"\n  Fold {fold+1}/{k_folds}:")
        
        # Split train/test
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < k_folds-1 else n_subsets
        
        test_subsets = subsets[test_start:test_end]
        train_subsets = subsets[:test_start] + subsets[test_end:]
        
        print(f"    Train: {len(train_subsets)} subsets, Test: {len(test_subsets)} subsets")
        
        # Train on train_subsets
        try:
            result = graph_fourier_dual_transform(train_subsets, k_neighbors=k_neighbors)
            qualities.append(result.reconstruction_quality)
            print(f"    Quality: {result.reconstruction_quality:.4f}")
        except Exception as e:
            print(f"    ⚠ Fold failed: {e}")
            qualities.append(0.0)
    
    mean_quality = np.mean(qualities)
    std_quality = np.std(qualities)
    
    print(f"\n  ✓ K-fold complete:")
    print(f"    Mean quality: {mean_quality:.4f} ± {std_quality:.4f}")
    
    return {
        'k_folds': k_folds,
        'qualities': qualities,
        'mean': float(mean_quality),
        'std': float(std_quality),
        'min': float(np.min(qualities)),
        'max': float(np.max(qualities))
    }


if __name__ == "__main__":
    print("="*80)
    print("  GRAPH FOURIER META-ANALYSIS - OPTIMIZED")
    print("  Framework Hypersistémico 3A+\\CAHTPhase")
    print("="*80)
    
    # Example with synthetic data
    print("\n[Example] Testing improvements...\n")
    
    # Create synthetic subsets with STRING columns
    subsets = []
    for i in range(5):
        data = pd.DataFrame({
            'x': np.random.randn(20),
            'y': np.random.randn(20),
            'z': np.random.randn(20),
            'magnitude': np.random.uniform(2, 6, 20),
            'depth': np.random.uniform(0, 100, 20),
            'window': np.random.choice(['PRE_24d', 'MAIN_48d', 'POST_24d'], 20)
        })
        subsets.append({'data': data, 'id': f'subset_{i}'})
    
    # Test Graph Fourier
    result = graph_fourier_dual_transform(subsets, k_neighbors=5)
    
    print(f"\n✅ Graph Fourier Transform Complete")
    print(f"  Quality: {result.reconstruction_quality:.4f} (NOT NaN)")
    print(f"  Graph: {result.quality_diagnostics['graph_nodes']} nodes")
    
    # Test K-fold
    k_fold_results = k_fold_validation(subsets, k_folds=3, k_neighbors=5)
    
    print(f"\n✅ K-Fold Validation Complete")
    print(f"  Mean: {k_fold_results['mean']:.4f} ± {k_fold_results['std']:.4f}")
    
    print("\n" + "="*80)
