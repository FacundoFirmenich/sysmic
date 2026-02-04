"""
================================================================================
METAMIXING MODULE (M2) - Aseptic Assignment Algorithm (3A)
================================================================================
Canonical tool for Sysmic SuperCore.

Generates pondered asymmetry mesh for non-egalitarian weight assignment
based on intrinsic characteristics of strategic subsets.

Foundation: Information-theoretic entropy maximization constrained by
observed distributions.

Author: Sysmic Framework
Status: Canonical Tool (Permanent)
================================================================================
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import minimize
from scipy.stats import entropy
from dataclasses import dataclass
import warnings

__all__ = [
    'AsepticAssignmentAlgorithm',
    'AsymmetryMesh',
    'optimize_subset_weights',
    'validate_asymmetry_mesh'
]


@dataclass
class AsymmetryMesh:
    """
    Pondered asymmetry mesh for subset weighting.
    
    Attributes:
        weights: Weight matrix (N_subsets × N_features)
        subset_ids: Identifiers for each subset
        feature_names: Names of features used
        optimization_score: Quality metric of optimization
        customization_params: Parameters used for this specific case
    """
    weights: np.ndarray
    subset_ids: List[str]
    feature_names: List[str]
    optimization_score: float
    customization_params: Dict[str, Any]


class AsepticAssignmentAlgorithm:
    """
    Algorithmic assignment of non-egalitarian weights.
    
    Core Principle:
    ---------------
    Weights are NOT equal by design. Each subset receives customized
    weights based on its particular characteristics, optimized to
    maximize information content while respecting observed distributions.
    
    Mathematical Foundation:
    ------------------------
    Maximize: H(W) = -Σ w_i log(w_i)  (entropy)
    Subject to:
        - Σ w_i = 1  (normalization)
        - w_i ≥ 0  (positivity)
        - Constraints from observed characteristics
    
    Usage:
    ------
    >>> aaa = AsepticAssignmentAlgorithm()
    >>> subsets = [{'data': df1, 'meta': {'type': 'control'}}, ...]
    >>> mesh = aaa.generate_asymmetry_mesh(subsets, context={'goal': 'fusion'})
    >>> print(mesh.weights)
    """
    
    def __init__(self, 
                 entropy_weight: float = 0.7,
                 constraint_weight: float = 0.3):
        """
        Initialize AAA.
        
        Args:
            entropy_weight: Weight for entropy maximization (0-1)
            constraint_weight: Weight for constraint satisfaction (0-1)
        """
        self.entropy_weight = entropy_weight
        self.constraint_weight = constraint_weight
        
    def generate_asymmetry_mesh(self, 
                                subsets: List[Dict], 
                                context: Dict) -> AsymmetryMesh:
        """
        Generate pondered asymmetry mesh.
        
        Process:
        1. Extract characteristics from each subset
        2. Compute optimization landscape
        3. Find optimal non-egalitarian weights
        4. Validate and return mesh
        
        Args:
            subsets: Strategic subsets with data and metadata
            context: Contextual information for customization
            
        Returns:
            AsymmetryMesh with optimized weights
        """
        print(f"\n[AAA] Generating asymmetry mesh for {len(subsets)} subsets...")
        
        # Extract characteristics
        characteristics = self._extract_characteristics(subsets)
        subset_ids = [s.get('id', f'subset_{i}') for i, s in enumerate(subsets)]
        
        # Generate feature matrix
        feature_matrix = self._build_feature_matrix(characteristics)
        feature_names = list(characteristics[0].keys())
        
        # Optimize weights
        weights = self._optimize_weights(feature_matrix, context)
        
        # Compute optimization score
        opt_score = self._compute_optimization_score(weights, feature_matrix)
        
        mesh = AsymmetryMesh(
            weights=weights,
            subset_ids=subset_ids,
            feature_names=feature_names,
            optimization_score=opt_score,
            customization_params=context
        )
        
        print(f"  ✓ Mesh generated: {weights.shape}")
        print(f"  ✓ Optimization score: {opt_score:.4f}")
        print(f"  ✓ Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
        
        return mesh
    
    def _extract_characteristics(self, subsets: List[Dict]) -> List[Dict]:
        """Extract intrinsic characteristics from each subset."""
        characteristics = []
        
        for subset in subsets:
            data = subset.get('data')
            meta = subset.get('meta', {})
            
            char = {}
            
            # Intrinsic characteristics
            if isinstance(data, pd.DataFrame):
                char['size'] = len(data)
                char['dimensionality'] = data.shape[1]
                char['density'] = data.notna().mean().mean()
                # Only numeric columns for variance
                numeric_data = data.select_dtypes(include=[np.number])
                char['variance'] = numeric_data.var().mean() if numeric_data.shape[1] > 0 else 0.0
            elif isinstance(data, np.ndarray):
                char['size'] = data.shape[0]
                char['dimensionality'] = data.shape[1] if data.ndim > 1 else 1
                char['density'] = np.sum(~np.isnan(data)) / data.size
                char['variance'] = np.nanvar(data)
            else:
                # Fallback for other types
                char['size'] = 1.0
                char['dimensionality'] = 1.0
                char['density'] = 1.0
                char['variance'] = 0.0
            
            # Metadata characteristics
            char['importance'] = meta.get('importance', 1.0)
            char['reliability'] = meta.get('reliability', 1.0)
            char['temporal_relevance'] = meta.get('temporal_relevance', 1.0)
            
            characteristics.append(char)
        
        return characteristics
    
    def _build_feature_matrix(self, characteristics: List[Dict]) -> np.ndarray:
        """Build normalized feature matrix from characteristics."""
        n_subsets = len(characteristics)
        feature_names = list(characteristics[0].keys())
        n_features = len(feature_names)
        
        matrix = np.zeros((n_subsets, n_features))
        
        for i, char in enumerate(characteristics):
            for j, fname in enumerate(feature_names):
                matrix[i, j] = char[fname]
        
        # Normalize each column to [0, 1]
        for j in range(n_features):
            col = matrix[:, j]
            col_min, col_max = col.min(), col.max()
            if col_max > col_min:
                matrix[:, j] = (col - col_min) / (col_max - col_min)
        
        return matrix
    
    def _optimize_weights(self, 
                         feature_matrix: np.ndarray, 
                         context: Dict) -> np.ndarray:
        """
        Optimize non-egalitarian weights.
        
        Objective: Maximize information (entropy) while respecting
        characteristic-based constraints.
        """
        n_subsets, n_features = feature_matrix.shape
        
        # Initial guess: slightly perturbed from egalitarian
        x0 = np.ones(n_subsets) / n_subsets + np.random.randn(n_subsets) * 0.01
        x0 = np.abs(x0)  # Ensure positive
        x0 /= x0.sum()  # Normalize
        
        def objective(w):
            """Negative entropy (we minimize, so negative for maximization)"""
            w_safe = np.clip(w, 1e-10, 1.0)  # Avoid log(0)
            H = -np.sum(w_safe * np.log(w_safe))  # Entropy
            
            # Constraint: weights should correlate with characteristics
            characteristic_score = np.mean([
                np.dot(w, feature_matrix[:, j]) 
                for j in range(n_features)
            ])
            
            # Combined objective
            return -(self.entropy_weight * H + 
                    self.constraint_weight * characteristic_score)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Normalization
        ]
        
        # Bounds
        bounds = [(0.0, 1.0) for _ in range(n_subsets)]
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        return result.x
    
    def _compute_optimization_score(self, 
                                    weights: np.ndarray, 
                                    feature_matrix: np.ndarray) -> float:
        """
        Compute quality score of optimization.
        
        Combines:
        - Entropy (diversity of weights)
        - Characteristic alignment
        - Non-egalitarian measure
        """
        # Entropy
        w_safe = np.clip(weights, 1e-10, 1.0)
        H = -np.sum(w_safe * np.log(w_safe))
        
        # Characteristic alignment
        alignment = np.mean([
            np.dot(weights, feature_matrix[:, j])
            for j in range(feature_matrix.shape[1])
        ])
        
        # Non-egalitarian measure (deviation from uniform)
        uniform = np.ones_like(weights) / len(weights)
        non_egal = np.linalg.norm(weights - uniform)
        
        # Combined score
        score = (H / np.log(len(weights)) + alignment + non_egal) / 3.0
        
        return score


def optimize_subset_weights(subsets: List[Dict], 
                           context: Optional[Dict] = None) ->AsymmetryMesh:
    """
    High-level interface for weight optimization.
    
    Args:
        subsets: Strategic subsets with data and metadata
        context: Optional context for customization
        
    Returns:
        AsymmetryMesh with optimized weights
    """
    if context is None:
        context = {}
    
    aaa = AsepticAssignmentAlgorithm()
    return aaa.generate_asymmetry_mesh(subsets, context)


def validate_asymmetry_mesh(mesh: AsymmetryMesh, 
                            expected_properties: Optional[Dict] = None) -> bool:
    """
    Validate asymmetry mesh meets requirements.
    
    Args:
        mesh: AsymmetryMesh to validate
        expected_properties: Optional dict with expected properties
        
    Returns:
        True if valid, False otherwise
    """
    # Basic validations
    assert np.all(mesh.weights >= 0), "Weights must be non-negative"
    assert np.isclose(mesh.weights.sum(), 1.0), "Weights must sum to 1.0"
    assert mesh.weights.shape[0] == len(mesh.subset_ids), "Dimension mismatch"
    
    # Non-egalitarian check
    uniform = np.ones_like(mesh.weights) / len(mesh.weights)
    deviation = np.linalg.norm(mesh.weights - uniform)
    assert deviation > 0.01, "Weights too close to egalitarian"
    
    # Optional property checks
    if expected_properties:
        if 'min_score' in expected_properties:
            assert mesh.optimization_score >= expected_properties['min_score']
    
    return True


if __name__ == "__main__":
    print("="*80)
    print("  AAA - Aseptic Assignment Algorithm")
    print("  Canonical Tool for Sysmic SuperCore")
    print("="*80)
    
    # Example usage
    print("\n[Example] Generating asymmetry mesh for 5 synthetic subsets...\n")
    
    # Create synthetic subsets
    subsets = []
    for i in range(5):
        data = pd.DataFrame(np.random.randn(100 * (i+1), 3))
        meta = {
            'id': f'subset_{i}',
            'importance': 1.0 + i * 0.2,
            'reliability': 0.9 - i * 0.1,
            'temporal_relevance': np.random.rand()
        }
        subsets.append({'data': data, 'meta': meta, 'id': f'subset_{i}'})
    
    # Generate mesh
    mesh = optimize_subset_weights(subsets, context={'goal': 'fusion', 'method': 'entropy_max'})
    
    # Validate
    is_valid = validate_asymmetry_mesh(mesh)
    
    print(f"\n✅ Asymmetry Mesh Generated and Validated")
    print(f"  Subsets: {len(mesh.subset_ids)}")
    print(f"  Features: {len(mesh.feature_names)}")
    print(f"  Optimization Score: {mesh.optimization_score:.4f}")
    print(f"\nWeights:")
    for sid, w in zip(mesh.subset_ids, mesh.weights):
        print(f"  {sid}: {w:.4f}")
    print("\n" + "="*80)
