"""
================================================================================
EMERGENT METRICS MODULE - Non-Reducible Analysis
================================================================================
Part of Framework Hypersistémico 3A+\CAHTPhase

Implements emergent metrics that CANNOT be reduced to individual analyses:
1. Trans-regional clustering (cross-event patterns)
2. Multivariate anomalies (outlier detection in high-D space)
3. Cross-event temporal patterns (evolution across events)

Author: Sysmic Framework
Status: Validation & Analysis Module
================================================================================
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings

__all__ = [
    'detect_transregional_clusters',
    'identify_multivariate_anomalies',
    'analyze_crossevent_temporal_patterns',
    'compute_emergent_complexity'
]


def detect_transregional_clusters(unified_data: np.ndarray,
                                  labels: Optional[np.ndarray] = None,
                                  method: str = 'dbscan',
                                  **kwargs) -> Dict:
    """
    Detect clusters that span multiple regions/events.
    
    Emergent property: Clusters that wouldn't be visible in individual analyses.
    
    Args:
        unified_data: Unified analysis data (N, features)
        labels: Optional event labels (N,) to track cross-event membership
        method: 'dbscan' or 'kmeans'
        **kwargs: Algorithm-specific parameters
        
    Returns:
        Dict with cluster assignments and cross-event statistics
    """
    print(f"\n[TRANS-REGIONAL CLUSTERING] Method: {method}")
    print(f"  Data shape: {unified_data.shape}")
    
    # Clustering
    if method == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    else:  # kmeans
        n_clusters = kwargs.get('n_clusters', 3)
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    cluster_labels = clusterer.fit_predict(unified_data)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"  ✓ Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise}")
    
    # Cross-event analysis (if event labels provided)
    cross_event_clusters = []
    if labels is not None:
        unique_events = np.unique(labels)
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:
                continue
            
            cluster_mask = cluster_labels == cluster_id
            cluster_events = labels[cluster_mask]
            unique_cluster_events = np.unique(cluster_events)
            
            if len(unique_cluster_events) > 1:
                # Cross-event cluster!
                cross_event_clusters.append({
                    'cluster_id': int(cluster_id),
                    'n_events': len(unique_cluster_events),
                    'events': unique_cluster_events.tolist(),
                    'size': int(cluster_mask.sum()),
                    'centroid': unified_data[cluster_mask].mean(axis=0)
                })
        
        print(f"  ✓ Cross-event clusters: {len(cross_event_clusters)}")
        for cec in cross_event_clusters:
            print(f"    Cluster {cec['cluster_id']}: {cec['n_events']} events, size={cec['size']}")
    
    return {
        'cluster_labels': cluster_labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'cross_event_clusters': cross_event_clusters,
        'method': method,
        'is_emergent': len(cross_event_clusters) > 0
    }


def identify_multivariate_anomalies(unified_data: np.ndarray,
                                     contamination: float = 0.1,
                                     method: str = 'elliptic_envelope') -> Dict:
    """
    Identify anomalies in high-dimensional space.
    
    Emergent property: Anomalies visible only in multivariate space,
    not in individual feature distributions.
    
    Args:
        unified_data: Unified analysis data (N, features)
        contamination: Expected proportion of outliers
        method: Anomaly detection method
        
    Returns:
        Dict with anomaly scores and indices
    """
    print(f"\n[MULTIVARIATE ANOMALIES] Method: {method}")
    print(f"  Data shape: {unified_data.shape}")
    
    if method == 'elliptic_envelope':
        detector = EllipticEnvelope(contamination=contamination, random_state=42)
        predictions = detector.fit_predict(unified_data)
        scores = detector.decision_function(unified_data)
    else:
        # Mahalanobis distance fallback
        mean = unified_data.mean(axis=0)
        cov = np.cov(unified_data.T)
        inv_cov = np.linalg.pinv(cov)
        
        scores = np.array([
            -np.sqrt((x - mean) @ inv_cov @ (x - mean))
            for x in unified_data
        ])
        threshold = np.percentile(scores, contamination * 100)
        predictions = np.where(scores < threshold, -1, 1)
    
    anomaly_indices = np.where(predictions == -1)[0]
    n_anomalies = len(anomaly_indices)
    
    print(f"  ✓ Anomalies detected: {n_anomalies} ({100*n_anomalies/len(unified_data):.1f}%)")
    
    # Analyze anomaly characteristics
    if n_anomalies > 0:
        normal_indices = np.where(predictions == 1)[0]
        
        # Feature-wise comparison
        anomaly_stats = {}
        for feat_idx in range(unified_data.shape[1]):
            anomaly_vals = unified_data[anomaly_indices, feat_idx]
            normal_vals = unified_data[normal_indices, feat_idx]
            
            # t-test
            if len(anomaly_vals) > 1 and len(normal_vals) > 1:
                t_stat, p_val = stats.ttest_ind(anomaly_vals, normal_vals)
                
                if p_val < 0.05:
                    anomaly_stats[f'feature_{feat_idx}'] = {
                        'anomaly_mean': float(anomaly_vals.mean()),
                        'normal_mean': float(normal_vals.mean()),
                        'p_value': float(p_val),
                        'significant': True
                    }
        
        print(f"  Significant feature differences: {len(anomaly_stats)}")
    else:
        anomaly_stats = {}
    
    return {
        'anomaly_indices': anomaly_indices.tolist(),
        'n_anomalies': n_anomalies,
        'anomaly_scores': scores.tolist(),
        'anomaly_stats': anomaly_stats,
        'contamination': contamination,
        'is_emergent': n_anomalies > 0
    }


def analyze_crossevent_temporal_patterns(unified_data: np.ndarray,
                                         timestamps: np.ndarray,
                                         event_labels: np.ndarray) -> Dict:
    """
    Analyze temporal evolution patterns across events.
    
    Emergent property: Temporal trends visible only when comparing
    multiple events, not within single event.
    
    Args:
        unified_data: Unified analysis data (N, features)
        timestamps: Temporal ordering (N,)
        event_labels: Event identifiers (N,)
        
    Returns:
        Dict with cross-event temporal statistics
    """
    print(f"\n[CROSS-EVENT TEMPORAL PATTERNS]")
    print(f"  Data shape: {unified_data.shape}")
    print(f"  Unique events: {len(np.unique(event_labels))}")
    
    patterns = {}
    
    # 1. PCA temporal trajectory
    pca = PCA(n_components=min(3, unified_data.shape[1]))
    pca_coords = pca.fit_transform(unified_data)
    
    print(f"  ✓ PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    
    # 2. Temporal correlation across events
    unique_events = np.unique(event_labels)
    
    if len(unique_events) > 1:
        # For each feature, compute correlation with time ACROSS events
        cross_correlations = []
        
        for feat_idx in range(unified_data.shape[1]):
            corr, p_val = stats.spearmanr(timestamps, unified_data[:, feat_idx])
            
            if not np.isnan(corr):
                cross_correlations.append({
                    'feature_idx': feat_idx,
                    'correlation': float(corr),
                    'p_value': float(p_val),
                    'significant': p_val < 0.05
                })
        
        significant_trends = [c for c in cross_correlations if c['significant']]
        print(f"  ✓ Significant temporal trends: {len(significant_trends)}/{len(cross_correlations)}")
        
        patterns['cross_correlations'] = cross_correlations
        patterns['n_significant_trends'] = len(significant_trends)
    
    # 3. Event-to-event similarity evolution
    event_centroids = []
    for event in unique_events:
        event_mask = event_labels == event
        centroid = unified_data[event_mask].mean(axis=0)
        event_centroids.append(centroid)
    
    if len(event_centroids) > 1:
        # Pairwise distances between event centroids
        centroid_distances = pdist(event_centroids, metric='euclidean')
        mean_inter_event_distance = float(centroid_distances.mean())
        std_inter_event_distance = float(centroid_distances.std())
        
        print(f"  ✓ Inter-event distance: {mean_inter_event_distance:.3f} ± {std_inter_event_distance:.3f}")
        
        patterns['event_centroids'] = [c.tolist() for c in event_centroids]
        patterns['inter_event_distance_mean'] = mean_inter_event_distance
        patterns['inter_event_distance_std'] = std_inter_event_distance
    
    patterns['pca_variance_explained'] = float(pca.explained_variance_ratio_.sum())
    patterns['pca_components'] = pca_coords.tolist()
    patterns['is_emergent'] = len(patterns.get('cross_correlations', [])) > 0
    
    return patterns


def compute_emergent_complexity(unified_data: np.ndarray,
                                individual_analyses: List[np.ndarray]) -> Dict:
    """
    Compute complexity metrics that emerge from unification.
    
    Compares complexity of unified analysis vs sum of individual analyses.
    
    Args:
        unified_data: Unified analysis result (N, features)
        individual_analyses: List of individual analysis results
        
    Returns:
        Dict with emergent complexity metrics
    """
    print(f"\n[EMERGENT COMPLEXITY]")
    
    # 1. Effective dimensionality (via PCA)
    pca_unified = PCA()
    pca_unified.fit(unified_data)
    
    # Intrinsic dimensionality: number of components explaining 95% variance
    cumsum_var = np.cumsum(pca_unified.explained_variance_ratio_)
    intrinsic_dim_unified = int(np.argmax(cumsum_var >= 0.95) + 1)
    
    print(f"  Unified intrinsic dimensionality: {intrinsic_dim_unified}")
    
    # 2. Individual complexities
    individual_dims = []
    for ind_data in individual_analyses:
        if len(ind_data) > 1 and ind_data.shape[1] > 1:
            # Filter NaN rows
            mask = ~np.isnan(ind_data).any(axis=1)
            clean_data = ind_data[mask]
            
            if len(clean_data) > 1:
                pca_ind = PCA()
                pca_ind.fit(clean_data)
                cumsum_var_ind = np.cumsum(pca_ind.explained_variance_ratio_)
                dim_ind = int(np.argmax(cumsum_var_ind >= 0.95) + 1)
                individual_dims.append(dim_ind)
    
    mean_individual_dim = np.mean(individual_dims) if individual_dims else 0
    
    print(f"  Mean individual dimensionality: {mean_individual_dim:.1f}")
    
    # 3. Emergent complexity ratio
    if mean_individual_dim > 0:
        complexity_ratio = intrinsic_dim_unified / mean_individual_dim
        is_emergent = complexity_ratio > 1.2  # 20% increase
    else:
        complexity_ratio = np.nan
        is_emergent = False
    
    print(f"  Complexity ratio: {complexity_ratio:.2f}")
    print(f"  Emergent complexity: {is_emergent}")
    
    return {
        'unified_intrinsic_dim': intrinsic_dim_unified,
        'mean_individual_dim': float(mean_individual_dim),
        'complexity_ratio': float(complexity_ratio) if not np.isnan(complexity_ratio) else None,
        'is_emergent': is_emergent,
        'individual_dims': individual_dims
    }


if __name__ == "__main__":
    print("="*80)
    print("  EMERGENT METRICS MODULE")
    print("  Sysmic Framework")
    print("="*80)
    
    # Synthetic test
    print("\n[Test] Synthetic multi-event data...\n")
    
    # 3 events with different characteristics
    event1 = np.random.randn(50, 5) + [0, 0, 0, 0, 0]
    event2 = np.random.randn(50, 5) + [2, 0, 0, 0, 0]
    event3 = np.random.randn(50, 5) + [0, 2, 0, 0, 0]
    
    unified = np.vstack([event1, event2, event3])
    labels = np.array([0]*50 + [1]*50 + [2]*50)
    timestamps = np.arange(150)
    
    # Test clustering
    clusters = detect_transregional_clusters(unified, labels=labels, method='dbscan', eps=1.0)
    
    # Test anomalies
    anomalies = identify_multivariate_anomalies(unified, contamination=0.1)
    
    # Test temporal patterns
    temporal = analyze_crossevent_temporal_patterns(unified, timestamps, labels)
    
    # Test emergent complexity
    complexity = compute_emergent_complexity(unified, [event1, event2, event3])
    
    print(f"\n✅ All emergent metrics demonstrated")
    print("\n" + "="*80)
