"""
Graph Theory Module for Seismic Network Analysis.

Applies graph/network theory to analyze fault topology, earthquake clustering,
and seismic network structure. Uses centrality measures, clustering coefficients,
and spectral methods to infer fault system architecture.

Research Base:
- Ramanujan graphs (optimal expanders, 2024-2025)
- Graph Neural Networks for seismicity (2024)
- Spectral graph theory for fault networks
- Centrality measures for seismic hazard assessment

Connection to Sysmic:
- Nodes = Earthquake events or spatial cells
- Edges = Proximity, temporal succession, or waveform similarity
- Network metrics → Fault system complexity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    import warnings
    warnings.warn(
        "NetworkX not available. Install: pip install networkx\n"
        "Some graph features will be limited.",
        ImportWarning
    )

__all__ = [
    "SeismicGraph",
    "compute_fault_network_centrality",
    "detect_graph_communities",
    "analyze_seismic_network_topology",
    "ramanu jan_graph_properties",
]


@dataclass
class GraphMetrics:
    """Metrics computed from seismic graph."""
    n_nodes: int
    n_edges: int
    avg_degree: float
    clustering_coefficient: float
    betweenness_centrality: Dict[int, float]
    eigenvector_centrality: Dict[int, float]
    n_components: int
    diameter: Optional[int] = None
    is_ramanujan_like: bool = False


class SeismicGraph:
    """
    Graph representation of seismic network.
    
    Philosophy:
    ----------
    Earthquakes form a complex network where:
    - Nodes = Individual events or spatial cells
    - Edges = Physical relationships (proximity, succession, similarity)
    
    Network properties reveal fault system architecture:
    - High centrality → Major fault zones
    - High clustering → Fault segments
    - Low betweenness → Isolated swarms
    - Ramanujan-like → Optimal stress distribution
    """
    
    def __init__(
        self,
        coordinates: np.ndarray,
        times: Optional[np.ndarray] = None,
        magnitudes: Optional[np.ndarray] = None,
        connection_radius: float = 10.0,  # km
        temporal_window: Optional[float] = None  # days
    ):
        """
        Initialize seismic graph.
        
        Args:
            coordinates: (N, 3) array of event locations [x, y, z]
            times: (N,) array of event times
            magnitudes: (N,) array of magnitudes
            connection_radius: Spatial distance for edge creation (km)
            temporal_window: Temporal window for edges (days, None = spatial only)
        """
        self.coordinates = coordinates
        self.times = times
        self.magnitudes = magnitudes
        self.connection_radius = connection_radius
        self.temporal_window = temporal_window
        
        self.n_nodes = len(coordinates)
        self.adjacency_matrix = None
        self.graph = None
        
        self._build_graph()
    
    def _build_graph(self):
        """Build adjacency matrix and networkx graph."""
        # Spatial proximity edges
        tree = cKDTree(self.coordinates)
        pairs = tree.query_pairs(r=self.connection_radius, output_type='ndarray')
        
        # Filter by temporal window if provided
        if self.times is not None and self.temporal_window is not None:
            temporal_mask = np.abs(
                self.times[pairs[:, 0]] - self.times[pairs[:, 1]]
            ) <= self.temporal_window
            pairs = pairs[temporal_mask]
        
        # Build sparse adjacency matrix
        row_ind = pairs[:, 0]
        col_ind = pairs[:, 1]
        data = np.ones(len(pairs))
        
        self.adjacency_matrix = csr_matrix(
            (data, (row_ind, col_ind)),
            shape=(self.n_nodes, self.n_nodes)
        )
        
        # Make symmetric (undirected graph)
        self.adjacency_matrix = self.adjacency_matrix + self.adjacency_matrix.T
        
        # Build NetworkX graph if available
        if NETWORKX_AVAILABLE:
            self.graph = nx.from_scipy_sparse_array(self.adjacency_matrix)
            
            # Add node attributes
            for i in range(self.n_nodes):
                attrs = {'coords': self.coordinates[i]}
                if self.times is not None:
                    attrs['time'] = self.times[i]
                if self.magnitudes is not None:
                    attrs['magnitude'] = self.magnitudes[i]
                self.graph.nodes[i].update(attrs)
    
    def compute_metrics(self) -> GraphMetrics:
        """
        Compute comprehensive graph metrics.
        
        Returns:
            GraphMetrics object
        """
        n_components, _ = connected_components(self.adjacency_matrix, directed=False)
        
        # Degree statistics
        degrees = np.array(self.adjacency_matrix.sum(axis=1)).flatten()
        avg_degree = degrees.mean()
        
        # Clustering coefficient (global)
        if NETWORKX_AVAILABLE and self.graph is not None:
            clustering_coef = nx.average_clustering(self.graph)
            
            # Centrality measures
            betweenness = nx.betweenness_centrality(self.graph)
            eigenvector = nx.eigenvector_centrality(self.graph, max_iter=1000)
            
            # Diameter (if connected)
            if n_components == 1:
                diameter = nx.diameter(self.graph)
            else:
                diameter = None
        else:
            # Fallback: simple clustering approximation
            clustering_coef = self._compute_clustering_coefficient()
            betweenness = {i: 0.0 for i in range(self.n_nodes)}
            eigenvector = {i: 1/self.n_nodes for i in range(self.n_nodes)}
            diameter = None
        
        # Check Ramanujan-like properties
        is_ramanujan = self._check_ramanujan_properties(degrees, avg_degree)
        
        return GraphMetrics(
            n_nodes=self.n_nodes,
            n_edges=self.adjacency_matrix.nnz // 2,  # Undirected
            avg_degree=avg_degree,
            clustering_coefficient=clustering_coef,
            betweenness_centrality=betweenness,
            eigenvector_centrality=eigenvector,
            n_components=n_components,
            diameter=diameter,
            is_ramanujan_like=is_ramanujan
        )
    
    def _compute_clustering_coefficient(self) -> float:
        """Compute global clustering coefficient without NetworkX."""
        # Local clustering for each node
        local_clustering = []
        
        for i in range(self.n_nodes):
            neighbors = self.adjacency_matrix[i].nonzero()[1]
            k_i = len(neighbors)
            
            if k_i < 2:
                continue
            
            # Count triangles
            triangles = 0
            for j in range(len(neighbors)):
                for l in range(j+1, len(neighbors)):
                    if self.adjacency_matrix[neighbors[j], neighbors[l]] > 0:
                        triangles += 1
            
            # Clustering coefficient
            possible_triangles = k_i * (k_i - 1) / 2
            if possible_triangles > 0:
                local_clustering.append(triangles / possible_triangles)
        
        return np.mean(local_clustering) if local_clustering else 0.0
    
    def _check_ramanujan_properties(self, degrees: np.ndarray, avg_degree: float) -> bool:
        """
        Check if graph exhibits Ramanujan-like properties.
        
        Ramanujan graph criterion:
        For k-regular graph, second eigenvalue λ₂ ≤ 2√(k-1)
        
        For non-regular graphs, we check approximate condition.
        """
        if avg_degree < 3:
            return False
        
        # Compute Laplacian eigenvalues
        from scipy.sparse.linalg import eigsh
        
        # Degree matrix
        D = csr_matrix((degrees, (np.arange(self.n_nodes), np.arange(self.n_nodes))))
        
        # Laplacian: L = D - A
        L = D - self.adjacency_matrix
        
        try:
            # Get smallest eigenvalues (Laplacian is positive semi-definite)
            eigenvalues = eigsh(L.asfptype(), k=min(6, self.n_nodes-1), which='SM', return_eigenvectors=False)
            eigenvalues = np.sort(eigenvalues)
            
            # Ramanujan bound: λ₂ ≤ 2√(k-1) where k = avg_degree
            ramanujan_bound = 2 * np.sqrt(avg_degree - 1)
            
            # Check if second-smallest eigenvalue (algebraic connectivity) is within bound
            if len(eigenvalues) > 1:
                return eigenvalues[1] <= ramanujan_bound * 1.1  # 10% tolerance
            
        except Exception:
            pass
        
        return False
    
    def identify_fault_zones(self, centrality_threshold: float = 0.8) -> List[int]:
        """
        Identify major fault zones using eigenvector centrality.
        
        High centrality nodes → Major fault intersections
        
        Args:
            centrality_threshold: Percentile threshold for identification
            
        Returns:
            List of node indices for major fault zones
        """
        metrics = self.compute_metrics()
        
        centralities = np.array(list(metrics.eigenvector_centrality.values()))
        threshold_value = np.percentile(centralities, centrality_threshold * 100)
        
        fault_zones = [i for i, c in metrics.eigenvector_centrality.items() 
                       if c >= threshold_value]
        
        return fault_zones
    
    def detect_communities(self, method: str = 'louvain') -> np.ndarray:
        """
        Detect seismic communities (fault segments, clusters).
        
        Args:
            method: 'louvain', 'spectral', or 'connected_components'
            
        Returns:
            Community labels for each node
        """
        if not NETWORKX_AVAILABLE:
            # Fallback to connected components
            n_comp, labels = connected_components(self.adjacency_matrix, directed=False)
            return labels
        
        if method == 'louvain':
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(self.graph)
                return np.array([communities[i] for i in range(self.n_nodes)])
            except ImportError:
                method = 'spectral'
        
        if method == 'spectral':
            from sklearn.cluster import SpectralClustering
            n_clusters = max(2, int(np.sqrt(self.n_nodes)))
            spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
            labels = spectral.fit_predict(self.adjacency_matrix.toarray())
            return labels
        
        # Default: connected components
        _, labels = connected_components(self.adjacency_matrix, directed=False)
        return labels


def compute_fault_network_centrality(
    coordinates: np.ndarray,
    connection_radius: float = 10.0
) -> Dict[str, np.ndarray]:
    """
    Compute centrality measures for fault network inference.
    
    Args:
        coordinates: Event coordinates
        connection_radius: Spatial connection threshold
        
    Returns:
        Dictionary with 'betweenness', 'eigenvector', 'degree' arrays
    """
    graph = SeismicGraph(coordinates, connection_radius=connection_radius)
    metrics = graph.compute_metrics()
    
    n = len(coordinates)
    betweenness = np.array([metrics.betweenness_centrality.get(i, 0.0) for i in range(n)])
    eigenvector = np.array([metrics.eigenvector_centrality.get(i, 0.0) for i in range(n)])
    degree = np.array(graph.adjacency_matrix.sum(axis=1)).flatten()
    
    return {
        'betweenness': betweenness,
        'eigenvector': eigenvector,
        'degree': degree
    }


def detect_graph_communities(
    coordinates: np.ndarray,
    connection_radius: float = 10.0,
    method: str = 'spectral'
) -> np.ndarray:
    """
    Detect seismic communities using graph clustering.
    
    Args:
        coordinates: Event coordinates
        connection_radius: Connection threshold
        method: Clustering method
        
    Returns:
        Community labels
    """
    graph = SeismicGraph(coordinates, connection_radius=connection_radius)
    return graph.detect_communities(method=method)


def analyze_seismic_network_topology(
    coordinates: np.ndarray,
    times: Optional[np.ndarray] = None,
    magnitudes: Optional[np.ndarray] = None,
    connection_radius: float = 10.0,
    temporal_window: Optional[float] = 1.0
) -> Tuple[GraphMetrics, Dict[str, any]]:
    """
    Comprehensive seismic network topology analysis.
    
    Args:
        coordinates: Event coordinates
        times: Event times
        magnitudes: Event magnitudes
        connection_radius: Spatial threshold
        temporal_window: Temporal threshold
        
    Returns:
        (GraphMetrics, analysis_dict)
    """
    graph = SeismicGraph(
        coordinates,
        times=times,
        magnitudes=magnitudes,
        connection_radius=connection_radius,
        temporal_window=temporal_window
    )
    
    metrics = graph.compute_metrics()
    fault_zones = graph.identify_fault_zones()
    communities = graph.detect_communities()
    
    analysis = {
        'fault_zones': fault_zones,
        'communities': communities,
        'n_communities': len(np.unique(communities)),
        'major_faults': len(fault_zones),
        'network_efficiency': 1 / (1 + metrics.avg_degree) if metrics.avg_degree > 0 else 0,
    }
    
    return metrics, analysis


def ramanujan_graph_properties(graph_metrics: GraphMetrics) -> Dict[str, any]:
    """
    Assess Ramanujan-like properties and their seismological implications.
    
    Ramanujan graphs = optimal expanders = efficient stress distribution
    
    Args:
        graph_metrics: GraphMetrics object
        
    Returns:
        Dictionary with Ramanujan assessment
    """
    assessment = {
        'is_ramanujan_like': graph_metrics.is_ramanujan_like,
        'avg_degree': graph_metrics.avg_degree,
        'clustering': graph_metrics.clustering_coefficient,
        'interpretation': ""
    }
    
    if graph_metrics.is_ramanujan_like:
        assessment['interpretation'] = (
            "Seismic network exhibits Ramanujan-like properties, suggesting:\n"
            "- Optimal stress distribution across fault network\n"
            "- Fast seismic wave mixing (rapid stress transfer)\n"
            "- Resilient structure (robust to link failures)\n"
            "- High network efficiency for earthquake propagation"
        )
    else:
        assessment['interpretation'] = (
            "Seismic network deviates from Ramanujan properties, implying:\n"
            "- Heterogeneous stress distribution\n"
            "- Structural bottlenecks or fault segmentation\n"
            "- Potential stress concentration zones\n"
            "- Non-optimal earthquake propagation pathways"
        )
    
    return assessment


if __name__ == "__main__":
    print("=" * 70)
    print("  GRAPH THEORY MODULE FOR SEISMIC NETWORK ANALYSIS")
    print("=" * 70)
    
    # Example: Random seismic network
    print("\n[Example] Seismic Network Analysis")
    np.random.seed(42)
    coords = np.random.randn(100, 3) * 50  # 100 events
    times = np.cumsum(np.random.exponential(0.5, 100))
    mags = np.random.uniform(3.0, 6.0, 100)
    
    metrics, analysis = analyze_seismic_network_topology(
        coords, times, mags,
        connection_radius=15.0,
        temporal_window=2.0
    )
    
    print(f"\nNetwork Metrics:")
    print(f"  Nodes: {metrics.n_nodes}")
    print(f"  Edges: {metrics.n_edges}")
    print(f"  Avg Degree: {metrics.avg_degree:.2f}")
    print(f"  Clustering Coefficient: {metrics.clustering_coefficient:.3f}")
    print(f"  Components: {metrics.n_components}")
    print(f"  Is Ramanujan-like: {metrics.is_ramanujan_like}")
    
    print(f"\nFault Network Analysis:")
    print(f"  Major fault zones identified: {analysis['major_faults']}")
    print(f"  Seismic communities detected: {analysis['n_communities']}")
    
    # Ramanujan assessment
    ram_props = ramanujan_graph_properties(metrics)
    print(f"\nRamanujan Properties:")
    print(ram_props['interpretation'])
    
    print("\n" + "=" * 70)
    print("✅ Graph theory module ready")
    print("✅ Fault network topology analysis operational")
    print("=" * 70)
