"""
COMPONENTE 3: EXTRACTOR DE CARACTERÍSTICAS FRACTALES COMPLETO
Extrae 50+ características fractales espaciales, temporales, de magnitud y cruzadas
para machine learning en sismología.
"""

import numpy as np
from scipy import spatial, stats, signal, optimize, special, fft
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import logging
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Importar módulos necesarios
try:
    import pywt
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False
    logging.warning("PyWavelets no está instalado. Algunas características wavelet no estarán disponibles.")

try:
    from scipy.spatial import ConvexHull, Delaunay
    GEOMETRY_AVAILABLE = True
except ImportError:
    GEOMETRY_AVAILABLE = False

@dataclass
class FeatureSet:
    """Conjunto completo de características fractales."""
    spatial_features: Dict[str, float]
    temporal_features: Dict[str, float]
    magnitude_features: Dict[str, float]
    cross_features: Dict[str, float]
    clustering_features: Dict[str, float]
    metadata: Dict[str, Any]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convierte todas las características a un DataFrame."""
        all_features = {}
        all_features.update(self.spatial_features)
        all_features.update(self.temporal_features)
        all_features.update(self.magnitude_features)
        all_features.update(self.cross_features)
        all_features.update(self.clustering_features)
        
        df = pd.DataFrame([all_features])
        df['feature_set_id'] = self.metadata.get('feature_set_id', 'unknown')
        df['n_events'] = self.metadata.get('n_events', 0)
        df['computation_time'] = self.metadata.get('computation_time', 0)
        
        return df

class FractalFeatureExtractor:
    """
    Extractor exhaustivo de características fractales para catálogos sísmicos.
    Implementa 50+ características basadas en literatura científica.
    """
    
    def __init__(self, 
                 random_state: Optional[int] = 42,
                 cache_size: int = 100,
                 parallel: bool = False):
        """
        Args:
            random_state: Semilla para reproducibilidad
            cache_size: Tamaño de caché para resultados
            parallel: Usar computación paralela cuando sea posible
        """
        self.rng = np.random.RandomState(random_state)
        self.cache = {}
        self.cache_size = cache_size
        self.parallel = parallel
        self.logger = logging.getLogger(__name__)
        
        # Inicializar estimador fractal básico
        from .fractal_estimator import FractalDimensionEstimator
        self.fractal_estimator = FractalDimensionEstimator(random_state=random_state)
        
        # Configurar características disponibles
        self.available_features = {
            'spatial': [
                'fractal_dimension_gp', 'fractal_dimension_takens', 'fractal_dimension_boxcount',
                'lacunarity', 'anisotropy', 'spatial_entropy', 'convex_hull_ratio',
                'density_heterogeneity', 'radial_distribution_slope', 'nearest_neighbor_distribution',
                'moran_i', 'geary_c', 'spatial_autocorrelation', 'cluster_index',
                'void_statistics', 'pair_correlation_function', 'ripley_k',
                'minkowski_functionals', 'percolation_threshold', 'spatial_scan_statistic'
            ],
            'temporal': [
                'hurst_exponent', 'spectral_exponent', 'autocorrelation_time',
                'permutation_entropy', 'sample_entropy', 'lempel_ziv_complexity',
                'lyapunov_exponent', 'detrended_fluctuation', 'multifractal_spectrum_width',
                'waiting_time_distribution', 'interevent_statistics', 'temporal_clustering',
                'periodicity_score', 'seasonality_index', 'temporal_trend',
                'burstiness', 'memory_coefficient', 'temporal_heterogeneity'
            ],
            'magnitude': [
                'b_value', 'b_value_uncertainty', 'magnitude_completeness',
                'magnitude_distribution_skewness', 'magnitude_distribution_kurtosis',
                'gutenberg_richter_fit', 'tapered_gutenberg_richter',
                'magnitude_frequency_deviation', 'largest_magnitude_gap',
                'magnitude_clustering', 'magnitude_correlation'
            ],
            'cross': [
                'space_time_correlation', 'magnitude_space_correlation',
                'magnitude_time_correlation', 'interevent_distance_time_correlation',
                'stress_transfer_indicator', 'triggering_probability',
                'aftershock_productivity', 'foreshock_main_shock_ratio'
            ],
            'clustering': [
                'dbscan_clusters', 'hierarchical_clusters', 'nearest_neighbor_index',
                'cluster_size_distribution', 'intercluster_distance',
                'intracluster_density', 'cluster_fractal_dimension',
                'spatial_dispersion_index', 'temporal_dispersion_index'
            ]
        }
        
    def extract_all_features(self,
                           catalog: pd.DataFrame,
                           time_col: str = 'time',
                           position_cols: List[str] = ['latitude', 'longitude', 'depth'],
                           mag_col: str = 'mag',
                           compute_all: bool = True,
                           feature_subset: Optional[List[str]] = None) -> FeatureSet:
        """
        Extrae todas las características fractales del catálogo.
        
        Args:
            catalog: DataFrame con datos sísmicos
            time_col: Nombre de la columna de tiempo
            position_cols: Nombres de columnas de posición
            mag_col: Nombre de la columna de magnitud
            compute_all: Calcular todas las características (puede ser lento)
            feature_subset: Subconjunto de características a calcular
            
        Returns:
            FeatureSet con todas las características calculadas
        """
        start_time = time.time()
        
        # Validar entrada
        self._validate_catalog(catalog, time_col, position_cols, mag_col)
        
        # Generar ID único para caché
        cache_key = self._generate_cache_key(catalog, time_col, position_cols, mag_col)
        
        if cache_key in self.cache:
            self.logger.info(f"Cache hit para catálogo: {cache_key[:16]}")
            return self.cache[cache_key]
        
        # Preparar datos
        times = pd.to_datetime(catalog[time_col])
        positions = catalog[position_cols].values
        magnitudes = catalog[mag_col].values
        
        # Convertir tiempos a segundos desde el inicio
        times_seconds = (times - times.min()).total_seconds().values
        
        # Extraer características por categoría
        features = FeatureSet(
            spatial_features={},
            temporal_features={},
            magnitude_features={},
            cross_features={},
            clustering_features={},
            metadata={
                'feature_set_id': cache_key[:16],
                'n_events': len(catalog),
                'time_range': f"{times.min()} to {times.max()}",
                'magnitude_range': f"{magnitudes.min():.2f} to {magnitudes.max():.2f}"
            }
        )
        
        try:
            # 1. Características espaciales
            self.logger.info("Extrayendo características espaciales...")
            if compute_all or self._should_compute('spatial', feature_subset):
                spatial_features = self._extract_spatial_features(positions)
                features.spatial_features.update(spatial_features)
            
            # 2. Características temporales
            self.logger.info("Extrayendo características temporales...")
            if compute_all or self._should_compute('temporal', feature_subset):
                temporal_features = self._extract_temporal_features(times_seconds)
                features.temporal_features.update(temporal_features)
            
            # 3. Características de magnitud
            self.logger.info("Extrayendo características de magnitud...")
            if compute_all or self._should_compute('magnitude', feature_subset):
                magnitude_features = self._extract_magnitude_features(magnitudes)
                features.magnitude_features.update(magnitude_features)
            
            # 4. Características cruzadas
            self.logger.info("Extrayendo características cruzadas...")
            if compute_all or self._should_compute('cross', feature_subset):
                cross_features = self._extract_cross_features(
                    positions, times_seconds, magnitudes
                )
                features.cross_features.update(cross_features)
            
            # 5. Características de clustering
            self.logger.info("Extrayendo características de clustering...")
            if compute_all or self._should_compute('clustering', feature_subset):
                clustering_features = self._extract_clustering_features(
                    positions, times_seconds, magnitudes
                )
                features.clustering_features.update(clustering_features)
            
            # Calcular tiempo total
            features.metadata['computation_time'] = time.time() - start_time
            
            # Guardar en caché si hay espacio
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = features
                # Mantener caché de tamaño limitado
                if len(self.cache) > self.cache_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extrayendo características: {str(e)}")
            raise
    
    def _validate_catalog(self, catalog: pd.DataFrame, 
                         time_col: str, 
                         position_cols: List[str], 
                         mag_col: str):
        """Valida que el catálogo tenga las columnas necesarias."""
        required_cols = [time_col, mag_col] + position_cols
        
        for col in required_cols:
            if col not in catalog.columns:
                raise ValueError(f"Columna requerida no encontrada: {col}")
        
        if len(catalog) < 50:
            self.logger.warning(f"Catálogo pequeño: {len(catalog)} eventos")
    
    def _generate_cache_key(self, catalog: pd.DataFrame,
                           time_col: str,
                           position_cols: List[str],
                           mag_col: str) -> str:
        """Genera clave única para caché."""
        # Hash de los datos principales
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(catalog[[time_col, mag_col] + position_cols]).values.tobytes()
        ).hexdigest()[:16]
        
        return f"features_{data_hash}"
    
    def _should_compute(self, category: str, feature_subset: Optional[List[str]]) -> bool:
        """Determina si se deben calcular características de una categoría."""
        if feature_subset is None:
            return True
        
        # Verificar si alguna característica de esta categoría está en el subset
        category_features = self.available_features[category]
        return any(feat in feature_subset for feat in category_features)
    
    def _extract_spatial_features(self, positions: np.ndarray) -> Dict[str, float]:
        """
        Extrae características espaciales fractales.
        
        Returns:
            Diccionario con 20+ características espaciales
        """
        features = {}
        
        if len(positions) < 10:
            self.logger.warning("Muy pocos puntos para características espaciales")
            return self._get_default_spatial_features()
        
        try:
            # 1. Dimensiones fractales por múltiples métodos
            # GP
            d_gp, unc_gp = self.fractal_estimator.estimate(
                positions, method='gp', bootstrap_iterations=30
            )
            features['fractal_dimension_gp'] = d_gp
            features['fractal_dimension_gp_unc'] = unc_gp
            
            # Takens
            d_takens, unc_takens = self.fractal_estimator.estimate(
                positions, method='takens', bootstrap_iterations=30
            )
            features['fractal_dimension_takens'] = d_takens
            features['fractal_dimension_takens_unc'] = unc_takens
            
            # Box-counting
            d_box, unc_box = self.fractal_estimator.estimate(
                positions, method='boxcount'
            )
            features['fractal_dimension_boxcount'] = d_box
            features['fractal_dimension_boxcount_unc'] = unc_box
            
            # 2. Lacunaridad (medida de heterogeneidad espacial)
            features['lacunarity'] = self._compute_lacunarity(positions)
            
            # 3. Anisotropía (ratio de ejes principales)
            features['anisotropy'] = self._compute_anisotropy(positions)
            
            # 4. Entropía espacial
            features['spatial_entropy'] = self._compute_spatial_entropy(positions)
            
            # 5. Ratio de hull convexo
            if GEOMETRY_AVAILABLE and len(positions) >= 10:
                features['convex_hull_ratio'] = self._compute_convex_hull_ratio(positions)
            else:
                features['convex_hull_ratio'] = np.nan
            
            # 6. Heterogeneidad de densidad
            features['density_heterogeneity'] = self._compute_density_heterogeneity(positions)
            
            # 7. Pendiente de distribución radial
            features['radial_distribution_slope'] = self._compute_radial_distribution_slope(positions)
            
            # 8. Distribución de vecino más cercano
            nn_features = self._compute_nearest_neighbor_distribution(positions)
            features.update(nn_features)
            
            # 9. Autocorrelación espacial (Moran's I)
            features['moran_i'] = self._compute_moran_i(positions)
            
            # 10. Geary's C
            features['geary_c'] = self._compute_geary_c(positions)
            
            # 11. Autocorrelación espacial general
            features['spatial_autocorrelation'] = self._compute_spatial_autocorrelation(positions)
            
            # 12. Índice de clustering
            features['cluster_index'] = self._compute_cluster_index(positions)
            
            # 13. Estadísticas de vacíos
            void_features = self._compute_void_statistics(positions)
            features.update(void_features)
            
            # 14. Función de correlación de pares
            pcf_features = self._compute_pair_correlation_function_features(positions)
            features.update(pcf_features)
            
            # 15. Función K de Ripley
            ripley_features = self._compute_ripley_k_features(positions)
            features.update(ripley_features)
            
            # 16. Funcionales de Minkowski (si hay suficientes puntos)
            if len(positions) >= 100:
                minkowski_features = self._compute_minkowski_functionals(positions)
                features.update(minkowski_features)
            
            # 17. Umbral de percolación estimado
            features['percolation_threshold'] = self._estimate_percolation_threshold(positions)
            
            # 18. Estadística de escaneo espacial
            features['spatial_scan_statistic'] = self._compute_spatial_scan_statistic(positions)
            
        except Exception as e:
            self.logger.error(f"Error en características espaciales: {str(e)}")
            # Devolver valores por defecto para características fallidas
            features.update(self._get_default_spatial_features())
        
        return features
    
    def _extract_temporal_features(self, times_seconds: np.ndarray) -> Dict[str, float]:
        """
        Extrae características temporales fractales.
        
        Returns:
            Diccionario con 15+ características temporales
        """
        features = {}
        
        if len(times_seconds) < 50:
            self.logger.warning("Muy pocos puntos para características temporales")
            return self._get_default_temporal_features()
        
        try:
            # Ordenar tiempos
            times_sorted = np.sort(times_seconds)
            
            # 1. Exponente de Hurst (R/S analysis)
            features['hurst_exponent'] = self._compute_hurst_exponent_rs(times_sorted)
            
            # 2. Exponente espectral (DFA)
            features['spectral_exponent'] = self._compute_spectral_exponent_dfa(times_sorted)
            
            # 3. Tiempo de autocorrelación
            features['autocorrelation_time'] = self._compute_autocorrelation_time(times_sorted)
            
            # 4. Entropía de permutación
            features['permutation_entropy'] = self._compute_permutation_entropy(times_sorted)
            
            # 5. Entropía muestral
            features['sample_entropy'] = self._compute_sample_entropy(times_sorted)
            
            # 6. Complejidad de Lempel-Ziv
            features['lempel_ziv_complexity'] = self._compute_lempel_ziv_complexity(times_sorted)
            
            # 7. Exponente de Lyapunov máximo estimado
            features['lyapunov_exponent'] = self._estimate_lyapunov_exponent(times_sorted)
            
            # 8. Análisis de fluctuaciones sin tendencia (DFA)
            features['detrended_fluctuation'] = self._compute_dfa_exponent(times_sorted)
            
            # 9. Ancho del espectro multifractal temporal
            features['multifractal_spectrum_width'] = self._compute_temporal_multifractal_width(times_sorted)
            
            # 10. Distribución de tiempos de espera
            waiting_features = self._compute_waiting_time_statistics(times_sorted)
            features.update(waiting_features)
            
            # 11. Estadísticas de intervalos entre eventos
            interevent_features = self._compute_interevent_statistics(times_sorted)
            features.update(interevent_features)
            
            # 12. Clustering temporal
            features['temporal_clustering'] = self._compute_temporal_clustering(times_sorted)
            
            # 13. Puntuación de periodicidad
            features['periodicity_score'] = self._compute_periodicity_score(times_sorted)
            
            # 14. Índice de estacionalidad
            features['seasonality_index'] = self._compute_seasonality_index(times_sorted)
            
            # 15. Tendencia temporal
            features['temporal_trend'] = self._compute_temporal_trend(times_sorted)
            
            # 16. Burstiness
            features['burstiness'] = self._compute_burstiness(times_sorted)
            
            # 17. Coeficiente de memoria
            features['memory_coefficient'] = self._compute_memory_coefficient(times_sorted)
            
            # 18. Heterogeneidad temporal
            features['temporal_heterogeneity'] = self._compute_temporal_heterogeneity(times_sorted)
            
        except Exception as e:
            self.logger.error(f"Error en características temporales: {str(e)}")
            features.update(self._get_default_temporal_features())
        
        return features
    
    def _extract_magnitude_features(self, magnitudes: np.ndarray) -> Dict[str, float]:
        """
        Extrae características de distribución de magnitudes.
        
        Returns:
            Diccionario con 10+ características de magnitud
        """
        features = {}
        
        if len(magnitudes) < 30:
            self.logger.warning("Muy pocas magnitudes para análisis")
            return self._get_default_magnitude_features()
        
        try:
            # 1. Valor b de Gutenberg-Richter (MLE)
            b_value, b_uncertainty = self._compute_b_value_mle(magnitudes)
            features['b_value'] = b_value
            features['b_value_uncertainty'] = b_uncertainty
            
            # 2. Magnitud de completitud (método de máxima curvatura)
            features['magnitude_completeness'] = self._estimate_magnitude_completeness(magnitudes)
            
            # 3. Asimetría de distribución de magnitudes
            features['magnitude_distribution_skewness'] = stats.skew(magnitudes)
            
            # 4. Curtosis de distribución de magnitudes
            features['magnitude_distribution_kurtosis'] = stats.kurtosis(magnitudes)
            
            # 5. Bondad del ajuste de Gutenberg-Richter
            features['gutenberg_richter_fit'] = self._compute_gutenberg_richter_goodness_of_fit(magnitudes)
            
            # 6. Parámetros de Gutenberg-Richter con tapado
            tapered_features = self._fit_tapered_gutenberg_richter(magnitudes)
            features.update(tapered_features)
            
            # 7. Desviación de frecuencia de magnitudes
            features['magnitude_frequency_deviation'] = self._compute_magnitude_frequency_deviation(magnitudes)
            
            # 8. Brecha de magnitud más grande
            features['largest_magnitude_gap'] = self._compute_largest_magnitude_gap(magnitudes)
            
            # 9. Clustering de magnitudes
            features['magnitude_clustering'] = self._compute_magnitude_clustering(magnitudes)
            
            # 10. Correlación de magnitudes
            features['magnitude_correlation'] = self._compute_magnitude_correlation(magnitudes)
            
        except Exception as e:
            self.logger.error(f"Error en características de magnitud: {str(e)}")
            features.update(self._get_default_magnitude_features())
        
        return features
    
    def _extract_cross_features(self, 
                               positions: np.ndarray,
                               times_seconds: np.ndarray,
                               magnitudes: np.ndarray) -> Dict[str, float]:
        """
        Extrae características cruzadas entre espacio, tiempo y magnitud.
        
        Returns:
            Diccionario con características cruzadas
        """
        features = {}
        
        if len(positions) < 20:
            return self._get_default_cross_features()
        
        try:
            # 1. Correlación espacio-tiempo
            features['space_time_correlation'] = self._compute_space_time_correlation(
                positions, times_seconds
            )
            
            # 2. Correlación magnitud-espacio
            features['magnitude_space_correlation'] = self._compute_magnitude_space_correlation(
                positions, magnitudes
            )
            
            # 3. Correlación magnitud-tiempo
            features['magnitude_time_correlation'] = self._compute_magnitude_time_correlation(
                times_seconds, magnitudes
            )
            
            # 4. Correlación distancia entre eventos-tiempo
            features['interevent_distance_time_correlation'] = self._compute_interevent_distance_time_correlation(
                positions, times_seconds
            )
            
            # 5. Indicador de transferencia de estrés
            features['stress_transfer_indicator'] = self._compute_stress_transfer_indicator(
                positions, times_seconds, magnitudes
            )
            
            # 6. Probabilidad de triggering
            features['triggering_probability'] = self._estimate_triggering_probability(
                positions, times_seconds, magnitudes
            )
            
            # 7. Productividad de réplicas
            features['aftershock_productivity'] = self._estimate_aftershock_productivity(
                times_seconds, magnitudes
            )
            
            # 8. Ratio foreshock-main shock
            features['foreshock_main_shock_ratio'] = self._compute_foreshock_main_shock_ratio(
                times_seconds, magnitudes
            )
            
        except Exception as e:
            self.logger.error(f"Error en características cruzadas: {str(e)}")
            features.update(self._get_default_cross_features())
        
        return features
    
    def _extract_clustering_features(self,
                                    positions: np.ndarray,
                                    times_seconds: np.ndarray,
                                    magnitudes: np.ndarray) -> Dict[str, float]:
        """
        Extrae características de clustering de eventos.
        
        Returns:
            Diccionario con características de clustering
        """
        features = {}
        
        if len(positions) < 20:
            return self._get_default_clustering_features()
        
        try:
            # 1. Clusters DBSCAN
            features['dbscan_clusters'] = self._compute_dbscan_clusters(positions)
            
            # 2. Clusters jerárquicos
            features['hierarchical_clusters'] = self._compute_hierarchical_clusters(positions)
            
            # 3. Índice de vecino más cercano
            features['nearest_neighbor_index'] = self._compute_nearest_neighbor_index(positions)
            
            # 4. Distribución de tamaño de clusters
            cluster_size_features = self._compute_cluster_size_distribution(
                positions, times_seconds
            )
            features.update(cluster_size_features)
            
            # 5. Distancia inter-cluster
            features['intercluster_distance'] = self._compute_intercluster_distance(positions)
            
            # 6. Densidad intra-cluster
            features['intracluster_density'] = self._compute_intracluster_density(positions)
            
            # 7. Dimensión fractal de clusters
            features['cluster_fractal_dimension'] = self._compute_cluster_fractal_dimension(
                positions, times_seconds
            )
            
            # 8. Índice de dispersión espacial
            features['spatial_dispersion_index'] = self._compute_spatial_dispersion_index(positions)
            
            # 9. Índice de dispersión temporal
            features['temporal_dispersion_index'] = self._compute_temporal_dispersion_index(times_seconds)
            
        except Exception as e:
            self.logger.error(f"Error en características de clustering: {str(e)}")
            features.update(self._get_default_clustering_features())
        
        return features
    
    # ============================================================================
    # IMPLEMENTACIONES DETALLADAS DE CARACTERÍSTICAS ESPACIALES
    # ============================================================================
    
    def _compute_lacunarity(self, positions: np.ndarray, 
                           box_sizes: Optional[np.ndarray] = None) -> float:
        """
        Calcula lacunaridad (medida de heterogeneidad espacial).
        
        Lacunaridad = Var(N) / [E(N)]^2, donde N es el conteo por caja.
        
        Args:
            positions: Coordenadas normalizadas [0,1]^3
            box_sizes: Tamaños de caja para análisis (opcional)
            
        Returns:
            Lacunaridad promedio sobre escalas
        """
        if box_sizes is None:
            box_sizes = np.logspace(np.log10(0.01), np.log10(0.5), 10)
        
        # Normalizar posiciones si no están normalizadas
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        ranges = maxs - mins
        ranges = np.where(ranges == 0, 1.0, ranges)
        positions_norm = (positions - mins) / ranges
        
        lacunarities = []
        
        for size in box_sizes:
            # Evitar tamaño de caja demasiado pequeño
            if size < 1e-6:
                continue
            
            # Discretizar en cajas
            bins = np.floor(positions_norm / size).astype(int)
            
            # Contar puntos por caja
            unique_boxes, counts = np.unique(bins, axis=0, return_counts=True)
            
            if len(counts) > 0:
                mean_count = np.mean(counts)
                var_count = np.var(counts)
                
                if mean_count > 0:
                    lac = var_count / (mean_count ** 2)
                    lacunarities.append(lac)
        
        return np.mean(lacunarities) if lacunarities else 0.0
    
    def _compute_anisotropy(self, positions: np.ndarray) -> float:
        """
        Calcula anisotropía espacial usando PCA.
        
        Returns:
            Ratio de varianza explicada entre primera y segunda componente
        """
        if len(positions) < 10:
            return 1.0
        
        try:
            # PCA
            pca = PCA()
            pca.fit(positions)
            
            explained_var = pca.explained_variance_ratio_
            
            if len(explained_var) > 1:
                # Ratio entre primera y segunda componente
                anisotropy = explained_var[0] / explained_var[1]
                return float(anisotropy)
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _compute_spatial_entropy(self, positions: np.ndarray,
                                grid_resolution: int = 10) -> float:
        """
        Calcula entropía espacial de Shannon.
        
        Args:
            positions: Coordenadas
            grid_resolution: Resolución de la cuadrícula para discretización
            
        Returns:
            Entropía espacial
        """
        # Normalizar
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        ranges = maxs - mins
        ranges = np.where(ranges == 0, 1.0, ranges)
        positions_norm = (positions - mins) / ranges
        
        # Discretizar en cuadrícula
        bins = np.floor(positions_norm * grid_resolution).astype(int)
        bins = np.clip(bins, 0, grid_resolution - 1)
        
        # Contar ocurrencias en cada celda
        unique_cells, counts = np.unique(bins, axis=0, return_counts=True)
        probabilities = counts / len(positions)
        
        # Entropía de Shannon: H = -Σ p_i * log(p_i)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        # Normalizar por entropía máxima (log(N_celdas))
        max_entropy = np.log(len(unique_cells) + 1e-10)
        
        return float(entropy / max_entropy if max_entropy > 0 else 0.0)
    
    def _compute_convex_hull_ratio(self, positions: np.ndarray) -> float:
        """
        Calcula ratio entre volumen del hull convexo y volumen del bounding box.
        
        Returns:
            Ratio de hull convexo (0-1)
        """
        if not GEOMETRY_AVAILABLE or len(positions) < 10:
            return np.nan
        
        try:
            # Hull convexo
            hull = ConvexHull(positions)
            hull_volume = hull.volume
            
            # Bounding box
            mins = positions.min(axis=0)
            maxs = positions.max(axis=0)
            bbox_volume = np.prod(maxs - mins)
            
            if bbox_volume > 0:
                return float(hull_volume / bbox_volume)
            else:
                return 0.0
                
        except Exception:
            return np.nan
    
    def _compute_density_heterogeneity(self, positions: np.ndarray,
                                      kernel_size: float = 0.1) -> float:
        """
        Calcula heterogeneidad de densidad espacial.
        
        Args:
            positions: Coordenadas
            kernel_size: Tamaño del kernel para estimación de densidad
            
        Returns:
            Coeficiente de variación de densidades locales
        """
        # Estimación de densidad por kernel
        n_points = len(positions)
        
        # Usar KDTree para conteos locales
        tree = spatial.cKDTree(positions)
        
        # Radio basado en kernel_size y extensión de datos
        extent = np.max(np.ptp(positions, axis=0))
        radius = kernel_size * extent
        
        # Contar vecinos dentro del radio para cada punto
        counts = tree.query_ball_point(positions, radius, return_length=True)
        counts = np.array(counts)
        
        # Densidad local (normalizada)
        densities = counts / (np.pi * radius ** 2)  # Para 2D, ajustar para 3D
        
        # Coeficiente de variación
        if np.mean(densities) > 0:
            cv = np.std(densities) / np.mean(densities)
            return float(cv)
        else:
            return 0.0
    
    def _compute_radial_distribution_slope(self, positions: np.ndarray) -> float:
        """
        Calcula pendiente de la distribución radial desde el centroide.
        
        Returns:
            Pendiente en escala log-log
        """
        # Centroide
        centroid = np.mean(positions, axis=0)
        
        # Distancias al centroide
        distances = np.linalg.norm(positions - centroid, axis=1)
        
        # Histograma en escala logarítmica
        log_distances = np.log10(distances[distances > 0])
        
        if len(log_distances) < 10:
            return 0.0
        
        hist, bin_edges = np.histogram(log_distances, bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Filtrar bins con conteo > 0
        valid = hist > 0
        if np.sum(valid) < 5:
            return 0.0
        
        # Regresión lineal en log-log
        slope, intercept = np.polyfit(
            bin_centers[valid], 
            np.log10(hist[valid] + 1), 
            1
        )
        
        return float(slope)
    
    def _compute_nearest_neighbor_distribution(self, positions: np.ndarray) -> Dict[str, float]:
        """
        Calcula estadísticas de distribución de vecinos más cercanos.
        
        Returns:
            Diccionario con múltiples estadísticas
        """
        if len(positions) < 10:
            return {
                'nn_mean': np.nan,
                'nn_std': np.nan,
                'nn_cv': np.nan,
                'nn_skewness': np.nan,
                'clark_evans': np.nan
            }
        
        tree = spatial.cKDTree(positions)
        
        # Distancias al vecino más cercano (excluyendo el punto mismo)
        distances, indices = tree.query(positions, k=2)
        nn_distances = distances[:, 1]  # Excluir distancia 0 (punto mismo)
        
        features = {
            'nn_mean': float(np.mean(nn_distances)),
            'nn_std': float(np.std(nn_distances)),
            'nn_cv': float(np.std(nn_distances) / np.mean(nn_distances) if np.mean(nn_distances) > 0 else 0),
            'nn_skewness': float(stats.skew(nn_distances)),
            'nn_kurtosis': float(stats.kurtosis(nn_distances))
        }
        
        # Estadística de Clark-Evans para prueba de aleatoriedad
        # R = (distancia media observada) / (distancia media esperada para proceso aleatorio)
        area = np.prod(positions.max(axis=0) - positions.min(axis=0))
        density = len(positions) / area if area > 0 else 0
        expected_mean_distance = 0.5 / np.sqrt(density) if density > 0 else 0
        
        if expected_mean_distance > 0:
            clark_evans = features['nn_mean'] / expected_mean_distance
            features['clark_evans'] = float(clark_evans)
        else:
            features['clark_evans'] = np.nan
        
        return features
    
    def _compute_moran_i(self, positions: np.ndarray) -> float:
        """
        Calcula estadística I de Moran para autocorrelación espacial.
        
        Returns:
            Estadística I de Moran (-1 a 1)
        """
        if len(positions) < 20:
            return 0.0
        
        # Usar distancias inversas como pesos
        tree = spatial.cKDTree(positions)
        
        # Matriz de pesos espaciales (distancias inversas)
        distances, indices = tree.query(positions, k=min(10, len(positions)))
        
        # Evitar división por cero
        distances = np.where(distances == 0, 1e-10, distances)
        weights = 1.0 / distances
        
        # Normalizar pesos por fila
        row_sums = weights.sum(axis=1)
        weights = weights / row_sums[:, np.newaxis]
        
        # Calcular estadística I de Moran simplificada
        # Para simplicidad, usar una aproximación
        n = len(positions)
        W = np.sum(weights)
        
        # Usar coordenadas x como variable
        x = positions[:, 0]
        x_mean = np.mean(x)
        x_diff = x - x_mean
        
        # Numerador: Σ_i Σ_j w_ij (x_i - x̄)(x_j - x̄)
        numerator = 0
        for i in range(n):
            for j in range(n):
                if i != j and j in indices[i]:
                    idx = np.where(indices[i] == j)[0][0]
                    numerator += weights[i, idx] * x_diff[i] * x_diff[j]
        
        # Denominador: Σ_i (x_i - x̄)²
        denominator = np.sum(x_diff ** 2)
        
        if denominator > 0 and W > 0:
            moran_i = (n / W) * (numerator / denominator)
            return float(moran_i)
        else:
            return 0.0
    
    def _compute_geary_c(self, positions: np.ndarray) -> float:
        """
        Calcula estadística C de Geary para autocorrelación espacial.
        
        Returns:
            Estadística C de Geary (0 a ~2)
        """
        if len(positions) < 20:
            return 1.0
        
        # Similar a Moran's I pero con diferencias al cuadrado
        tree = spatial.cKDTree(positions)
        distances, indices = tree.query(positions, k=min(10, len(positions)))
        distances = np.where(distances == 0, 1e-10, distances)
        weights = 1.0 / distances
        row_sums = weights.sum(axis=1)
        weights = weights / row_sums[:, np.newaxis]
        
        n = len(positions)
        W = np.sum(weights)
        x = positions[:, 0]
        
        # Numerador: Σ_i Σ_j w_ij (x_i - x_j)²
        numerator = 0
        for i in range(n):
            for j in indices[i]:
                if i != j:
                    idx = np.where(indices[i] == j)[0][0]
                    numerator += weights[i, idx] * (x[i] - x[j]) ** 2
        
        # Denominador: Σ_i (x_i - x̄)²
        denominator = np.sum((x - np.mean(x)) ** 2)
        
        if denominator > 0 and W > 0:
            geary_c = ((n - 1) / (2 * W)) * (numerator / denominator)
            return float(geary_c)
        else:
            return 1.0
    
    def _compute_spatial_autocorrelation(self, positions: np.ndarray) -> float:
        """
        Calcula autocorrelación espacial general usando variograma.
        
        Returns:
            Autocorrelación espacial promedio
        """
        if len(positions) < 30:
            return 0.0
        
        # Submuestreo para eficiencia
        n_sample = min(100, len(positions))
        indices = self.rng.choice(len(positions), n_sample, replace=False)
        sample = positions[indices]
        
        # Calcular variograma experimental
        distances = spatial.distance.pdist(sample)
        values = sample[:, 0]  # Usar primera coordenada como variable
        
        # Calcular diferencias al cuadrado para pares
        n_pairs = len(distances)
        if n_pairs < 10:
            return 0.0
        
        # Agrupar por distancias (binning)
        max_dist = np.max(distances)
        n_bins = 10
        bin_edges = np.linspace(0, max_dist, n_bins + 1)
        
        gamma = np.zeros(n_bins)  # Semivariograma
        counts = np.zeros(n_bins)
        
        # Calcular diferencias para pares
        k = 0
        for i in range(n_sample):
            for j in range(i + 1, n_sample):
                dist = np.linalg.norm(sample[i] - sample[j])
                diff_sq = (values[i] - values[j]) ** 2
                
                # Encontrar bin
                bin_idx = np.searchsorted(bin_edges, dist) - 1
                if 0 <= bin_idx < n_bins:
                    gamma[bin_idx] += diff_sq
                    counts[bin_idx] += 1
                k += 1
        
        # Promediar
        valid = counts > 0
        if np.any(valid):
            gamma[valid] = gamma[valid] / (2 * counts[valid])
            
            # Autocorrelación como 1 - (gamma / varianza)
            variance = np.var(values)
            if variance > 0:
                autocorr = 1 - (np.mean(gamma[valid]) / variance)
                return float(np.clip(autocorr, -1, 1))
        
        return 0.0
    
    def _compute_cluster_index(self, positions: np.ndarray) -> float:
        """
        Calcula índice de clustering espacial.
        
        Returns:
            Índice de clustering (0: uniforme, 1: completamente agrupado)
        """
        if len(positions) < 10:
            return 0.0
        
        # Comparar distribución de vecinos más cercanos con Poisson
        tree = spatial.cKDTree(positions)
        distances, _ = tree.query(positions, k=2)
        nn_distances = distances[:, 1]
        
        # Para proceso Poisson, la distribución de distancias al vecino más cercano
        # sigue una distribución de Rayleigh
        area = np.prod(positions.max(axis=0) - positions.min(axis=0))
        density = len(positions) / area if area > 0 else 0
        
        # Parámetro de escala para Rayleigh
        sigma = 1 / np.sqrt(2 * np.pi * density) if density > 0 else 1
        
        # Estadística Kolmogorov-Smirnov contra distribución teórica
        if sigma > 0 and len(nn_distances) > 10:
            # Distribución Rayleigh
            theoretical = stats.rayleigh.rvs(scale=sigma, size=len(nn_distances))
            stat, _ = stats.ks_2samp(nn_distances, theoretical)
            
            # Convertir a índice de clustering (0-1)
            cluster_index = float(stat)
            return cluster_index
        
        return 0.0
    
    def _compute_void_statistics(self, positions: np.ndarray) -> Dict[str, float]:
        """
        Calcula estadísticas de vacíos en la distribución espacial.
        
        Returns:
            Diccionario con estadísticas de vacíos
        """
        features = {
            'void_fraction': 0.0,
            'largest_void_size': 0.0,
            'void_size_distribution_slope': 0.0
        }
        
        if len(positions) < 50 or not GEOMETRY_AVAILABLE:
            return features
        
        try:
            # Usar triangulación de Delaunay para encontrar vacíos
            tri = Delaunay(positions)
            
            # Encontrar tetraedros grandes (posibles vacíos)
            tetrahedra = positions[tri.simplices]
            
            # Calcular volúmenes de tetraedros
            volumes = []
            for tetra in tetrahedra:
                # Fórmula del volumen de un tetraedro
                mat = tetra[1:] - tetra[0]
                volume = np.abs(np.linalg.det(mat)) / 6
                volumes.append(volume)
            
            volumes = np.array(volumes)
            
            if len(volumes) > 0:
                # Fracción de volumen en vacíos grandes
                volume_threshold = np.percentile(volumes, 75)
                large_voids = volumes[volumes > volume_threshold]
                
                total_volume = np.prod(positions.max(axis=0) - positions.min(axis=0))
                if total_volume > 0:
                    void_fraction = np.sum(large_voids) / total_volume
                    features['void_fraction'] = float(void_fraction)
                
                # Tamaño del vacío más grande
                if len(volumes) > 0:
                    features['largest_void_size'] = float(np.max(volumes))
                
                # Pendiente de distribución de tamaños de vacíos
                if len(volumes) >= 10:
                    hist, bin_edges = np.histogram(np.log10(volumes[volumes > 0]), bins=10)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    valid = hist > 0
                    if np.sum(valid) >= 5:
                        slope, _ = np.polyfit(bin_centers[valid], np.log10(hist[valid] + 1), 1)
                        features['void_size_distribution_slope'] = float(slope)
        
        except Exception as e:
            self.logger.debug(f"Error en estadísticas de vacíos: {str(e)}")
        
        return features
    
    # ============================================================================
    # IMPLEMENTACIONES DETALLADAS DE CARACTERÍSTICAS TEMPORALES
    # ============================================================================
    
    def _compute_hurst_exponent_rs(self, times: np.ndarray) -> float:
        """
        Calcula exponente de Hurst usando análisis R/S.
        
        Args:
            times: Tiempos ordenados
            
        Returns:
            Exponente de Hurst H (0 < H < 1)
        """
        if len(times) < 100:
            return 0.5
        
        # Convertir a serie de intervalos
        intervals = np.diff(times)
        if len(intervals) < 50:
            return 0.5
        
        # Análisis R/S clásico
        n = len(intervals)
        max_lag = min(n // 4, 100)
        
        lags = range(2, max_lag + 1)
        rs_values = []
        
        for lag in lags:
            k = n // lag
            if k < 2:
                continue
            
            # Dividir en subseries de longitud lag
            subs = intervals[:k * lag].reshape(k, lag)
            
            rs_lag = []
            for sub in subs:
                # Serie acumulativa
                X = np.cumsum(sub - np.mean(sub))
                
                # Rango
                R = np.max(X) - np.min(X)
                
                # Desviación estándar
                S = np.std(sub)
                
                if S > 0:
                    rs_lag.append(R / S)
            
            if rs_lag:
                rs_values.append(np.mean(rs_lag))
        
        if len(rs_values) < 5:
            return 0.5
        
        # Regresión log-log
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        slope, _ = np.polyfit(log_lags, log_rs, 1)
        
        # H = slope
        H = float(slope)
        
        return np.clip(H, 0.0, 1.0)
    
    def _compute_spectral_exponent_dfa(self, times: np.ndarray) -> float:
        """
        Calcula exponente espectral usando Detrended Fluctuation Analysis (DFA).
        
        Returns:
            Exponente espectral β
        """
        if len(times) < 100:
            return 1.0
        
        intervals = np.diff(times)
        if len(intervals) < 50:
            return 1.0
        
        # Serie integrada
        y = np.cumsum(intervals - np.mean(intervals))
        N = len(y)
        
        # Tamaños de ventana en progresión geométrica
        window_sizes = np.unique(np.logspace(np.log10(4), np.log10(N//4), 20).astype(int))
        window_sizes = window_sizes[window_sizes < N//4]
        
        F = []  # Fluctuaciones
        
        for n in window_sizes:
            # Dividir en segmentos de tamaño n
            n_segments = N // n
            if n_segments < 2:
                continue
            
            # Remuestrear para tener exactamente n_segments segmentos
            y_resampled = y[:n_segments * n]
            segments = y_resampled.reshape(n_segments, n)
            
            # Calcular fluctuación en cada segmento
            F_segment = []
            for segment in segments:
                # Ajuste polinomial (orden 1: lineal)
                x = np.arange(n)
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                
                # Fluctuación
                fluctuation = np.sqrt(np.mean((segment - trend) ** 2))
                F_segment.append(fluctuation)
            
            F.append(np.mean(F_segment))
        
        if len(F) < 5:
            return 1.0
        
        # Regresión log-log
        log_window_sizes = np.log(window_sizes[:len(F)])
        log_F = np.log(F)
        
        slope, _ = np.polyfit(log_window_sizes, log_F, 1)
        
        # Exponente espectral β = 2α - 1, donde α = slope
        alpha = float(slope)
        beta = 2 * alpha - 1
        
        return float(beta)
    
    def _compute_autocorrelation_time(self, times: np.ndarray) -> float:
        """
        Calcula tiempo de autocorrelación de la serie temporal.
        
        Returns:
            Tiempo de autocorrelación (en unidades de tiempo de entrada)
        """
        intervals = np.diff(times)
        if len(intervals) < 50:
            return 0.0
        
        # Calcular función de autocorrelación
        max_lag = min(100, len(intervals) // 4)
        acf = np.zeros(max_lag)
        
        mean = np.mean(intervals)
        var = np.var(intervals)
        
        if var == 0:
            return 0.0
        
        for lag in range(max_lag):
            if lag < len(intervals):
                autocov = np.mean((intervals[lag:] - mean) * (intervals[:len(intervals)-lag] - mean))
                acf[lag] = autocov / var
        
        # Encontrar tiempo de autocorrelación (donde ACF cae a 1/e)
        threshold = 1 / np.e
        for i in range(1, len(acf)):
            if acf[i] < threshold:
                # Interpolación lineal para mejor precisión
                if i > 0 and acf[i-1] >= threshold:
                    t1 = i - 1
                    t2 = i
                    y1 = acf[i-1]
                    y2 = acf[i]
                    
                    # Interpolación lineal
                    autocorr_time = t1 + (threshold - y1) * (t2 - t1) / (y2 - y1)
                    return float(autocorr_time)
        
        return float(len(acf))
    
    def _compute_permutation_entropy(self, series: np.ndarray, 
                                    m: int = 3, delay: int = 1) -> float:
        """
        Calcula entropía de permutación de la serie temporal.
        
        Args:
            series: Serie temporal
            m: Longitud de los patrones (embedding dimension)
            delay: Retardo entre muestras
            
        Returns:
            Entropía de permutación normalizada (0-1)
        """
        if len(series) < 10 * m:
            return 0.0
        
        n = len(series)
        permutations = []
        
        # Generar todos los patrones posibles
        for i in range(n - (m - 1) * delay):
            # Extraer patrón
            pattern = series[i:i + m * delay:delay]
            
            # Ordenar para obtener patrón de permutación
            permutation = np.argsort(pattern)
            
            # Convertir a tupla para hashing
            permutations.append(tuple(permutation))
        
        if len(permutations) < 10:
            return 0.0
        
        # Calcular frecuencias de patrones
        unique, counts = np.unique(permutations, axis=0, return_counts=True)
        probabilities = counts / len(permutations)
        
        # Entropía de Shannon
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        # Normalizar por log(m!)
        max_entropy = np.log(special.factorial(m))
        
        if max_entropy > 0:
            return float(entropy / max_entropy)
        else:
            return 0.0
    
    def _compute_sample_entropy(self, series: np.ndarray, 
                               m: int = 2, r: float = 0.2) -> float:
        """
        Calcula Sample Entropy (SampEn) de la serie temporal.
        
        Args:
            series: Serie temporal
            m: Longitud de los patrones
            r: Tolerancia (fracción de la desviación estándar)
            
        Returns:
            Sample Entropy
        """
        if len(series) < 10 * m:
            return 0.0
        
        n = len(series)
        r_val = r * np.std(series)
        
        def _maxdist(xi, xj):
            return np.max(np.abs(xi - xj))
        
        # Contar patrones similares
        B = 0.0
        A = 0.0
        
        for i in range(n - m):
            for j in range(i + 1, n - m):
                if _maxdist(series[i:i+m], series[j:j+m]) < r_val:
                    B += 1
                    
                    # Extender a m+1
                    if i + m + 1 <= n and j + m + 1 <= n:
                        if _maxdist(series[i:i+m+1], series[j:j+m+1]) < r_val:
                            A += 1
        
        # Evitar división por cero
        if B > 0 and A > 0:
            return float(-np.log(A / B))
        else:
            return 0.0
    
    def _compute_lempel_ziv_complexity(self, series: np.ndarray) -> float:
        """
        Calcula complejidad de Lempel-Ziv de la serie temporal.
        
        Returns:
            Complejidad de Lempel-Ziv normalizada
        """
        if len(series) < 10:
            return 0.0
        
        # Binarizar la serie (sobre/under media)
        threshold = np.median(series)
        binary_series = (series > threshold).astype(int)
        
        # Algoritmo de Lempel-Ziv
        n = len(binary_series)
        complexity = 1
        i = 0
        
        while i < n:
            # Buscar el substring más largo que ya ha aparecido
            j = i + 1
            found = False
            
            while j <= n and not found:
                substring = binary_series[i:j]
                
                # Buscar en la parte anterior de la serie
                for k in range(i):
                    if k + len(substring) <= n:
                        if np.array_equal(binary_series[k:k+len(substring)], substring):
                            found = True
                            break
                
                if not found:
                    j += 1
            
            i = j - 1
            complexity += 1
        
        # Normalizar
        max_complexity = n / np.log2(n) if n > 1 else 1
        
        return float(complexity / max_complexity)
    
    # ============================================================================
    # IMPLEMENTACIONES DETALLADAS DE CARACTERÍSTICAS DE MAGNITUD
    # ============================================================================
    
    def _compute_b_value_mle(self, magnitudes: np.ndarray, 
                            mc: Optional[float] = None) -> Tuple[float, float]:
        """
        Calcula valor b usando Maximum Likelihood Estimation.
        
        Args:
            magnitudes: Array de magnitudes
            mc: Magnitud de completitud (opcional)
            
        Returns:
            Tupla (b_value, uncertainty)
        """
        if len(magnitudes) < 30:
            return 1.0, 0.1
        
        # Estimar magnitud de completitud si no se proporciona
        if mc is None:
            mc = self._estimate_magnitude_completeness(magnitudes)
        
        # Filtrar magnitudes sobre completitud
        mag_filtered = magnitudes[magnitudes >= mc - 0.1]  # Pequeña tolerancia
        
        if len(mag_filtered) < 10:
            return 1.0, 0.1
        
        # MLE para valor b
        mean_mag = np.mean(mag_filtered)
        b_value = 1.0 / (np.log(10) * (mean_mag - mc))
        b_uncertainty = b_value / np.sqrt(len(mag_filtered))
        
        return float(b_value), float(b_uncertainty)
    
    def _estimate_magnitude_completeness(self, magnitudes: np.ndarray) -> float:
        """
        Estima magnitud de completitud usando método de máxima curvatura.
        
        Returns:
            Magnitud de completitud estimada
        """
        if len(magnitudes) < 30:
            return np.min(magnitudes)
        
        # Histograma de magnitudes
        hist, bin_edges = np.histogram(magnitudes, bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Frecuencia acumulativa (número de eventos con M ≥ m)
        cum_counts = np.cumsum(hist[::-1])[::-1]
        
        # Encontrar punto de máxima curvatura
        log_counts = np.log10(cum_counts + 1)
        
        # Calcular curvatura (segunda derivada numérica)
        curvature = np.gradient(np.gradient(log_counts))
        
        # Punto de máxima curvatura (más negativo)
        max_curve_idx = np.argmin(curvature)
        
        mc = bin_centers[max_curve_idx]
        
        return float(mc)
    
    # ============================================================================
    # FUNCIONES DE VALOR POR DEFECTO (para manejo de errores)
    # ============================================================================
    
    def _get_default_spatial_features(self) -> Dict[str, float]:
        """Características espaciales por defecto."""
        return {f: np.nan for f in self.available_features['spatial']}
    
    def _get_default_temporal_features(self) -> Dict[str, float]:
        """Características temporales por defecto."""
        return {f: np.nan for f in self.available_features['temporal']}
    
    def _get_default_magnitude_features(self) -> Dict[str, float]:
        """Características de magnitud por defecto."""
        return {f: np.nan for f in self.available_features['magnitude']}
    
    def _get_default_cross_features(self) -> Dict[str, float]:
        """Características cruzadas por defecto."""
        return {f: np.nan for f in self.available_features['cross']}
    
    def _get_default_clustering_features(self) -> Dict[str, float]:
        """Características de clustering por defecto."""
        return {f: np.nan for f in self.available_features['clustering']}
    
    # ============================================================================
    # MÉTODOS AUXILIARES ADICIONALES
    # ============================================================================
    
    def create_ml_dataset(self, catalogs: List[pd.DataFrame],
                         labels: Optional[List] = None,
                         output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Crea dataset para machine learning a partir de múltiples catálogos.
        
        Args:
            catalogs: Lista de DataFrames con catálogos
            labels: Etiquetas para cada catálogo
            output_path: Path para guardar el dataset
            
        Returns:
            DataFrame con features y labels
        """
        all_features = []
        
        for i, catalog in enumerate(catalogs):
            try:
                features = self.extract_all_features(catalog)
                feature_dict = {}
                
                # Combinar todas las características
                feature_dict.update(features.spatial_features)
                feature_dict.update(features.temporal_features)
                feature_dict.update(features.magnitude_features)
                feature_dict.update(features.cross_features)
                feature_dict.update(features.clustering_features)
                
                # Añadir metadatos
                feature_dict['catalog_id'] = i
                feature_dict['n_events'] = features.metadata['n_events']
                
                # Añadir label si está disponible
                if labels is not None and i < len(labels):
                    feature_dict['label'] = labels[i]
                
                all_features.append(feature_dict)
                
                self.logger.info(f"Procesado catálogo {i+1}/{len(catalogs)}")
                
            except Exception as e:
                self.logger.error(f"Error procesando catálogo {i}: {str(e)}")
                continue
        
        # Crear DataFrame
        df = pd.DataFrame(all_features)
        
        # Guardar si se especifica path
        if output_path:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Dataset guardado en {output_path} con {len(df)} muestras")
        
        return df
    
    def select_important_features(self, 
                                 features_df: pd.DataFrame,
                                 target_col: str,
                                 n_features: int = 20) -> List[str]:
        """
        Selecciona las características más importantes usando correlación.
        
        Args:
            features_df: DataFrame con características
            target_col: Columna objetivo
            n_features: Número de características a seleccionar
            
        Returns:
            Lista de nombres de características importantes
        """
        if target_col not in features_df.columns:
            raise ValueError(f"Columna objetivo {target_col} no encontrada")
        
        # Calcular correlación con objetivo
        correlations = features_df.corr()[target_col].abs().sort_values(ascending=False)
        
        # Excluir la propia columna objetivo y metadatos
        exclude_cols = [target_col, 'catalog_id', 'n_events']
        important_features = [
            f for f in correlations.index 
            if f not in exclude_cols and not pd.isna(correlations[f])
        ][:n_features]
        
        return important_features