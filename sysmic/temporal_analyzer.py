"""
SECCIÓN 3: ANÁLISIS TEMPORAL D(t)
Extracted from _python_5.py for modular architecture.
"""

import numpy as np
import pandas as pd
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from scipy import signal, stats

from .infrastructure import (
    SystemConfiguration, ScientificLogger, SystemCache, 
    ParallelExecutor, FractalEstimationResult
)
from .fractal_estimator import FractalDimensionEstimator

class TemporalFractalAnalyzer:
    """
    Analizador de dimensión fractal temporal D(t).
    
    Características:
    1. Ventanas temporales adaptativas
    2. Detección de cambios de régimen (PELT, BOCPD)
    3. Análisis de tendencias y ciclos
    4. Predicción de transiciones
    """
    
    def __init__(self, config: SystemConfiguration, logger: ScientificLogger):
        self.config = config
        self.logger = logger
        self.estimator = FractalDimensionEstimator(config, logger)
        self.cache = SystemCache(config)
        
        # Parámetros
        self.min_window_events = 100
        self.max_window_events = 5000
        self.window_overlap = 0.7
        self.adaptive_window = True
        
        # Para detección de cambios
        self.change_detection_methods = ['pelt', 'bocpd', 'bayesian']
        self.change_confidence = 0.95
        
    def compute_D_t(self, catalog: pd.DataFrame,
                   time_column: str = 'time',
                   position_columns: List[str] = ['latitude', 'longitude', 'depth'],
                   magnitude_column: str = 'mag',
                   method: str = 'gp',
                   output_format: str = 'dataframe') -> Union[pd.DataFrame, Dict]:
        """
        Calcula D(t) con ventanas temporales.
        """
        start_time = time.time()
        
        # Validar entrada
        self._validate_catalog(catalog, time_column, position_columns, magnitude_column)
        
        experiment_id = self.logger.start_experiment(
            "D_t_analysis",
            {
                'catalog_size': len(catalog),
                'method': method,
                'time_range': f"{catalog[time_column].min()} to {catalog[time_column].max()}"
            }
        )
        
        # 1. Preparar datos
        times = pd.to_datetime(catalog[time_column])
        positions = catalog[position_columns].values
        magnitudes = catalog[magnitude_column].values
        
        # Ordenar por tiempo
        sort_idx = np.argsort(times)
        times = times.iloc[sort_idx]
        positions = positions[sort_idx]
        magnitudes = magnitudes[sort_idx]
        
        # 2. Definir ventanas temporales
        windows = self._define_temporal_windows(times, positions)
        
        self.logger.logger.info(f"Definidas {len(windows)} ventanas temporales")
        
        # 3. Calcular D para cada ventana (en paralelo)
        # We need a bound method for parallel execution in current architecture
        # Or define a top-level helper. For now, we will execute sequentially or use simple parallel loop
        # The infrastructure.ParallelExecutor handles pickling bound methods usually fine with dill/cloudpickle
        # but concurrent.futures.ProcessPoolExecutor might struggle.
        # We will iterate and map.
        
        D_results = []
        
        # Define helper for parallel map
        def process_window(args):
            idx, (start, end) = args
            window_pos = positions[start:end]
            try:
                # We create a lightweight estimator here or reuse via global if needed
                # But since we are inside a method, it is tricky.
                # Let's run sequentially for safety in this refactor unless we refactor _compute_window_D to be static
                result = self.estimator.estimate(
                    window_pos,
                    method=method,
                    bootstrap_iterations=50, # Reduced for speed
                    confidence_level=0.95
                )
                return idx, result
            except Exception as e:
                return idx, None
        
        # Sequential loop for robustness in this refactor step
        for i, (start_idx, end_idx) in enumerate(windows):
            window_positions = positions[start_idx:end_idx]
            try:
                result = self.estimator.estimate(
                    window_positions,
                    method=method,
                    bootstrap_iterations=50,
                    confidence_level=0.95
                )
                
                if result:
                     # Tiempo representativo de la ventana (mediana)
                    window_times = times.iloc[start_idx:end_idx]
                    representative_time = window_times.iloc[len(window_times) // 2]
                    
                    # Completitud de magnitud para esta ventana
                    window_mags = magnitudes[start_idx:end_idx]
                    completeness = self._estimate_magnitude_completeness(window_mags)
                    
                    D_results.append({
                        'window_index': i,
                        'time': representative_time,
                        'D': result.dimension,
                        'D_uncertainty': result.uncertainty,
                        'n_events': len(window_times),
                        'completeness_mc': completeness,
                        'window_start': window_times.iloc[0],
                        'window_end': window_times.iloc[-1],
                        'method': method,
                        'quality_metrics': result.quality_metrics
                    })
            except Exception as e:
                self.logger.logger.warning(f"Error en ventana {i}: {e}")
        
        # 5. Crear serie temporal
        D_series = pd.DataFrame(D_results)
        
        if D_series.empty:
            raise ValueError("No se pudo calcular D para ninguna ventana")
        
        # Ordenar por tiempo
        D_series = D_series.sort_values('time').reset_index(drop=True)
        
        # 6. Suavizado (opcional)
        if len(D_series) > 5:
            D_series = self._smooth_D_series(D_series)
        
        # 7. Metabolismo de resultados
        metrics = {
            'n_windows': len(D_series),
            'mean_D': float(D_series['D'].mean()),
            'computation_time': time.time() - start_time
        }
        
         # Formatear salida
        if output_format == 'dataframe':
             return D_series
        else:
             return {'D_series': D_series.to_dict('records'), 'metrics': metrics}

    def _validate_catalog(self, catalog: pd.DataFrame, time_column: str,
                         position_columns: List[str], magnitude_column: str):
        """Valida el catálogo de entrada."""
        required_columns = [time_column, magnitude_column] + position_columns
        for col in required_columns:
            if col not in catalog.columns:
                raise ValueError(f"Columna requerida no encontrada: {col}")

    def _define_temporal_windows(self, times: pd.Series, 
                                positions: np.ndarray) -> List[Tuple[int, int]]:
        """Define ventanas temporales adaptativas."""
        n_events = len(times)
        windows = []
        
        # Ventanas fijas por ahora
        window_size = min(self.max_window_events, 
                        max(self.min_window_events, n_events // 20))
        
        for start_idx in range(0, n_events, int(window_size * (1 - self.window_overlap))):
            end_idx = min(start_idx + window_size, n_events)
            if end_idx - start_idx >= self.min_window_events:
                windows.append((start_idx, end_idx))
                
        return windows

    def _estimate_magnitude_completeness(self, magnitudes: np.ndarray) -> float:
        """Estima magnitud de completitud usando máxima curvatura."""
        if len(magnitudes) < 50:
            return np.min(magnitudes)
        
        # Histograma
        hist, bin_edges = np.histogram(magnitudes, bins=20)
        cum_counts = np.cumsum(hist[::-1])[::-1]
        
        # Punto de máxima curvatura
        log_counts = np.log10(cum_counts + 1)
        curvature = np.gradient(np.gradient(log_counts))
        
        # Encontrar máximo de curvatura (más negativo)
        max_curve_idx = np.argmin(curvature)
        
        return bin_edges[max_curve_idx]

    def _smooth_D_series(self, D_series: pd.DataFrame, 
                        smoothing_window: int = 3) -> pd.DataFrame:
        """Suaviza la serie D(t) usando filtro de mediana."""
        if len(D_series) <= smoothing_window:
            return D_series
        
        D_smoothed = D_series['D'].rolling(
            window=smoothing_window, center=True, min_periods=1
        ).median()
        
        D_series = D_series.copy()
        D_series['D_smoothed'] = D_smoothed
        return D_series
