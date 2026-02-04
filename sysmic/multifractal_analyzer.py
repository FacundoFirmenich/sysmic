"""
COMPONENTE 2: ESPECTRO MULTIFRACTAL 3D COMPLETO
Implementación correcta para datos espaciales, sin proyección incorrecta a 1D
Métodos: Box-Counting 3D, Correlation Integral 3D, Moment Method 3D
Validación con fractales matemáticos conocidos
"""

import numpy as np
from scipy import spatial, stats, optimize, ndimage
from typing import Dict, List, Tuple, Optional, Union
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
import hashlib

@dataclass
class MultifractalResult:
    """Resultado estructurado de análisis multifractal."""
    q_values: np.ndarray
    D_q: np.ndarray
    tau_q: np.ndarray
    alpha: np.ndarray
    f_alpha: np.ndarray
    method: str
    quality_metrics: Dict
    spectrum_analysis: Dict

class Multifractal3DAnalyzer:
    """
    Analizador multifractal para datos espaciales 3D.
    Implementa métodos correctos para nubes de puntos 3D.
    
    Referencias:
    1. Halsey et al., "Fractal measures and their singularities", 1986
    2. Chhabra & Jensen, "Direct determination of f(α)", 1989
    3. Schertzer & Lovejoy, "Non-linear variability in geophysics", 1987
    """
    
    def __init__(self, 
                 q_min: float = -10.0,
                 q_max: float = 10.0,
                 n_q_points: int = 41,
                 n_scales: int = 20,
                 min_scale: float = 0.01,
                 max_scale: float = 0.5,
                 cache_size: int = 128):
        """
        Args:
            q_min: Valor mínimo de q (momentos negativos)
            q_max: Valor máximo de q (momentos positivos)
            n_q_points: Número de puntos en el espectro
            n_scales: Número de escalas para análisis
            min_scale: Escala mínima (fracción del extent)
            max_scale: Escala máxima (fracción del extent)
            cache_size: Tamaño de caché para resultados
        """
        self.q_values = np.linspace(q_min, q_max, n_q_points)
        self.n_scales = n_scales
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.cache = {}
        self.cache_size = cache_size
        self.logger = logging.getLogger(__name__)
        
        # Configurar tipos de métodos disponibles
        self.available_methods = {
            'boxcounting_3d': self._boxcounting_multifractal_3d,
            'correlation_3d': self._correlation_multifractal_3d,
            'moment_3d': self._moment_method_3d,
            'chhabra_jensen': self._chhabra_jensen_method
        }
        
    def compute_multifractal_spectrum(self,
                                     coordinates: np.ndarray,
                                     method: str = 'boxcounting_3d',
                                     normalize_coords: bool = True,
                                     compute_quality: bool = True,
                                     parallel: bool = False) -> MultifractalResult:
        """
        Calcula espectro multifractal D(q) para datos espaciales 3D.
        
        Args:
            coordinates: Array (N, 3) de coordenadas
            method: Método de cálculo
            normalize_coords: Normalizar a cubo unitario [0,1]^3
            compute_quality: Calcular métricas de calidad
            parallel: Usar paralelización para cálculos intensivos
            
        Returns:
            MultifractalResult con espectro completo y métricas
        """
        # Validar entrada
        coordinates = self._validate_and_preprocess(coordinates, normalize_coords)
        
        # Generar clave de caché
        cache_key = self._generate_cache_key(coordinates, method)
        
        # Verificar caché
        if cache_key in self.cache:
            self.logger.debug(f"Cache hit for key: {cache_key[:16]}")
            return self.cache[cache_key]
        
        # Seleccionar método
        if method not in self.available_methods:
            raise ValueError(
                f"Método desconocido: {method}. "
                f"Disponibles: {list(self.available_methods.keys())}"
            )
        
        # Calcular espectro
        self.logger.info(f"Calculando espectro multifractal con método: {method}")
        start_time = time.time()
        
        try:
            if parallel:
                result = self._compute_parallel(coordinates, method)
            else:
                result = self.available_methods[method](coordinates)
            
            # Calcular métricas de calidad
            if compute_quality:
                quality = self._compute_spectrum_quality(result, coordinates)
                result.quality_metrics = quality
            
            # Análisis del espectro
            analysis = self._analyze_spectrum(result.D_q, result.tau_q, 
                                            result.alpha, result.f_alpha)
            result.spectrum_analysis = analysis
            
            # Tiempo de ejecución
            result.computation_time = time.time() - start_time
            
            # Guardar en caché (si hay espacio)
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = result
                # Crear caché LRU manualmente
                if len(self.cache) > self.cache_size:
                    # Eliminar la entrada más antigua
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en cálculo multifractal: {str(e)}")
            raise
    
    def _validate_and_preprocess(self, 
                               coordinates: np.ndarray,
                               normalize: bool) -> np.ndarray:
        """Valida y preprocesa coordenadas."""
        # Validar forma
        if coordinates.ndim != 2:
            raise ValueError(f"Coordinates debe ser 2D, shape actual: {coordinates.shape}")
        
        if coordinates.shape[1] < 2:
            raise ValueError(f"Se requieren al menos 2 dimensiones, recibidas: {coordinates.shape[1]}")
        
        # Asegurar que sea float64
        coords = np.asarray(coordinates, dtype=np.float64)
        
        # Filtrar valores no finitos
        valid_mask = np.isfinite(coords).all(axis=1)
        if not np.all(valid_mask):
            n_invalid = np.sum(~valid_mask)
            self.logger.warning(f"Filtrando {n_invalid} puntos no finitos")
            coords = coords[valid_mask]
        
        if len(coords) < 100:
            raise ValueError(f"Muy pocos puntos válidos: {len(coords)}. Se requieren al menos 100.")
        
        # Normalizar a cubo unitario si se solicita
        if normalize:
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            ranges = maxs - mins
            
            # Evitar división por cero
            ranges = np.where(ranges == 0, 1.0, ranges)
            
            coords = (coords - mins) / ranges
        
        return coords
    
    def _generate_cache_key(self, 
                          coordinates: np.ndarray, 
                          method: str) -> str:
        """Genera clave única para caché."""
        # Hash de coordenadas y parámetros
        coord_hash = hashlib.md5(coordinates.tobytes()).hexdigest()[:16]
        param_hash = hashlib.md5(
            f"{method}_{self.q_values.tobytes()}_{self.n_scales}".encode()
        ).hexdigest()[:16]
        
        return f"{coord_hash}_{param_hash}"
    
    def _boxcounting_multifractal_3d(self, coordinates: np.ndarray) -> MultifractalResult:
        """
        Método de box-counting 3D generalizado.
        Implementa: partition function Z(q, ε) = Σ_i μ_i(ε)^q
        """
        # Generar escalas en progresión geométrica
        scales = np.logspace(
            np.log10(self.min_scale),
            np.log10(self.max_scale),
            self.n_scales
        )
        
        n_points = len(coordinates)
        
        # Precalcular discretizaciones para cada escala
        discretizations = []
        for scale in scales:
            bins = np.floor(coordinates / scale).astype(int)
            discretizations.append(bins)
        
        # Calcular Z(q, ε) para cada q y escala
        Z_matrix = np.zeros((len(self.q_values), len(scales)))
        
        for i, q in enumerate(self.q_values):
            for j, scale in enumerate(scales):
                bins = discretizations[j]
                
                # Calcular probabilidades por caja
                unique_boxes, counts = np.unique(bins, axis=0, return_counts=True)
                probabilities = counts / n_points
                
                # Regularización para q negativos
                prob_reg = probabilities + 1e-10
                prob_reg = prob_reg / prob_reg.sum()
                
                # Función de partición
                if np.abs(q - 1.0) < 1e-10:
                    # Caso límite q=1: entropy
                    Z_q = -np.sum(probabilities * np.log(prob_reg))
                else:
                    Z_q = np.sum(prob_reg ** q)
                
                Z_matrix[i, j] = Z_q
        
        # Calcular τ(q) por regresión log-log
        log_scales = np.log(scales)
        tau_q = np.zeros(len(self.q_values))
        tau_q_uncertainty = np.zeros(len(self.q_values))
        
        for i, q in enumerate(self.q_values):
            valid = Z_matrix[i, :] > 0
            if np.sum(valid) < 5:
                tau_q[i] = np.nan
                tau_q_uncertainty[i] = np.nan
                continue
            
            # Regresión lineal ponderada
            log_Z = np.log(Z_matrix[i, valid])
            scale_subset = log_scales[valid]
            
            # Ponderar por calidad del ajuste
            weights = 1.0 / (np.abs(log_Z - np.mean(log_Z)) + 0.1)
            
            # Ajuste robusto usando Theil-Sen para q=2
            if np.abs(q - 2.0) < 1e-2:
                # Para q=2 usar Theil-Sen como referencia
                slope_ts = self._theil_sen_slope(scale_subset, log_Z)
                tau_q[i] = slope_ts
            else:
                # Para otros q usar regresión lineal ponderada
                coeffs = np.polyfit(scale_subset, log_Z, 1, w=weights)
                tau_q[i] = coeffs[0]
            
            # Calcular incertidumbre mediante bootstrap
            tau_bootstrap = []
            for _ in range(50):  # Bootstrap rápido
                indices = np.random.choice(len(scale_subset), len(scale_subset), replace=True)
                coeffs_boot = np.polyfit(scale_subset[indices], log_Z[indices], 1)
                tau_bootstrap.append(coeffs_boot[0])
            
            tau_q_uncertainty[i] = np.std(tau_bootstrap)
        
        # Calcular D(q) = τ(q)/(q-1) para q≠1
        D_q = np.zeros_like(tau_q)
        for i, q in enumerate(self.q_values):
            if np.abs(q - 1.0) < 1e-10:
                # Caso q=1: dimensión de información (derivada de τ)
                if i > 0 and i < len(tau_q) - 1:
                    # Derivada numérica central
                    dtau_dq = (tau_q[i+1] - tau_q[i-1]) / (self.q_values[i+1] - self.q_values[i-1])
                    D_q[i] = dtau_dq
                else:
                    D_q[i] = np.nan
            else:
                D_q[i] = tau_q[i] / (q - 1)
        
        # Calcular espectro de singularidades f(α)
        # α = dτ/dq, f(α) = qα - τ(q)
        alpha = np.zeros_like(tau_q)
        f_alpha = np.zeros_like(tau_q)
        
        # Calcular derivada numérica de τ(q)
        for i in range(len(tau_q)):
            if i == 0:
                # Diferencia hacia adelante
                dtau = tau_q[i+1] - tau_q[i]
                dq = self.q_values[i+1] - self.q_values[i]
            elif i == len(tau_q) - 1:
                # Diferencia hacia atrás
                dtau = tau_q[i] - tau_q[i-1]
                dq = self.q_values[i] - self.q_values[i-1]
            else:
                # Diferencia central
                dtau = tau_q[i+1] - tau_q[i-1]
                dq = self.q_values[i+1] - self.q_values[i-1]
            
            if dq != 0:
                alpha[i] = dtau / dq
                f_alpha[i] = self.q_values[i] * alpha[i] - tau_q[i]
            else:
                alpha[i] = np.nan
                f_alpha[i] = np.nan
        
        return MultifractalResult(
            q_values=self.q_values.copy(),
            D_q=D_q,
            tau_q=tau_q,
            alpha=alpha,
            f_alpha=f_alpha,
            method='boxcounting_3d',
            quality_metrics={
                'tau_uncertainty': tau_q_uncertainty,
                'n_scales_used': self.n_scales,
                'min_scale': float(self.min_scale),
                'max_scale': float(self.max_scale)
            },
            spectrum_analysis={}  # Se llenará después
        )
    
    def _correlation_multifractal_3d(self, coordinates: np.ndarray) -> MultifractalResult:
        """
        Método de correlación 3D generalizado.
        Usa función de correlación generalizada C(q, r).
        """
        n_points = len(coordinates)
        
        # Submuestreo para matrices de distancia grandes
        if n_points > 2000:
            sample_size = 2000
            indices = np.random.choice(n_points, sample_size, replace=False)
            sample = coordinates[indices]
            self.logger.info(f"Submuestreando a {sample_size} puntos para cálculo de correlación")
        else:
            sample = coordinates
            sample_size = n_points
        
        # Calcular matriz de distancias (triangular superior)
        self.logger.info("Calculando matriz de distancias...")
        dist_matrix = spatial.distance.pdist(sample)
        
        # Determinar rangos de escalas adaptativos
        min_dist = np.percentile(dist_matrix, 1)
        max_dist = np.percentile(dist_matrix, 50)  # Usar percentil 50 para evitar outliers
        min_dist = max(min_dist, 1e-10)
        
        scales = np.logspace(
            np.log10(min_dist),
            np.log10(max_dist),
            self.n_scales
        )
        
        # Precalcular conteos para cada escala
        self.logger.info("Precalculando conteos por escala...")
        counts_by_scale = []
        
        for r in scales:
            # Contar pares dentro de distancia r
            count = np.sum(dist_matrix <= r)
            counts_by_scale.append(count)
        
        counts_by_scale = np.array(counts_by_scale)
        total_pairs = len(dist_matrix)
        
        # Normalizar a probabilidades acumulativas
        C_r = counts_by_scale / total_pairs
        
        # Calcular Z(q, r) = C(q, r) para diferentes q
        Z_matrix = np.zeros((len(self.q_values), len(scales)))
        
        for i, q in enumerate(self.q_values):
            if np.abs(q - 1.0) < 1e-10:
                # Caso q=1: usar -C(r) * log(C(r))
                valid = C_r > 0
                Z_matrix[i, valid] = -C_r[valid] * np.log(C_r[valid] + 1e-10)
            else:
                Z_matrix[i, :] = C_r ** q
        
        # Calcular τ(q) por regresión
        log_scales = np.log(scales)
        tau_q = np.zeros(len(self.q_values))
        
        for i, q in enumerate(self.q_values):
            valid = (Z_matrix[i, :] > 0) & np.isfinite(Z_matrix[i, :])
            if np.sum(valid) < 5:
                tau_q[i] = np.nan
                continue
            
            # Regresión lineal en región lineal
            log_Z = np.log(Z_matrix[i, valid])
            scale_subset = log_scales[valid]
            
            # Usar regresión robusta
            slope, _ = np.polyfit(scale_subset, log_Z, 1)
            tau_q[i] = slope
        
        # Calcular D(q)
        D_q = np.zeros_like(tau_q)
        for i, q in enumerate(self.q_values):
            if np.abs(q - 1.0) < 1e-10:
                if i > 0 and i < len(tau_q) - 1:
                    dtau_dq = (tau_q[i+1] - tau_q[i-1]) / (self.q_values[i+1] - self.q_values[i-1])
                    D_q[i] = dtau_dq
                else:
                    D_q[i] = np.nan
            else:
                D_q[i] = tau_q[i] / (q - 1)
        
        # Calcular α y f(α)
        alpha, f_alpha = self._compute_alpha_falpha(tau_q)
        
        return MultifractalResult(
            q_values=self.q_values.copy(),
            D_q=D_q,
            tau_q=tau_q,
            alpha=alpha,
            f_alpha=f_alpha,
            method='correlation_3d',
            quality_metrics={
                'n_points_used': sample_size,
                'min_dist': float(min_dist),
                'max_dist': float(max_dist),
                'scales': scales
            },
            spectrum_analysis={}
        )
    
    def _moment_method_3d(self, coordinates: np.ndarray) -> MultifractalResult:
        """
        Método de momentos 3D para datos discretos.
        Partición del espacio en celdas y cálculo de momentos.
        """
        # Generar malla 3D de diferentes resoluciones
        resolutions = 2 ** np.arange(2, 8)  # Resoluciones: 4, 8, 16, 32, 64, 128
        
        n_points = len(coordinates)
        Z_matrix = np.zeros((len(self.q_values), len(resolutions)))
        
        for j, res in enumerate(resolutions):
            # Discretizar en malla res x res x res
            bins = np.floor(coordinates * res).astype(int)
            bins = np.clip(bins, 0, res - 1)
            
            # Contar puntos por celda
            unique_cells, counts = np.unique(bins, axis=0, return_counts=True)
            probabilities = counts / n_points
            
            # Calcular Z(q, ε) para cada q
            for i, q in enumerate(self.q_values):
                if np.abs(q - 1.0) < 1e-10:
                    # Entropía de Shannon
                    prob_reg = probabilities + 1e-10
                    prob_reg = prob_reg / prob_reg.sum()
                    Z_q = -np.sum(probabilities * np.log(prob_reg))
                else:
                    Z_q = np.sum(probabilities ** q)
                
                Z_matrix[i, j] = Z_q
        
        # Escalas ε = 1/res (tamaño de celda)
        scales = 1.0 / resolutions
        
        # Calcular τ(q) por regresión
        log_scales = np.log(scales)
        tau_q = np.zeros(len(self.q_values))
        
        for i, q in enumerate(self.q_values):
            valid = Z_matrix[i, :] > 0
            if np.sum(valid) < 3:
                tau_q[i] = np.nan
                continue
            
            log_Z = np.log(Z_matrix[i, valid])
            scale_subset = log_scales[valid]
            
            slope, _ = np.polyfit(scale_subset, log_Z, 1)
            tau_q[i] = slope
        
        # Calcular D(q), α, f(α)
        D_q = self._compute_Dq_from_tauq(tau_q)
        alpha, f_alpha = self._compute_alpha_falpha(tau_q)
        
        return MultifractalResult(
            q_values=self.q_values.copy(),
            D_q=D_q,
            tau_q=tau_q,
            alpha=alpha,
            f_alpha=f_alpha,
            method='moment_3d',
            quality_metrics={
                'resolutions': resolutions.tolist(),
                'n_cells_used': [len(np.unique(np.floor(coordinates * res), axis=0)) 
                                for res in resolutions]
            },
            spectrum_analysis={}
        )
    
    def _chhabra_jensen_method(self, coordinates: np.ndarray) -> MultifractalResult:
        """
        Implementación del método directo de Chhabra & Jensen (1989).
        Calcula α y f(α) directamente sin calcular τ(q) primero.
        """
        n_points = len(coordinates)
        
        # Generar escalas
        scales = np.logspace(
            np.log10(self.min_scale),
            np.log10(self.max_scale),
            self.n_scales
        )
        
        # Para cada q, calcular α(q) y f(q) directamente
        alpha_q = np.zeros(len(self.q_values))
        f_q = np.zeros(len(self.q_values))
        
        for i, q in enumerate(self.q_values):
            # Calcular medidas μ_i(q, ε) para cada escala
            alpha_by_scale = []
            f_by_scale = []
            
            for scale in scales:
                # Discretizar
                bins = np.floor(coordinates / scale).astype(int)
                unique_boxes, counts = np.unique(bins, axis=0, return_counts=True)
                probabilities = counts / n_points
                
                # Regularizar
                prob_reg = probabilities + 1e-10
                prob_reg = prob_reg / prob_reg.sum()
                
                # Calcular medidas μ_i(q, ε)
                if np.abs(q - 1.0) < 1e-10:
                    # Caso límite q=1
                    mu_i = -probabilities * np.log(prob_reg)
                    mu_i = mu_i / np.sum(mu_i + 1e-10)
                    alpha_i = -np.log(prob_reg) / np.log(scale)
                else:
                    mu_i = (prob_reg ** q) / np.sum(prob_reg ** q)
                    alpha_i = np.log(prob_reg) / np.log(scale)
                
                # α(q) = Σ μ_i(q, ε) * α_i
                alpha_scale = np.sum(mu_i * alpha_i)
                
                # f(q) = Σ μ_i(q, ε) * log(μ_i(q, ε)) / log(ε)
                f_scale = np.sum(mu_i * np.log(mu_i + 1e-10)) / np.log(scale)
                
                alpha_by_scale.append(alpha_scale)
                f_by_scale.append(f_scale)
            
            # Promediar sobre escalas (excluyendo valores extremos)
            alpha_by_scale = np.array(alpha_by_scale)
            f_by_scale = np.array(f_by_scale)
            
            # Filtrar valores no finitos
            valid = np.isfinite(alpha_by_scale) & np.isfinite(f_by_scale)
            if np.sum(valid) < 3:
                alpha_q[i] = np.nan
                f_q[i] = np.nan
            else:
                alpha_q[i] = np.mean(alpha_by_scale[valid])
                f_q[i] = np.mean(f_by_scale[valid])
        
        # Calcular D(q) = (q*α(q) - f(q)) / (q-1) para q≠1
        D_q = np.zeros_like(alpha_q)
        for i, q in enumerate(self.q_values):
            if np.abs(q - 1.0) < 1e-10:
                D_q[i] = f_q[i]  # Para q=1, D(1) = f(α(1))
            else:
                D_q[i] = (q * alpha_q[i] - f_q[i]) / (q - 1)
        
        # Calcular τ(q) = q*α(q) - f(q)
        tau_q = self.q_values * alpha_q - f_q
        
        return MultifractalResult(
            q_values=self.q_values.copy(),
            D_q=D_q,
            tau_q=tau_q,
            alpha=alpha_q,
            f_alpha=f_q,
            method='chhabra_jensen',
            quality_metrics={
                'n_scales_used': len(scales),
                'alpha_range': [float(np.nanmin(alpha_q)), float(np.nanmax(alpha_q))],
                'f_range': [float(np.nanmin(f_q)), float(np.nanmax(f_q))]
            },
            spectrum_analysis={}
        )
    
    def _theil_sen_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """Estimador Theil-Sen robusto para pendientes."""
        n = len(x)
        slopes = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if x[j] != x[i]:
                    slope = (y[j] - y[i]) / (x[j] - x[i])
                    slopes.append(slope)
        
        if len(slopes) == 0:
            return 0.0
        
        return float(np.median(slopes))
    
    def _compute_Dq_from_tauq(self, tau_q: np.ndarray) -> np.ndarray:
        """Calcula D(q) a partir de τ(q)."""
        D_q = np.zeros_like(tau_q)
        
        for i, q in enumerate(self.q_values):
            if np.abs(q - 1.0) < 1e-10:
                # Caso q=1: derivada numérica
                if i > 0 and i < len(tau_q) - 1:
                    dtau_dq = (tau_q[i+1] - tau_q[i-1]) / (self.q_values[i+1] - self.q_values[i-1])
                    D_q[i] = dtau_dq
                else:
                    D_q[i] = np.nan
            else:
                D_q[i] = tau_q[i] / (q - 1)
        
        return D_q
    
    def _compute_alpha_falpha(self, tau_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula α y f(α) a partir de τ(q) usando derivadas numéricas."""
        alpha = np.zeros_like(tau_q)
        f_alpha = np.zeros_like(tau_q)
        
        n = len(tau_q)
        
        for i in range(n):
            # Derivada dτ/dq
            if i == 0:
                dtau = tau_q[i+1] - tau_q[i]
                dq = self.q_values[i+1] - self.q_values[i]
            elif i == n - 1:
                dtau = tau_q[i] - tau_q[i-1]
                dq = self.q_values[i] - self.q_values[i-1]
            else:
                dtau = tau_q[i+1] - tau_q[i-1]
                dq = self.q_values[i+1] - self.q_values[i-1]
            
            if dq != 0:
                alpha[i] = dtau / dq
                f_alpha[i] = self.q_values[i] * alpha[i] - tau_q[i]
            else:
                alpha[i] = np.nan
                f_alpha[i] = np.nan
        
        return alpha, f_alpha
    
    def _compute_spectrum_quality(self,
                                 result: MultifractalResult,
                                 coordinates: np.ndarray) -> Dict:
        """Calcula métricas de calidad del espectro multifractal."""
        quality = {}
        
        # 1. Completitud del espectro
        valid_D = np.isfinite(result.D_q)
        completeness = np.sum(valid_D) / len(result.D_q)
        quality['spectrum_completeness'] = float(completeness)
        
        # 2. Consistencia monótona (D(q) debe ser no creciente)
        if completeness > 0.5:
            valid_indices = np.where(valid_D)[0]
            valid_q = self.q_values[valid_indices]
            valid_Dq = result.D_q[valid_indices]
            
            # Ordenar por q
            sort_idx = np.argsort(valid_q)
            sorted_q = valid_q[sort_idx]
            sorted_Dq = valid_Dq[sort_idx]
            
            # Calcular diferencia entre D(q) consecutivos
            diffs = np.diff(sorted_Dq)
            non_increasing = np.sum(diffs <= 0.1) / len(diffs)  # Tolerancia pequeña
            
            quality['monotonicity_score'] = float(non_increasing)
        else:
            quality['monotonicity_score'] = np.nan
        
        # 3. Suavidad del espectro
        if completeness > 0.7:
            valid_Dq = result.D_q[valid_D]
            # Calcular variación de segundo orden
            curvature = np.abs(np.gradient(np.gradient(valid_Dq)))
            smoothness = 1.0 / (1.0 + np.mean(curvature))
            quality['smoothness_score'] = float(smoothness)
        else:
            quality['smoothness_score'] = np.nan
        
        # 4. Consistencia con propiedades teóricas
        # D(0) ≥ D(1) ≥ D(2) (para fractales monofractales)
        idx_0 = np.argmin(np.abs(self.q_values - 0))
        idx_1 = np.argmin(np.abs(self.q_values - 1))
        idx_2 = np.argmin(np.abs(self.q_values - 2))
        
        if all(np.isfinite([result.D_q[idx_0], result.D_q[idx_1], result.D_q[idx_2]])):
            theory_score = 0
            if result.D_q[idx_0] >= result.D_q[idx_1]:
                theory_score += 1
            if result.D_q[idx_1] >= result.D_q[idx_2]:
                theory_score += 1
            quality['theory_consistency'] = theory_score / 2
        else:
            quality['theory_consistency'] = np.nan
        
        # 5. Error de ajuste para τ(q)
        if 'tau_uncertainty' in result.quality_metrics:
            tau_unc = result.quality_metrics['tau_uncertainty']
            valid_tau_unc = tau_unc[np.isfinite(tau_unc)]
            if len(valid_tau_unc) > 0:
                quality['average_tau_uncertainty'] = float(np.mean(valid_tau_unc))
            else:
                quality['average_tau_uncertainty'] = np.nan
        
        return quality
    
    def _analyze_spectrum(self,
                         D_q: np.ndarray,
                         tau_q: np.ndarray,
                         alpha: np.ndarray,
                         f_alpha: np.ndarray) -> Dict:
        """Análisis e interpretación del espectro multifractal."""
        analysis = {}
        
        # 1. Valores específicos importantes
        # D0: Dimensión de capacidad (box-counting)
        idx_0 = np.argmin(np.abs(self.q_values - 0))
        if np.isfinite(D_q[idx_0]):
            analysis['D0'] = float(D_q[idx_0])
        else:
            analysis['D0'] = np.nan
        
        # D1: Dimensión de información (Shannon)
        idx_1 = np.argmin(np.abs(self.q_values - 1))
        if np.isfinite(D_q[idx_1]):
            analysis['D1'] = float(D_q[idx_1])
            analysis['D1_info_dimension'] = True
        else:
            analysis['D1'] = np.nan
            analysis['D1_info_dimension'] = False
        
        # D2: Dimensión de correlación (Grassberger-Procaccia)
        idx_2 = np.argmin(np.abs(self.q_values - 2))
        if np.isfinite(D_q[idx_2]):
            analysis['D2'] = float(D_q[idx_2])
        else:
            analysis['D2'] = np.nan
        
        # 2. Ancho del espectro
        valid_D = D_q[np.isfinite(D_q)]
        if len(valid_D) > 0:
            analysis['spectrum_width'] = float(np.max(valid_D) - np.min(valid_D))
            analysis['D_min'] = float(np.min(valid_D))
            analysis['D_max'] = float(np.max(valid_D))
        else:
            analysis['spectrum_width'] = np.nan
            analysis['D_min'] = np.nan
            analysis['D_max'] = np.nan
        
        # 3. Asimetría
        if len(valid_D) > 0:
            median_D = np.median(valid_D)
            left_half = valid_D[valid_D <= median_D]
            right_half = valid_D[valid_D >= median_D]
            
            if len(left_half) > 0 and len(right_half) > 0:
                skewness = (np.mean(right_half) - median_D) - (median_D - np.mean(left_half))
                analysis['spectrum_skewness'] = float(skewness)
            else:
                analysis['spectrum_skewness'] = np.nan
        else:
            analysis['spectrum_skewness'] = np.nan
        
        # 4. Clasificación
        if analysis['spectrum_width'] is not np.nan:
            width = analysis['spectrum_width']
            if width < 0.1:
                classification = "Monofractal"
                interpretation = "El sistema exhibe una única dimensión fractal, " \
                               "característica de procesos simples o auto-similares."
            elif width < 0.5:
                classification = "Débilmente multifractal"
                interpretation = "Existe cierta multifractalidad, pero dominada " \
                               "por unos pocos exponentes de singularidad."
            else:
                classification = "Fuertemente multifractal"
                interpretation = "El sistema exhibe un amplio rango de exponentes " \
                               "de singularidad, indicando heterogeneidad compleja."
        else:
            classification = "Indeterminado"
            interpretation = "No se pudo determinar la multifractalidad debido a " \
                           "datos insuficientes o errores en el cálculo."
        
        analysis['classification'] = classification
        analysis['interpretation'] = interpretation
        
        # 5. Relación D1 vs D2 (para monofractales deberían ser iguales)
        if np.isfinite(analysis['D1']) and np.isfinite(analysis['D2']):
            D1_D2_diff = np.abs(analysis['D1'] - analysis['D2'])
            analysis['D1_D2_difference'] = float(D1_D2_diff)
            if D1_D2_diff < 0.1:
                analysis['monofractal_consistency'] = "Alta"
            elif D1_D2_diff < 0.3:
                analysis['monofractal_consistency'] = "Moderada"
            else:
                analysis['monofractal_consistency'] = "Baja"
        else:
            analysis['D1_D2_difference'] = np.nan
            analysis['monofractal_consistency'] = "Indeterminado"
        
        # 6. Espectro de singularidades
        valid_alpha = alpha[np.isfinite(alpha) & np.isfinite(f_alpha)]
        valid_f = f_alpha[np.isfinite(alpha) & np.isfinite(f_alpha)]
        
        if len(valid_alpha) > 0:
            analysis['alpha_min'] = float(np.min(valid_alpha))
            analysis['alpha_max'] = float(np.max(valid_alpha))
            analysis['alpha_range'] = float(analysis['alpha_max'] - analysis['alpha_min'])
            analysis['f_max'] = float(np.max(valid_f))
            
            # Punto donde f(α) es máximo (dimensión de Hausdorff)
            max_f_idx = np.argmax(valid_f)
            analysis['alpha_at_fmax'] = float(valid_alpha[max_f_idx])
            analysis['fmax_value'] = float(valid_f[max_f_idx])
        else:
            analysis['alpha_min'] = np.nan
            analysis['alpha_max'] = np.nan
            analysis['alpha_range'] = np.nan
            analysis['f_max'] = np.nan
            analysis['alpha_at_fmax'] = np.nan
            analysis['fmax_value'] = np.nan
        
        return analysis
    
    def validate_with_known_fractals(self) -> Dict:
        """
        Valida el analizador con fractales matemáticos de dimensión conocida.
        
        Returns:
            Diccionario con resultados de validación
        """
        validation_results = {}
        
        # 1. Conjunto de Cantor 3D (D ≈ log(8)/log(3) ≈ 1.8928)
        cantor_3d = self._generate_cantor_3d(iterations=5)
        result_cantor = self.compute_multifractal_spectrum(cantor_3d, method='boxcounting_3d')
        D0_cantor = result_cantor.spectrum_analysis.get('D0', np.nan)
        error_cantor = np.abs(D0_cantor - 1.8928) if np.isfinite(D0_cantor) else np.nan
        
        validation_results['cantor_3d'] = {
            'expected_D0': 1.8928,
            'measured_D0': D0_cantor,
            'absolute_error': error_cantor,
            'relative_error': error_cantor / 1.8928 if error_cantor is not np.nan else np.nan,
            'classification': result_cantor.spectrum_analysis.get('classification', 'Unknown'),
            'passed': error_cantor < 0.1 if error_cantor is not np.nan else False
        }
        
        # 2. Alfombra de Sierpinski 3D (D ≈ log(20)/log(3) ≈ 2.7268)
        sierpinski_3d = self._generate_sierpinski_carpet_3d(iterations=4)
        result_sierpinski = self.compute_multifractal_spectrum(sierpinski_3d, method='boxcounting_3d')
        D0_sierpinski = result_sierpinski.spectrum_analysis.get('D0', np.nan)
        error_sierpinski = np.abs(D0_sierpinski - 2.7268) if np.isfinite(D0_sierpinski) else np.nan
        
        validation_results['sierpinski_3d'] = {
            'expected_D0': 2.7268,
            'measured_D0': D0_sierpinski,
            'absolute_error': error_sierpinski,
            'relative_error': error_sierpinski / 2.7268 if error_sierpinski is not np.nan else np.nan,
            'classification': result_sierpinski.spectrum_analysis.get('classification', 'Unknown'),
            'passed': error_sierpinski < 0.15 if error_sierpinski is not np.nan else False
        }
        
        # 3. Plano 2D en 3D (D = 2.0)
        plane_3d = self._generate_plane_3d(n_points=2000)
        result_plane = self.compute_multifractal_spectrum(plane_3d, method='boxcounting_3d')
        D0_plane = result_plane.spectrum_analysis.get('D0', np.nan)
        error_plane = np.abs(D0_plane - 2.0) if np.isfinite(D0_plane) else np.nan
        
        validation_results['plane_3d'] = {
            'expected_D0': 2.0,
            'measured_D0': D0_plane,
            'absolute_error': error_plane,
            'relative_error': error_plane / 2.0 if error_plane is not np.nan else np.nan,
            'classification': result_plane.spectrum_analysis.get('classification', 'Unknown'),
            'passed': error_plane < 0.1 if error_plane is not np.nan else False
        }
        
        # 4. Línea 1D en 3D (D = 1.0)
        line_3d = self._generate_line_3d(n_points=1000)
        result_line = self.compute_multifractal_spectrum(line_3d, method='boxcounting_3d')
        D0_line = result_line.spectrum_analysis.get('D0', np.nan)
        error_line = np.abs(D0_line - 1.0) if np.isfinite(D0_line) else np.nan
        
        validation_results['line_3d'] = {
            'expected_D0': 1.0,
            'measured_D0': D0_line,
            'absolute_error': error_line,
            'relative_error': error_line / 1.0 if error_line is not np.nan else np.nan,
            'classification': result_line.spectrum_analysis.get('classification', 'Unknown'),
            'passed': error_line < 0.1 if error_line is not np.nan else False
        }
        
        # Resumen de validación
        passed_tests = sum(1 for v in validation_results.values() if v['passed'])
        total_tests = len(validation_results)
        
        validation_results['summary'] = {
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'validation_status': 'PASSED' if passed_tests == total_tests else 'PARTIAL',
            'average_relative_error': np.nanmean([
                v['relative_error'] for v in validation_results.values() 
                if v['relative_error'] is not np.nan
            ])
        }
        
        return validation_results
    
    def _generate_cantor_3d(self, iterations: int = 5) -> np.ndarray:
        """Genera conjunto de Cantor 3D."""
        points = [np.array([0.5, 0.5, 0.5])]  # Punto inicial
        
        for _ in range(iterations):
            new_points = []
            for p in points:
                for dx in [0, 1/3, 2/3]:
                    for dy in [0, 1/3, 2/3]:
                        for dz in [0, 1/3, 2/3]:
                            # Solo mantener puntos donde al menos una coordenada es 1/3
                            if (dx == 1/3) or (dy == 1/3) or (dz == 1/3):
                                new_points.append(p + np.array([dx, dy, dz]) / (3 ** (iterations - 1)))
            points = new_points
        
        return np.array(points)
    
    def _generate_sierpinski_carpet_3d(self, iterations: int = 4) -> np.ndarray:
        """Genera alfombra de Sierpinski 3D (Menger sponge)."""
        points = [np.array([0.5, 0.5, 0.5])]
        
        for _ in range(iterations):
            new_points = []
            for p in points:
                for dx in [0, 1/3, 2/3]:
                    for dy in [0, 1/3, 2/3]:
                        for dz in [0, 1/3, 2/3]:
                            # Eliminar el cubo central y los centros de las caras
                            if not ((dx == 1/3 and dy == 1/3) or
                                   (dx == 1/3 and dz == 1/3) or
                                   (dy == 1/3 and dz == 1/3)):
                                new_points.append(p + np.array([dx, dy, dz]) / (3 ** (iterations - 1)))
            points = new_points
        
        return np.array(points)
    
    def _generate_plane_3d(self, n_points: int = 2000) -> np.ndarray:
        """Genera plano 2D en espacio 3D."""
        np.random.seed(42)
        x = np.random.rand(n_points)
        y = np.random.rand(n_points)
        z = np.zeros(n_points)  # Todos en z=0
        return np.column_stack([x, y, z])
    
    def _generate_line_3d(self, n_points: int = 1000) -> np.ndarray:
        """Genera línea 1D en espacio 3D."""
        np.random.seed(42)
        t = np.linspace(0, 1, n_points)
        x = t
        y = np.zeros_like(t)
        z = np.zeros_like(t)
        return np.column_stack([x, y, z])
    
    def _compute_parallel(self, coordinates: np.ndarray, method: str) -> MultifractalResult:
        """Ejecuta cálculo en paralelo si está disponible."""
        try:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            
            n_cores = multiprocessing.cpu_count()
            self.logger.info(f"Ejecutando en paralelo con {n_cores} cores")
            
            # Dividir trabajo por valores de q
            n_q = len(self.q_values)
            q_chunks = np.array_split(self.q_values, n_cores)
            
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                futures = []
                for chunk in q_chunks:
                    future = executor.submit(
                        self._compute_chunk,
                        coordinates, method, chunk
                    )
                    futures.append(future)
                
                # Combinar resultados
                results = [f.result() for f in futures]
                
                # Reconstruir arrays completos
                all_D_q = np.concatenate([r.D_q for r in results])
                all_tau_q = np.concatenate([r.tau_q for r in results])
                all_alpha = np.concatenate([r.alpha for r in results])
                all_f_alpha = np.concatenate([r.f_alpha for r in results])
                
                # Ordenar por q
                sort_idx = np.argsort(np.concatenate([chunk for chunk in q_chunks]))
                
                return MultifractalResult(
                    q_values=self.q_values.copy(),
                    D_q=all_D_q[sort_idx],
                    tau_q=all_tau_q[sort_idx],
                    alpha=all_alpha[sort_idx],
                    f_alpha=all_f_alpha[sort_idx],
                    method=f"{method}_parallel",
                    quality_metrics={'parallel_cores': n_cores},
                    spectrum_analysis={}
                )
                
        except ImportError:
            self.logger.warning("Parallel execution not available, falling back to serial")
            return self.available_methods[method](coordinates)
    
    def _compute_chunk(self, coordinates: np.ndarray, method: str, q_chunk: np.ndarray) -> MultifractalResult:
        """Calcula un chunk del espectro (para paralelización)."""
        # Crear analizador temporal para este chunk
        temp_analyzer = Multifractal3DAnalyzer(
            q_min=np.min(q_chunk),
            q_max=np.max(q_chunk),
            n_q_points=len(q_chunk),
            n_scales=self.n_scales,
            min_scale=self.min_scale,
            max_scale=self.max_scale
        )
        
        # Calcular solo para este chunk
        result = temp_analyzer.available_methods[method](coordinates)
        
        return result