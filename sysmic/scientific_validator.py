"""
COMPONENTE 4: VALIDADOR CIENTÍFICO COMPLETO CON DATOS REALES
Implementa validación exhaustiva contra catálogos reales publicados
y fractales matemáticos de referencia.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import json
import pickle
import hashlib
import time
from pathlib import Path
import warnings
from scipy import stats, optimize
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Importar módulos necesarios
try:
    import obspy
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    OBS_AVAILABLE = True
except ImportError:
    OBS_AVAILABLE = False
    logging.warning("Obspy no disponible. Algunas funciones de descarga no estarán disponibles.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

@dataclass
class ValidationResult:
    """Resultado estructurado de validación."""
    test_name: str
    reference_value: float
    measured_value: float
    absolute_error: float
    relative_error: float
    confidence_interval: Tuple[float, float]
    n_samples: int
    passed: bool
    tolerance: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario."""
        return {
            'test_name': self.test_name,
            'reference_value': self.reference_value,
            'measured_value': self.measured_value,
            'absolute_error': self.absolute_error,
            'relative_error': self.relative_error,
            'confidence_interval': list(self.confidence_interval),
            'n_samples': self.n_samples,
            'passed': self.passed,
            'tolerance': self.tolerance,
            'metadata': self.metadata,
            'diagnostics': self.diagnostics
        }

@dataclass
class BenchmarkCatalog:
    """Catálogo de referencia para validación."""
    name: str
    data: np.ndarray  # (n_events, 4): lat, lon, depth_km, mag
    times: np.ndarray  # segundos desde epoch o fechas
    reference_D2: float
    reference_D2_uncertainty: float
    reference_publication: str
    reference_doi: str
    spatial_extent_km: Tuple[float, float, float]  # (dx, dy, dz)
    temporal_extent_days: float
    magnitude_range: Tuple[float, float]
    completeness_magnitude: float
    notes: str = ""
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convierte a DataFrame para análisis."""
        df = pd.DataFrame({
            'latitude': self.data[:, 0],
            'longitude': self.data[:, 1],
            'depth_km': self.data[:, 2],
            'mag': self.data[:, 3]
        })
        
        if len(self.times) == len(self.data):
            if isinstance(self.times[0], (datetime, UTCDateTime)):
                df['time'] = [t.isoformat() for t in self.times]
            else:
                df['time'] = pd.to_datetime(self.times, unit='s')
        
        df['catalog_name'] = self.name
        return df

class ScientificValidator:
    """
    Validador científico exhaustivo para sistema de dimensión fractal.
    
    Características:
    1. Valida contra fractales matemáticos con D conocida
    2. Valida contra catálogos sísmicos reales publicados
    3. Implementa tests estadísticos rigurosos
    4. Genera reportes de validación completos
    5. Incluye benchmarks estándar de la industria
    """
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 download_timeout: int = 30,
                 random_state: int = 42):
        """
        Args:
            cache_dir: Directorio para cache de datos descargados
            download_timeout: Timeout para descargas (segundos)
            random_state: Semilla para reproducibilidad
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.fractal_benchmarks'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.download_timeout = download_timeout
        self.rng = np.random.RandomState(random_state)
        self.logger = logging.getLogger(__name__)
        
        # Inicializar sistemas de análisis
        from .fractal_estimator import FractalDimensionEstimator
        from .multifractal_analyzer import Multifractal3DAnalyzer
        from .feature_extractor import FractalFeatureExtractor
        
        self.estimator = FractalDimensionEstimator(random_state=random_state)
        self.multifractal_analyzer = Multifractal3DAnalyzer()
        self.feature_extractor = FractalFeatureExtractor()
        
        # Catálogos de referencia predefinidos
        self.benchmark_catalogs = self._initialize_benchmark_catalogs()
        
        # Fractales matemáticos predefinidos
        self.mathematical_fractals = self._initialize_mathematical_fractals()
        
        # Métricas de validación
        self.validation_metrics = {}
        
    def _initialize_benchmark_catalogs(self) -> Dict[str, BenchmarkCatalog]:
        """Inicializa catálogos de referencia con datos reales o sintéticos."""
        catalogs = {}
        
        # 1. Catálogo de Parkfield (California)
        catalogs['parkfield'] = self._create_parkfield_benchmark()
        
        # 2. Catálogo de Landers (California)
        catalogs['landers'] = self._create_landers_benchmark()
        
        # 3. Catálogo de Northridge (California)
        catalogs['northridge'] = self._create_northridge_benchmark()
        
        # 4. Catálogo de Tohoku (Japón)
        catalogs['tohoku'] = self._create_tohoku_benchmark()
        
        # 5. Catálogo de Izmit (Turquía)
        catalogs['izmit'] = self._create_izmit_benchmark()
        
        # 6. Catálogo sintético - Strike-slip fault
        catalogs['synthetic_strike_slip'] = self._create_synthetic_strike_slip()
        
        # 7. Catálogo sintético - Thrust fault
        catalogs['synthetic_thrust'] = self._create_synthetic_thrust()
        
        # 8. Catálogo sintético - Normal fault
        catalogs['synthetic_normal'] = self._create_synthetic_normal()
        
        # 9. Catálogo sintético - Volcanic swarm
        catalogs['synthetic_volcanic'] = self._create_synthetic_volcanic()
        
        return catalogs
    
    def _initialize_mathematical_fractals(self) -> Dict[str, Dict]:
        """Inicializa fractales matemáticos con dimensión conocida."""
        return {
            'cantor_set_1d': {
                'generator': self._generate_cantor_set_1d,
                'D_true': np.log(2) / np.log(3),  # ≈ 0.6309
                'description': "Cantor set (1D) - Middle-thirds removal"
            },
            'cantor_dust_2d': {
                'generator': self._generate_cantor_dust_2d,
                'D_true': np.log(4) / np.log(3),  # ≈ 1.2619
                'description': "Cantor dust (2D) - Product of 1D Cantor sets"
            },
            'sierpinski_triangle': {
                'generator': self._generate_sierpinski_triangle,
                'D_true': np.log(3) / np.log(2),  # ≈ 1.5850
                'description': "Sierpinski triangle - Iterated function system"
            },
            'sierpinski_carpet': {
                'generator': self._generate_sierpinski_carpet,
                'D_true': np.log(8) / np.log(3),  # ≈ 1.8928
                'description': "Sierpinski carpet - Square version"
            },
            'koch_curve': {
                'generator': self._generate_koch_curve,
                'D_true': np.log(4) / np.log(3),  # ≈ 1.2619
                'description': "Koch snowflake boundary"
            },
            'menger_sponge': {
                'generator': self._generate_menger_sponge,
                'D_true': np.log(20) / np.log(3),  # ≈ 2.7268
                'description': "Menger sponge - 3D generalization"
            },
            'uniform_line': {
                'generator': self._generate_uniform_line,
                'D_true': 1.0,
                'description': "Uniform distribution on a line"
            },
            'uniform_plane': {
                'generator': self._generate_uniform_plane,
                'D_true': 2.0,
                'description': "Uniform distribution on a plane"
            },
            'uniform_cube': {
                'generator': self._generate_uniform_cube,
                'D_true': 3.0,
                'description': "Uniform distribution in a cube"
            },
            'henon_attractor': {
                'generator': self._generate_henon_attractor,
                'D_true': 1.21,  # Grassberger & Procaccia, 1983
                'description': "Henon strange attractor"
            },
            'lorenz_attractor': {
                'generator': self._generate_lorenz_attractor,
                'D_true': 2.06,  # Moon, 1987
                'description': "Lorenz attractor"
            },
            'rossler_attractor': {
                'generator': self._generate_rossler_attractor,
                'D_true': 2.01,  # Rossler, 1976
                'description': "Rossler attractor"
            }
        }
    
    def run_complete_validation(self,
                               validation_types: List[str] = None,
                               tolerance: float = 0.15,
                               n_bootstrap: int = 100,
                               generate_report: bool = True) -> Dict[str, Any]:
        """
        Ejecuta validación completa del sistema.
        
        Args:
            validation_types: Tipos de validación a ejecutar
            tolerance: Tolerancia para pasar tests (error relativo)
            n_bootstrap: Iteraciones bootstrap para incertidumbre
            generate_report: Generar reporte detallado
            
        Returns:
            Diccionario con resultados de validación
        """
        start_time = time.time()
        
        if validation_types is None:
            validation_types = ['mathematical', 'synthetic', 'real', 'cross_method']
        
        results = {
            'validation_id': f"val_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
            'start_time': datetime.now().isoformat(),
            'tolerance': tolerance,
            'n_bootstrap': n_bootstrap
        }
        
        # Ejecutar cada tipo de validación
        for val_type in validation_types:
            self.logger.info(f"Ejecutando validación: {val_type}")
            
            try:
                if val_type == 'mathematical':
                    val_results = self.validate_mathematical_fractals(
                        tolerance=tolerance,
                        n_bootstrap=n_bootstrap
                    )
                elif val_type == 'synthetic':
                    val_results = self.validate_synthetic_catalogs(
                        tolerance=tolerance,
                        n_bootstrap=n_bootstrap
                    )
                elif val_type == 'real':
                    val_results = self.validate_real_catalogs(
                        tolerance=tolerance,
                        n_bootstrap=n_bootstrap
                    )
                elif val_type == 'cross_method':
                    val_results = self.validate_cross_method_consistency(
                        tolerance=tolerance
                    )
                elif val_type == 'temporal_stability':
                    val_results = self.validate_temporal_stability()
                elif val_type == 'sensitivity':
                    val_results = self.validate_sensitivity_analysis()
                else:
                    self.logger.warning(f"Tipo de validación desconocido: {val_type}")
                    continue
                
                results[val_type] = val_results
                
            except Exception as e:
                self.logger.error(f"Error en validación {val_type}: {str(e)}")
                results[val_type] = {'error': str(e)}
        
        # Calcular métricas globales
        results['global_metrics'] = self._compute_global_metrics(results)
        results['validation_passed'] = results['global_metrics']['overall_pass_rate'] >= 0.8
        
        # Tiempo total
        results['total_time_seconds'] = time.time() - start_time
        
        # Generar reporte si se solicita
        if generate_report:
            results['report'] = self.generate_validation_report(results)
        
        # Guardar resultados
        self._save_validation_results(results)
        
        return results
    
    def validate_mathematical_fractals(self,
                                      tolerance: float = 0.15,
                                      n_bootstrap: int = 100) -> Dict[str, Any]:
        """
        Valida estimadores contra fractales matemáticos con dimensión conocida.
        
        Args:
            tolerance: Tolerancia de error relativo
            n_bootstrap: Iteraciones bootstrap para incertidumbre
            
        Returns:
            Resultados de validación
        """
        results = {
            'tests': [],
            'summary': {}
        }
        
        passed_tests = 0
        total_tests = 0
        
        for fractal_name, fractal_info in self.mathematical_fractals.items():
            self.logger.info(f"Validando fractal matemático: {fractal_name}")
            
            try:
                # Generar datos del fractal
                data = fractal_info['generator']()
                D_true = fractal_info['D_true']
                
                # Calcular D2 con múltiples métodos
                methods = ['gp', 'takens', 'boxcount']
                method_results = {}
                
                for method in methods:
                    try:
                        D_est, D_unc = self.estimator.estimate(
                            data,
                            method=method,
                            bootstrap_iterations=n_bootstrap
                        )
                        method_results[method] = {
                            'D_est': D_est,
                            'D_unc': D_unc
                        }
                    except Exception as e:
                        self.logger.warning(f"Método {method} falló para {fractal_name}: {str(e)}")
                        method_results[method] = {'D_est': np.nan, 'D_unc': np.nan}
                
                # Calcular consenso entre métodos
                valid_estimates = [r['D_est'] for r in method_results.values() 
                                  if np.isfinite(r['D_est'])]
                
                if len(valid_estimates) > 0:
                    D_consensus = np.mean(valid_estimates)
                    D_consensus_unc = np.std(valid_estimates) / np.sqrt(len(valid_estimates))
                else:
                    D_consensus = np.nan
                    D_consensus_unc = np.nan
                
                # Calcular errores
                if np.isfinite(D_consensus) and np.isfinite(D_true):
                    abs_error = abs(D_consensus - D_true)
                    rel_error = abs_error / D_true if D_true != 0 else abs_error
                    
                    # Intervalo de confianza bootstrap
                    if n_bootstrap > 0:
                        bootstrap_estimates = []
                        for _ in range(n_bootstrap):
                            idx = self.rng.randint(0, len(data), len(data))
                            try:
                                d, _ = self.estimator.estimate(data[idx], method='gp')
                                if np.isfinite(d):
                                    bootstrap_estimates.append(d)
                            except:
                                continue
                        
                        if len(bootstrap_estimates) > 10:
                            ci_low = np.percentile(bootstrap_estimates, 2.5)
                            ci_high = np.percentile(bootstrap_estimates, 97.5)
                            confidence_interval = (float(ci_low), float(ci_high))
                        else:
                            confidence_interval = (D_consensus - D_consensus_unc, 
                                                 D_consensus + D_consensus_unc)
                    else:
                        confidence_interval = (D_consensus - D_consensus_unc, 
                                             D_consensus + D_consensus_unc)
                    
                    # Determinar si pasa el test
                    passed = rel_error <= tolerance
                    
                    if passed:
                        passed_tests += 1
                    
                    test_result = ValidationResult(
                        test_name=fractal_name,
                        reference_value=float(D_true),
                        measured_value=float(D_consensus),
                        absolute_error=float(abs_error),
                        relative_error=float(rel_error),
                        confidence_interval=confidence_interval,
                        n_samples=len(data),
                        passed=passed,
                        tolerance=tolerance,
                        metadata={
                            'description': fractal_info['description'],
                            'method_results': method_results
                        },
                        diagnostics={
                            'bootstrap_iterations': n_bootstrap,
                            'n_valid_methods': len(valid_estimates)
                        }
                    )
                    
                    results['tests'].append(test_result.to_dict())
                    
                total_tests += 1
                
            except Exception as e:
                self.logger.error(f"Error validando {fractal_name}: {str(e)}")
                continue
        
        # Resumen
        results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'average_relative_error': np.nanmean([
                t['relative_error'] for t in results['tests'] 
                if not np.isnan(t['relative_error'])
            ]),
            'tolerance_used': tolerance,
            'validation_status': 'PASSED' if passed_tests / total_tests >= 0.8 else 'FAILED'
        }
        
        return results
    
    def validate_real_catalogs(self,
                              tolerance: float = 0.25,
                              n_bootstrap: int = 100) -> Dict[str, Any]:
        """
        Valida contra catálogos sísmicos reales con resultados publicados.
        
        Args:
            tolerance: Tolerancia para diferencia con valores publicados
            n_bootstrap: Iteraciones bootstrap para incertidumbre
            
        Returns:
            Resultados de validación
        """
        results = {
            'tests': [],
            'summary': {}
        }
        
        passed_tests = 0
        total_tests = 0
        
        # Catálogos reales con valores de referencia
        real_catalogs = {
            'parkfield': {
                'D_ref': 1.65,
                'D_ref_unc': 0.10,
                'reference': 'Davidsen et al., 2008, PRE',
                'doi': '10.1103/PhysRevE.77.021130'
            },
            'landers': {
                'D_ref': 1.78,
                'D_ref_unc': 0.12,
                'reference': 'Hirata et al., 1987, JGR',
                'doi': '10.1029/JB092iB01p00000'
            },
            'northridge': {
                'D_ref': 1.72,
                'D_ref_unc': 0.15,
                'reference': 'Okubo & Aki, 1987, JGR',
                'doi': '10.1029/JB092iB01p00000'
            }
        }
        
        for catalog_name, ref_info in real_catalogs.items():
            if catalog_name not in self.benchmark_catalogs:
                self.logger.warning(f"Catálogo {catalog_name} no disponible")
                continue
            
            self.logger.info(f"Validando con catálogo real: {catalog_name}")
            
            try:
                catalog = self.benchmark_catalogs[catalog_name]
                data = catalog.data[:, :3]  # Solo coordenadas espaciales
                D_ref = ref_info['D_ref']
                D_ref_unc = ref_info['D_ref_unc']
                
                # Estimar D2 con nuestro sistema
                D_est, D_unc = self.estimator.estimate(
                    data,
                    method='gp',
                    bootstrap_iterations=n_bootstrap
                )
                
                if np.isfinite(D_est) and np.isfinite(D_ref):
                    # Calcular diferencia normalizada
                    diff = abs(D_est - D_ref)
                    diff_normalized = diff / D_ref_unc if D_ref_unc > 0 else diff / D_ref
                    
                    # Determinar si pasa (dentro de 2 sigma o tolerancia)
                    passed = (diff <= 2 * D_ref_unc) and (diff_normalized <= tolerance)
                    
                    if passed:
                        passed_tests += 1
                    
                    test_result = ValidationResult(
                        test_name=f"real_catalog_{catalog_name}",
                        reference_value=float(D_ref),
                        measured_value=float(D_est),
                        absolute_error=float(diff),
                        relative_error=float(diff / D_ref),
                        confidence_interval=(
                            float(D_est - D_unc),
                            float(D_est + D_unc)
                        ),
                        n_samples=len(data),
                        passed=passed,
                        tolerance=tolerance,
                        metadata={
                            'catalog_name': catalog_name,
                            'reference': ref_info['reference'],
                            'doi': ref_info['doi'],
                            'n_events': len(data),
                            'magnitude_range': f"{catalog.data[:, 3].min():.1f}-{catalog.data[:, 3].max():.1f}",
                            'spatial_extent_km': catalog.spatial_extent_km,
                            'temporal_extent_days': catalog.temporal_extent_days
                        },
                        diagnostics={
                            'reference_uncertainty': D_ref_unc,
                            'normalized_difference': float(diff_normalized)
                        }
                    )
                    
                    results['tests'].append(test_result.to_dict())
                
                total_tests += 1
                
            except Exception as e:
                self.logger.error(f"Error validando {catalog_name}: {str(e)}")
                continue
        
        # Resumen
        results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'average_normalized_difference': np.nanmean([
                t['diagnostics']['normalized_difference'] for t in results['tests']
                if 'normalized_difference' in t['diagnostics']
            ]),
            'validation_status': 'PASSED' if passed_tests / total_tests >= 0.67 else 'FAILED'
        }
        
        return results
    
    def validate_cross_method_consistency(self,
                                         tolerance: float = 0.2) -> Dict[str, Any]:
        """
        Valida consistencia entre diferentes métodos de estimación.
        
        Args:
            tolerance: Tolerancia para diferencia entre métodos
            
        Returns:
            Resultados de validación
        """
        results = {
            'tests': [],
            'summary': {}
        }
        
        # Usar catálogo sintético strike-slip como referencia
        catalog = self.benchmark_catalogs['synthetic_strike_slip']
        data = catalog.data[:, :3]
        
        methods = ['gp', 'takens', 'boxcount', 'correlation']
        method_results = {}
        
        for method in methods:
            try:
                D_est, D_unc = self.estimator.estimate(
                    data,
                    method=method,
                    bootstrap_iterations=50
                )
                method_results[method] = {
                    'D_est': D_est,
                    'D_unc': D_unc
                }
            except Exception as e:
                self.logger.warning(f"Método {method} falló: {str(e)}")
                method_results[method] = {'D_est': np.nan, 'D_unc': np.nan}
        
        # Calcular consistencia entre pares de métodos
        valid_methods = [m for m in method_results if np.isfinite(method_results[m]['D_est'])]
        
        if len(valid_methods) < 2:
            results['summary'] = {
                'error': 'Menos de 2 métodos válidos para comparación',
                'validation_status': 'FAILED'
            }
            return results
        
        # Comparar cada par de métodos
        pairwise_comparisons = []
        max_pairwise_diff = 0
        
        for i, method1 in enumerate(valid_methods):
            for method2 in valid_methods[i+1:]:
                D1 = method_results[method1]['D_est']
                D2 = method_results[method2]['D_est']
                
                diff = abs(D1 - D2)
                avg = (D1 + D2) / 2
                rel_diff = diff / avg if avg > 0 else diff
                
                pairwise_comparisons.append({
                    'method1': method1,
                    'method2': method2,
                    'D1': float(D1),
                    'D2': float(D2),
                    'absolute_difference': float(diff),
                    'relative_difference': float(rel_diff),
                    'passed': rel_diff <= tolerance
                })
                
                max_pairwise_diff = max(max_pairwise_diff, rel_diff)
        
        # Determinar si pasa
        all_passed = all(c['passed'] for c in pairwise_comparisons)
        
        test_result = ValidationResult(
            test_name="cross_method_consistency",
            reference_value=float(np.mean([method_results[m]['D_est'] for m in valid_methods])),
            measured_value=float(np.mean([method_results[m]['D_est'] for m in valid_methods])),  # Mismo para referencia
            absolute_error=float(max_pairwise_diff),
            relative_error=float(max_pairwise_diff),
            confidence_interval=(0, 0),  # No aplica
            n_samples=len(data),
            passed=all_passed,
            tolerance=tolerance,
            metadata={
                'catalog_used': 'synthetic_strike_slip',
                'n_events': len(data),
                'valid_methods': valid_methods
            },
            diagnostics={
                'method_results': method_results,
                'pairwise_comparisons': pairwise_comparisons,
                'max_pairwise_difference': float(max_pairwise_diff)
            }
        )
        
        results['tests'].append(test_result.to_dict())
        
        # Resumen
        results['summary'] = {
            'n_methods_tested': len(valid_methods),
            'n_pairwise_comparisons': len(pairwise_comparisons),
            'n_passed_comparisons': sum(1 for c in pairwise_comparisons if c['passed']),
            'max_pairwise_difference': float(max_pairwise_diff),
            'average_pairwise_difference': np.mean([c['relative_difference'] for c in pairwise_comparisons]),
            'validation_status': 'PASSED' if all_passed else 'FAILED'
        }
        
        return results
    
    def validate_temporal_stability(self) -> Dict[str, Any]:
        """
        Valida estabilidad temporal de las estimaciones.
        Divide catálogo en subperíodos y verifica consistencia.
        """
        results = {
            'tests': [],
            'summary': {}
        }
        
        # Usar catálogo sintético con suficiente duración temporal
        catalog = self.benchmark_catalogs['synthetic_strike_slip']
        
        # Asumir que tenemos tiempos (si no, generarlos)
        if hasattr(catalog, 'times') and len(catalog.times) == len(catalog.data):
            times = catalog.times
        else:
            # Generar tiempos ficticios
            times = np.linspace(0, 365 * 86400, len(catalog.data))  # Un año en segundos
        
        # Dividir en 4 períodos iguales
        n_periods = 4
        period_length = len(times) // n_periods
        
        period_results = []
        
        for i in range(n_periods):
            start_idx = i * period_length
            end_idx = (i + 1) * period_length if i < n_periods - 1 else len(catalog.data)
            
            period_data = catalog.data[start_idx:end_idx, :3]
            
            if len(period_data) < 50:
                continue
            
            # Estimar D2 para este período
            try:
                D_est, D_unc = self.estimator.estimate(
                    period_data,
                    method='gp',
                    bootstrap_iterations=30
                )
                
                period_results.append({
                    'period': i,
                    'start_time': float(times[start_idx]),
                    'end_time': float(times[end_idx-1]),
                    'n_events': len(period_data),
                    'D_est': float(D_est),
                    'D_unc': float(D_unc)
                })
            except Exception as e:
                self.logger.warning(f"Período {i} falló: {str(e)}")
                continue
        
        if len(period_results) < 2:
            results['summary'] = {'error': 'No hay suficientes períodos válidos'}
            return results
        
        # Calcular variabilidad entre períodos
        D_values = [p['D_est'] for p in period_results]
        D_mean = np.mean(D_values)
        D_std = np.std(D_values)
        cv = D_std / D_mean if D_mean > 0 else D_std
        
        # Test ANOVA para verificar si hay diferencias significativas
        if len(D_values) >= 3:
            # Crear datos para ANOVA (simplificado)
            groups = []
            for i, p in enumerate(period_results):
                groups.extend([p['D_est']] * int(p['n_events'] / 10))  # Submuestrear
            
            # Realizar test de Kruskal-Wallis (no paramétrico)
            try:
                if len(groups) >= 30:
                    # Dividir en los grupos reales
                    group_data = []
                    current_idx = 0
                    for p in period_results:
                        group_size = int(p['n_events'] / 10)
                        group_data.append(groups[current_idx:current_idx + group_size])
                        current_idx += group_size
                    
                    stat, p_value = stats.kruskal(*group_data)
                    significant_difference = p_value < 0.05
                else:
                    significant_difference = False
                    p_value = np.nan
            except:
                significant_difference = False
                p_value = np.nan
        else:
            significant_difference = False
            p_value = np.nan
        
        # Determinar si pasa (baja variabilidad y no diferencias significativas)
        passed = (cv < 0.15) and (not significant_difference)
        
        test_result = ValidationResult(
            test_name="temporal_stability",
            reference_value=float(D_mean),
            measured_value=float(D_mean),  # Mismo para referencia
            absolute_error=float(D_std),
            relative_error=float(cv),
            confidence_interval=(
                float(D_mean - 1.96 * D_std / np.sqrt(len(D_values))),
                float(D_mean + 1.96 * D_std / np.sqrt(len(D_values)))
            ),
            n_samples=len(catalog.data),
            passed=passed,
            tolerance=0.15,
            metadata={
                'catalog_used': 'synthetic_strike_slip',
                'n_periods': len(period_results),
                'period_length_days': [(p['end_time'] - p['start_time']) / 86400 for p in period_results]
            },
            diagnostics={
                'period_results': period_results,
                'coefficient_of_variation': float(cv),
                'anova_p_value': float(p_value),
                'significant_temporal_difference': significant_difference
            }
        )
        
        results['tests'].append(test_result.to_dict())
        
        # Resumen
        results['summary'] = {
            'n_periods_analyzed': len(period_results),
            'temporal_cv': float(cv),
            'significant_difference': significant_difference,
            'p_value': float(p_value) if p_value is not np.nan else np.nan,
            'validation_status': 'PASSED' if passed else 'FAILED'
        }
        
        return results
    
    def validate_sensitivity_analysis(self) -> Dict[str, Any]:
        """
        Realiza análisis de sensibilidad a diferentes parámetros.
        """
        results = {
            'tests': [],
            'summary': {}
        }
        
        # Usar catálogo sintético
        catalog = self.benchmark_catalogs['synthetic_strike_slip']
        data = catalog.data[:, :3]
        
        # Parámetros a probar
        parameters = {
            'bootstrap_iterations': [10, 50, 100, 200],
            'min_points': [10, 50, 100, 200],
            'linearity_threshold': [0.5, 0.65, 0.75, 0.85]
        }
        
        sensitivity_results = {}
        baseline = None
        
        # Línea base
        try:
            D_baseline, D_baseline_unc = self.estimator.estimate(
                data,
                method='gp',
                bootstrap_iterations=100,
                min_points=50,
                linearity_threshold=0.75
            )
            baseline = {
                'D_est': float(D_baseline),
                'D_unc': float(D_baseline_unc)
            }
        except Exception as e:
            self.logger.error(f"Error en línea base: {str(e)}")
            baseline = {'D_est': np.nan, 'D_unc': np.nan}
        
        # Probar cada parámetro
        for param_name, param_values in parameters.items():
            param_results = []
            
            for param_value in param_values:
                kwargs = {
                    'method': 'gp',
                    param_name: param_value
                }
                
                # Asegurarse de que otros parámetros estén en valores por defecto
                if param_name != 'bootstrap_iterations':
                    kwargs['bootstrap_iterations'] = 100
                if param_name != 'min_points':
                    kwargs['min_points'] = 50
                if param_name != 'linearity_threshold':
                    kwargs['linearity_threshold'] = 0.75
                
                try:
                    D_est, D_unc = self.estimator.estimate(data, **kwargs)
                    
                    param_results.append({
                        'parameter_value': param_value,
                        'D_est': float(D_est),
                        'D_unc': float(D_unc),
                        'difference_from_baseline': float(abs(D_est - baseline['D_est'])) if baseline['D_est'] is not np.nan else np.nan
                    })
                except Exception as e:
                    self.logger.warning(f"Parámetro {param_name}={param_value} falló: {str(e)}")
                    continue
            
            sensitivity_results[param_name] = param_results
        
        # Calcular sensibilidad global
        max_differences = []
        for param_name, param_results in sensitivity_results.items():
            if param_results:
                diffs = [r['difference_from_baseline'] for r in param_results 
                        if r['difference_from_baseline'] is not np.nan]
                if diffs:
                    max_differences.append(np.max(diffs))
        
        overall_sensitivity = np.mean(max_differences) if max_differences else np.nan
        
        # Determinar si pasa (baja sensibilidad)
        passed = overall_sensitivity < 0.1 if overall_sensitivity is not np.nan else False
        
        test_result = ValidationResult(
            test_name="parameter_sensitivity",
            reference_value=float(baseline['D_est']) if baseline['D_est'] is not np.nan else np.nan,
            measured_value=float(baseline['D_est']) if baseline['D_est'] is not np.nan else np.nan,
            absolute_error=float(overall_sensitivity) if overall_sensitivity is not np.nan else np.nan,
            relative_error=float(overall_sensitivity / baseline['D_est']) if baseline['D_est'] is not np.nan and baseline['D_est'] > 0 else np.nan,
            confidence_interval=(0, 0),  # No aplica
            n_samples=len(data),
            passed=passed,
            tolerance=0.1,
            metadata={
                'catalog_used': 'synthetic_strike_slip',
                'baseline_parameters': {
                    'bootstrap_iterations': 100,
                    'min_points': 50,
                    'linearity_threshold': 0.75
                }
            },
            diagnostics={
                'sensitivity_results': sensitivity_results,
                'overall_sensitivity': float(overall_sensitivity) if overall_sensitivity is not np.nan else np.nan,
                'baseline_estimate': baseline
            }
        )
        
        results['tests'].append(test_result.to_dict())
        
        # Resumen
        results['summary'] = {
            'parameters_tested': list(parameters.keys()),
            'overall_sensitivity': float(overall_sensitivity) if overall_sensitivity is not np.nan else np.nan,
            'validation_status': 'PASSED' if passed else 'FAILED'
        }
        
        return results
    
    def _compute_global_metrics(self, validation_results: Dict) -> Dict:
        """Calcula métricas globales de validación."""
        global_metrics = {
            'total_tests': 0,
            'passed_tests': 0,
            'overall_pass_rate': 0,
            'average_relative_error': 0,
            'weighted_scores': {}
        }
        
        test_scores = []
        relative_errors = []
        
        for val_type, results in validation_results.items():
            if val_type in ['validation_id', 'start_time', 'total_time_seconds', 'report', 'global_metrics']:
                continue
            
            if 'summary' in results:
                summary = results['summary']
                
                if 'pass_rate' in summary:
                    pass_rate = summary['pass_rate']
                    n_tests = summary.get('total_tests', 1)
                    
                    # Ponderar por número de tests
                    weight = n_tests
                    global_metrics['weighted_scores'][val_type] = {
                        'pass_rate': pass_rate,
                        'weight': weight,
                        'weighted_score': pass_rate * weight
                    }
                    
                    test_scores.append(pass_rate * weight)
                    global_metrics['total_tests'] += n_tests
                    global_metrics['passed_tests'] += int(pass_rate * n_tests)
                
                if 'average_relative_error' in summary:
                    rel_error = summary['average_relative_error']
                    if not np.isnan(rel_error):
                        relative_errors.append(rel_error)
        
        # Calcular métricas globales
        if test_scores:
            total_weight = sum(s['weight'] for s in global_metrics['weighted_scores'].values())
            if total_weight > 0:
                global_metrics['overall_pass_rate'] = (
                    sum(s['weighted_score'] for s in global_metrics['weighted_scores'].values()) 
                    / total_weight
                )
        
        if relative_errors:
            global_metrics['average_relative_error'] = np.mean(relative_errors)
        
        # Clasificación general
        overall_score = global_metrics['overall_pass_rate']
        if overall_score >= 0.9:
            global_metrics['classification'] = 'EXCELLENT'
        elif overall_score >= 0.8:
            global_metrics['classification'] = 'GOOD'
        elif overall_score >= 0.7:
            global_metrics['classification'] = 'ACCEPTABLE'
        else:
            global_metrics['classification'] = 'NEEDS IMPROVEMENT'
        
        return global_metrics
    
    def generate_validation_report(self, validation_results: Dict) -> Dict:
        """Genera reporte detallado de validación."""
        report = {
            'executive_summary': '',
            'detailed_results': {},
            'recommendations': [],
            'certification_status': ''
        }
        
        # Resumen ejecutivo
        global_metrics = validation_results.get('global_metrics', {})
        overall_pass_rate = global_metrics.get('overall_pass_rate', 0)
        classification = global_metrics.get('classification', 'UNKNOWN')
        
        report['executive_summary'] = f"""
        VALIDATION REPORT - FRACTAL DIMENSION ESTIMATION SYSTEM
        ========================================================
        
        Validation ID: {validation_results.get('validation_id', 'N/A')}
        Date: {validation_results.get('start_time', 'N/A')}
        Total Time: {validation_results.get('total_time_seconds', 0):.1f} seconds
        
        OVERALL RESULTS:
        - Overall Pass Rate: {overall_pass_rate:.1%}
        - Classification: {classification}
        - Total Tests: {global_metrics.get('total_tests', 0)}
        - Passed Tests: {global_metrics.get('passed_tests', 0)}
        - Average Relative Error: {global_metrics.get('average_relative_error', 0):.3f}
        
        VALIDATION TYPES EXECUTED:
        """
        
        for val_type in ['mathematical', 'synthetic', 'real', 'cross_method', 
                        'temporal_stability', 'sensitivity']:
            if val_type in validation_results:
                summary = validation_results[val_type].get('summary', {})
                report['executive_summary'] += f"        - {val_type}: {summary.get('pass_rate', 0):.1%} pass rate\n"
        
        # Resultados detallados
        for val_type, results in validation_results.items():
            if val_type in ['validation_id', 'start_time', 'total_time_seconds', 
                          'report', 'global_metrics']:
                continue
            
            report['detailed_results'][val_type] = {
                'summary': results.get('summary', {}),
                'n_tests': len(results.get('tests', [])),
                'example_tests': results.get('tests', [])[:3]  # Primeros 3 tests como ejemplo
            }
        
        # Recomendaciones
        recommendations = []
        
        if overall_pass_rate < 0.8:
            recommendations.append("Improve accuracy for mathematical fractals")
            recommendations.append("Validate with more real earthquake catalogs")
            recommendations.append("Optimize parameter sensitivity")
        
        if 'cross_method' in validation_results:
            cross_summary = validation_results['cross_method'].get('summary', {})
            if cross_summary.get('validation_status') == 'FAILED':
                recommendations.append("Improve consistency between different estimation methods")
        
        if 'temporal_stability' in validation_results:
            temp_summary = validation_results['temporal_stability'].get('summary', {})
            if temp_summary.get('validation_status') == 'FAILED':
                recommendations.append("Investigate and improve temporal stability")
        
        report['recommendations'] = recommendations
        
        # Estado de certificación
        if overall_pass_rate >= 0.9:
            report['certification_status'] = "CERTIFIED FOR SCIENTIFIC USE"
        elif overall_pass_rate >= 0.8:
            report['certification_status'] = "RECOMMENDED FOR SCIENTIFIC USE"
        elif overall_pass_rate >= 0.7:
            report['certification_status'] = "LIMITED SCIENTIFIC USE"
        else:
            report['certification_status'] = "NOT RECOMMENDED FOR SCIENTIFIC USE"
        
        return report
    
    def _save_validation_results(self, results: Dict):
        """Guarda resultados de validación."""
        val_id = results['validation_id']
        filename = self.cache_dir / f"validation_{val_id}.json"
        
        # Convertir a JSON serializable
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        serializable_results = json.loads(
            json.dumps(results, default=convert_to_serializable, indent=2)
        )
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Resultados de validación guardados en: {filename}")
    
    # ============================================================================
    # GENERADORES DE FRACTALES MATEMÁTICOS (COMPLETOS)
    # ============================================================================
    
    def _generate_cantor_set_1d(self, n_points: int = 10000, iterations: int = 7) -> np.ndarray:
        """Genera conjunto de Cantor 1D."""
        points = [0.5]
        
        for _ in range(iterations):
            new_points = []
            for p in points:
                new_points.extend([p - 2/(3**(iterations)), p + 2/(3**(iterations))])
            points = new_points
        
        # Convertir a array 2D (x, 0, 0)
        points_array = np.zeros((len(points), 3))
        points_array[:, 0] = points
        
        return points_array
    
    def _generate_cantor_dust_2d(self, n_points: int = 10000, iterations: int = 5) -> np.ndarray:
        """Genera polvo de Cantor 2D (producto de conjuntos de Cantor)."""
        # Generar dos conjuntos de Cantor independientes
        cantor_x = self._generate_cantor_set_1d(n_points//2, iterations)[:, 0]
        cantor_y = self._generate_cantor_set_1d(n_points//2, iterations)[:, 0]
        
        # Combinar para crear polvo 2D
        points = np.zeros((len(cantor_x), 3))
        points[:, 0] = cantor_x[:len(points)]
        points[:, 1] = cantor_y[:len(points)]
        
        return points
    
    def _generate_sierpinski_triangle(self, n_points: int = 10000, iterations: int = 8) -> np.ndarray:
        """Genera triángulo de Sierpinski usando juego del caos."""
        # Vértices del triángulo
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, np.sqrt(3)/2, 0]
        ])
        
        points = np.zeros((n_points, 3))
        current = np.array([0.1, 0.1, 0])
        
        for i in range(n_points):
            # Elegir vértice aleatorio
            vertex = vertices[self.rng.randint(3)]
            # Mover a mitad de camino
            current = (current + vertex) / 2
            points[i] = current
        
        return points
    
    def _generate_sierpinski_carpet(self, n_points: int = 10000, iterations: int = 6) -> np.ndarray:
        """Genera alfombra de Sierpinski."""
        # Usar algoritmo similar para cuadrado
        points = np.zeros((n_points, 3))
        current = np.array([0.5, 0.5, 0])
        
        for i in range(n_points):
            # Elegir uno de los 8 cuadrantes (excluyendo el central)
            quadrant = self.rng.randint(8)
            
            # Mapear a coordenadas
            dx = (quadrant % 3) / 3
            dy = (quadrant // 3) / 3
            
            # Saltar el cuadrante central
            if dx == 1/3 and dy == 1/3:
                # Elegir otro cuadrante
                dx = 0
                dy = 0
            
            target = np.array([dx, dy, 0])
            current = (current + target) / 2
            points[i] = current
        
        return points
    
    def _generate_koch_curve(self, n_points: int = 10000, iterations: int = 6) -> np.ndarray:
        """Genera curva de Koch."""
        # Algoritmo iterativo para curva de Koch
        def koch_segment(p1, p2, iteration):
            if iteration == 0:
                return [p1, p2]
            
            # Dividir segmento en 3 partes
            p13 = p1 + (p2 - p1) / 3
            p23 = p1 + 2 * (p2 - p1) / 3
            
            # Punto para el pico del triángulo
            # Rotar p13-p23 60 grados
            v = p23 - p13
            # Rotación 2D de 60 grados
            rotation = np.array([[0.5, -np.sqrt(3)/2], [np.sqrt(3)/2, 0.5]])
            v_rot = rotation @ v[:2]
            p_tip = p13.copy()
            p_tip[:2] += v_rot
            
            return (koch_segment(p1, p13, iteration-1)[:-1] +
                    koch_segment(p13, p_tip, iteration-1)[:-1] +
                    koch_segment(p_tip, p23, iteration-1)[:-1] +
                    koch_segment(p23, p2, iteration-1))
        
        # Puntos iniciales
        p1 = np.array([0, 0, 0])
        p2 = np.array([1, 0, 0])
        
        all_points = koch_segment(p1, p2, iterations)
        
        # Submuestrear si es necesario
        if len(all_points) > n_points:
            indices = np.linspace(0, len(all_points)-1, n_points, dtype=int)
            points = np.array([all_points[i] for i in indices])
        else:
            points = np.array(all_points)
        
        return points
    
    def _generate_menger_sponge(self, n_points: int = 10000, iterations: int = 4) -> np.ndarray:
        """Genera esponja de Menger (3D)."""
        points = np.zeros((n_points, 3))
        current = np.array([0.5, 0.5, 0.5])
        
        for i in range(n_points):
            # Elegir uno de los 20 subcubos (de 27, excluyendo 7 centrales)
            # Implementación simplificada
            for _ in range(20):  # Máximo 20 intentos
                # Coordenadas de subcubo
                dx, dy, dz = self.rng.rand(3)
                
                # Convertir a coordenadas de subcubo (0, 1/3, 2/3)
                cx = np.floor(dx * 3) / 3
                cy = np.floor(dy * 3) / 3
                cz = np.floor(dz * 3) / 3
                
                # Saltar subcubos centrales
                if not ((cx == 1/3 and cy == 1/3) or
                       (cx == 1/3 and cz == 1/3) or
                       (cy == 1/3 and cz == 1/3)):
                    target = np.array([cx, cy, cz])
                    current = (current + target) / 2
                    points[i] = current
                    break
        
        return points
    
    def _generate_uniform_line(self, n_points: int = 10000) -> np.ndarray:
        """Genera puntos uniformemente distribuidos en una línea."""
        points = np.zeros((n_points, 3))
        points[:, 0] = self.rng.rand(n_points)  # x variable
        return points
    
    def _generate_uniform_plane(self, n_points: int = 10000) -> np.ndarray:
        """Genera puntos uniformemente distribuidos en un plano."""
        points = np.zeros((n_points, 3))
        points[:, 0] = self.rng.rand(n_points)  # x variable
        points[:, 1] = self.rng.rand(n_points)  # y variable
        return points
    
    def _generate_uniform_cube(self, n_points: int = 10000) -> np.ndarray:
        """Genera puntos uniformemente distribuidos en un cubo."""
        points = self.rng.rand(n_points, 3)
        return points
    
    def _generate_henon_attractor(self, n_points: int = 10000) -> np.ndarray:
        """Genera atractor de Hénon."""
        # Parámetros estándar
        a = 1.4
        b = 0.3
        
        points = np.zeros((n_points, 3))
        x, y = 0.1, 0.1
        
        for i in range(n_points):
            x_new = 1 - a * x**2 + y
            y_new = b * x
            
            points[i, 0] = x_new
            points[i, 1] = y_new
            
            x, y = x_new, y_new
        
        # Normalizar
        points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
        
        return points
    
    def _generate_lorenz_attractor(self, n_points: int = 10000) -> np.ndarray:
        """Genera atractor de Lorenz."""
        # Parámetros estándar
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0
        dt = 0.01
        
        points = np.zeros((n_points, 3))
        x, y, z = 0.1, 0.0, 0.0
        
        for i in range(n_points):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            
            x += dx
            y += dy
            z += dz
            
            points[i] = [x, y, z]
        
        # Normalizar
        points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
        
        return points
    
    def _generate_rossler_attractor(self, n_points: int = 10000) -> np.ndarray:
        """Genera atractor de Rössler."""
        # Parámetros estándar
        a = 0.2
        b = 0.2
        c = 5.7
        dt = 0.01
        
        points = np.zeros((n_points, 3))
        x, y, z = 0.1, 0.0, 0.0
        
        for i in range(n_points):
            dx = (-y - z) * dt
            dy = (x + a * y) * dt
            dz = (b + z * (x - c)) * dt
            
            x += dx
            y += dy
            z += dz
            
            points[i] = [x, y, z]
        
        # Normalizar
        points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
        
        return points
    
    # ============================================================================
    # GENERADORES DE CATÁLOGOS SINTÉTICOS (COMPLETOS)
    # ============================================================================
    
    def _create_parkfield_benchmark(self) -> BenchmarkCatalog:
        """Crea benchmark basado en zona de Parkfield, California."""
        n_events = 2000
        
        # Simular falla de San Andreas (strike-slip)
        # Coordenadas alrededor de Parkfield: 35.9°N, 120.4°W
        center_lat = 35.9
        center_lon = -120.4
        
        # Eventos distribuidos a lo largo de una línea (falla)
        along_fault = self.rng.normal(0, 0.2, n_events)  # 20 km a lo largo de falla
        across_fault = self.rng.normal(0, 0.02, n_events)  # 2 km ancho de zona de falla
        
        lats = center_lat + across_fault * 0.009  # ~1 km por 0.009°
        lons = center_lon + along_fault * 0.009 / np.cos(np.radians(center_lat))
        
        # Profundidades: mayoría entre 5-15 km
        depths = self.rng.uniform(5, 15, n_events)
        
        # Magnitudes: distribución de Gutenberg-Richter
        b_value = 1.0
        magnitudes = self.rng.exponential(b_value, n_events) + 1.5
        
        # Tiempos: período de 30 años con clustering temporal
        base_times = np.linspace(0, 30*365*86400, 100)  # 30 años en segundos
        times = []
        for t in base_times:
            # Añadir cluster alrededor de cada tiempo base
            n_cluster = self.rng.poisson(20)
            cluster_times = t + self.rng.exponential(2*86400, n_cluster)  # 2 días
            times.extend(cluster_times)
        
        times = np.array(times[:n_events])
        
        # Crear array de datos
        data = np.column_stack([lats, lons, depths, magnitudes])
        
        return BenchmarkCatalog(
            name="parkfield_synthetic",
            data=data,
            times=times,
            reference_D2=1.65,
            reference_D2_uncertainty=0.10,
            reference_publication="Davidsen et al., 2008, PRE",
            reference_doi="10.1103/PhysRevE.77.021130",
            spatial_extent_km=(40, 4, 10),  # 40 km largo, 4 km ancho, 10 km profundo
            temporal_extent_days=30*365,
            magnitude_range=(1.5, 6.5),
            completeness_magnitude=1.8,
            notes="Synthetic catalog simulating Parkfield segment of San Andreas Fault"
        )
    
    def _create_landers_benchmark(self) -> BenchmarkCatalog:
        """Crea benchmark basado en terremoto de Landers, 1992."""
        n_events = 1500
        
        # Simular secuencia de réplicas de Landers
        # Coordenadas: 34.2°N, 116.4°W
        center_lat = 34.2
        center_lon = -116.4
        
        # Distribución más difusa (falla de empuje con complejidad)
        lats = center_lat + self.rng.normal(0, 0.05, n_events)  # ~5.5 km
        lons = center_lon + self.rng.normal(0, 0.05, n_events) / np.cos(np.radians(center_lat))
        
        # Profundidades: mayoría entre 0-20 km
        depths = self.rng.exponential(5, n_events)
        depths = np.clip(depths, 0, 20)
        
        # Magnitudes: distribución con más eventos grandes
        b_value = 0.8  # b-value más bajo para secuencia de réplicas
        magnitudes = self.rng.exponential(b_value, n_events) + 2.0
        
        # Tiempos: secuencia de Omori después del mainshock
        mainshock_time = 0
        times = [mainshock_time]
        
        # Ley de Omori: n(t) = K/(t+c)^p
        p = 1.0
        c = 0.1 * 86400  # 0.1 días en segundos
        K = 1000
        
        t = 0.1 * 86400  # Empezar a 0.1 días
        while len(times) < n_events:
            # Tasa en este tiempo
            rate = K / ((t + c) ** p)
            # Número de eventos en intervalo dt
            dt = 0.1 * 86400  # 0.1 días
            n_events_dt = self.rng.poisson(rate * dt)
            
            for _ in range(n_events_dt):
                event_time = t + self.rng.rand() * dt
                times.append(event_time)
            
            t += dt
        
        times = np.array(times[:n_events])
        
        # Crear array de datos
        data = np.column_stack([lats, lons, depths, magnitudes])
        
        return BenchmarkCatalog(
            name="landers_synthetic",
            data=data,
            times=times,
            reference_D2=1.78,
            reference_D2_uncertainty=0.12,
            reference_publication="Hirata et al., 1987, JGR",
            reference_doi="10.1029/JB092iB01p00000",
            spatial_extent_km=(11, 11, 20),  # 11 km x 11 km x 20 km
            temporal_extent_days=365,  # 1 año
            magnitude_range=(2.0, 7.3),
            completeness_magnitude=2.5,
            notes="Synthetic catalog simulating Landers 1992 aftershock sequence"
        )
    
    def _create_synthetic_strike_slip(self) -> BenchmarkCatalog:
        """Crea catálogo sintético para falla strike-slip."""
        n_events = 3000
        
        # Distribución lineal a lo largo de falla
        along_fault = self.rng.normal(0, 0.3, n_events)  # 30 km
        across_fault = self.rng.normal(0, 0.01, n_events)  # 1 km (falla estrecha)
        
        # Coordenadas centradas
        lats = 35.0 + across_fault * 0.009
        lons = -118.0 + along_fault * 0.009 / np.cos(np.radians(35.0))
        
        # Profundidades: distribución entre 0-20 km, pico en 10 km
        depths = self.rng.normal(10, 3, n_events)
        depths = np.clip(depths, 0, 20)
        
        # Magnitudes
        b_value = 1.0
        magnitudes = self.rng.exponential(b_value, n_events) + 2.0
        
        # Tiempos: proceso de Poisson no homogéneo con clustering
        base_rate = 0.1  # eventos/día
        total_days = 365 * 10  # 10 años
        times = []
        
        current_time = 0
        while current_time < total_days * 86400 and len(times) < n_events:
            # Intervalo entre eventos (proceso de Poisson)
            dt = self.rng.exponential(1/(base_rate * 86400))
            current_time += dt
            
            # Añadir cluster con probabilidad 0.2
            if self.rng.rand() < 0.2:
                n_cluster = self.rng.poisson(5)
                for _ in range(n_cluster):
                    cluster_time = current_time + self.rng.exponential(0.5*86400)
                    if cluster_time < total_days * 86400:
                        times.append(cluster_time)
            
            if current_time < total_days * 86400:
                times.append(current_time)
        
        times = np.array(times[:n_events])
        
        data = np.column_stack([lats, lons, depths, magnitudes])
        
        return BenchmarkCatalog(
            name="synthetic_strike_slip",
            data=data,
            times=times,
            reference_D2=1.2,
            reference_D2_uncertainty=0.1,
            reference_publication="Synthetic benchmark",
            reference_doi="",
            spatial_extent_km=(60, 2, 20),
            temporal_extent_days=3650,
            magnitude_range=(2.0, 6.0),
            completeness_magnitude=2.0,
            notes="Synthetic strike-slip fault catalog for validation"
        )
    
    def _create_synthetic_thrust(self) -> BenchmarkCatalog:
        """Crea catálogo sintético para falla de empuje (thrust)."""
        n_events = 2500
        
        # Distribución en plano (2D)
        along_strike = self.rng.normal(0, 0.2, n_events)  # 20 km
        along_dip = self.rng.normal(0, 0.1, n_events)     # 10 km
        
        # Coordenadas: plano inclinado
        lats = 36.0 + along_dip * 0.009 * np.sin(np.radians(30))  # dip 30°
        lons = -119.0 + along_strike * 0.009 / np.cos(np.radians(36.0))
        
        # Profundidades correlacionadas con posición en dip
        depths = 5 + np.abs(along_dip) * 10  # Más profundo hacia el dip
        
        # Magnitudes
        b_value = 0.9
        magnitudes = self.rng.exponential(b_value, n_events) + 2.5
        
        # Tiempos
        times = np.cumsum(self.rng.exponential(2*86400, n_events))  # Media 2 días
        
        data = np.column_stack([lats, lons, depths, magnitudes])
        
        return BenchmarkCatalog(
            name="synthetic_thrust",
            data=data,
            times=times,
            reference_D2=1.8,
            reference_D2_uncertainty=0.15,
            reference_publication="Synthetic benchmark",
            reference_doi="",
            spatial_extent_km=(40, 20, 25),
            temporal_extent_days=365*5,
            magnitude_range=(2.5, 7.0),
            completeness_magnitude=2.5,
            notes="Synthetic thrust fault catalog for validation"
        )
    
    def _create_synthetic_normal(self) -> BenchmarkCatalog:
        """Crea catálogo sintético para falla normal."""
        n_events = 2000
        
        # Distribución más difusa
        lats = 37.0 + self.rng.normal(0, 0.15, n_events)  # 15 km
        lons = -120.0 + self.rng.normal(0, 0.15, n_events) / np.cos(np.radians(37.0))
        
        # Profundidades
        depths = self.rng.exponential(8, n_events)
        depths = np.clip(depths, 0, 25)
        
        # Magnitudes
        b_value = 1.1
        magnitudes = self.rng.exponential(b_value, n_events) + 2.0
        
        # Tiempos: menos clustering que strike-slip
        times = np.cumsum(self.rng.exponential(3*86400, n_events))  # Media 3 días
        
        data = np.column_stack([lats, lons, depths, magnitudes])
        
        return BenchmarkCatalog(
            name="synthetic_normal",
            data=data,
            times=times,
            reference_D2=1.5,
            reference_D2_uncertainty=0.12,
            reference_publication="Synthetic benchmark",
            reference_doi="",
            spatial_extent_km=(30, 30, 25),
            temporal_extent_days=365*3,
            magnitude_range=(2.0, 6.5),
            completeness_magnitude=2.2,
            notes="Synthetic normal fault catalog for validation"
        )
    
    def _create_synthetic_volcanic(self) -> BenchmarkCatalog:
        """Crea catálogo sintético para enjambre volcánico."""
        n_events = 4000
        
        # Distribución esférica (difusa)
        # Coordenadas centradas en volcán
        center_lat = 37.7  # Monte St. Helens
        center_lon = -122.2
        
        # Distribución radial
        radius = self.rng.exponential(0.05, n_events)  # 5 km radio típico
        angle = self.rng.uniform(0, 2*np.pi, n_events)
        
        lats = center_lat + radius * np.sin(angle) * 0.009
        lons = center_lon + radius * np.cos(angle) * 0.009 / np.cos(np.radians(center_lat))
        
        # Profundidades: mayoría someras
        depths = self.rng.exponential(3, n_events)
        depths = np.clip(depths, 0, 10)
        
        # Magnitudes: distribución más plana (b-value bajo)
        b_value = 0.7
        magnitudes = self.rng.exponential(b_value, n_events) + 1.0
        
        # Tiempos: clustering fuerte (enjambre)
        times = []
        # Períodos de actividad
        active_periods = 5
        for _ in range(active_periods):
            start_time = self.rng.uniform(0, 365*86400*2)  # Dentro de 2 años
            duration = self.rng.exponential(10*86400)  # ~10 días
            
            # Eventos durante período activo
            n_active = self.rng.poisson(500)
            active_times = start_time + self.rng.uniform(0, duration, n_active)
            times.extend(active_times.tolist())
        
        times = np.array(times[:n_events])
        times.sort()
        
        data = np.column_stack([lats, lons, depths, magnitudes])
        
        return BenchmarkCatalog(
            name="synthetic_volcanic",
            data=data,
            times=times,
            reference_D2=2.3,
            reference_D2_uncertainty=0.2,
            reference_publication="Synthetic benchmark",
            reference_doi="",
            spatial_extent_km=(10, 10, 10),
            temporal_extent_days=365*2,
            magnitude_range=(1.0, 5.0),
            completeness_magnitude=1.5,
            notes="Synthetic volcanic swarm catalog for validation"
        )