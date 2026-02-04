"""
SECCIÓN 1: CONFIGURACIÓN DEL SISTEMA Y UTILIDADES
Extracted from _python_5.py for modular architecture.
"""

import numpy as np
import pandas as pd
import logging
import time
import json
import pickle
import hashlib
import warnings
import traceback
import sys
import math
import itertools
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache, wraps
from collections import defaultdict, OrderedDict
import concurrent.futures
import multiprocessing
from contextlib import contextmanager

class SystemMode(Enum):
    """Modos de operación del sistema."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    CERTIFIED = "certified"

class CertificationLevel(Enum):
    """Niveles de certificación ISO."""
    LEVEL_0 = "basic"           # Uso básico, sin certificación
    LEVEL_1 = "scientific"      # Uso científico, validado
    LEVEL_2 = "commercial"      # Uso comercial, auditado
    LEVEL_3 = "certified"       # Certificado ISO 9001:2015
    LEVEL_4 = "nuclear"         # Para aplicaciones críticas (nuclear, médica)

@dataclass
class SystemConfiguration:
    """Configuración completa del sistema."""
    # Identificación
    system_id: str = field(default_factory=lambda: f"SYSMIC-{int(time.time())}")
    version: str = "6.0.0"
    release_date: str = "2025-01-15"
    
    # Certificación
    certification_level: CertificationLevel = CertificationLevel.LEVEL_3
    certification_id: Optional[str] = None
    license_key: Optional[str] = None
    license_expiry: Optional[datetime] = None
    
    # Rendimiento
    max_workers: int = field(default_factory=lambda: multiprocessing.cpu_count())
    cache_size_mb: int = 2048  # 2GB de caché
    memory_limit_mb: Optional[int] = 8192  # 8GB límite
    enable_gpu: bool = False
    gpu_device_id: int = 0
    
    # Precisión
    floating_point_precision: str = "float64"  # "float32", "float64", "float128"
    bootstrap_iterations: int = 1000
    monte_carlo_samples: int = 10000
    convergence_threshold: float = 1e-6
    max_iterations: int = 1000
    
    # Almacenamiento
    data_directory: Path = field(default_factory=lambda: Path.home() / ".fractal_system")
    cache_directory: Path = field(default_factory=lambda: Path.home() / ".fractal_system" / "cache")
    reports_directory: Path = field(default_factory=lambda: Path.home() / ".fractal_system" / "reports")
    exports_directory: Path = field(default_factory=lambda: Path.home() / ".fractal_system" / "exports")
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = field(default_factory=lambda: Path.home() / ".fractal_system" / "system.log")
    enable_telemetry: bool = False
    telemetry_endpoint: Optional[str] = None
    
    # Validación
    auto_validation: bool = True
    validation_threshold: float = 0.85  # 85% de tests deben pasar
    benchmark_update_frequency: str = "weekly"  # daily, weekly, monthly
    
    # Reportes
    generate_executive_summary: bool = True
    generate_technical_report: bool = True
    generate_latex_report: bool = False
    generate_pdf_report: bool = False
    report_language: str = "es"  # es, en, fr, de
    
    # Seguridad
    encrypt_sensitive_data: bool = True
    encryption_key: Optional[str] = None
    audit_log_enabled: bool = True
    
    def __post_init__(self):
        """Crear directorios necesarios."""
        directories = [
            self.data_directory,
            self.cache_directory,
            self.reports_directory,
            self.exports_directory
        ]
        
        for directory in directories:
            # Simple check to avoid errors if path is not proper
            if isinstance(directory, Path):
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
        
        # Configurar certificación
        if self.certification_level == CertificationLevel.LEVEL_3:
            self.certification_id = self._generate_certification_id()
        
        # Configurar límite de memoria
        if self.memory_limit_mb:
            self._set_memory_limit()
    
    def _generate_certification_id(self) -> str:
        """Genera ID de certificación único."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        system_hash = hashlib.sha256(f"{self.system_id}{self.version}".encode()).hexdigest()[:16]
        return f"ISO9001-2025-SYSMIC-{timestamp}-{system_hash}"
    
    def _set_memory_limit(self):
        """Configura límite de memoria para el proceso."""
        try:
            import resource
            # Convertir MB a bytes
            soft_limit = self.memory_limit_mb * 1024 * 1024
            hard_limit = self.memory_limit_mb * 1024 * 1024 * 2
            resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard_limit))
        except (ImportError, ValueError):
            pass  # No disponible en Windows o error en el límite

class ScientificLogger:
    """
    Sistema de logging científico con formato para papers y auditoría.
    """
    
    def __init__(self, config: SystemConfiguration):
        self.config = config
        self.metrics = OrderedDict()
        self.audit_trail = []
        self.experiment_id = None
        self.setup_logging()
        
    def setup_logging(self):
        """Configura el sistema de logging."""
        # Crear logger principal
        self.logger = logging.getLogger("Sysmic")
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Evitar propagación a root logger
        self.logger.propagate = False
        
        # Formato detallado
        log_format = logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        console_handler.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Remove existing handlers to avoid duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        self.logger.addHandler(console_handler)
        
        # Handler para archivo
        if self.config.log_file:
            try:
                file_handler = logging.FileHandler(self.config.log_file, encoding='utf-8')
                file_handler.setFormatter(log_format)
                file_handler.setLevel(logging.DEBUG)  # Archivo tiene todo
                self.logger.addHandler(file_handler)
            except Exception:
                pass
        
        # Handler para auditoría
        if self.config.audit_log_enabled:
            try:
                audit_file = self.config.data_directory / "audit.log"
                audit_handler = logging.FileHandler(audit_file, encoding='utf-8')
                audit_format = logging.Formatter(
                    '%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                audit_handler.setFormatter(audit_format)
                audit_handler.setLevel(logging.INFO)
                self.logger.addHandler(audit_handler)
            except Exception:
                pass
    
    def start_experiment(self, experiment_name: str, parameters: Dict):
        """Inicia un nuevo experimento científico."""
        self.experiment_id = f"exp_{int(time.time())}_{hashlib.md5(experiment_name.encode()).hexdigest()[:8]}"
        
        self.logger.info(f"Iniciando experimento: {experiment_name}")
        self.logger.info(f"ID del experimento: {self.experiment_id}")
        self.logger.info(f"Parámetros: {json.dumps(parameters, indent=2)}")
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'START_EXPERIMENT',
            'experiment_id': self.experiment_id,
            'experiment_name': experiment_name,
            'parameters': parameters
        })
        
        return self.experiment_id
    
    def log_metric(self, name: str, value: Any, unit: Optional[str] = None, 
                   uncertainty: Optional[float] = None, confidence_interval: Optional[Tuple] = None,
                   description: Optional[str] = None):
        """
        Registra una métrica científica con incertidumbre y documentación.
        """
        metric_data = {
            'value': value,
            'unit': unit,
            'uncertainty': uncertainty,
            'confidence_interval': confidence_interval,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'experiment_id': self.experiment_id
        }
        
        self.metrics[name] = metric_data
        
        # Log detallado
        if uncertainty is not None and confidence_interval is not None:
            self.logger.info(f"Métrica '{name}': {value} ± {uncertainty} {unit or ''} "
                           f"(CI: {confidence_interval[0]} - {confidence_interval[1]})")
        elif uncertainty is not None:
            self.logger.info(f"Métrica '{name}': {value} ± {uncertainty} {unit or ''}")
        else:
            self.logger.info(f"Métrica '{name}': {value} {unit or ''}")
    
    def log_warning(self, message: str, context: Optional[Dict] = None):
        """Registra una advertencia con contexto."""
        warning_data = {
            'message': message,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'experiment_id': self.experiment_id
        }
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'WARNING',
            **warning_data
        })
        
        self.logger.warning(f"{message} | Contexto: {context}")
    
    def log_error(self, error: Exception, context: Optional[Dict] = None):
        """Registra un error con traza completa."""
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'experiment_id': self.experiment_id
        }
        
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'ERROR',
            **error_data
        })
        
        self.logger.error(f"Error: {error} | Contexto: {context}")
        self.logger.debug(f"Traceback completo: {traceback.format_exc()}")
    
    def generate_latex_table(self, caption: str = "Resultados del Experimento", 
                            label: str = "tab:results") -> str:
        """
        Genera una tabla LaTeX para papers científicos.
        """
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append(f"\\caption{{{caption}}}")
        latex.append(f"\\label{{{label}}}")
        latex.append("\\begin{tabular}{lccc}")
        latex.append("\\hline")
        latex.append("Métrica & Valor & Incertidumbre & Unidad \\\\")
        latex.append("\\hline")
        
        for name, data in self.metrics.items():
            value = data['value']
            unc = data['uncertainty']
            unit = data['unit'] or ''
            
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            
            if unc is not None:
                unc_str = f"${value_str} \\pm {unc:.4f}$"
            else:
                unc_str = f"${value_str}$"
            
            latex.append(f"{name} & {unc_str} & {unit} \\\\")
        
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def generate_json_report(self) -> Dict:
        """Genera reporte completo en formato JSON."""
        return {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics,
            'audit_trail': self.audit_trail[-100:],  # Últimos 100 eventos
            'summary': self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict:
        """Genera resumen estadístico de las métricas."""
        numeric_metrics = {}
        
        for name, data in self.metrics.items():
            if isinstance(data['value'], (int, float)):
                numeric_metrics[name] = data['value']
        
        if numeric_metrics:
            values = list(numeric_metrics.values())
            return {
                'n_metrics': len(numeric_metrics),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        else:
            return {'n_metrics': 0}

class SystemCache:
    """
    Sistema de caché inteligente con persistencia y gestión de memoria.
    """
    
    def __init__(self, config: SystemConfiguration):
        self.config = config
        self.cache = OrderedDict()
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'size_bytes': 0,
            'max_size_bytes': config.cache_size_mb * 1024 * 1024
        }
        self.persistence_file = config.cache_directory / "cache.pkl"
        
        # Cargar caché persistente si existe
        self._load_persistent_cache()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor de la caché."""
        if key in self.cache:
            self.cache.move_to_end(key)  # LRU: mover al final
            self.cache_stats['hits'] += 1
            return self.cache[key]['value']
        else:
            self.cache_stats['misses'] += 1
            return default
    
    def set(self, key: str, value: Any, size_bytes: Optional[int] = None):
        """Establece un valor en la caché."""
        if size_bytes is None:
            # Estimación del tamaño
            size_bytes = self._estimate_size(value)
        
        # Verificar si hay espacio
        while (self.cache_stats['size_bytes'] + size_bytes > self.cache_stats['max_size_bytes'] 
               and self.cache):
            # Eliminar el elemento menos usado recientemente
            oldest_key, oldest_item = self.cache.popitem(last=False)
            self.cache_stats['size_bytes'] -= oldest_item['size_bytes']
        
        # Almacenar
        self.cache[key] = {
            'value': value,
            'size_bytes': size_bytes,
            'timestamp': time.time(),
            'access_count': 0
        }
        self.cache_stats['size_bytes'] += size_bytes
    
    @lru_cache(maxsize=1000)
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Genera clave de caché única para argumentos."""
        # Hash de todos los argumentos
        arg_hash = hashlib.md5()
        
        for arg in args:
            if hasattr(arg, 'tobytes'):
                arg_hash.update(arg.tobytes())
            else:
                arg_hash.update(str(arg).encode())
        
        for key, value in sorted(kwargs.items()):
            arg_hash.update(f"{key}:{value}".encode())
        
        return f"cache_{arg_hash.hexdigest()}"
    
    def cache_decorator(self, maxsize: int = 128):
        """
        Decorador para cachear resultados de funciones.
        
        Args:
            maxsize: Tamaño máximo de la caché para esta función
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generar clave
                cache_key = self._generate_cache_key(
                    func.__name__,
                    *args,
                    **{k: v for k, v in kwargs.items() if k != 'cache_key'}
                )
                
                # Verificar caché
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Ejecutar función
                result = func(*args, **kwargs)
                
                # Almacenar en caché
                self.set(cache_key, result)
                
                return result
            
            return wrapper
        return decorator
    
    def _estimate_size(self, obj: Any) -> int:
        """Estima el tamaño en bytes de un objeto."""
        try:
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, pd.DataFrame):
                return obj.memory_usage(deep=True).sum()
            elif isinstance(obj, dict):
                return sum(self._estimate_size(v) for v in obj.values())
            elif isinstance(obj, list):
                return sum(self._estimate_size(v) for v in obj)
            else:
                return sys.getsizeof(obj)
        except Exception:
            return 1024  # Fallback
    
    def _load_persistent_cache(self):
        """Carga caché persistente desde disco."""
        if self.persistence_file.exists():
            try:
                with open(self.persistence_file, 'rb') as f:
                    persistent_cache = pickle.load(f)
                
                # Filtrar entradas expiradas (más de 7 días)
                current_time = time.time()
                expired_keys = []
                
                for key, item in persistent_cache.items():
                    if current_time - item['timestamp'] > 7 * 24 * 3600:  # 7 días
                        expired_keys.append(key)
                    else:
                        self.cache[key] = item
                        self.cache_stats['size_bytes'] += item['size_bytes']
                
                # Eliminar expirados
                for key in expired_keys:
                    del persistent_cache[key]
                
                # Guardar sin expirados
                self._save_persistent_cache(persistent_cache)
                
                logging.info(f"Caché persistente cargada: {len(self.cache)} items")
                
            except Exception as e:
                logging.warning(f"No se pudo cargar caché persistente: {e}")
    
    def _save_persistent_cache(self, cache_data: Optional[Dict] = None):
        """Guarda caché persistente en disco."""
        try:
            if cache_data is None:
                cache_data = dict(self.cache)
            
            with open(self.persistence_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            logging.warning(f"No se pudo guardar caché persistente: {e}")
    
    def clear(self):
        """Limpia la caché."""
        self.cache.clear()
        self.cache_stats['size_bytes'] = 0
        self.cache_stats['hits'] = 0
        self.cache_stats['misses'] = 0
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas de la caché."""
        hit_rate = (self.cache_stats['hits'] / 
                   (self.cache_stats['hits'] + self.cache_stats['misses'])
                   if (self.cache_stats['hits'] + self.cache_stats['misses']) > 0 else 0)
        
        return {
            **self.cache_stats,
            'hit_rate': hit_rate,
            'items_count': len(self.cache),
            'memory_usage_mb': self.cache_stats['size_bytes'] / (1024 * 1024),
            'memory_limit_mb': self.cache_stats['max_size_bytes'] / (1024 * 1024)
        }

class ParallelExecutor:
    """
    Ejecutor paralelo inteligente con gestión de recursos.
    """
    
    def __init__(self, config: SystemConfiguration):
        self.config = config
        self.executor = None
        self.max_workers = config.max_workers
        self.task_queue = []
        self.results_cache = {}
        
    @contextmanager
    def get_executor(self, executor_type: str = "process"):
        """
        Context manager para ejecutor paralelo.
        
        Args:
            executor_type: "process" para CPU intensivo, "thread" para I/O
        """
        if executor_type == "process":
            executor_class = concurrent.futures.ProcessPoolExecutor
        else:
            executor_class = concurrent.futures.ThreadPoolExecutor
        
        executor = None
        try:
            executor = executor_class(max_workers=self.max_workers)
            yield executor
        finally:
            if executor:
                executor.shutdown(wait=True)
    
    def parallel_map(self, func: Callable, iterable: Sequence, 
                    chunk_size: Optional[int] = None, 
                    executor_type: str = "process",
                    progress_bar: bool = False) -> List:
        """
        Ejecuta función en paralelo sobre un iterable.
        """
        if chunk_size is None:
            # Calcular chunk size óptimo
            chunk_size = max(1, len(iterable) // (self.max_workers * 4))
        
        # Dividir en chunks
        chunks = []
        for i in range(0, len(iterable), chunk_size):
            chunks.append(iterable[i:i + chunk_size])
        
        # Función wrapper para procesar chunks
        def process_chunk(chunk):
            return [func(item) for item in chunk]
        
        # Ejecutar en paralelo
        with self.get_executor(executor_type) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            
            results = []
            for future in futures:
                results.extend(future.result())
        
        return results
    
    def execute_tasks(self, tasks: List[Dict], 
                     priority: bool = False) -> List:
        """
        Ejecuta una lista de tareas con opción de prioridad.
        """
        if priority:
            tasks.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        with self.get_executor("process") as executor:
            futures = []
            for task in tasks:
                future = executor.submit(task['func'], *task.get('args', []), 
                                        **task.get('kwargs', {}))
                futures.append((task.get('id', None), future))
            
            # Recolectar resultados
            results = []
            for task_id, future in futures:
                try:
                    result = future.result(timeout=task.get('timeout', 3600))
                    results.append({'id': task_id, 'result': result, 'status': 'success'})
                except concurrent.futures.TimeoutError:
                    results.append({'id': task_id, 'result': None, 'status': 'timeout'})
                except Exception as e:
                    results.append({'id': task_id, 'result': str(e), 'status': 'error'})
        
        return results

@dataclass
class FractalEstimationResult:
    """Resultado estructurado de estimación de dimensión fractal."""
    dimension: float
    uncertainty: float
    method: str
    n_points: int
    scaling_region: Optional[Tuple[float, float]] = None
    correlation_coefficient: Optional[float] = None
    residuals: Optional[np.ndarray] = None
    bootstrap_distribution: Optional[np.ndarray] = None
    convergence_diagnostics: Optional[Dict] = None
    quality_metrics: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario."""
        return {
            'dimension': self.dimension,
            'uncertainty': self.uncertainty,
            'method': self.method,
            'n_points': self.n_points,
            'scaling_region': self.scaling_region,
            'correlation_coefficient': self.correlation_coefficient,
            'quality_metrics': self.quality_metrics
        }
