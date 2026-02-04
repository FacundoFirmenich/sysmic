"""
COMPONENTE 5: SISTEMA INTEGRADO SYSMIC
Sistema unificado que integra todos los componentes modulares.
Proporciona una API profesional para análisis fractal de catálogos sísmicos.
"""

import numpy as np
import pandas as pd
import logging
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field, asdict

# Importar infraestructura y tipos
from .infrastructure import (
    SystemConfiguration, 
    CertificationLevel, 
    ScientificLogger, 
    SystemCache
)
from .fractal_estimator import FractalDimensionEstimator
from .temporal_analyzer import TemporalFractalAnalyzer
from .multifractal_analyzer import Multifractal3DAnalyzer
from .hierarchical_bayesian import HierarchicalBayesianFractal
from .fractal_feature_extractor import FractalFeatureExtractor
from .scientific_validator import ScientificValidator
from .gravitas import GravitationalAnomalyDetector

# Experimental imports (optional)
try:
    from .experimental import ramanujan
    HAS_EXPERIMENTAL = True
except ImportError:
    HAS_EXPERIMENTAL = False

@dataclass
class AnalysisResult:
    """Resultado completo de análisis."""
    
    analysis_id: str
    timestamp: str
    catalog_info: Dict[str, Any]
    config: Dict[str, Any]
    
    # Resultados por módulo
    fractal_dimension: Optional[Dict] = None
    temporal_analysis: Optional[Dict] = None
    multifractal_spectrum: Optional[Dict] = None
    bayesian_analysis: Optional[Dict] = None
    features: Optional[Dict] = None
    validation: Optional[Dict] = None
    
    # Reportes
    executive_summary: Optional[str] = None
    technical_report: Optional[Dict] = None
    recommendations: Optional[List[str]] = None
    
    # Metadatos
    computation_time: float = 0.0
    success: bool = False
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario."""
        return asdict(self)
    
    def save(self, filepath: str):
        """Guarda resultados a archivo."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def generate_report(self, format: str = 'text') -> str:
        """Genera reporte en formato especificado."""
        if format == 'text':
            return self._generate_text_report()
        elif format == 'json':
            return json.dumps(self.to_dict(), indent=2, default=str)
        else:
            return "Report format not supported"

    def _generate_text_report(self) -> str:
        """Genera reporte de texto."""
        report = []
        report.append("=" * 80)
        report.append("SYSMIC - ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Analysis ID: {self.analysis_id}")
        report.append(f"Timestamp: {self.timestamp}")
        report.append(f"Catalog: {self.catalog_info.get('n_events', 0)} events")
        report.append(f"Computation Time: {self.computation_time:.2f} seconds")
        report.append(f"Success: {'Yes' if self.success else 'No'}")
        return "\n".join(report)

class Sysmic:
    """
    Sistema integrado de análisis fractal profesional.
    Integra estimación D, multifractales, features, bayesiano y validación.
    """
    
    def __init__(self, config: Optional[SystemConfiguration] = None):
        self.config = config or SystemConfiguration()
        
        # Configurar logging
        self.logger = ScientificLogger(self.config)
        self.logger.logger.info(f"Sysmic inicializado (Nivel: {self.config.certification_level})")
        
        # Inicializar caché
        self.cache = SystemCache(self.config)
        
        # Inicializar módulos
        self.modules = self._initialize_modules()
        
    def _initialize_modules(self) -> Dict[str, Any]:
        """Inicializa los módulos según el nivel de certificación."""
        modules = {}
        # Mapeo de niveles:
        # LEVEL_0 (Basic) -> Estimator
        # LEVEL_1 (Scientific) -> Estimator, Multifractal, Validator
        # LEVEL_2 (Commercial) -> + Features
        # LEVEL_3 (Certified) -> + Bayesian
        
        level = self.config.certification_level
        
        self.logger.logger.info(f"Inicializando módulos para nivel {level}")
        
        # 1. Estimador (Siempre presente)
        modules['estimator'] = FractalDimensionEstimator(self.config, self.logger)
        
        # 2. Análisis Temporal (Siempre presente en Pro)
        modules['temporal'] = TemporalFractalAnalyzer(self.config, self.logger)
        
        # 3. Multifractal (Scientific+)
        if level != CertificationLevel.LEVEL_0:
            modules['multifractal'] = Multifractal3DAnalyzer(
                q_min=-10.0, q_max=10.0, n_q_points=41, n_scales=20
            )
            
        # 4. Validator (Scientific+)
        if level != CertificationLevel.LEVEL_0:
            modules['validator'] = ScientificValidator(
                tolerance=self.config.validation_threshold
            )
            
        # 5. Features (Commercial+)
        if level in [CertificationLevel.LEVEL_2, CertificationLevel.LEVEL_3, CertificationLevel.LEVEL_4]:
            modules['feature_extractor'] = FractalFeatureExtractor(
                random_state=42
            )
            
        # 6. Bayesian (Certified+)
        if level in [CertificationLevel.LEVEL_3, CertificationLevel.LEVEL_4]:
            modules['bayesian'] = HierarchicalBayesianFractal(
                n_samples=self.config.monte_carlo_samples
            )
            
        return modules
        
    def analyze_catalog(self,
                       catalog: pd.DataFrame,
                       analysis_types: Optional[List[str]] = None,
                       output_format: str = 'all') -> AnalysisResult:
        """
        Ejecuta análisis completo de un catálogo sísmico.
        """
        start_time = time.time()
        
        # Validar catálogo
        self._validate_catalog(catalog)
        
        # ID de análisis
        analysis_id = self._generate_analysis_id(catalog)
        
        # Verificar caché
        cached = self.cache.get(analysis_id)
        if cached:
            self.logger.logger.info(f"Usando análisis en caché: {analysis_id}")
            return cached
            
        # Info del catálogo
        catalog_info = {
            'n_events': len(catalog),
            'time_range': (catalog['time'].min().isoformat(), catalog['time'].max().isoformat())
        }
        
        result = AnalysisResult(
            analysis_id=analysis_id,
            timestamp=datetime.now().isoformat(),
            catalog_info=catalog_info,
            config=asdict(self.config) if hasattr(self.config, 'to_dict') else str(self.config)
        )
        
        try:
            if analysis_types is None:
                analysis_types = ['fractal_dimension', 'temporal', 'multifractal', 'validation']
            
            # Ejecutar análisis
            if 'fractal_dimension' in analysis_types:
                result.fractal_dimension = self._analyze_fractal_dimension(catalog)
                
            if 'temporal' in analysis_types:
                result.temporal_analysis = self._analyze_temporal(catalog)
                
            if 'multifractal' in analysis_types and 'multifractal' in self.modules:
                result.multifractal_spectrum = self._analyze_multifractal(catalog)
                
            if 'bayesian' in analysis_types and 'bayesian' in self.modules:
                result.bayesian_analysis = self._analyze_bayesian(catalog)
                
            if 'features' in analysis_types and 'feature_extractor' in self.modules:
                result.features = self._extract_features(catalog)
                
            if 'validation' in analysis_types and 'validator' in self.modules:
                result.validation = self._validate_analysis(result)
                
            result.success = True
            
        except Exception as e:
            self.logger.log_error(e, {'analysis_id': analysis_id})
            result.success = False
            result.errors.append(str(e))
            
        result.computation_time = time.time() - start_time
        
        # Cachear
        if result.success:
            self.cache.set(analysis_id, result)
            
        return result

    def _validate_catalog(self, catalog: pd.DataFrame):
        """Valida formato del catálogo."""
        required = ['time', 'latitude', 'longitude', 'depth', 'mag']
        for col in required:
            if col not in catalog.columns:
                raise ValueError(f"Missing column: {col}")
        
    def _generate_analysis_id(self, catalog: pd.DataFrame) -> str:
        """Genera ID único."""
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(catalog[['time', 'latitude', 'longitude']]).values.tobytes()
        ).hexdigest()[:12]
        return f"analysis_{data_hash}"
        
    def _analyze_fractal_dimension(self, catalog: pd.DataFrame) -> Dict:
        """Ejecuta estimador de dimensión fractal."""
        positions = catalog[['latitude', 'longitude', 'depth']].values
        res = self.modules['estimator'].estimate(positions, method='gp')
        return res.to_dict()
        
    def _analyze_temporal(self, catalog: pd.DataFrame) -> Dict:
        """Ejecuta análisis temporal D(t)."""
        return self.modules['temporal'].compute_D_t(catalog)
        
    def _analyze_multifractal(self, catalog: pd.DataFrame) -> Dict:
        """Ejecuta análisis multifractal."""
        positions = catalog[['latitude', 'longitude', 'depth']].values
        try:
            res = self.modules['multifractal'].compute_multifractal_spectrum(
                positions, method='boxcounting_3d', compute_quality=True
            )
            # Adapt output if needed
            return asdict(res) if not hasattr(res, 'to_dict') else res.to_dict() or asdict(res)
        except:
             # Fallback if method name differs in _python_2 class
             return {'error': 'Multifractal analysis failed'}
             
    def _analyze_bayesian(self, catalog: pd.DataFrame) -> Dict:
        positions = catalog[['latitude', 'longitude', 'depth']].values
        return self.modules['bayesian'].estimate(positions)
        
    def _extract_features(self, catalog: pd.DataFrame) -> Dict:
        return self.modules['feature_extractor'].extract(catalog)
        
    def _validate_analysis(self, result: AnalysisResult) -> Dict:
        # Construct validation input
        return self.modules['validator'].validate(result.fractal_dimension)

if __name__ == '__main__':
    # Simple CLI test
    print("Sysmic v6.0 Initialized")
    config = SystemConfiguration()
    system = Sysmic(config)
    print(f"Modules loaded: {list(system.modules.keys())}")
