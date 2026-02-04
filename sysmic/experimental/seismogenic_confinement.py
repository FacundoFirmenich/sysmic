"""
SEISMIC FRACTAL ANALYSIS SYSTEM v3.0
Advanced implementation of seismogenic confinement theory and universal precursor detection
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class SeismogenicConfinementTheory:
    """
    Implementation of the Fundamental Law of Seismogenic Confinement
    Based on: "Seismogenic Confinement: A Universal Fractal Law Governing Lithospheric Rupture Dynamics"
    """
    
    def __init__(self):
        self.quantum_regime_boundaries = {
            'QUANTUM_CONFINEMENT_I': (1.25, 1.45, "Induced/Anthropogenic Seismicity"),
            'QUANTUM_CONFINEMENT_II': (1.45, 1.70, "Transform Fault Systems"), 
            'QUANTUM_CONFINEMENT_III': (1.70, 2.00, "Subduction Interface Baseline"),
            'QUANTUM_CONFINEMENT_IV': (2.00, 2.20, "Transitional Geometries"),
            'QUANTUM_CONFINEMENT_V': (2.20, 2.36, "Collisional Volumetric Deformation"),
            'QUANTUM_CONFINEMENT_VI': (2.36, 3.00, "Mantle Plume Maximum Entropy")
        }
        
        self.universal_precursor_thresholds = {
            'DIMENSIONAL_COLLAPSE_SIGNATURE': 1.00,
            'SILENT_LOCKING_QUIESCENCE': 5,
            'CRITICAL_STRESS_ACCUMULATION': 0.70
        }
    
    def quantum_regime_classification(self, fractal_dimension: float, 
                                    b_value: Optional[float] = None,
                                    spatial_parameters: Optional[Dict] = None) -> Dict:
        """
        Classifies tectonic regimes according to quantum confinement hierarchy
        Returns comprehensive regime characterization
        """
        regime_characterization = {
            'fractal_dimension': fractal_dimension,
            'quantum_regime': None,
            'confinement_intensity': self._calculate_confinement_intensity(fractal_dimension),
            'theoretical_interpretation': None,
            'predictive_implications': []
        }
        
        # Quantum regime determination
        for regime_id, (d2_min, d2_max, description) in self.quantum_regime_boundaries.items():
            if d2_min <= fractal_dimension < d2_max:
                regime_characterization['quantum_regime'] = regime_id
                regime_characterization['theoretical_interpretation'] = description
                break
        
        if not regime_characterization['quantum_regime']:
            regime_characterization['quantum_regime'] = 'UNDEFINED_QUANTUM_STATE'
            regime_characterization['theoretical_interpretation'] = 'Quantum regime beyond defined boundaries'
        
        # Refinement with stress accumulation parameters
        if b_value:
            stress_characterization = self._characterize_stress_state(b_value, fractal_dimension)
            regime_characterization.update(stress_characterization)
        
        # Predictive implications
        regime_characterization['predictive_implications'] = self._generate_predictive_implications(
            regime_characterization
        )
        
        return regime_characterization
    
    def _calculate_confinement_intensity(self, d2: float) -> float:
        """Calculates quantum confinement intensity (0-1 scale)"""
        normalized_confinement = (d2 - 1.25) / (2.36 - 1.25)
        quantum_intensity = 1 - max(0, min(1, normalized_confinement))
        return round(quantum_intensity, 4)
    
    def _characterize_stress_state(self, b_value: float, d2: float) -> Dict:
        """Characterizes stress accumulation state"""
        stress_parameters = {
            'b_value': b_value,
            'stress_accumulation_index': max(0, (1 - b_value/1.0) * 2),
            'rheological_state': None,
            'failure_propensity': None
        }
        
        if b_value < 0.8:
            stress_parameters['rheological_state'] = 'CRITICAL_STRESS_ACCUMULATION'
            stress_parameters['failure_propensity'] = 'HIGH'
        elif b_value < 1.0:
            stress_parameters['rheological_state'] = 'MODERATE_STRESS_ACCUMULATION' 
            stress_parameters['failure_propensity'] = 'MEDIUM'
        else:
            stress_parameters['rheological_state'] = 'DIFFUSE_STRESS_DISTRIBUTION'
            stress_parameters['failure_propensity'] = 'LOW'
        
        return stress_parameters
    
    def _generate_predictive_implications(self, regime_data: Dict) -> List[str]:
        """Generates theoretical predictive implications"""
        implications = []
        regime = regime_data['quantum_regime']
        confinement = regime_data['confinement_intensity']
        
        if regime.startswith('QUANTUM_CONFINEMENT_I'):
            implications.extend([
                "High susceptibility to fluid-induced triggering",
                "Potential for rapid stress transfer",
                "Anthropogenic signature detectable"
            ])
        elif regime.startswith('QUANTUM_CONFINEMENT_III'):
            implications.extend([
                "Characteristic earthquake behavior expected",
                "Megathrust rupture potential present",
                "Tsunami generation capability confirmed"
            ])
        elif regime.startswith('QUANTUM_CONFINEMENT_V'):
            implications.extend([
                "Volumetric deformation patterns dominant",
                "Complex rupture propagation expected",
                "Multiple fault interaction probable"
            ])
        
        if confinement > 0.7:
            implications.append("High confinement suggests localized rupture nucleation")
        elif confinement < 0.3:
            implications.append("Low confinement indicates distributed deformation")
            
        return implications