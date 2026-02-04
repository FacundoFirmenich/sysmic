class QuantumEfficiencyParadoxAnalyzer:
    """
    Analyzes the Quantum Efficiency Paradox in tectonic systems
    Examines D2 efficiency across different tectonic regimes
    """
    
    def __init__(self):
        self.paradox_threshold = 0.85
        self.system_complexity_metrics = ['d2', 'b_value', 'event_density', 'spatial_entropy']
        
    def analyze_efficiency_paradox(self, system_a: Dict, system_b: Dict) -> Dict:
        """
        Analyzes efficiency paradox between two tectonic systems
        """
        complexity_comparison = self._compare_system_complexity(system_a, system_b)
        efficiency_metrics = self._calculate_efficiency_metrics(system_a, system_b)
        paradox_strength = self._compute_paradox_strength(complexity_comparison, efficiency_metrics)
        
        analysis_results = {
            'paradox_detected': paradox_strength > self.paradox_threshold,
            'paradox_strength': paradox_strength,
            'complexity_comparison': complexity_comparison,
            'efficiency_metrics': efficiency_metrics,
            'theoretical_implications': self._generate_theoretical_implications(
                paradox_strength, complexity_comparison
            ),
            'quantum_interpretation': self._quantum_interpretation(system_a, system_b, paradox_strength)
        }
        
        return analysis_results
    
    def _compare_system_complexity(self, system_a: Dict, system_b: Dict) -> Dict:
        """Comprehensive comparison of system complexity"""
        complexity_metrics = {}
        
        for metric in self.system_complexity_metrics:
            value_a = system_a.get(metric, 0)
            value_b = system_b.get(metric, 0)
            
            if value_a == 0 or value_b == 0:
                complexity_metrics[metric] = 'INCOMPARABLE'
                continue
                
            ratio = value_a / value_b
            if ratio > 1.2:
                complexity_metrics[metric] = 'A_MORE_COMPLEX'
            elif ratio < 0.8:
                complexity_metrics[metric] = 'B_MORE_COMPLEX' 
            else:
                complexity_metrics[metric] = 'COMPARABLE_COMPLEXITY'
        
        # Overall complexity assessment
        a_complexity = self._compute_overall_complexity(system_a)
        b_complexity = self._compute_overall_complexity(system_b)
        
        complexity_metrics['overall_complexity'] = {
            'system_a': a_complexity,
            'system_b': b_complexity,
            'complexity_ratio': a_complexity / b_complexity
        }
        
        return complexity_metrics
    
    def _compute_overall_complexity(self, system: Dict) -> float:
        """Computes overall system complexity metric"""
        complexity_components = []
        
        # Fractal dimension component (weighted highest)
        d2_complexity = system.get('d2', 1.0) / 3.0  # Normalized to maximum 3.0
        complexity_components.append(d2_complexity * 0.4)
        
        # Spatial entropy component
        spatial_entropy = system.get('spatial_entropy', 0.5)
        complexity_components.append(spatial_entropy * 0.3)
        
        # Event density component
        event_density = min(1.0, system.get('event_density', 0) / 1000)  # Normalized
        complexity_components.append(event_density * 0.2)
        
        # b-value complexity component
        b_complexity = 1 - abs(system.get('b_value', 1.0) - 1.0)  # Peak at b=1.0
        complexity_components.append(b_complexity * 0.1)
        
        return sum(complexity_components)
    
    def _calculate_efficiency_metrics(self, system_a: Dict, system_b: Dict) -> Dict:
        """Calculates efficiency metrics for both systems"""
        efficiency_a = system_a.get('d2', 1.0) / self._compute_overall_complexity(system_a)
        efficiency_b = system_b.get('d2', 1.0) / self._compute_overall_complexity(system_b)
        
        return {
            'system_a_efficiency': efficiency_a,
            'system_b_efficiency': efficiency_b, 
            'efficiency_ratio': efficiency_a / efficiency_b,
            'efficiency_difference': efficiency_a - efficiency_b
        }
    
    def _compute_paradox_strength(self, complexity: Dict, efficiency: Dict) -> float:
        """Computes the strength of the efficiency paradox"""
        # Paradox exists when more complex system has lower D2 but higher efficiency
        complexity_ratio = complexity['overall_complexity']['complexity_ratio']
        efficiency_ratio = efficiency['efficiency_ratio']
        
        if complexity_ratio > 1.1 and efficiency_ratio < 0.9:
            # System A more complex but less efficient - strong paradox
            paradox_strength = min(1.0, (complexity_ratio - 1) * (1 - efficiency_ratio) * 5)
        elif complexity_ratio < 0.9 and efficiency_ratio > 1.1:
            # System B more complex but less efficient - strong paradox
            paradox_strength = min(1.0, (1 - complexity_ratio) * (efficiency_ratio - 1) * 5)
        else:
            paradox_strength = 0.0
            
        return paradox_strength
    
    def _generate_theoretical_implications(self, paradox_strength: float, complexity: Dict) -> List[str]:
        """Generates theoretical implications of efficiency paradox"""
        implications = []
        
        if paradox_strength > 0.8:
            implications.extend([
                "Quantum mechanical constraints dominate tectonic efficiency",
                "Cold slab systems exhibit organized rupture propagation",
                "Warm continental systems show chaotic energy dissipation",
                "Fundamental rheological differences in deformation mechanisms"
            ])
        elif paradox_strength > 0.5:
            implications.extend([
                "Moderate efficiency paradox suggests transitional behavior",
                "Mixed deformation mechanisms operating simultaneously", 
                "Complex energy partitioning between different scales"
            ])
        
        return implications
    
    def _quantum_interpretation(self, system_a: Dict, system_b: Dict, paradox_strength: float) -> str:
        """Provides quantum mechanical interpretation of the paradox"""
        if paradox_strength > 0.7:
            return (
                "The observed efficiency paradox suggests quantum confinement effects "
                "where cold oceanic slabs maintain coherent wavefunction propagation "
                "while warm continental systems experience quantum decoherence "
                "leading to inefficient energy transfer and chaotic fracturing."
            )
        else:
            return "System behavior consistent with classical continuum mechanics."