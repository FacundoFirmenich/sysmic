"""
================================================================================
CONSISTENCY INDICATORS MODULE - Valid for Small & Large N
================================================================================
Part of Framework Hypersistémico 3A+\CAHTPhase

PHILOSOPHY:
- N pequeño NO es defecto, es condición de trabajo
- Consistency ≠ external validity (generalization)
- Consistency = internal coherence of results

INDICATORS:
1. Cross-Fold Consistency (variance, stability)
2. Residual Consistency (pattern, distribution)
3. Complexity Consistency (signal vs reconstruction)
4. Metric Consistency (multiple metrics agree?)

OUTPUT: Consistency score (0-1) + interpretation válida para cualquier N

Author: Sysmic Framework
Status: Core Validation - N-Agnostic
================================================================================
"""
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats

__all__ = [
    'compute_consistency_indicators',
    'interpret_consistency',
    'consistency_report'
]


def compute_consistency_indicators(
    kfold_qualities: List[float],
    residual_stats: Dict,
    complexity_stats: Dict,
    reconstruction_quality: float
) -> Dict:
    """
    Compute consistency indicators válidos para cualquier N.
    
    NO penaliza N pequeño. Mide coherencia interna.
    
    Args:
        kfold_qualities: K-fold quality scores
        residual_stats: From residual_analysis
        complexity_stats: From assess_reconstruction_complexity
        reconstruction_quality: Overall quality
        
    Returns:
        Dict with consistency indicators (0-1 scale)
    """
    print(f"\n[CONSISTENCY INDICATORS] N-agnostic validation")
    
    indicators = {}
    
    # ========================================================================
    # 1. CROSS-FOLD CONSISTENCY
    # ========================================================================
    # Measures: Are K-fold results stable?
    # Interpretation: 
    #   - High variance = INCONSISTENT (random/unstable)
    #   - Low variance = CONSISTENT (stable pattern)
    #   - Zero variance with N pequeño = PERFECTLY CONSISTENT (not suspicious)
    
    kfold_mean = np.mean(kfold_qualities)
    kfold_std = np.std(kfold_qualities)
    kfold_variance = np.var(kfold_qualities)
    
    # Coefficient of variation (normalized by mean)
    if kfold_mean > 0:
        cv = kfold_std / kfold_mean
    else:
        cv = 0.0
    
    # Consistency score: 1.0 = perfect consistency (low CV)
    # CV < 0.05 → excellent consistency
    # CV < 0.10 → good consistency
    # CV > 0.20 → poor consistency
    
    if cv < 0.05:
        crossfold_consistency = 1.0
        crossfold_interpretation = "EXCELLENT: Results highly stable"
    elif cv < 0.10:
        crossfold_consistency = 0.9 - (cv - 0.05) * 10  # Linear decay 0.9 → 0.4
        crossfold_interpretation = "GOOD: Results reasonably stable"
    elif cv < 0.20:
        crossfold_consistency = 0.4 - (cv - 0.10) * 2  # Linear decay 0.4 → 0.2
        crossfold_interpretation = "MODERATE: Some instability"
    else:
        crossfold_consistency = max(0.0, 0.2 - (cv - 0.20))
        crossfold_interpretation = "POOR: Results unstable"
    
    indicators['crossfold'] = {
        'consistency_score': float(crossfold_consistency),
        'cv': float(cv),
        'mean': float(kfold_mean),
        'std': float(kfold_std),
        'variance': float(kfold_variance),
        'interpretation': crossfold_interpretation,
        'n_agnostic': True  # This metric is valid for any N
    }
    
    print(f"  Cross-fold consistency: {crossfold_consistency:.3f} (CV={cv:.4f})")
    
    # ========================================================================
    # 2. RESIDUAL CONSISTENCY
    # ========================================================================
    # Measures: Are residuals well-behaved (Gaussian, uncorrelated)?
    # Interpretation:
    #   - Gaussian residuals = model captures all structure
    #   - Autocorrelated residuals = model misses temporal structure
    #   - Mean≠0 = systematic bias
    
    residual_quality = residual_stats.get('quality_score', 0.0)
    residual_error = residual_stats.get('relative_error', 1.0)
    
    # Consistency score based on residual quality
    # High quality → high consistency
    
    if residual_quality >= 0.75:
        residual_consistency = residual_quality
        residual_interpretation = "EXCELLENT: Residuals well-behaved"
    elif residual_quality >= 0.50:
        residual_consistency = residual_quality
        residual_interpretation = "GOOD: Residuals acceptable"
    elif residual_error < 0.01:
        # Special case: very low error but poor quality
        # This suggests TRIVIAL problem (everything is constant)
        residual_consistency = 0.3
        residual_interpretation = "SUSPICIOUS: Error too low (trivial signal?)"
    else:
        residual_consistency = residual_quality
        residual_interpretation = "POOR: Residuals show problems"
    
    indicators['residual'] = {
        'consistency_score': float(residual_consistency),
        'quality_score': float(residual_quality),
        'relative_error': float(residual_error),
        'interpretation': residual_interpretation,
        'n_agnostic': True
    }
    
    print(f"  Residual consistency: {residual_consistency:.3f} (quality={residual_quality:.3f})")
    
    # ========================================================================
    # 3. COMPLEXITY CONSISTENCY
    # ========================================================================
    # Measures: Does reconstructed signal match original complexity?
    # Interpretation:
    #   - Complexity ratio ~1.0 = consistent
    #   - Ratio << 1.0 = oversimplified (information loss)
    #   - Ratio >> 1.0 = overcomplicated (hallucination)
    
    complexity_ratio = complexity_stats.get('complexity_ratio', 0.0)
    is_trivial = complexity_stats.get('is_trivial', False)
    
    # Consistency score: peaks at ratio=1.0
    if is_trivial:
        complexity_consistency = 0.2
        complexity_interpretation = "POOR: Reconstruction trivial (oversimplified)"
    elif 0.8 <= complexity_ratio <= 1.2:
        complexity_consistency = 1.0
        complexity_interpretation = "EXCELLENT: Complexity preserved"
    elif 0.5 <= complexity_ratio < 0.8 or 1.2 < complexity_ratio <= 1.5:
        # Moderate deviation
        deviation = abs(complexity_ratio - 1.0)
        complexity_consistency = max(0.5, 1.0 - deviation)
        complexity_interpretation = "GOOD: Minor complexity shift"
    else:
        # Large deviation
        complexity_consistency = max(0.0, 0.5 - abs(complexity_ratio - 1.0) * 0.5)
        complexity_interpretation = "POOR: Significant complexity mismatch"
    
    indicators['complexity'] = {
        'consistency_score': float(complexity_consistency),
        'complexity_ratio': float(complexity_ratio),
        'is_trivial': is_trivial,
        'interpretation': complexity_interpretation,
        'n_agnostic': True
    }
    
    print(f"  Complexity consistency: {complexity_consistency:.3f} (ratio={complexity_ratio:.3f})")
    
    # ========================================================================
    # 4. METRIC CONSISTENCY
    # ========================================================================
    # Measures: Do multiple metrics agree on quality?
    # Interpretation:
    #   - All metrics high → consistent high quality
    #   - Metrics disagree → inconsistent (red flag)
    
    metric_values = [
        kfold_mean,  # K-fold quality
        1.0 - residual_error,  # Residual quality (inverted error)
        complexity_ratio,  # Complexity preserved
        reconstruction_quality  # Overall quality
    ]
    
    # Normalize to 0-1
    normalized_metrics = []
    for m in metric_values:
        if 0 <= m <= 1:
            normalized_metrics.append(m)
        elif m > 1:
            # Complexity ratio can be >1
            normalized_metrics.append(min(1.0, 2.0 - m))
    
    # Agreement: std of normalized metrics
    # Low std → metrics agree
    metric_std = np.std(normalized_metrics)
    metric_mean = np.mean(normalized_metrics)
    
    if metric_std < 0.1:
        metric_consistency = 1.0
        metric_interpretation = "EXCELLENT: All metrics agree"
    elif metric_std < 0.2:
        metric_consistency = 0.8
        metric_interpretation = "GOOD: Metrics mostly agree"
    elif metric_std < 0.3:
        metric_consistency = 0.5
        metric_interpretation = "MODERATE: Some metric disagreement"
    else:
        metric_consistency = max(0.0, 0.5 - (metric_std - 0.3))
        metric_interpretation = "POOR: Metrics disagree (red flag)"
    
    indicators['metric_agreement'] = {
        'consistency_score': float(metric_consistency),
        'metric_std': float(metric_std),
        'metric_mean': float(metric_mean),
        'interpretation': metric_interpretation,
        'n_agnostic': True
    }
    
    print(f"  Metric consistency: {metric_consistency:.3f} (std={metric_std:.3f})")
    
    # ========================================================================
    # OVERALL CONSISTENCY SCORE
    # ========================================================================
    # Weighted average of all indicators
    
    weights = {
        'crossfold': 0.3,
        'residual': 0.3,
        'complexity': 0.2,
        'metric_agreement': 0.2
    }
    
    overall_score = sum(
        indicators[key]['consistency_score'] * weights[key]
        for key in weights.keys()
    )
    
    print(f"\n  OVERALL CONSISTENCY: {overall_score:.3f}")
    
    return {
        'indicators': indicators,
        'overall_score': float(overall_score),
        'weights': weights,
        'n_samples': len(kfold_qualities) if kfold_qualities else 0
    }


def interpret_consistency(consistency_result: Dict, n_samples: int) -> Dict:
    """
    Interpret consistency results with N-aware context.
    
    KEY: Does NOT penalize small N, but provides context.
    
    Args:
        consistency_result: From compute_consistency_indicators
        n_samples: Number of samples/events
        
    Returns:
        Dict with interpretation
    """
    overall_score = consistency_result['overall_score']
    
    # Consistency tiers (independent of N)
    if overall_score >= 0.90:
        tier = "TIER-1 EXCELLENT"
        description = "Results are internally consistent and robust"
    elif overall_score >= 0.75:
        tier = "TIER-2 GOOD"
        description = "Results show good internal consistency"
    elif overall_score >= 0.60:
        tier = "TIER-3 ACCEPTABLE"
        description = "Results are reasonably consistent with minor concerns"
    elif overall_score >= 0.40:
        tier = "TIER-4 MODERATE"
        description = "Results show moderate consistency issues"
    else:
        tier = "TIER-5 POOR"
        description = "Results have significant consistency problems"
    
    # N-aware context (NOT penalty, just context)
    if n_samples < 10:
        n_context = f"N={n_samples} (small sample): Results are CONSISTENT but LIMITED in scope. External validation needed for generalization."
    elif n_samples < 50:
        n_context = f"N={n_samples} (moderate sample): Results are CONSISTENT with reasonable scope. Some external validation recommended."
    else:
        n_context = f"N={n_samples} (large sample): Results are CONSISTENT with broad scope. High confidence in generalization."
    
    # Specific recommendations
    recommendations = []
    
    indicators = consistency_result['indicators']
    
    if indicators['crossfold']['consistency_score'] < 0.7:
        recommendations.append("Investigate K-fold instability (high variance)")
    
    if indicators['residual']['consistency_score'] < 0.7:
        recommendations.append("Review residual properties (may indicate model issues)")
    
    if indicators['complexity']['consistency_score'] < 0.7:
        if indicators['complexity']['is_trivial']:
            recommendations.append("CRITICAL: Signal is trivial (explains Quality=1.0)")
        else:
            recommendations.append("Complexity mismatch detected (information loss/gain)")
    
    if indicators['metric_agreement']['consistency_score'] < 0.7:
        recommendations.append("CRITICAL: Metrics disagree (investigate cause)")
    
    if not recommendations:
        recommendations.append("No concerns detected. Framework operating as expected.")
    
    return {
        'overall_score': overall_score,
        'tier': tier,
        'description': description,
        'n_context': n_context,
        'recommendations': recommendations,
        'is_n_agnostic': True
    }


def consistency_report(consistency_result: Dict, interpretation: Dict) -> str:
    """
    Generate markdown consistency report.
    
    Args:
        consistency_result: From compute_consistency_indicators
        interpretation: From interpret_consistency
        
    Returns:
        Markdown report string
    """
    report = f"""
# CONSISTENCY INDICATORS REPORT

## Overall Assessment

**Consistency Score**: {interpretation['overall_score']:.3f} / 1.0  
**Tier**: {interpretation['tier']}  
**Description**: {interpretation['description']}

**Sample Context**: {interpretation['n_context']}

---

## Individual Indicators

"""
    
    indicators = consistency_result['indicators']
    
    for key, data in indicators.items():
        name = key.replace('_', ' ').title()
        score = data['consistency_score']
        interp = data['interpretation']
        
        report += f"### {name}\n\n"
        report += f"- **Score**: {score:.3f}\n"
        report += f"- **Interpretation**: {interp}\n"
        
        if key == 'crossfold':
            report += f"- **CV**: {data['cv']:.4f}\n"
            report += f"- **Variance**: {data['variance']:.6f}\n"
        elif key == 'residual':
            report += f"- **Relative Error**: {data['relative_error']:.4f}\n"
        elif key == 'complexity':
            report += f"- **Complexity Ratio**: {data['complexity_ratio']:.3f}\n"
            report += f"- **Is Trivial**: {data['is_trivial']}\n"
        elif key == 'metric_agreement':
            report += f"- **Metric Std**: {data['metric_std']:.3f}\n"
        
        report += "\n"
    
    report += "---\n\n## Recommendations\n\n"
    
    for i, rec in enumerate(interpretation['recommendations'], 1):
        report += f"{i}. {rec}\n"
    
    report += f"\n---\n\n*Report generated: Framework Hypersistémico 3A+\\CAHTPhase*\n"
    report += f"*N-Agnostic Validation: Valid for any sample size*\n"
    
    return report


if __name__ == "__main__":
    print("="*80)
    print("  CONSISTENCY INDICATORS MODULE")
    print("  N-Agnostic Validation")
    print("="*80)
    
    # Test with N pequeño (like Katyusha)
    print("\n[Test] N=4 case (small sample)...\n")
    
    consistency_result = compute_consistency_indicators(
        kfold_qualities=[1.0, 1.0, 1.0, 1.0],  # Perfect consistency
        residual_stats={'quality_score': 0.85, 'relative_error': 0.05},
        complexity_stats={'complexity_ratio': 0.92, 'is_trivial': False},
        reconstruction_quality=1.0
    )
    
    interpretation = interpret_consistency(consistency_result, n_samples=4)
    
    report = consistency_report(consistency_result, interpretation)
    
    print("\n" + "="*80)
    print(report)
    print("="*80)
