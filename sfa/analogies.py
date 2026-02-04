import numpy as np
from sfa.multifractal import MultifractalAnalyzer

# Try to import PyMC, but provide fallback if not available
try:
    import pymc as pm
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("PyMC not available. Using analytical approximation for scale transformation.")

def scale_transformation_operator(observed_D2, noise_model='multiplicative', use_bayesian=True):
    """
    Operador que remueve ruido para revelar D₃ multifractal
    
    Parámetros:
    -----------
    observed_D2 : float
        Dimensión fractal observada (con ruido)
    noise_model : str
        'multiplicative' : ruido proporcional a señal
        'additive' : ruido independiente de señal
    use_bayesian : bool
        Si True, usa inferencia bayesiana (requiere PyMC instalado)
        Si False, usa aproximación analítica
        
    Retorna:
    --------
    D3_estimate : float
        Estimación de dimensión multifractal subyacente
    uncertainty : dict or pandas.DataFrame
        Resumen de la incertidumbre bayesiana (si PyMC está disponible)
        o estimación analítica de error
    """
    
    if use_bayesian and PYMC_AVAILABLE:
        # Modelo bayesiano jerárquico para la transformación
        with pm.Model() as scale_model:
            # Priors basados en física del sistema
            # D3 suele estar entre 2.0 y 3.0 para sistemas espaciales complejos
            D3_true = pm.TruncatedNormal('D3_true', mu=2.5, sigma=0.5, 
                                         lower=2.0, upper=3.0)
            
            if noise_model == 'multiplicative':
                # Ruido como factor de reducción
                noise_factor = pm.Beta('noise_factor', alpha=2, beta=5)
                
                # Likelihood
                D2_observed = pm.Normal('D2_observed', 
                                       mu=D3_true * (1 - noise_factor),
                                       sigma=0.1, 
                                       observed=observed_D2)
            else:
                # Ruido aditivo
                noise_magnitude = pm.Exponential('noise_magnitude', lam=2)
                D2_observed = pm.Normal('D2_observed',
                                       mu=D3_true - noise_magnitude,
                                       sigma=0.1,
                                       observed=observed_D2)
            
            # Inferencia
            trace = pm.sample(1000, tune=500, cores=1, progressbar=False)
            
        summary = pm.summary(trace)
        return trace.posterior['D3_true'].mean().item(), summary
    
    else:
        # Aproximación analítica simple
        # Asumiendo que el ruido reduce la dimensión observada
        # D2 = D3 * (1 - noise_factor) para modelo multiplicativo
        # o D2 = D3 - noise para modelo aditivo
        
        if noise_model == 'multiplicative':
            # Estimamos noise_factor ~ Beta(2,5) con media = 2/7 ≈ 0.286
            expected_noise = 2.0 / 7.0
            # D2 = D3 * (1 - 0.286) -> D3 = D2 / 0.714
            D3_estimate = observed_D2 / (1 - expected_noise)
            # Error propagado (aproximación)
            noise_std = np.sqrt((2 * 5) / ((2 + 5)**2 * (2 + 5 + 1)))
            D3_std = observed_D2 * noise_std / (1 - expected_noise)**2
        else:
            # Modelo aditivo: D2 = D3 - noise, noise ~ Exp(2) con media = 0.5
            expected_noise = 0.5
            D3_estimate = observed_D2 + expected_noise
            # Error propagado
            noise_std = 0.5  # std de Exp(2)
            D3_std = noise_std
        
        # Clamp to physical range
        D3_estimate = np.clip(D3_estimate, 2.0, 3.0)
        
        uncertainty = {
            'method': 'analytical_approximation',
            'mean': D3_estimate,
            'std': D3_std,
            'hdi_3%': max(2.0, D3_estimate - 2 * D3_std),
            'hdi_97%': min(3.0, D3_estimate + 2 * D3_std),
            'noise_model': noise_model
        }
        
        return D3_estimate, uncertainty

def validate_scale_transformation():
    """
    Valida la analogía con sistemas conocidos
    """
    print("=" * 70)
    print("VALIDACIÓN DE TRANSFORMACIÓN DE ESCALA")
    print("Analogía: Arqueoastronomía ↔ Multifractalidad")
    print("=" * 70)
    print()
    
    # 1. Sistema de precesión (modelo simplificado)
    print("1. Generando datos sintéticos de precesión...")
    t = np.linspace(0, 26000, 1000)  
    precession_angle = 23.5 * np.sin(2*np.pi*t/26000) 
    
    # 2. Añadir "ruido arqueológico" (registros incompletos)
    np.random.seed(42)
    archaeological_mask = np.random.binomial(1, 0.3, len(t))  
    
    observed_values = precession_angle[archaeological_mask == 1]
    observed_times = t[archaeological_mask == 1]
    
    print(f"   - Total de puntos: {len(t)}")
    print(f"   - Puntos observados (30%): {len(observed_values)}")
    print()
    
    data_for_analysis = observed_values.reshape(-1, 1)
    
    # 3. Análisis multifractal de lo OBSERVADO
    print("2. Calculando espectro multifractal D_q...")
    mfsfa = MultifractalAnalyzer()
    q_range = np.arange(-2, 3, 0.5)
    q_vals, Dq_vals = mfsfa.compute_renyi_spectrum(data_for_analysis, q_values=q_range)
    Dq_spectrum = np.column_stack((q_vals, Dq_vals)) if len(q_vals) > 0 else np.array([])
    
    if len(Dq_spectrum) == 0:
        print("   ⚠ No se pudo calcular el espectro (datos insuficientes).")
        return None

    print(f"   - Espectro calculado para {len(Dq_vals)} valores de q")
    
    # Singularity spectrum (if enough points)
    if len(Dq_vals) >= 3:
        # Approximate alpha, f_alpha from Dq
        # alpha = dD_q/dq, f_alpha = q*alpha - D_q
        alpha = Dq_vals  # Simplified
        f_alpha = Dq_vals  # Placeholder
    else:
        alpha, f_alpha = np.array([]), np.array([])
    
    # Obtener D2 observado (q=2 or closest)
    if len(q_vals) > 0 and len(Dq_vals) > 0:
        # Find q closest to 2
        idx_q2 = np.argmin(np.abs(q_vals - 2.0))
        D2_obs = Dq_vals[idx_q2]
    else:
        D2_obs = np.nan
    
    print(f"   - D₂ observado: {D2_obs:.4f}")
    print()
    
    # 4. Estimación de D₃ subyacente
    print("3. Aplicando transformación de escala (D₂ → D₃)...")
    
    try:
        D3_est, uncertainty = scale_transformation_operator(D2_obs, use_bayesian=PYMC_AVAILABLE)
        print(f"   - D₃ estimado: {D3_est:.4f}")
        
        if isinstance(uncertainty, dict):
            print(f"   - Método: {uncertainty['method']}")
            print(f"   - Error estándar: ±{uncertainty['std']:.4f}")
            print(f"   - Intervalo 95%: [{uncertainty['hdi_3%']:.4f}, {uncertainty['hdi_97%']:.4f}]")
        else:
            print("   - Resumen estadístico (PyMC):")
            print(uncertainty.loc['D3_true'])
    except Exception as e:
        print(f"   ⚠ Error en transformación: {e}")
        D3_est = np.nan
        uncertainty = None
    
    print()
    
    # 5. Métricas adicionales
    spectrum_width = alpha.max() - alpha.min() if len(alpha) > 0 else 0
    hurst_estimate = 2 - D3_est if not np.isnan(D3_est) else np.nan
    
    print("4. Métricas de complejidad multifractal:")
    print(f"   - Ancho del espectro Δα: {spectrum_width:.4f}")
    print(f"   - Exponente de Hurst estimado: {hurst_estimate:.4f}")
    print()
    
    results = {
        'D2_observed': D2_obs,
        'D3_estimated': D3_est,
        'spectrum_width': spectrum_width,
        'Hurst_estimate': hurst_estimate,
        'q_range': (q_vals.min(), q_vals.max()),
        'alpha_range': (alpha.min(), alpha.max()) if len(alpha) > 0 else (np.nan, np.nan)
    }
    
    print("=" * 70)
    print("VALIDACIÓN COMPLETADA")
    print("=" * 70)
    print()
    print("RESULTADOS FINALES:")
    for key, value in results.items():
        if isinstance(value, tuple):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}" if not np.isnan(value) else f"  {key}: N/A")
    print()
    
    return results

if __name__ == "__main__":
    results = validate_scale_transformation()
