"""
SYSMIC WEB INTERFACE
Powered by Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from pathlib import Path

# Configuración de página
st.set_page_config(
    page_title="Sysmic Framework Pro",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar lógica central (si está instalada en el path)
try:
    import sysmic
    from sysmic.system import Sysmic
    from sysmic.infrastructure import SystemConfiguration, CertificationLevel
    HAS_CORE = True
except ImportError:
    st.error("Sysmic Core not found. Please install the package first.")
    HAS_CORE = False

# --- HEADER & SIDEBAR ---
with st.sidebar:
    st.title("🌋 SYSMIC")
    st.caption("Advanced Geophysical Framework")
    st.markdown("---")
    
    level = st.select_slider(
        "Certificación",
        options=["LEVEL_0", "LEVEL_1", "LEVEL_2", "LEVEL_3", "LEVEL_4"],
        value="LEVEL_1"
    )
    
    st.markdown("---")
    st.info(f"System Mode: {level}")

# --- MAIN CONTENT ---
st.title("Geophysical Analysis Dashboard")
st.markdown("### Professional Seismic Fractal Analysis")

if HAS_CORE:
    # 1. Upload
    uploaded_file = st.file_uploader("Upload Seismic Catalog (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            catalog = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(catalog)} events")
            
            with st.expander("Preview Data"):
                st.dataframe(catalog.head())
                
            # 2. Configuración
            col1, col2 = st.columns(2)
            with col1:
                run_fractal = st.checkbox("Fractal Dimension", value=True)
                run_temporal = st.checkbox("Temporal Analysis", value=True)
            with col2:
                run_multi = st.checkbox("Multifractal Spectrum", value=level != "LEVEL_0")
                run_valid = st.checkbox("Scientific Validation", value=level != "LEVEL_0")
            
            # 3. Ejecución
            if st.button("Run Sysmic Analysis", type="primary"):
                # Configurar sistema
                config = SystemConfiguration(
                    certification_level=CertificationLevel[level]
                )
                system = Sysmic(config)
                
                # Definir tipos
                types = []
                if run_fractal: types.append('fractal_dimension')
                if run_temporal: types.append('temporal')
                if run_multi: types.append('multifractal')
                if run_valid: types.append('validation')
                
                with st.spinner("Processing geophysical data..."):
                    start = time.time()
                    result = system.analyze_catalog(catalog, analysis_types=types)
                    duration = time.time() - start
                
                if result.success:
                    st.balloons()
                    st.success(f"Analysis Complete in {duration:.2f}s")
                    
                    # Resultados principales
                    st.markdown("### Key Metrics")
                    m1, m2, m3 = st.columns(3)
                    
                    if result.fractal_dimension:
                         cons = result.fractal_dimension.get('consensus', {})
                         d_val = cons.get('value', 0)
                         m1.metric("Fractal Dimension (D)", f"{d_val:.3f}")
                         
                         unc = cons.get('uncertainty', 0)
                         m2.metric("Uncertainty (σ)", f"{unc:.3f}")
                    
                    m3.metric("Events Analyzed", len(catalog))
                    
                    # Tabs para detalles
                    tab1, tab2, tab3 = st.tabs(["Temporal", "Multifractal", "Report"])
                    
                    with tab1:
                        if result.temporal_analysis:
                             st.json(result.temporal_analysis)
                        else:
                            st.info("No temporal analysis run")
                            
                    with tab2:
                        if result.multifractal_spectrum:
                             st.json(result.multifractal_spectrum)
                        else:
                            st.info("No multifractal analysis run")
                            
                    with tab3:
                        report = result.generate_report()
                        st.text_area("Full Report", report, height=400)
                        st.download_button("Download Report", report, file_name=f"{result.analysis_id}.txt")
                        
                else:
                    st.error("Analysis Failed")
                    for err in result.errors:
                        st.warning(err)
                        
        except Exception as e:
            st.error(f"Error reading file: {e}")
            
else:
    st.warning("Framework core offline.")
