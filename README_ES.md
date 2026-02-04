# 🌋 SYSMIC: Framework de Análisis Geofísico

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-green.svg)](https://hub.docker.com/)
[![Open Science](https://badges.frapsoft.com/os/v1/open-science.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)

> **"Revelando la arquitectura oculta de los sistemas tectónicos."**

**Sysmic** (anteriormente FractalSystemPro) es un framework computacional de vanguardia para el análisis sísmico avanzado. Integra geometría fractal, espectros multifractales, dinámica temporal e inferencia bayesiana para proporcionar una comprensión holística de la física de los terremotos.

---

## 🌟 Características Clave

### 🔬 Núcleo Científico
- **Dimensión Fractal (D₂)**: Estimación robusta de Grassberger-Procaccia con incertidumbre por bootstrap.
- **Espectro Multifractal (Dq)**: Análisis completo del espectro de singularidades $f(\alpha)$.
- **Dinámica Temporal**: Evolución $D(t)$ y detección de desaceleración crítica ("critical slowing down").
- **Validación Bayesiana**: Evaluación probabilística de las estimaciones dimensionales.
- **Análisis Topológico**: Detección de comunidades basada en grafos en redes sísmicas.

### 💻 Interfaces Modernas
- **CLI Interactivo**: Interfaz de terminal profesional con visualización ASCII art.
- **Panel Web**: GUI potenciada por Streamlit para exploración de datos sencilla.
- **Estudio en la Nube**: Notebook unificado de Google Colab para ejecución instantánea.
- **Contenerizado**: Soporte total de Docker para despliegue reproducible.

---

## 🚀 Inicio Rápido

### 1. Instalación

```bash
git clone https://github.com/TuUsuario/Sysmic.git
cd Sysmic
pip install -r requirements.txt
```

### 2. Modo Interactivo (CLI)

Accede a la interfaz de línea de comandos profesional:

```bash
python -m sysmic.interactive
```

### 3. Panel Web (Streamlit)

Lanza el servidor web local:

```bash
streamlit run streamlit_app.py
```

### 4. Despliegue con Docker

Ejecuta el entorno contenerizado:

```bash
docker build -t sysmic .
docker run -it sysmic
```

---

## 📊 Metodología Científica

Sysmic implementa un riguroso **Protocolo de Validación Triple**:

1.  **Consistencia Geométrica**: El ancho del espectro $D_q$ ($\Delta\alpha$) debe superar el piso de ruido.
2.  **Estabilidad Temporal**: $\Delta D_2 / \Delta t$ debe permanecer acotado durante la quiescencia.
3.  **Significancia Estadística**: Pruebas con datos sustitutos (aleatorización) con $p < 0.05$.

Para formulaciones matemáticas detalladas, vea [MATHEMATICAL_APPENDIX.md](MATHEMATICAL_APPENDIX.md).

---

## 🌍 Manifiesto de Ciencia Abierta

Creemos que el conocimiento encerrado no es conocimiento, es un secreto.
Sysmic se basa en tres pilares:

1.  **Transparencia**: Cada algoritmo está abierto para inspección.
2.  **Reproducibilidad**: Los entornos contenerizados aseguran resultados consistentes.
3.  **Accesibilidad**: Herramientas diseñadas para todos, desde estudiantes hasta investigadores de primer nivel.

---

## 🤝 Contribuyendo

¡Damos la bienvenida a las contribuciones! Por favor lea [CONTRIBUTING.md](CONTRIBUTING.md) para detalles sobre nuestro código de conducta y el proceso de envío.

---

## 📜 Licencia

Este proyecto está licenciado bajo la **GNU General Public License v3.0**.
Vea [LICENSE](LICENSE) para más detalles.

**Fuentes de Datos:**
1. **United States Geological Survey (USGS)**
2. **National Research Institute for Earth Science and Disaster Resilience (NIED)**, Japón (Hi-net/F-net High-Sensitivity Seismograph Network). Acceso proporcionado bajo protocolos de investigación registrados.
