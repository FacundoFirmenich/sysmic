---
title: 'Sysmic: Un Marco de Inteligencia de Código Abierto para Tomografía Fractal de Alta Precisión y Monitoreo en Tiempo Real'
tags:
  - Python
  - sismología
  - dimensión fractal
  - inferencia bayesiana
  - monitoreo en tiempo real
  - aprendizaje automático
  - análisis topológico de datos
  - hi-net
authors:
  - name: Facundo Firmenich
    orcid: 0009-0002-6578-3811
    affiliation: 1
  - name: Pau Firmenich
    affiliation: 1
  - name: León Firmenich
    affiliation: 1
affiliations:
  - name: Centro de Estudios del Sur (CEDESUR)
    index: 1
date: 04 de Febrero de 2026
bibliography: paper.bib
---

# Resumen

Comprender la compleja organización espacio-temporal de la sismicidad es crítico tanto para la geofísica fundamental como para la evaluación de riesgos. **Sysmic** (anteriormente SFA) es un marco computacional integral de código abierto que trasciende la estimación fractal tradicional. Integra **Tomografía Fractal** de alta precisión, **Inferencia Bayesiana**, **Análisis Topológico de Datos (TDA)** y **Monitoreo en Tiempo Real** en una plataforma de inteligencia unificada.

Diseñado para manejar los conjuntos de datos masivos de redes modernas de alta sensibilidad (ej. NIED Hi-Net), Sysmic resuelve la paradoja de proyección geométrica ("geometric projection paradox") de larga data en sismología al proporcionar herramientas calibradas con precisión para distinguir estructuras de falla genuinas de artefactos observacionales. Más allá del análisis retrospectivo, el marco incluye los motores **Gravitas** y **Nexus**, permitiendo modelado predictivo e inteligencia de investigación automatizada.

# Declaración de Necesidad

La sismología cuantitativa enfrenta un "diluvio de datos" donde los métodos estadísticos tradicionales, agnósticos al catálogo, a menudo fallan en extraer señales significativas de conjuntos de datos dominados por ruido. Los investigadores carecen específicamente de:
1.  **Robustez contra artefactos de precisión:** Los algoritmos estándar a menudo arrojan dimensiones "volumétricas" espurias ($D \approx 3$) simplemente debido a la incertidumbre de localización ($\sigma$).
2.  **Análisis multiparadigma integrado:** Las herramientas suelen estar aisladas: el análisis fractal, el agrupamiento topológico y la inferencia bayesiana rara vez coexisten en un solo flujo de trabajo.
3.  **Capacidad de tiempo real:** La mayoría de los códigos fractales están diseñados para el análisis estático y retrospectivo de catálogos, no para flujos de monitoreo continuo.

Sysmic aborda estas necesidades proporcionando un ecosistema modular que trata la calidad de los datos ($\sigma$, $M_c$, Red $Q_{eff}$) como un ciudadano de primera clase en el proceso de inferencia. Originalmente desarrollado para apoyar los avances en "Tomografía Fractal y la Planarización Dependiente de la Profundidad de la Sismicidad" (Firmenich et al., 2026), ha evolucionado hacia un motor de propósito general para el análisis sísmico avanzado.

# Arquitectura y Módulos Clave

El marco Sysmic está estructurado en módulos de alto rendimiento diseñados para escalabilidad y extensibilidad:

### 1. Motores Centrales (`sysmic.core`, `sysmic.bayesian`)
*   **Estimador Fractal Calibrado por Precisión:** Implementa un algoritmo Grassberger-Procaccia optimizado ($O(N \log N)$ vía k-d trees) capaz de procesar >200,000 eventos en segundos.
*   **Detección de Barrera de Información de Fisher:** Calcula automáticamente el umbral de precisión crítico $\sigma_c$ donde la inferencia se vuelve dominada por el prior.
*   **Suite de Inferencia Bayesiana:** Estimación basada en MCMC de la dimensionalidad latente ($D_3$) utilizando muestreadores `emcee` y `dynesty`, incorporando "Triple-Validación" (Divergencia KL, Concentración de Frontera, Invarianza de Escala).

### 2. Gravitas: Inteligencia en Tiempo Real (`sysmic.gravitas`)
*   **Monitoreo Continuo:** Un motor dedicado para el cálculo en tiempo real de la evolución de la dimensión fractal y el valor $b$, capaz de ingerir datos en streaming.
*   **Integración de RL Adaptativo:** Soporte experimental para Aprendizaje por Refuerzo (Thompson Sampling) para optimizar dinámicamente los parámetros de la ventana de monitoreo.
*   **Analítica Predictiva:** Implementa algoritmos de detección de precursores basados en fluctuaciones de dimensión fractal ($\Delta D_2$) y medidas de entropía.

### 3. Nexus: Automatización de Investigación (`sysmic.nexus`)
*   **Generación Automatizada de Insights:** Algoritmos para escanear vastos espacios de parámetros e identificar anomalías estadísticamente significativas sin intervención manual.
*   **Inteligencia de Proyecto:** Una capa de meta-análisis que integra resultados a través de diferentes regiones tectónicas y escalas, facilitando estudios comparativos globales.

### 4. Interoperabilidad Avanzada (`sysmic.integration`)
*   **Red Sísmica Global (WSG):** Conectores para formatos estándar (ObsPy, QuakeML) y catálogos principales (USGS, ISC-GEM, JMA).
*   **Preparado para Sismología Planetaria:** Módulos diseñados para la ingesta de datos planetarios (ej. InSight Mars, futura misión Dragonfly), aprovechando la robustez del marco ante datos dispersos/ruidosos.
*   **Análisis Topológico:** Integración con bibliotecas de teoría de grafos para calcular modularidad de red y brechas de dimensión topológica ($\Delta D = D_2 - D_{graph}$).

### 5. Interfaces Interactivas
*   **CLI y TUI:** Interfaz de línea de comandos profesional con marca en arte ASCII para despliegue rápido.
*   **Panel Web:** Interfaz gráfica basada en Streamlit para exploración de datos y visualización de resultados.
*   **Cloud Studios:** Notebooks de Colab optimizados para análisis nativo en la nube.

# Recursos Pedagógicos

Para democratizar la geofísica avanzada, Sysmic incluye una biblioteca `resources/multimedia` que contiene:
*   **Videos Didácticos:** visualizaciones de geometría terrestre 8-bit vs. 4K, ilusiones sísmicas y arquitectura fractal.
*   **Documentos Técnicos:** reportes PDF sobre resolución fractal, geometría de alta resolución y tomografía sísmica.
*   **Soporte Multilingüe:** Materiales disponibles en Inglés, Español y Portugués para apoyar la educación global.

# Eficiencia Computacional y la Democratización de la Ciencia

Una filosofía de diseño central de Sysmic es la eliminación de la barrera específica entre la capacidad científica de alto impacto y el acceso a infraestructura de supercomputación. Para demostrar empíricamente esto, todo el desarrollo, las pruebas y la ejecución de este marco—incluyendo el procesamiento de >190,000 eventos Hi-Net, inferencia Bayesiana MCMC y simulaciones Gravitas en tiempo real—se restringió estrictamente a una **laptop de consumo de rango medio de 2016** (Intel Core i5-6200U, 8GB DDR3 RAM, GPU dedicada de 128MB).

Esta restricción deliberada de hardware sirve como validación de la optimización del marco (ej. indexación espacial k-d tree, broadcasting vectorizado) y desafía la suposición de que la sismología de "Big Data" requiere recursos computacionales masivos. Sysmic prueba que con algoritmos optimizados, el descubrimiento científico de primer nivel es accesible en hardware heredado ubicuo.

# Matemáticas y Validación

Sysmic va más allá de la estimación de caja negra implementando salvaguardas matemáticas rigurosas:

**La Barrera de Información de Fisher:**
$$ \sigma_c = -\frac{\lambda}{2} \log\left( \frac{k \cdot \mathcal{I}_{prior} \cdot \sigma_{D_2}^2 \cdot N_{pairs}}{Q_{eff} \cdot N_{eff}} \right) $$
Este criterio derivado previene el artefacto de "Saturación Bayesiana", asegurando que las salidas reflejen la realidad física en lugar de suposiciones a priori.

**Corrección de Grafo Espectral:**
$$ D_{graph} \approx 2 + \frac{\log N_{planes}}{\log(L/\ell)} - \beta_{topo} \lambda_{gap} $$
Permite la reconciliación de la dimensionalidad geométrica y topológica.

# Casos de Uso de Ejemplo

1.  **Tomografía de Alta Precisión:** Resolviendo la estructura multi-planar "baraja de cartas" de la zona de subducción de Japón (validación Hi-Net).
2.  **Clasificación Tectónica Global:** Discriminación de regímenes tectónicos (Rifting vs. Colisión) basada en sus firmas fractales específicas.
3.  **Nowcasting Sísmico:** Monitoreo de caídas de $D_2$ en tiempo real como un proxy potencial para la acumulación de estrés (motor Gravitas).

# Disponibilidad y Ciencia Abierta

Sysmic es de código abierto (GPLv3) y está diseñado para la era de la Ciencia Abierta. Incluye pruebas unitarias exhaustivas, conjuntos de datos de validación sintética (ruido isotrópico vs. anisotrópico) y soporte Docker para entornos reproducibles. Este marco tiene como objetivo democratizar el acceso a herramientas de sismología estadística de alta gama, cerrando la brecha entre la complejidad teórica y la utilidad operativa.

# Agradecimientos

Esta investigación se basa principalmente en las políticas de datos abiertos del **United States Geological Survey (USGS)**, cuyo acceso irrestricto constituye la columna vertebral de nuestro análisis. Extendemos también nuestra gratitud al **International Seismological Centre (ISC)** por la rápida provisión del **ISC-GEM Global Instrumental Earthquake Catalogue (Ver. 11.0)**, disponible mediante credenciales bajo demanda. Finalmente, agradecemos explícitamente al **National Research Institute for Earth Science and Disaster Resilience (NIED)**, Japón, por otorgar acceso a los datos de la red de sismógrafos de alta sensibilidad (Hi-net/F-net) tras un riguroso proceso de aprobación para investigadores registrados; estos datos de acceso restringido fueron cruciales para resolver estructuras profundas de subducción.

# Declaración de Desarrollo Asistido por IA y Reflexión Crítica

El desarrollo de este marco utilizó varios Grandes Modelos de Lenguaje (LLMs) para acelerar el prototipado y estructurar salidas intermedias. Los modelos utilizados, en orden descendente de intensidad y relevancia, fueron:

1.  **Antigravity (impulsado por Gemini 3 Pro)**
2.  **DeepSeek v3.2**
3.  **Claude 4.5 Sonnet**
4.  **Google NotebookLM**
5.  **Qwen 3 Max**

**Exclusión de ChatGPT 5.2:**
Adicionalmente, se evaluó **ChatGPT 5.2**. Este caso resulta particularmente remarcable no por su utilidad, sino por su desempeño decepcionante. A pesar de ser la única herramienta del conjunto respaldada por una suscripción de pago (nivel base, ~$20-25/mes), su persistente falta de fiabilidad obligó a su exclusión definitiva del pool operativo de inteligencias artificiales.

**Evaluación Crítica:**
Si bien estas herramientas proporcionaron una utilidad innegable en la maquetación de estructuras de código y la articulación de borradores iniciales, acelerando notablemente la producción, su contribución estuvo estrictamente limitada por deficiencias significativas. A pesar de instrucciones explícitas y severas con respecto a la integridad de los datos, los modelos exhibieron una tendencia persistente hacia la alucinación y la invención de datos ficticios. En consecuencia, resultaron conducentes solo bajo una supervisión constante y línea por línea de todo el código, traducciones y escritura académica. Además, contrariamente a las especificaciones de los proveedores, observamos ventanas de contexto efectivo estrechas y una degradación progresiva de la memoria a largo plazo dentro de las sesiones. Esta experiencia delinea una frontera clara entre el aporte neto *esperado* de los agentes de IA y el valor *genuino* que ofrecen actualmente, el cual sigue dependiendo de una verificación humana exhaustiva.

# Referencias
