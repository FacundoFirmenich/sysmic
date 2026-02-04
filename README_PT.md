# 🌋 SYSMIC: Framework de Análise Geofísica

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-green.svg)](https://hub.docker.com/)
[![Open Science](https://badges.frapsoft.com/os/v1/open-science.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)

> **"Revelando a arquitetura oculta dos sistemas tectônicos."**

**Sysmic** (anteriormente FractalSystemPro) é um framework computacional de ponta ("state-of-the-art") para análise sísmica avançada. Integra geometria fractal, espectros multifractais, dinâmica temporal e inferência bayesiana para fornecer uma compreensão holística da física de terremotos.

---

## 🌟 Recursos Principais

### 🔬 Núcleo Científico ("Scientific Core")
- **Dimensão Fractal (D₂)**: Estimativa robusta de Grassberger-Procaccia com incerteza bootstrap.
- **Espectro Multifractal (Dq)**: Análise completa do espectro de singularidade $f(\alpha)$.
- **Dinâmica Temporal**: Evolução $D(t)$ e detecção de desaceleração crítica ("critical slowing down").
- **Validação Bayesiana**: Avaliação probabilística de estimativas dimensionais.
- **Análise Topológica**: Detecção de comunidades baseada em grafos em redes sísmicas.

### 💻 Interfaces Modernas
- **CLI Interativo**: Interface de terminal profissional com visualização ASCII art.
- **Painel Web**: GUI baseada em Streamlit para fácil exploração de dados.
- **Estúdio na Nuvem**: Notebook unificado do Google Colab para execução instantânea.
- **Conteinerizado**: Suporte total ao Docker para implantação reprodutível.

---

## 🚀 Início Rápido

### 1. Instalação

```bash
git clone https://github.com/SeuUsuario/Sysmic.git
cd Sysmic
pip install -r requirements.txt
```

### 2. Modo Interativo (CLI)

Acesse a interface de linha de comando profissional:

```bash
python -m sysmic.interactive
```

### 3. Painel Web (Streamlit)

Inicie o servidor web local:

```bash
streamlit run streamlit_app.py
```

### 4. Implantação Docker

Execute o ambiente conteinerizado:

```bash
docker build -t sysmic .
docker run -it sysmic
```

---

## 📊 Metodologia Científica

O Sysmic implementa um rigoroso **Protocolo de Validação Tripla**:

1.  **Consistência Geométrica**: A largura do espectro $D_q$ ($\Delta\alpha$) deve exceder o piso de ruído.
2.  **Estabilidade Temporal**: $\Delta D_2 / \Delta t$ deve permanecer limitado durante a quiescência.
3.  **Significância Estatística**: Testes com dados substitutos (embaralhamento aleatório) com $p < 0.05$.

Para formulações matemáticas detalhadas, veja [MATHEMATICAL_APPENDIX.md](MATHEMATICAL_APPENDIX.md).

---

## 🌍 Manifesto de Ciência Aberta

Acreditamos que o conhecimento trancado não é conhecimento, é um segredo.
O Sysmic é construído sobre três pilares:

1.  **Transparência**: Cada algoritmo está aberto para inspeção.
2.  **Reprodutibilidade**: Ambientes conteinerizados garantem resultados consistentes.
3.  **Acessibilidade**: Ferramentas projetadas para todos, de estudantes a pesquisadores de alto nível.

---

## 🤝 Contribuindo

Congratulamo-nos com contribuições! Leia [CONTRIBUTING.md](CONTRIBUTING.md) para detalhes sobre nosso código de conduta e processo de envio.

---

## 📜 Licença

Este projeto está licenciado sob a **GNU General Public License v3.0**.
Veja [LICENSE](LICENSE) para detalhes.

**Fontes de Dados:**
1. **United States Geological Survey (USGS)**
2. **National Research Institute for Earth Science and Disaster Resilience (NIED)**, Japão (Hi-net/F-net High-Sensitivity Seismograph Network). Acesso fornecido sob protocolos de pesquisa registrados.
3. **International Seismological Centre (ISC)**: ISC-GEM Global Instrumental Earthquake Catalogue (Ver. 11.0).
