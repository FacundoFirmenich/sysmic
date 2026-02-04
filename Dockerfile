# Base image optimizada para ciencia de datos
FROM python:3.11-slim

# Metadatos
LABEL maintainer="Sysmic Framework Team"
LABEL version="6.0.0"
LABEL description="Sysmic Geophysical Framework - Professional Edition"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements (si existiera, sino instalar directo)
# RUN pip install --no-cache-dir -r requirements.txt

# Instalar dependencias core
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scipy>=1.10.0 \
    matplotlib>=3.7.0 \
    scikit-learn>=1.2.0 \
    joblib

# Copiar código fuente
COPY . /app

# Usuario no root por seguridad
RUN useradd -m sysmicuser
USER sysmicuser

# Punto de entrada por defecto: CLI interactiva
CMD ["python", "-m", "sysmic.interactive"]
