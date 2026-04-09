FROM python:3.11-slim

LABEL maintainer="Facundo Firmenich <f.firmenich@cedesur.org>"
LABEL description="Sysmic: Fractal Tomography of Seismicity"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
