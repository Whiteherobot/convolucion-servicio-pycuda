# Imagen del servicio de convolución. Bloque 2 - Jordy Romero
# Base -devel porque PyCUDA necesita nvcc; el digest fija la imagen exacta.
FROM nvidia/cuda:13.4.3-devel-ubuntu24.04@sha256:5626a7d0ecafe3e7c115a452f6b05f20f214f8509346196a7835b205dc5b7fd9

# PyCUDA se compila al instalarlo.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 exige un entorno virtual para pip.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

WORKDIR /servicio
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Sin privilegios de root.
RUN useradd --system --no-create-home servicio
USER servicio

EXPOSE 8000
# Sano solo con la GPU cargada: /api/salud da 200 antes, con "gpu": null.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python3 -c "import json, urllib.request as u; assert json.load(u.urlopen('http://localhost:8000/api/salud', timeout=4))['gpu']"

# Un solo proceso: el contexto CUDA vive en un hilo.
CMD ["gunicorn", "-w", "1", "--threads", "8", "-b", "0.0.0.0:8000", "app.servidor:crear_app()"]
