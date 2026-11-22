# Imagen del servicio de convolución. Bloque 2 - Jordy Romero
# Base -devel (no -runtime): PyCUDA llama a nvcc para compilar el kernel al arrancar.
# Etiqueta y digest: la etiqueta se puede mover, el digest no.
FROM nvidia/cuda:13.4.3-devel-ubuntu24.04@sha256:5626a7d0ecafe3e7c115a452f6b05f20f214f8509346196a7835b205dc5b7fd9

# build-essential y python3-dev: PyCUDA se compila al instalarlo con pip.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 no deja instalar con pip fuera de un entorno virtual.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

WORKDIR /servicio
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# El servicio no necesita ser root.
RUN useradd --system --no-create-home servicio
USER servicio

EXPOSE 8000
# Sano solo cuando la GPU ya cargó: /api/salud responde 200 desde antes, pero con "gpu": null.
# El primer arranque compila el kernel, por eso el periodo de gracia.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python3 -c "import json, urllib.request as u; assert json.load(u.urlopen('http://localhost:8000/api/salud', timeout=4))['gpu']"

# Un solo proceso: el contexto CUDA vive en un único hilo trabajador.
CMD ["gunicorn", "-w", "1", "--threads", "8", "-b", "0.0.0.0:8000", "app.servidor:crear_app()"]
