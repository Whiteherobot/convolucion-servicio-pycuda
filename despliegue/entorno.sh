#!/usr/bin/env bash
# Anota el entorno del equipo y de la imagen para el informe. Bloque 2 - Jordy Romero.
# --entrypoint evita el banner de NVIDIA que ensucia la salida.
# Uso: despliegue/entorno.sh [imagen]   (por defecto convolucion-servicio:p03)
imagen="${1:-convolucion-servicio:p03}"

echo "## Anfitrión"
echo "CPU:        $(lscpu | sed -n 's/^Model name: *//p')"
echo "RAM:        $(free -h | awk '/^Mem:/ {print $2}')"
echo "SO:         $(. /etc/os-release && echo "$PRETTY_NAME"), kernel $(uname -r)"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Controlador NVIDIA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
echo "Docker:     $(docker --version)"
echo "NVIDIA Container Toolkit: $(nvidia-ctk --version | head -1)"

echo
echo "## Imagen $imagen (CUDA de la imagen, no del anfitrión)"
echo "ID y digest: $(docker images --digests --format '{{.ID}} {{.Digest}}' "$imagen")"
echo "nvcc:       $(docker run --rm --entrypoint nvcc "$imagen" --version | grep release)"
echo "Python:     $(docker run --rm --entrypoint python3 "$imagen" --version)"
echo "PyCUDA:     $(docker run --rm --entrypoint pip "$imagen" show pycuda 2>/dev/null | sed -n 's/^Version: //p')"
echo "Commit:     $(git rev-parse --short HEAD)"
