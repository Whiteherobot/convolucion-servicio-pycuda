#!/usr/bin/env bash
# Todo en un comando, en la laptop de Jordy (Linux + GPU NVIDIA + Docker). Bloque 3 - Andrés Morocho.
# Uso, desde la raíz del repo:  bash experimentos/ejecutar_todo.sh
# Al final hace commit de resultados/<corrida>/ (con tu usuario de git); solo queda hacer `git push`.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGEN=convolucion-servicio:p03
CONTENEDOR=conv-p03
URL=http://localhost:8000
paso() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
falla() { printf '\n\033[1;31mERROR: %s\033[0m\nCopia esta salida y envíala tal cual.\n' "$*"; exit 1; }

paso "1/7 Comprobaciones previas"
command -v docker >/dev/null || falla "no está docker"
command -v nvidia-smi >/dev/null || falla "no está nvidia-smi (controlador NVIDIA)"
[ -d /opt/cuda/bin ] && export PATH=/opt/cuda/bin:$PATH        # nvcc del anfitrión (Arch)
command -v nvcc >/dev/null || falla "no está nvcc en el anfitrión (PyCUDA directo lo necesita)"
git status --short --untracked-files=no | grep -q . && falla "hay cambios sin commit; deja el repo igual al de GitHub (git stash)"
echo "Commit: $(git rev-parse --short HEAD) en $(git rev-parse --abbrev-ref HEAD)"

paso "2/7 Entorno de Python del anfitrión (.venv)"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
PY=.venv/bin/python

paso "3/7 Imágenes de prueba (datos/)"
if ! sha256sum -c --quiet datos/SHA256SUMS 2>/dev/null; then
    echo "Faltan o no coinciden: se generan desde la práctica 02 (datos/README.md)"
    rm -rf /tmp/p02 && git clone -q https://github.com/Whiteherobot/convolucion-pycuda.git /tmp/p02
    (cd /tmp/p02 && "$OLDPWD/$PY" practica01/scripts/preparar_imagenes.py practica01/datos/original.jpg)
    cp /tmp/p02/datos/imagen_*.pgm datos/
fi
sha256sum -c datos/SHA256SUMS || falla "las imágenes no coinciden con datos/SHA256SUMS"

paso "4/7 Imagen Docker (unos 4 minutos la primera vez)"
mkdir -p resultados
docker build -t "$IMAGEN" . 2>&1 | tee resultados/build.log || falla "docker build"
docker run --rm --gpus all "$IMAGEN" nvidia-smi 2>&1 | tee resultados/nvidia-smi_contenedor.txt \
    || falla "el contenedor no ve la GPU"

paso "5/7 Servicio"
docker rm -f "$CONTENEDOR" >/dev/null 2>&1 || true
docker run -d --name "$CONTENEDOR" --gpus all --memory=4g -p 8000:8000 "$IMAGEN" >/dev/null
for _ in $(seq 1 90); do
    curl -fs "$URL/api/salud" 2>/dev/null | grep -q '"gpu": *"' && break
    sleep 2
done
curl -fs "$URL/api/salud" | grep -q '"gpu": *"' || { docker logs --tail 40 "$CONTENEDOR"; falla "el servicio no cargó la GPU"; }
echo "Servicio listo: $(curl -fs "$URL/api/salud")"

paso "6/7 Pruebas del Bloque 2 (pytest contra el contenedor)"
PRUEBAS=resultados/pytest_$(date +%Y%m%d-%H%M%S).txt
$PY -m pytest -v -rxXs 2>&1 | tee "$PRUEBAS" || echo "(hay pruebas que fallan: quedan en $PRUEBAS; se sigue con las mediciones)"

paso "7/7 Experimentos (unos 15 minutos: no uses la laptop, cargador conectado, apps cerradas)"
$PY experimentos/correr.py --url "$URL" --imagen "$IMAGEN" --contenedor "$CONTENEDOR" --python-host "$PY" \
    || falla "experimentos/correr.py"

CORRIDA=$(ls -d resultados/2* | sort | tail -1)
# Evidencia que pide la guía: build, GPU en el anfitrión y en el contenedor, imagen, pruebas y registro.
cp "$PRUEBAS" "$CORRIDA/pytest.txt"
mv resultados/build.log resultados/nvidia-smi_contenedor.txt "$CORRIDA/"
nvidia-smi > "$CORRIDA/nvidia-smi_anfitrion.txt" 2>&1 || true
{ docker images --digests "$IMAGEN"; docker ps --filter name="$CONTENEDOR"; } > "$CORRIDA/docker.txt" 2>&1 || true
docker logs "$CONTENEDOR" > "$CORRIDA/servicio.log" 2>&1 || true
git add "$CORRIDA"
git commit -q -m "Bloque 3: resultados de la corrida $(basename "$CORRIDA") en la laptop de Jordy" \
    || falla "no se pudo hacer commit de $CORRIDA"
git log --oneline -1
paso "Listo. Ahora ejecuta:  git push"
