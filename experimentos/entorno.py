"""Verifica y anota el entorno antes de medir. Bloque 3 - Andrés Morocho.

Uso (solo verificar, sin medir):
    python experimentos/entorno.py --url http://localhost:8000 --imagen convolucion-servicio:p03 --contenedor conv-p03

Problemas (bloquean la medición): faltan las imágenes o su hash no coincide, el servicio no responde o no
tiene GPU. Avisos (se anotan pero no bloquean): cambios sin commit, docker o nvidia-smi no disponibles.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cliente  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent


def comando(*args, tiempo_maximo: float = 120) -> str | None:
    """Ejecuta un comando y devuelve su salida, o None si no existe o falla."""
    if shutil.which(args[0]) is None:
        return None
    try:
        salida = subprocess.run(args, capture_output=True, text=True, timeout=tiempo_maximo, cwd=RAIZ)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return salida.stdout.strip() if salida.returncode == 0 else None


def verificar_datos(problemas: list) -> dict:
    """Comprueba los .pgm contra datos/SHA256SUMS."""
    hashes = {}
    for linea in (RAIZ / "datos" / "SHA256SUMS").read_text().splitlines():
        esperado, nombre = linea.split()
        ruta = RAIZ / nombre
        if not ruta.is_file():
            problemas.append(f"Falta {nombre}: generarlo según datos/README.md")
            continue
        real = hashlib.sha256(ruta.read_bytes()).hexdigest()
        if real != esperado:
            problemas.append(f"El hash de {nombre} no coincide con datos/SHA256SUMS")
        hashes[nombre] = real
    return hashes


def info_anfitrion() -> dict:
    cpu, ram = platform.processor() or None, None
    if Path("/proc/cpuinfo").is_file():
        m = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text())
        cpu = m.group(1).strip() if m else cpu
        m = re.search(r"MemTotal:\s*(\d+)", Path("/proc/meminfo").read_text())
        ram = f"{int(m.group(1)) / 1024 ** 2:.1f} GiB" if m else None
    try:
        pycuda = importlib.metadata.version("pycuda")
    except importlib.metadata.PackageNotFoundError:
        pycuda = None
    nvcc = comando("nvcc", "--version")
    return {
        "so": platform.platform(),
        "cpu": cpu,
        "ram": ram,
        "gpu": comando("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"),
        "controlador_nvidia": comando("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        "nvcc_anfitrion": _release(nvcc),
        "python_anfitrion": platform.python_version(),
        "pycuda_anfitrion": pycuda,
        "docker": comando("docker", "--version"),
        "nvidia_container_toolkit": (comando("nvidia-ctk", "--version") or "").split("\n")[0] or None,
    }


def info_imagen(imagen: str, avisos: list) -> dict:
    """Versiones dentro de la imagen (CUDA de la imagen, no la del anfitrión)."""
    identificador = comando("docker", "image", "inspect", "--format", "{{.Id}}", imagen)
    if identificador is None:
        avisos.append(f"No se pudo inspeccionar la imagen {imagen}: no habrá modo directo_contenedor")
        return {"nombre": imagen, "id": None}
    pip = comando("docker", "run", "--rm", "--entrypoint", "pip", imagen, "show", "pycuda") or ""
    return {
        "nombre": imagen,
        "id": identificador,
        "nvcc": _release(comando("docker", "run", "--rm", "--entrypoint", "nvcc", imagen, "--version")),
        "python": comando("docker", "run", "--rm", "--entrypoint", "python3", imagen, "--version"),
        "pycuda": next((l.split(": ", 1)[1] for l in pip.splitlines() if l.startswith("Version:")), None),
    }


def info_git(avisos: list) -> dict:
    commit = comando("git", "rev-parse", "HEAD")
    cambios = comando("git", "status", "--porcelain", "--untracked-files=no")
    if cambios:
        avisos.append("Hay cambios sin commit: el commit anotado no describe exactamente el código medido")
    return {"commit": commit, "rama": comando("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "cambios_sin_commit": bool(cambios)}


def recolectar(url: str, imagen: str | None, contenedor: str | None) -> tuple[dict, list, list]:
    """Devuelve (entorno, problemas, avisos)."""
    problemas, avisos = [], []
    hashes = verificar_datos(problemas)

    try:
        estado = cliente.salud(url)
        if estado.get("gpu") is None:
            problemas.append(f"El servicio no tiene GPU: {estado.get('error')}")
    except Exception as error:
        estado = None
        problemas.append(f"El servicio no responde en {url}: {error}")

    if contenedor and comando("docker", "inspect", "--format", "{{.State.Running}}", contenedor) != "true":
        avisos.append(f"El contenedor {contenedor} no está corriendo: no se podrá medir el arranque en frío")

    anfitrion = info_anfitrion()
    host = urlparse(url).hostname
    entorno = {
        "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
        "url_servicio": url,
        "cliente": ("mismo equipo que el servidor (localhost)" if host in ("localhost", "127.0.0.1")
                    else f"otro equipo de la red hacia {host}"),
        "anfitrion": anfitrion,
        "imagen": info_imagen(imagen, avisos) if imagen else None,
        "contenedor": contenedor,
        "git": info_git(avisos),
        "salud": estado,
        "datos_sha256": hashes,
    }
    imagen_nvcc = (entorno["imagen"] or {}).get("nvcc")
    entorno["etiqueta"] = " | ".join(str(x) for x in (
        (estado or {}).get("gpu") or anfitrion["gpu"],
        f"controlador {anfitrion['controlador_nvidia']}",
        f"CUDA imagen {imagen_nvcc}",
        f"commit {(entorno['git']['commit'] or '?')[:7]}",
    ))
    return entorno, problemas, avisos


def _release(texto: str | None) -> str | None:
    m = re.search(r"release ([\d.]+), V([\d.]+)", texto or "")
    return m.group(2) if m else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--imagen", default="convolucion-servicio:p03")
    p.add_argument("--contenedor", default="conv-p03")
    args = p.parse_args()
    entorno, problemas, avisos = recolectar(args.url, args.imagen, args.contenedor)
    print(json.dumps(entorno, indent=2, ensure_ascii=False))
    for a in avisos:
        print("AVISO:", a)
    for pr in problemas:
        print("PROBLEMA:", pr)
    print("Listo para medir." if not problemas else "NO se puede medir todavía.")
    sys.exit(1 if problemas else 0)


if __name__ == "__main__":
    main()
