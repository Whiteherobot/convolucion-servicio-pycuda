"""Herramientas comunes de las pruebas. Bloque 2 - Jordy Romero.

Las pruebas hablan con el servicio por HTTP (contenedor o gunicorn en el anfitrión):
SERVICIO_URL cambia la dirección, por defecto http://localhost:8000.
"""
import io
import os

import numpy as np
import pytest
import requests
from PIL import Image

URL = os.environ.get("SERVICIO_URL", "http://localhost:8000")
SEMILLA = 2026


def imagen_aleatoria(ancho: int, alto: int) -> np.ndarray:
    """Valores enteros 0..255; la semilla fija hace la prueba repetible."""
    generador = np.random.default_rng(SEMILLA)
    return generador.integers(0, 256, size=(alto, ancho)).astype(np.float32)


def a_png(matriz: np.ndarray) -> bytes:
    salida = io.BytesIO()
    Image.fromarray(matriz.astype(np.uint8), mode="L").save(salida, format="PNG")
    return salida.getvalue()


def enviar(datos: bytes, filtro: str = "media", mascara: int = 3, **extra):
    """POST /api/procesar con los campos del contrato."""
    campos = {"filtro": filtro, "mascara": str(mascara), **extra}
    return requests.post(URL + "/api/procesar", files={"imagen": ("imagen", datos)},
                         data=campos, timeout=120)


def descargar_crudo(respuesta_json: dict) -> np.ndarray:
    """Baja el resultado float32 sin recortar (.npy) de una solicitud correcta."""
    crudo = requests.get(URL + respuesta_json["url_crudo"], timeout=120)
    return np.load(io.BytesIO(crudo.content))


@pytest.fixture(scope="session", autouse=True)
def servicio():
    """Las pruebas se saltan si el servicio no está arriba o no tiene GPU."""
    try:
        salud = requests.get(URL + "/api/salud", timeout=5).json()
    except requests.ConnectionError:
        pytest.skip(f"No hay servicio en {URL}. Ver despliegue/README.md")
    if salud["gpu"] is None:
        pytest.skip(f"El servicio no tiene GPU: {salud['error']}")
    return salud


@pytest.fixture(scope="session")
def gpu():
    """PyCUDA directo en este proceso, para comparar contra lo que devuelve el servicio."""
    from app.modelo.pycuda_conv import ConvolucionPyCUDA
    return ConvolucionPyCUDA()
