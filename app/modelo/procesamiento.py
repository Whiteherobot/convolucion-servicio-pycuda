"""Procesamiento compartido directo y web. Bloque 1 - Michael Lata."""
from io import BytesIO

import numpy as np
from PIL import Image

from app.modelo import imagen, mascaras


def procesar(gpu, imagen: np.ndarray, filtro: str, k: int,
             hilos: int = 256):
    """Crea la máscara y ejecuta la convolución en GPU."""
    mascara = mascaras.crear(filtro, k)
    return gpu.ejecutar(imagen, mascara, hilos)


def decodificar(datos: bytes) -> np.ndarray:
    """Decodifica una imagen como escala de grises float32."""
    with Image.open(BytesIO(datos)) as original:
        gris = original.convert("L")
        return np.ascontiguousarray(np.asarray(gris, dtype=np.float32))


def codificar_png(resultado) -> bytes:
    """Convierte el resultado al rango visual y lo codifica como PNG."""
    visual = imagen.a_rango_visual(resultado)
    salida = BytesIO()
    Image.fromarray(visual, mode="L").save(salida, format="PNG")
    return salida.getvalue()
