"""Limpieza de resultados vencidos con varios hilos. Bloque 2 - Jordy Romero."""
import os
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.modelo.solicitudes import ColaGPU, VIDA_RESULTADO_S

ARCHIVOS = 1000
HILOS = 8


@pytest.mark.xfail(reason="Bloque 1: limpiar_viejos() no tolera que otro hilo borre el mismo archivo "
                   "(FileNotFoundError, que en el servicio es un 500)", strict=False)
def test_varios_hilos_pueden_limpiar_a_la_vez():
    carpeta = tempfile.mkdtemp()
    vencido = time.time() - VIDA_RESULTADO_S - 60
    for numero in range(ARCHIVOS):
        ruta = Path(carpeta) / f"{numero:032x}.npy"
        ruta.touch()
        os.utime(ruta, (vencido, vencido))

    cola_falsa = SimpleNamespace(directorio=carpeta)  # limpiar_viejos solo usa este atributo
    fallos = []

    def limpiar():
        try:
            ColaGPU.limpiar_viejos(cola_falsa)
        except Exception as error:
            fallos.append(error)

    hilos = [threading.Thread(target=limpiar) for _ in range(HILOS)]
    for hilo in hilos:
        hilo.start()
    for hilo in hilos:
        hilo.join()

    assert fallos == []
