"""Imagen 1001x777, que no es múltiplo de 128/256/512 hilos. Bloque 2 - Jordy Romero."""
import math

import pytest

from app.modelo import mascaras, procesamiento
from app.modelo.validacion import TOLERANCIA, errores
from conftest import a_png, descargar_crudo, enviar, imagen_aleatoria

ANCHO, ALTO = 1001, 777


@pytest.mark.parametrize("hilos", [128, 256, 512])
def test_cobertura_total_con_cualquier_numero_de_hilos(gpu, hilos):
    entrada = imagen_aleatoria(ANCHO, ALTO)
    respuesta = enviar(a_png(entrada), "laplaciano", 3, hilos=str(hilos))
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert (datos["ancho"], datos["alto"]) == (ANCHO, ALTO)
    assert datos["bloques"] == math.ceil(ANCHO * ALTO / hilos)

    esperado, _ = gpu.ejecutar(entrada, mascaras.crear("laplaciano"), hilos)
    salida = descargar_crudo(datos)
    maximo, _ = errores(esperado, salida)
    assert maximo < TOLERANCIA
    # El último píxel es el que cubre el bloque parcial.
    assert salida[-1, -1] == pytest.approx(esperado[-1, -1], abs=TOLERANCIA)
