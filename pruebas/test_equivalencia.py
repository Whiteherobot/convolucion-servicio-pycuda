"""PyCUDA directo contra el servicio, mismo procesar. Bloque 2 - Jordy Romero.

Se compara el .npy (float32 sin recortar) con lo que da PyCUDA en este proceso.
"""
from pathlib import Path

import pytest

from app.modelo import procesamiento
from app.modelo.validacion import TOLERANCIA, errores
from conftest import a_png, descargar_crudo, enviar, imagen_aleatoria

FILTROS = [("media", 9), ("nitidez", 3), ("laplaciano", 3)]
IMAGEN_2000 = Path(__file__).resolve().parents[1] / "datos" / "imagen_2000.pgm"


def comparar(datos: bytes, filtro: str, k: int, gpu) -> None:
    respuesta = enviar(datos, filtro, k)
    assert respuesta.status_code == 200
    por_servicio = descargar_crudo(respuesta.json())

    imagen = procesamiento.decodificar(datos)
    directo, _ = procesamiento.procesar(gpu, imagen, filtro, k)
    maximo, medio = errores(directo, por_servicio)
    assert maximo < TOLERANCIA
    assert medio < TOLERANCIA


@pytest.mark.parametrize("filtro,k", FILTROS)
def test_directo_igual_a_servicio(gpu, filtro, k):
    comparar(a_png(imagen_aleatoria(600, 400)), filtro, k, gpu)


@pytest.mark.skipif(not IMAGEN_2000.exists(), reason="falta datos/imagen_2000.pgm (ver datos/README.md)")
@pytest.mark.parametrize("filtro,k", FILTROS)
def test_directo_igual_a_servicio_con_pgm_real(gpu, filtro, k):
    comparar(IMAGEN_2000.read_bytes(), filtro, k, gpu)
