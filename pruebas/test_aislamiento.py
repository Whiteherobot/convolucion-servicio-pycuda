"""Solicitudes simultáneas: cada una recibe su resultado; con la cola llena, 503. Bloque 2 - Jordy Romero."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

from app.modelo import mascaras
from app.modelo.validacion import TOLERANCIA, errores
from conftest import URL, a_png, descargar_crudo, enviar, imagen_aleatoria

IMAGEN_4000 = Path(__file__).resolve().parents[1] / "datos" / "imagen_4000.pgm"
SOLICITUDES_A_LA_VEZ = 12  # con 8 hilos en gunicorn: 1 en la GPU + 4 en cola + 3 rechazadas


def test_dos_solicitudes_a_la_vez_reciben_cada_una_su_resultado(gpu):
    # Tamaños y filtros distintos: si se cruzaran los resultados, la forma o los valores no coincidirían.
    solicitudes = [(imagen_aleatoria(600, 400), "laplaciano", 3),
                   (imagen_aleatoria(300, 700), "media", 9)]
    with ThreadPoolExecutor(2) as pool:
        respuestas = list(pool.map(lambda s: enviar(a_png(s[0]), s[1], s[2]), solicitudes))

    for (entrada, filtro, k), respuesta in zip(solicitudes, respuestas):
        assert respuesta.status_code == 200
        datos = respuesta.json()
        assert (datos["alto"], datos["ancho"]) == entrada.shape
        esperado, _ = gpu.ejecutar(entrada, mascaras.crear(filtro, k))
        maximo, _ = errores(esperado, descargar_crudo(datos))
        assert maximo < TOLERANCIA

    assert respuestas[0].json()["id"] != respuestas[1].json()["id"]


@pytest.mark.skipif(not IMAGEN_4000.exists(), reason="falta datos/imagen_4000.pgm (ver datos/README.md)")
def test_con_la_cola_llena_responde_503_y_el_servicio_sigue():
    # Imagen grande y media 21: cada solicitud ocupa la GPU el tiempo suficiente para llenar la cola.
    datos = IMAGEN_4000.read_bytes()
    with ThreadPoolExecutor(SOLICITUDES_A_LA_VEZ) as pool:
        codigos = list(pool.map(lambda _: enviar(datos, "media", 21).status_code,
                                range(SOLICITUDES_A_LA_VEZ)))

    assert 503 in codigos
    assert 200 in codigos
    assert set(codigos) <= {200, 503}
    assert requests.get(URL + "/api/salud", timeout=5).status_code == 200
