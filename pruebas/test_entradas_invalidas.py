"""Entradas inválidas: código, mensaje y que el servicio siga respondiendo. Bloque 2 - Jordy Romero."""
import io

import numpy as np
import pytest
import requests
from PIL import Image

from conftest import URL, a_png, enviar, imagen_aleatoria

MB = 1024 * 1024
PNG_VALIDO = a_png(imagen_aleatoria(50, 50))


def imagen_gif() -> bytes:
    salida = io.BytesIO()
    Image.new("L", (10, 10)).save(salida, format="GIF")
    return salida.getvalue()


# (nombre, archivo, filtro, mascara, extra, código esperado, texto del error)
CASOS = [
    ("no_es_imagen", b"esto es texto, no una imagen", "media", 3, {}, 400, "no es una imagen"),
    ("formato_no_permitido", imagen_gif(), "media", 3, {}, 400, "Formato"),
    ("lado_mayor_al_limite", a_png(np.zeros((1, 6001))), "media", 3, {}, 413, "demasiado grande"),
    ("archivo_mayor_a_40MB", bytes(42 * MB), "media", 3, {}, 413, "demasiado grande"),
    ("filtro_desconocido", PNG_VALIDO, "sobel", 3, {}, 400, "Filtro desconocido"),
    ("media_con_mascara_5", PNG_VALIDO, "media", 5, {}, 400, "Máscara no permitida"),
    ("nitidez_con_mascara_9", PNG_VALIDO, "nitidez", 9, {}, 400, "Máscara no permitida"),
    ("mascara_no_numerica", PNG_VALIDO, "media", "abc", {}, 400, "Máscara inválida"),
    ("hilos_cero", PNG_VALIDO, "media", 3, {"hilos": "0"}, 400, "Hilos"),
    ("hilos_mayor_a_1024", PNG_VALIDO, "media", 3, {"hilos": "2048"}, 400, "Hilos"),
]


@pytest.mark.parametrize("archivo,filtro,mascara,extra,codigo,texto",
                         [caso[1:] for caso in CASOS], ids=[caso[0] for caso in CASOS])
def test_entrada_invalida_se_rechaza_y_el_servicio_sigue(archivo, filtro, mascara, extra, codigo, texto):
    respuesta = enviar(archivo, filtro, mascara, **extra)
    assert respuesta.status_code == codigo
    assert texto in respuesta.json()["error"]

    assert requests.get(URL + "/api/salud", timeout=5).status_code == 200
    assert enviar(PNG_VALIDO, "media", 3).status_code == 200


@pytest.mark.xfail(strict=True, reason="Bloque 1: validar() solo lee la cabecera; la imagen truncada "
                   "falla recién en la GPU y responde 500 en vez de 400")
def test_imagen_truncada_responde_400():
    respuesta = enviar(PNG_VALIDO[:100], "media", 3)
    assert respuesta.status_code == 400


def test_sin_archivo_responde_400():
    respuesta = requests.post(URL + "/api/procesar", data={"filtro": "media", "mascara": "3"}, timeout=10)
    assert respuesta.status_code == 400
    assert "Falta el archivo" in respuesta.json()["error"]


def test_resultado_inexistente_responde_404():
    respuesta = requests.get(URL + "/api/resultado/" + "0" * 32 + ".png", timeout=10)
    assert respuesta.status_code == 404
