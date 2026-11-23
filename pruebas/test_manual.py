"""Matrices pequeñas con valores calculados a mano. Bloque 2 - Jordy Romero."""
import numpy as np
import pytest

from app.modelo.validacion import TOLERANCIA
from conftest import a_png, descargar_crudo, enviar

# matriz de la figura 1
FIGURA1 = np.array([[35, 40, 41, 45, 50],
                    [40, 40, 42, 46, 52],
                    [42, 46, 50, 55, 55],
                    [48, 52, 56, 58, 60],
                    [56, 60, 65, 70, 75]], dtype=np.float32)


def test_laplaciano_figura1():
    respuesta = enviar(a_png(FIGURA1), "laplaciano", 3)
    assert respuesta.status_code == 200
    salida = descargar_crudo(respuesta.json())
    # centro (1,1): 40 + 46 + 40 + 42 - 4*40 = 8
    assert salida[1, 1] == pytest.approx(8, abs=TOLERANCIA)
    # esquina: 40 + 40 - 4*35 = -60 (el .npy no recorta negativos)
    assert salida[0, 0] == pytest.approx(-60, abs=TOLERANCIA)


def test_media_con_ceros_fuera_de_la_imagen():
    unos = np.ones((5, 7), dtype=np.float32)
    respuesta = enviar(a_png(unos), "media", 3)
    assert respuesta.status_code == 200
    salida = descargar_crudo(respuesta.json())
    # vecinos dentro de la imagen: esquina 4 de 9, borde 6 de 9, interior 9 de 9
    assert salida[0, 0] == pytest.approx(4 / 9, abs=TOLERANCIA)
    assert salida[0, 3] == pytest.approx(6 / 9, abs=TOLERANCIA)
    assert salida[2, 3] == pytest.approx(1.0, abs=TOLERANCIA)
