"""Solicitudes simultáneas: cada una recibe su resultado; con la cola llena, 503. Bloque 2 - Jordy Romero."""
import io
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from app.modelo import mascaras
from app.modelo.pycuda_conv import TiemposGPU
from app.modelo.validacion import TOLERANCIA, errores
from app.servidor import crear_app
from conftest import a_png, descargar_crudo, enviar, imagen_aleatoria


class GpuLenta:
    """GPU falsa: tarda 0.3 s por solicitud, lo justo para que la cola se llene."""

    contexto_ms = 0.0
    compilacion_ms = 0.0
    contexto = SimpleNamespace(pop=lambda: None)

    def ejecutar(self, imagen, mascara, hilos):
        time.sleep(0.3)
        return np.zeros_like(imagen), TiemposGPU()


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


def test_con_la_cola_llena_responde_503_y_el_servicio_sigue(monkeypatch):
    # Se prueba la app en este proceso con una GPU falsa: con gunicorn --threads 4 nunca hay más de
    # 4 solicitudes dentro a la vez y la cola (4 en espera + 1 en la GPU) no llega a llenarse por HTTP.
    monkeypatch.setattr("app.modelo.solicitudes.ConvolucionPyCUDA", GpuLenta)
    app = crear_app()
    cola = app.extensions["cola_gpu"]
    datos = a_png(imagen_aleatoria(50, 50))

    def solicitar(_):
        campos = {"imagen": (io.BytesIO(datos), "imagen.png"), "filtro": "media", "mascara": "3"}
        return app.test_client().post("/api/procesar", data=campos).status_code

    try:
        with ThreadPoolExecutor(12) as pool:
            codigos = list(pool.map(solicitar, range(12)))
        assert 503 in codigos
        assert 200 in codigos
        assert set(codigos) <= {200, 503}
        assert app.test_client().get("/api/salud").status_code == 200
    finally:
        cola.detener()
        cola.hilo.join(timeout=10)
