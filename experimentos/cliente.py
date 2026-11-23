"""Cliente HTTP que mide una solicitud de punta a punta. Bloque 3 - Andrés Morocho.

El tiempo del cliente va desde justo antes del POST hasta tener el PNG completo (incluye la descarga de
url_resultado). Se mide con time.perf_counter() en el reloj de esta máquina; nunca se resta contra los
tiempos del servidor, que se guardan tal como los devuelve la respuesta.
"""
import time

import requests

TIEMPO_MAXIMO_S = 300
CAMPOS_SERVIDOR = ("espera_cola", "decodificacion", "h2d", "kernel", "d2h",
                   "gpu_total", "codificacion", "servidor")


def salud(url: str, tiempo_maximo: float = 5) -> dict:
    """GET /api/salud. Lanza una excepción si el servicio no responde."""
    respuesta = requests.get(url + "/api/salud", timeout=tiempo_maximo)
    respuesta.raise_for_status()
    return respuesta.json()


def enviar(url: str, datos: bytes, nombre: str, filtro: str, mascara: int,
           hilos: int = 256, guardar_png=None) -> dict:
    """Envía una imagen y descarga el PNG. Devuelve una fila con tiempos en ms y el estado.

    Nunca lanza excepciones por errores HTTP o de red: quedan en 'estado', 'codigo_http' y 'error'
    para que el experimento siga y el fallo se registre.
    """
    fila = {"id_solicitud": "", "estado": "error", "codigo_http": "", "error": "",
            "bloques": "", "cliente_ms": "", "descarga_ms": "",
            **{c + "_ms": "" for c in CAMPOS_SERVIDOR}}
    campos = {"filtro": filtro, "mascara": str(mascara), "hilos": str(hilos)}

    inicio = time.perf_counter()
    try:
        respuesta = requests.post(url + "/api/procesar", files={"imagen": (nombre, datos)},
                                  data=campos, timeout=TIEMPO_MAXIMO_S)
        fila["codigo_http"] = respuesta.status_code
        cuerpo = _json(respuesta)
        if respuesta.status_code != 200:
            fila["error"] = cuerpo.get("error", respuesta.text[:200])
            fila["cliente_ms"] = _ms(inicio)
            return fila

        inicio_descarga = time.perf_counter()
        png = requests.get(url + cuerpo["url_resultado"], timeout=TIEMPO_MAXIMO_S)
        fin = time.perf_counter()
    except requests.RequestException as error:
        fila["error"] = f"{type(error).__name__}: {error}"[:300]
        fila["cliente_ms"] = _ms(inicio)
        return fila

    fila["id_solicitud"] = cuerpo["id"]
    fila["bloques"] = cuerpo["bloques"]
    for campo in CAMPOS_SERVIDOR:
        fila[campo + "_ms"] = cuerpo["tiempos_ms"].get(campo, "")
    fila["cliente_ms"] = round((fin - inicio) * 1000, 3)
    fila["descarga_ms"] = round((fin - inicio_descarga) * 1000, 3)

    if png.status_code != 200 or not png.content.startswith(b"\x89PNG"):
        fila["codigo_http"] = png.status_code
        fila["error"] = "La descarga del PNG falló o no es un PNG"
        return fila
    fila["estado"] = "ok"
    if guardar_png is not None:
        with open(guardar_png, "wb") as archivo:
            archivo.write(png.content)
    return fila


def _json(respuesta) -> dict:
    try:
        return respuesta.json()
    except ValueError:
        return {}


def _ms(inicio: float) -> float:
    return round((time.perf_counter() - inicio) * 1000, 3)
