"""Validación y cola de procesamiento GPU. Bloque 1 - Michael Lata."""
import atexit
import io
import math
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
import pycuda.driver as cuda

from app.modelo.procesamiento import codificar_png, decodificar, procesar
from app.modelo.pycuda_conv import ConvolucionPyCUDA


MAX_BYTES = 40 * 1024 * 1024
MAX_LADO = 6000
FORMATOS = {"PNG", "JPEG", "PPM", "BMP", "TIFF"}
FILTROS = {"media": (3, 9, 21), "nitidez": (3,), "laplaciano": (3,)}
MAX_COLA = 4  # solicitudes en espera; una más puede estar en la GPU
VIDA_RESULTADO_S = 600
HILOS_POR_DEFECTO = 256

Image.MAX_IMAGE_PIXELS = MAX_LADO * MAX_LADO + 1


class ErrorSolicitud(Exception):
    """Error HTTP sencillo con código y mensaje."""

    def __init__(self, codigo: int, mensaje: str):
        super().__init__(mensaje)
        self.codigo = codigo


def validar(datos: bytes, filtro, mascara_txt, hilos_txt):
    """Valida el archivo y los parámetros de una solicitud."""
    if len(datos) > MAX_BYTES:
        raise ErrorSolicitud(413, "Archivo demasiado grande")

    try:
        imagen_pil = Image.open(io.BytesIO(datos))
        formato = imagen_pil.format
        ancho, alto = imagen_pil.size
        imagen_pil.close()
    except Exception:
        raise ErrorSolicitud(400, "El archivo no es una imagen válida")

    if formato not in FORMATOS:
        raise ErrorSolicitud(400, "Formato de imagen no permitido")
    if ancho > MAX_LADO or alto > MAX_LADO:
        raise ErrorSolicitud(413, "Imagen demasiado grande")

    if filtro not in FILTROS:
        raise ErrorSolicitud(400, "Filtro desconocido")

    if mascara_txt is None or str(mascara_txt).strip() == "":
        k = 3
    else:
        try:
            k = int(mascara_txt)
        except (TypeError, ValueError):
            raise ErrorSolicitud(400, "Máscara inválida")
    if k not in FILTROS[filtro]:
        raise ErrorSolicitud(400, "Máscara no permitida para el filtro")

    if hilos_txt is None or str(hilos_txt).strip() == "":
        hilos = HILOS_POR_DEFECTO
    else:
        try:
            hilos = int(hilos_txt)
        except (TypeError, ValueError):
            raise ErrorSolicitud(400, "Hilos inválidos")
    if hilos < 1 or hilos > 1024:
        raise ErrorSolicitud(400, "Hilos debe estar entre 1 y 1024")

    return filtro, k, hilos


class ColaGPU:
    """Cola serializada con un único contexto PyCUDA."""

    def __init__(self):
        self.cola = queue.Queue(maxsize=MAX_COLA)
        self.listo = threading.Event()
        self.gpu = None
        self.error_gpu = None
        self.contexto_ms = None
        self.compilacion_ms = None
        self.directorio = tempfile.mkdtemp(prefix="convolucion_")
        atexit.register(shutil.rmtree, self.directorio, True)  # borra temporales al salir
        self.hilo = threading.Thread(target=self._trabajador, daemon=True)
        self.hilo.start()

    def _trabajador(self):
        try:
            self.gpu = ConvolucionPyCUDA()
            try:
                atexit.unregister(self.gpu.contexto.pop)
            except Exception:
                pass
            self.contexto_ms = round(self.gpu.contexto_ms, 3)
            self.compilacion_ms = round(self.gpu.compilacion_ms, 3)
        except Exception as error:
            self.error_gpu = str(error)
            self.gpu = None
        finally:
            self.listo.set()

        while True:
            trabajo = self.cola.get()
            if trabajo is None:
                if self.gpu is not None:
                    self.gpu.contexto.pop()
                return

            trabajo["espera_cola"] = (time.perf_counter() - trabajo["inicio_cola"]) * 1000
            try:
                if self.gpu is None:
                    motivo = self.error_gpu or "motivo desconocido"
                    raise ErrorSolicitud(503, "GPU no disponible: " + motivo)

                inicio = time.perf_counter()
                arreglo = decodificar(trabajo["datos"])
                tiempo_decodificacion = (time.perf_counter() - inicio) * 1000

                resultado, tiempos = procesar(
                    self.gpu, arreglo, trabajo["filtro"],
                    trabajo["k"], trabajo["hilos"])

                inicio = time.perf_counter()
                png = codificar_png(resultado)
                tiempo_codificacion = (time.perf_counter() - inicio) * 1000

                solicitud_id = uuid.uuid4().hex
                alto, ancho = resultado.shape
                np.save(os.path.join(self.directorio, solicitud_id + ".npy"),
                        np.ascontiguousarray(resultado, dtype=np.float32))
                with open(os.path.join(self.directorio, solicitud_id + ".png"), "wb") as archivo:
                    archivo.write(png)

                trabajo["respuesta"] = {
                    "id": solicitud_id,
                    "filtro": trabajo["filtro"],
                    "mascara": trabajo["k"],
                    "ancho": int(ancho),
                    "alto": int(alto),
                    "hilos": trabajo["hilos"],
                    "bloques": math.ceil(ancho * alto / trabajo["hilos"]),
                    "tiempos_ms": {
                        "espera_cola": round(trabajo["espera_cola"], 3),
                        "decodificacion": round(tiempo_decodificacion, 3),
                        "h2d": round(tiempos.h2d_ms, 3),
                        "kernel": round(tiempos.kernel_ms, 3),
                        "d2h": round(tiempos.d2h_ms, 3),
                        "gpu_total": round(tiempos.total_ms, 3),
                        "codificacion": round(tiempo_codificacion, 3),
                        "servidor": 0.0,
                    },
                    "url_resultado": "/api/resultado/" + solicitud_id + ".png",
                    "url_crudo": "/api/resultado/" + solicitud_id + ".npy",
                }
            except ErrorSolicitud as error:
                trabajo["error"] = error
            except Exception as error:
                trabajo["error"] = ErrorSolicitud(
                    500, "Falla de procesamiento GPU: " + str(error))
            finally:
                trabajo["evento"].set()

    def enviar(self, datos, filtro, k, hilos):
        if not self.listo.wait(timeout=120):
            raise ErrorSolicitud(503, "GPU no disponible: tiempo de espera agotado")
        if self.gpu is None:
            motivo = self.error_gpu or "motivo desconocido"
            raise ErrorSolicitud(503, "GPU no disponible: " + motivo)

        trabajo = {
            "datos": datos,
            "filtro": filtro,
            "k": k,
            "hilos": hilos,
            "inicio_cola": time.perf_counter(),
            "evento": threading.Event(),
        }
        try:
            self.cola.put_nowait(trabajo)
        except queue.Full:
            raise ErrorSolicitud(503, "Cola llena, intenta de nuevo")

        trabajo["evento"].wait()
        if "error" in trabajo:
            raise trabajo["error"]
        return trabajo["respuesta"]

    def detener(self):
        """Detiene al trabajador de manera limpia."""
        self.cola.put(None)

    def limpiar_viejos(self):
        ahora = time.time()
        for archivo in Path(self.directorio).glob("*"):
            if archivo.is_file() and ahora - archivo.stat().st_mtime > VIDA_RESULTADO_S:
                archivo.unlink()

    def ruta_resultado(self, id_texto: str, ext: str):
        if not re.fullmatch(r"[0-9a-f]{32}", id_texto or ""):
            return None
        if ext not in ("png", "npy"):
            return None
        ruta = os.path.join(self.directorio, id_texto + "." + ext)
        if not os.path.isfile(ruta):
            return None
        ahora = time.time()
        if ahora - os.path.getmtime(ruta) > VIDA_RESULTADO_S:
            os.unlink(ruta)
            return None
        return ruta

    def salud(self):
        nombre = None
        if self.gpu is not None:
            try:
                nombre = cuda.Device(0).name()
            except Exception:
                nombre = None
        return {
            "gpu": nombre,
            "error": self.error_gpu,
            "contexto_ms": self.contexto_ms,
            "compilacion_ms": self.compilacion_ms,
            "en_cola": self.cola.qsize(),
            "max_cola": MAX_COLA,
            "limites": {
                "archivo_bytes": MAX_BYTES,
                "lado_px": MAX_LADO,
                "formatos": sorted(FORMATOS),
                "vida_resultado_s": VIDA_RESULTADO_S,
            },
        }
