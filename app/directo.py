"""CLI de convolución directa en GPU. Bloque 1 - Michael Lata."""
import atexit
import argparse
import json
from pathlib import Path

import numpy as np

from app.modelo import procesamiento
from app.modelo.imagen import guardar_visual
from app.modelo.pycuda_conv import ConvolucionPyCUDA


def argumento():
    """Lee los argumentos de línea de comandos."""
    analizador = argparse.ArgumentParser()
    analizador.add_argument("entrada")
    analizador.add_argument("salida")
    analizador.add_argument("filtro")
    analizador.add_argument("k", type=int)
    analizador.add_argument("--hilos", type=int, default=256)
    analizador.add_argument("--repeticiones", type=int, default=1)
    return analizador.parse_args()


def principal():
    """Ejecuta calentamiento y repeticiones."""
    args = argumento()
    gpu = ConvolucionPyCUDA()
    print(json.dumps({
        "contexto_ms": round(gpu.contexto_ms, 3),
        "compilacion_ms": round(gpu.compilacion_ms, 3),
    }, ensure_ascii=False))

    imagen = procesamiento.decodificar(open(args.entrada, "rb").read())
    resultado, tiempos = procesamiento.procesar(
        gpu, imagen, args.filtro, args.k, args.hilos)
    print(json.dumps({
        "calentamiento": True,
        "h2d": round(tiempos.h2d_ms, 3),
        "kernel": round(tiempos.kernel_ms, 3),
        "d2h": round(tiempos.d2h_ms, 3),
        "gpu_total": round(tiempos.total_ms, 3),
    }, ensure_ascii=False))

    for repeticion in range(1, args.repeticiones + 1):
        resultado, tiempos = procesamiento.procesar(
            gpu, imagen, args.filtro, args.k, args.hilos)
        print(json.dumps({
            "repeticion": repeticion,
            "h2d": round(tiempos.h2d_ms, 3),
            "kernel": round(tiempos.kernel_ms, 3),
            "d2h": round(tiempos.d2h_ms, 3),
            "gpu_total": round(tiempos.total_ms, 3),
        }, ensure_ascii=False))

    guardar_visual(args.salida, resultado)
    np.save(Path(args.salida).with_suffix(".npy"),
            np.ascontiguousarray(resultado, dtype=np.float32))
    try:
        atexit.unregister(gpu.contexto.pop)
    except Exception:
        pass
    gpu.contexto.pop()


if __name__ == "__main__":
    principal()
