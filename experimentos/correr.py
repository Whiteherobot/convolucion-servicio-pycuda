"""Corre los experimentos de la práctica 03 y guarda CSV. Bloque 3 - Andrés Morocho.

Un factor a la vez y 256 hilos fijos. Cada configuración: 1 calentamiento (repeticion 0, se guarda pero no
entra en las estadísticas) y N repeticiones. Ver experimentos/README.md para el procedimiento completo.

    python experimentos/correr.py                         # todo, con los valores por defecto
    python experimentos/correr.py --experimentos integracion,resolucion
"""
import argparse
import csv
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import requests  # noqa: E402

import cliente  # noqa: E402
import entorno as modulo_entorno  # noqa: E402
from app.modelo.validacion import TOLERANCIA, errores  # noqa: E402  (misma fórmula de la P02)

RAIZ = modulo_entorno.RAIZ
HILOS = 256
EXPERIMENTOS = ("validacion", "frio", "integracion", "resolucion", "filtros", "concurrencia")
COLUMNAS = [
    "corrida", "entorno", "experimento", "modo", "filtro", "resolucion", "mascara", "hilos", "bloques",
    "repeticion", "calentamiento", "concurrencia", "ronda", "cliente_num", "id_solicitud",
    "espera_cola_ms", "decodificacion_ms", "h2d_ms", "kernel_ms", "d2h_ms", "gpu_total_ms",
    "codificacion_ms", "servidor_ms", "descarga_ms", "cliente_ms", "contexto_ms", "compilacion_ms",
    "arranque_ms", "estado", "codigo_http", "error", "marca_tiempo",
]
COLUMNAS_VALIDACION = ["corrida", "entorno", "filtro", "mascara", "resolucion", "id_solicitud", "error_max",
                       "error_medio", "tolerancia", "cumple", "estado", "error"]
COLUMNAS_RONDAS = ["corrida", "entorno", "concurrencia", "ronda", "solicitudes", "exitosas", "fallidas",
                   "duracion_ms", "exitosas_por_s"]


class Registro:
    """Escribe mediciones.csv, rondas.csv y errores.log a medida que llegan (si algo se corta, no se pierde)."""

    def __init__(self, carpeta: Path, corrida: str, etiqueta: str):
        self.carpeta, self.base = carpeta, {"corrida": corrida, "entorno": etiqueta}
        self.candado = threading.Lock()
        for nombre, columnas in (("mediciones.csv", COLUMNAS), ("rondas.csv", COLUMNAS_RONDAS)):
            with open(carpeta / nombre, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(columnas)

    def fila(self, **datos):
        fila = {**dict.fromkeys(COLUMNAS, ""), **self.base, "hilos": HILOS,
                "marca_tiempo": datetime.now().astimezone().isoformat(timespec="milliseconds"), **datos}
        fila["calentamiento"] = "si" if fila["repeticion"] == 0 else "no"
        with self.candado:
            self._agregar("mediciones.csv", COLUMNAS, fila)
            if fila["estado"] != "ok":
                self.error(fila)
        marca = "calentamiento" if fila["repeticion"] == 0 else f"rep {fila['repeticion']}"
        print(f"  {fila['experimento']:<12} {fila['modo']:<18} {fila['filtro']} {fila['mascara']} "
              f"{fila['resolucion']} {marca:<13} {fila['estado']:<5} "
              f"kernel={fila['kernel_ms']} cliente={fila['cliente_ms']}", flush=True)

    def ronda(self, **datos):
        with self.candado:
            self._agregar("rondas.csv", COLUMNAS_RONDAS, {**self.base, **datos})

    def error(self, datos):
        with open(self.carpeta / "errores.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(datos, ensure_ascii=False, default=str) + "\n")

    def _agregar(self, nombre, columnas, fila):
        with open(self.carpeta / nombre, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, columnas, extrasaction="ignore").writerow(fila)


class Experimentos:
    def __init__(self, args, registro: Registro):
        self.args, self.reg = args, registro
        self.imagenes = {}

    def datos(self, resolucion: int) -> bytes:
        if resolucion not in self.imagenes:
            self.imagenes[resolucion] = (RAIZ / "datos" / f"imagen_{resolucion}.pgm").read_bytes()
        return self.imagenes[resolucion]

    # --- Servicio -----------------------------------------------------------------------------------
    def servicio(self, experimento, filtro, mascara, resolucion, repeticion, **extra):
        fila = cliente.enviar(self.args.url, self.datos(resolucion), f"imagen_{resolucion}.pgm",
                              filtro, mascara, HILOS)
        self.reg.fila(experimento=experimento, modo="servicio", filtro=filtro, mascara=mascara,
                      resolucion=resolucion, repeticion=repeticion, concurrencia=extra.pop("concurrencia", 1),
                      **extra, **fila)
        return fila

    def serie_servicio(self, experimento, filtro, mascara, resolucion):
        for repeticion in range(0, self.args.repeticiones + 1):
            self.servicio(experimento, filtro, mascara, resolucion, repeticion)

    # --- PyCUDA directo (anfitrión o contenedor) -----------------------------------------------------
    def serie_directa(self, modo, filtro, mascara, resolucion):
        entrada = f"datos/imagen_{resolucion}.pgm"
        resto = [filtro, str(mascara), "--hilos", str(HILOS), "--repeticiones", str(self.args.repeticiones)]
        if modo == "directo_host":
            orden = [self.args.python_host, "-m", "app.directo", entrada,
                     str(Path(tempfile.gettempdir()) / "directo_host.png"), *resto]
        else:
            orden = ["docker", "run", "--rm", "--gpus", "all", "--memory=4g",
                     "-v", f"{RAIZ / 'datos'}:/servicio/datos:ro", "--entrypoint", "python3",
                     self.args.imagen, "-m", "app.directo", entrada, "/tmp/directo.png", *resto]
        print("  $", " ".join(orden), flush=True)
        comun = dict(experimento="integracion", modo=modo, filtro=filtro, mascara=mascara,
                     resolucion=resolucion, concurrencia=1, bloques=-(-resolucion * resolucion // HILOS))
        try:
            salida = subprocess.run(orden, cwd=RAIZ, capture_output=True, text=True, timeout=900)
        except (OSError, subprocess.TimeoutExpired) as error:
            self.reg.fila(**comun, repeticion=1, estado="error", error=str(error)[:300])
            return
        lineas = [json.loads(l) for l in salida.stdout.splitlines() if l.startswith("{")]
        if salida.returncode != 0 or not lineas:
            self.reg.fila(**comun, repeticion=1, estado="error", codigo_http=f"salida {salida.returncode}",
                          error=(salida.stderr.strip().splitlines() or ["sin salida"])[-1][:300])
            self.reg.error({"modo": modo, "orden": orden, "stderr": salida.stderr[-3000:]})
            return
        arranque = lineas[0]
        for linea in lineas[1:]:
            self.reg.fila(**comun, id_solicitud=f"{modo}-{uuid.uuid4().hex[:12]}",
                          repeticion=0 if linea.get("calentamiento") else linea["repeticion"],
                          h2d_ms=linea["h2d"], kernel_ms=linea["kernel"], d2h_ms=linea["d2h"],
                          gpu_total_ms=linea["gpu_total"], contexto_ms=arranque.get("contexto_ms", ""),
                          compilacion_ms=arranque.get("compilacion_ms", ""), estado="ok")

    # --- Validación ---------------------------------------------------------------------------------
    def validacion(self):
        """Equivalencia PyCUDA directo vs .npy del servicio (antes del recorte) y PNG de evidencia, 2000²."""
        evidencia = self.reg.carpeta / "evidencia"
        evidencia.mkdir(exist_ok=True)
        ruta = self.reg.carpeta / "validacion.csv"
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(COLUMNAS_VALIDACION)
        for filtro, mascara in (("media", 9), ("nitidez", 3), ("laplaciano", 3)):
            fila = {**self.reg.base, "filtro": filtro, "mascara": mascara, "resolucion": 2000,
                    "tolerancia": TOLERANCIA}
            try:
                salida = Path(tempfile.gettempdir()) / f"validacion_{filtro}{mascara}.png"
                proceso = subprocess.run(
                    [self.args.python_host, "-m", "app.directo", "datos/imagen_2000.pgm", str(salida), filtro,
                     str(mascara), "--hilos", str(HILOS), "--repeticiones", "1"],
                    cwd=RAIZ, capture_output=True, text=True, timeout=600)
                if proceso.returncode != 0:
                    raise RuntimeError("app.directo: " + (proceso.stderr.strip().splitlines() or ["?"])[-1])
                directo = np.load(salida.with_suffix(".npy"))
                respuesta = cliente.enviar(self.args.url, self.datos(2000), "imagen_2000.pgm", filtro, mascara,
                                           HILOS, guardar_png=evidencia / f"{filtro}{mascara}.png")
                if respuesta["estado"] != "ok":
                    raise RuntimeError(f"servicio {respuesta['codigo_http']}: {respuesta['error']}")
                crudo = requests.get(f"{self.args.url}/api/resultado/{respuesta['id_solicitud']}.npy", timeout=300)
                crudo.raise_for_status()
                maximo, medio = errores(directo, np.load(io.BytesIO(crudo.content)))
                fila.update(id_solicitud=respuesta["id_solicitud"], error_max=maximo, error_medio=medio,
                            cumple="si" if maximo < TOLERANCIA and medio < TOLERANCIA else "no", estado="ok")
            except Exception as error:
                fila.update(estado="error", error=f"{type(error).__name__}: {error}"[:300])
                self.reg.error({"experimento": "validacion", **fila})
            with open(ruta, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, COLUMNAS_VALIDACION, extrasaction="ignore").writerow(fila)
            print(f"  validacion   {filtro} {mascara}  {fila['estado']}  error_max={fila.get('error_max', '')} "
                  f"error_medio={fila.get('error_medio', '')}", flush=True)

    # --- Experimentos -------------------------------------------------------------------------------
    def frio(self):
        """Reinicia el contenedor y mide la primera solicitud. Fuera de las mediciones estables."""
        if not self.args.contenedor:
            print("  Sin --contenedor: se omite el arranque en frío.")
            return
        for repeticion in range(1, self.args.reinicios + 1):
            arranque_ms = self.reiniciar()
            if arranque_ms is None:
                self.reg.fila(experimento="frio", modo="servicio", repeticion=repeticion, estado="error",
                              error="El servicio no quedó listo tras reiniciar el contenedor")
                continue
            estado = cliente.salud(self.args.url)
            self.servicio("frio", "media", 9, 4000, repeticion, arranque_ms=arranque_ms,
                          contexto_ms=estado["contexto_ms"], compilacion_ms=estado["compilacion_ms"])

    def integracion(self):
        modos = [m for m in ("directo_host", "directo_contenedor") if m not in self.args.omitir_modos]
        for modo in modos:
            self.serie_directa(modo, "media", 9, 4000)
        self.serie_servicio("integracion", "media", 9, 4000)

    def resolucion(self):
        for resolucion in (2000, 4000, 6000):
            self.serie_servicio("resolucion", "media", 9, resolucion)

    def filtros(self):
        for filtro, mascara in (("media", 3), ("media", 9), ("media", 21), ("nitidez", 3), ("laplaciano", 3)):
            self.serie_servicio("filtros", filtro, mascara, 4000)

    def concurrencia(self):
        # Directorio temporal nuevo: evita el error de limpiar_viejos() con archivos vencidos (Bloque 1).
        if self.args.contenedor:
            self.reiniciar()
        for clientes in (1, 2):
            self.servicio("concurrencia", "media", 9, 4000, 0, concurrencia=clientes)
            for ronda in range(1, self.args.rondas + 1):
                self.ronda_concurrente(clientes, ronda)

    def ronda_concurrente(self, clientes, ronda):
        """Todos los clientes envían a la vez; rendimiento = exitosas / duración total de la ronda."""
        salida, inicio = threading.Event(), [0.0]
        resultados = [None] * clientes

        def trabajar(i):
            salida.wait()
            resultados[i] = self.servicio("concurrencia", "media", 9, 4000, ronda, concurrencia=clientes,
                                          ronda=ronda, cliente_num=i + 1)

        hilos = [threading.Thread(target=trabajar, args=(i,)) for i in range(clientes)]
        for h in hilos:
            h.start()
        inicio[0] = time.perf_counter()
        salida.set()
        for h in hilos:
            h.join()
        duracion = (time.perf_counter() - inicio[0]) * 1000
        exitosas = sum(r["estado"] == "ok" for r in resultados)
        self.reg.ronda(concurrencia=clientes, ronda=ronda, solicitudes=clientes, exitosas=exitosas,
                       fallidas=clientes - exitosas, duracion_ms=round(duracion, 3),
                       exitosas_por_s=round(exitosas / (duracion / 1000), 4))

    def reiniciar(self) -> float | None:
        """docker restart y espera a que /api/salud tenga GPU. Devuelve ms hasta quedar listo."""
        print(f"  Reiniciando {self.args.contenedor}...", flush=True)
        inicio = time.perf_counter()
        subprocess.run(["docker", "restart", self.args.contenedor], capture_output=True, timeout=120)
        while time.perf_counter() - inicio < 180:
            try:
                if cliente.salud(self.args.url, 2).get("gpu"):
                    return round((time.perf_counter() - inicio) * 1000, 3)
            except Exception:
                pass
            time.sleep(0.5)
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--imagen", default="convolucion-servicio:p03")
    p.add_argument("--contenedor", default="conv-p03", help="vacío para no reiniciar (sin arranque en frío)")
    p.add_argument("--python-host", default=sys.executable, help="Python con PyCUDA para directo_host")
    p.add_argument("--experimentos", default=",".join(EXPERIMENTOS))
    p.add_argument("--omitir-modos", default="", help="p. ej. directo_host si no se puede en el anfitrión")
    p.add_argument("--repeticiones", type=int, default=5)
    p.add_argument("--rondas", type=int, default=5)
    p.add_argument("--reinicios", type=int, default=3)
    p.add_argument("--forzar", action="store_true", help="medir aunque la verificación encuentre problemas")
    args = p.parse_args()
    elegidos = [e.strip() for e in args.experimentos.split(",") if e.strip()]
    args.omitir_modos = [m.strip() for m in args.omitir_modos.split(",") if m.strip()]
    if desconocidos := set(elegidos) - set(EXPERIMENTOS):
        p.error(f"experimentos desconocidos: {desconocidos}")

    entorno, problemas, avisos = modulo_entorno.recolectar(args.url, args.imagen, args.contenedor)
    for a in avisos:
        print("AVISO:", a)
    for pr in problemas:
        print("PROBLEMA:", pr)
    if problemas and not args.forzar:
        sys.exit("No se mide: corrige los problemas (o usa --forzar y explícalo en el informe).")

    corrida = datetime.now().strftime("%Y%m%d-%H%M%S")
    carpeta = RAIZ / "resultados" / corrida
    carpeta.mkdir(parents=True)
    entorno.update(corrida=corrida, parametros=vars(args), problemas=problemas, avisos=avisos)
    (carpeta / "entorno.json").write_text(json.dumps(entorno, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Corrida {corrida} -> {carpeta.relative_to(RAIZ)}\nEntorno: {entorno['etiqueta']}")

    experimentos = Experimentos(args, Registro(carpeta, corrida, entorno["etiqueta"]))
    for nombre in elegidos:
        print(f"\n== {nombre}", flush=True)
        getattr(experimentos, nombre)()
    print(f"\nListo. Comprime y comparte la carpeta {carpeta.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
