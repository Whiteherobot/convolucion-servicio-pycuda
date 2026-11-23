#!/usr/bin/env python3
"""Tablas y gráficas del informe a partir de una corrida. Bloque 3 - Andrés Morocho.

Uso: python experimentos/graficas.py [resultados/<corrida>] [--salida resultados]
     (sin argumento toma la corrida más reciente de resultados/)

Entradas (las deja experimentos/correr.py): entorno.json, mediciones.csv, rondas.csv, validacion.csv,
  evidencia/*.png (y datos/imagen_2000.pgm para la figura de evidencia)
Salidas:
  <corrida>/resumen.csv   mediana, p25, p75, CV, mínimo y máximo de cada tiempo por configuración
  <salida>/tablas/*.tex   tablas para el informe (\\tabla{...}); no editarlas a mano
  <salida>/graficas/*.png gráficas para el informe y las diapositivas
Si falta un experimento, solo se omiten sus salidas, con el motivo.

Definiciones:
  solo cuentan filas con estado «ok» y calentamiento «no»; el arranque en frío va en su propia tabla
  «resto del servidor» = servidor - (cola + decodificación + h2d + kernel + d2h + codificación):
      validación, lectura del multipart y escritura de .png/.npy
  «fuera del servidor» = cliente - servidor (HTTP, red local y descarga del PNG). Se restan duraciones,
      nunca marcas de tiempo de relojes distintos
  en las barras apiladas cada segmento es la mediana de su componente (no tienen por qué sumar la
      mediana del total, que se rotula aparte)
"""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
METRICAS = ["espera_cola", "decodificacion", "h2d", "kernel", "d2h", "gpu_total", "codificacion",
            "resto", "servidor", "fuera", "cliente"]
# Paleta validada (5 tonos, P02 + morado con más croma); los segmentos sin cálculo van en gris con trama.
SEGMENTOS = [
    ("espera_cola", "Espera en cola", "#6E6E6E", "xx"),
    ("decodificacion", "Decodificación", "#4C72B0", None),
    ("h2d", "CPU→GPU", "#55A868", None),
    ("kernel", "Kernel", "#C44E52", None),
    ("d2h", "GPU→CPU", "#7A5CC0", None),
    ("codificacion", "Codificación PNG", "#DD8452", None),
    ("resto", "Resto del servidor", "#A8A8A8", "//"),
    ("fuera", "Fuera del servidor (HTTP y descarga)", "#D0D0D0", ".."),
]
NOMBRE_MODO = {"directo_host": "Directo (anfitrión)", "directo_contenedor": "Directo (contenedor)",
               "servicio": "Servicio (HTTP)"}
ORDEN_FILTROS = [("media", 3), ("media", 9), ("media", 21), ("nitidez", 3), ("laplaciano", 3)]

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})


class FaltanDatos(Exception):
    """Salida que no se puede generar con lo que hay (se omite con aviso)."""


# ── Carga y estadística ─────────────────────────────────────────────────────────────────────────

def numero(valor):
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def leer_csv(ruta: Path) -> list[dict]:
    if not ruta.is_file():
        return []
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def preparar(fila: dict) -> dict:
    """Convierte los tiempos a float y calcula «resto» y «fuera»."""
    t = {m: numero(fila.get(m + "_ms")) for m in METRICAS if m not in ("resto", "fuera")}
    if t["servidor"] is not None:
        partes = [t[m] or 0.0 for m in ("espera_cola", "decodificacion", "h2d", "kernel", "d2h", "codificacion")]
        t["resto"] = max(t["servidor"] - sum(partes), 0.0)
        t["fuera"] = max(t["cliente"] - t["servidor"], 0.0) if t["cliente"] is not None else None
    else:
        t["resto"] = t["fuera"] = None
    return {**fila, "t": t}


def estable(filas, **filtro) -> list[dict]:
    """Filas válidas para estadística: ok, sin calentamiento y que cumplen el filtro."""
    return [f for f in filas if f["estado"] == "ok" and f["calentamiento"] == "no"
            and all(str(f[k]) == str(v) for k, v in filtro.items())]


def est(valores) -> dict | None:
    v = sorted(x for x in valores if x is not None)
    if not v:
        return None
    media = statistics.fmean(v)
    p25, _, p75 = statistics.quantiles(v, n=4, method="inclusive") if len(v) > 1 else (v[0],) * 3
    return {"n": len(v), "mediana": statistics.median(v), "p25": p25, "p75": p75, "min": v[0], "max": v[-1],
            "cv": 100 * statistics.stdev(v) / media if len(v) > 1 and media else 0.0}


def mediana(filas, metrica):
    e = est(f["t"][metrica] for f in filas)
    return e["mediana"] if e else None


# ── Formato LaTeX ───────────────────────────────────────────────────────────────────────────────

def ms(v) -> str:
    if v is None:
        return "--"
    if v >= 100:
        return f"{v:.0f}"
    return f"{v:.3g}" if v >= 0.01 else f"{v:.2g}"


def tex(texto) -> str:
    texto = str(texto)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
        texto = texto.replace(a, b) if a != "\\" else texto
    return texto.replace("×", r"$\times$").replace("→", r"$\to$")


def escribir_tabla(ruta: Path, columnas: str, encabezado: list[str], filas: list[list[str]]):
    lineas = ["% Generado por experimentos/graficas.py; no editar a mano.",
              rf"\begin{{tabular}}{{{columnas}}}", r"\toprule", " & ".join(tex(e) for e in encabezado) + r" \\", r"\midrule"]
    lineas += [" & ".join(f) + r" \\" for f in filas]
    lineas += [r"\bottomrule", r"\end{tabular}", ""]
    ruta.write_text("\n".join(lineas), encoding="utf-8")


def cv(filas, metrica) -> str:
    e = est(f["t"][metrica] for f in filas)
    return f"{e['cv']:.1f}" if e else "--"


# ── Tablas ──────────────────────────────────────────────────────────────────────────────────────

def tabla_entorno(entorno: dict, ruta: Path):
    a, img, git = entorno.get("anfitrion", {}), entorno.get("imagen") or {}, entorno.get("git", {})
    salud = entorno.get("salud") or {}
    filas = [
        ("GPU (servicio)", salud.get("gpu")), ("GPU y memoria (nvidia-smi)", a.get("gpu")),
        ("Controlador NVIDIA (anfitrión)", a.get("controlador_nvidia")), ("CPU", a.get("cpu")),
        ("RAM", a.get("ram")), ("Sistema operativo", a.get("so")), ("Docker", a.get("docker")),
        ("NVIDIA Container Toolkit", a.get("nvidia_container_toolkit")),
        ("CUDA de la imagen (nvcc)", img.get("nvcc")), ("Python / PyCUDA (imagen)",
                                                        f"{img.get('python')} / {img.get('pycuda')}"),
        ("CUDA del anfitrión (nvcc)", a.get("nvcc_anfitrion")),
        ("Python / PyCUDA (anfitrión)", f"{a.get('python_anfitrion')} / {a.get('pycuda_anfitrion')}"),
        ("Imagen Docker", f"{img.get('nombre')} ({(img.get('id') or '?').replace('sha256:', '')[:12]})"),
        ("Commit medido", (git.get("commit") or "?")[:7] + (" (con cambios sin commit)"
                                                           if git.get("cambios_sin_commit") else "")),
        ("Cliente", entorno.get("cliente")), ("Fecha de la corrida", entorno.get("fecha")),
    ]
    def valor(v):
        return "no disponible" if v in (None, "", "None / None") else str(v).replace("None", "?")
    escribir_tabla(ruta, "lp{9cm}", ["Componente", "Valor"], [[tex(k), tex(valor(v))] for k, v in filas])


def tabla_frio(filas, ruta: Path):
    frias = sorted((f for f in filas if f["experimento"] == "frio"), key=lambda f: int(f["repeticion"]))
    if not frias:
        raise FaltanDatos("no hay filas del arranque en frío")
    referencia = estable(filas, experimento="integracion", modo="servicio")
    cuerpo = [[f["repeticion"], ms(numero(f["arranque_ms"])), ms(numero(f["contexto_ms"])),
               ms(numero(f["compilacion_ms"])), ms(f["t"]["servidor"]), ms(f["t"]["cliente"]),
               f["estado"]] for f in frias]
    cuerpo.append([r"\midrule Estable (mediana)", "--", "--", "--", ms(mediana(referencia, "servidor")),
                   ms(mediana(referencia, "cliente")), "ok"])
    escribir_tabla(ruta, "lrrrrrl", ["Reinicio", "Listo", "Contexto", "Compilación", "Servidor",
                                     "Cliente", "Estado"], cuerpo)


def tabla_integracion(filas, ruta: Path):
    cuerpo = []
    for modo in NOMBRE_MODO:
        g = estable(filas, experimento="integracion", modo=modo)
        if g:
            cuerpo.append([NOMBRE_MODO[modo]] + [ms(mediana(g, m)) for m in
                                                 ("h2d", "kernel", "d2h", "gpu_total", "servidor", "cliente")]
                          + [cv(g, "gpu_total"), str(len(g))])
    if not cuerpo:
        raise FaltanDatos("no hay filas de integración")
    escribir_tabla(ruta, "lrrrrrrrr", ["Modo", "CPU→GPU", "Kernel", "GPU→CPU", "GPU total", "Servidor",
                                       "Cliente", "CV GPU (%)", "n"], [[tex(c) for c in f] for f in cuerpo])


def _tabla_servicio(filas, ruta, experimento, grupos, nombre):
    cuerpo = []
    for etiqueta, filtro in grupos:
        g = estable(filas, experimento=experimento, **filtro)
        if g:
            cuerpo.append([etiqueta] + [ms(mediana(g, m)) for m in
                                        ("kernel", "gpu_total", "decodificacion", "codificacion",
                                         "servidor", "cliente")] + [cv(g, "cliente"), str(len(g))])
    if not cuerpo:
        raise FaltanDatos(f"no hay filas de {experimento}")
    escribir_tabla(ruta, "lrrrrrrrr", [nombre, "Kernel", "GPU total", "Decodif.", "Codif.", "Servidor",
                                       "Cliente", "CV cliente (%)", "n"], [[tex(c) for c in f] for f in cuerpo])


def grupos_resolucion():
    return [(f"{r}×{r}", {"resolucion": r}) for r in (2000, 4000, 6000)]


def grupos_filtros():
    return [(f"{f} {k}×{k}", {"filtro": f, "mascara": k}) for f, k in ORDEN_FILTROS]


def tabla_concurrencia(filas, rondas, ruta: Path):
    cuerpo = []
    for c in (1, 2):
        todas = [f for f in filas if f["experimento"] == "concurrencia" and f["concurrencia"] == str(c)
                 and f["calentamiento"] == "no"]
        g = estable(filas, experimento="concurrencia", concurrencia=c)
        r = [x for x in rondas if x["concurrencia"] == str(c)]
        if not todas:
            continue
        e = est(x["t"]["cliente"] for x in g)
        tasa = est(numero(x["exitosas_por_s"]) for x in r)
        cuerpo.append([str(c), f"{len(g)}/{len(todas)}",
                       f"{ms(e['mediana'])} ({ms(e['p25'])}--{ms(e['p75'])})" if e else "--",
                       ms(mediana(g, "espera_cola")), ms(mediana(g, "servidor")),
                       f"{tasa['mediana']:.2f}" if tasa else "--", str(len(r))])
    if not cuerpo:
        raise FaltanDatos("no hay filas de concurrencia")
    escribir_tabla(ruta, "crrrrrr", ["Clientes", "Exitosas", "Cliente (p25--p75)", "Espera cola", "Servidor",
                                     "Exitosas/s", "Rondas"], cuerpo)


def tabla_estado(filas, ruta: Path):
    conteo = defaultdict(lambda: defaultdict(int))
    for f in filas:
        clave = "ok" if f["estado"] == "ok" else f"error {f['codigo_http'] or 'sin código'}"
        conteo[f["experimento"]][clave] += 1
    cuerpo = [[e, str(c.get("ok", 0)), str(sum(v for k, v in c.items() if k != "ok")),
               tex(", ".join(f"{k}: {v}" for k, v in c.items() if k != "ok") or "--")]
              for e, c in conteo.items()]
    escribir_tabla(ruta, "lrrl", ["Experimento", "Correctas", "Con error", "Detalle"], cuerpo)


def tabla_validacion(corrida: Path, ruta: Path):
    filas = leer_csv(corrida / "validacion.csv")
    if not filas:
        raise FaltanDatos("no hay validacion.csv")

    def cientifico(v):
        x = numero(v)
        return "--" if x is None else ("0" if x == 0 else f"{x:.1e}")
    cuerpo = [[tex(f"{f['filtro']} {f['mascara']}×{f['mascara']}"), cientifico(f["error_max"]),
               cientifico(f["error_medio"]), cientifico(f["tolerancia"]),
               f["cumple"] if f["estado"] == "ok" else tex("error: " + f["error"][:40])] for f in filas]
    escribir_tabla(ruta, "lrrrl", ["Filtro", "Error máximo", "Error medio", "Tolerancia", "Cumple"], cuerpo)


def grafica_evidencia(corrida: Path, ruta: Path, recorte: int = 300):
    """Original y resultado del servicio para los tres filtros (2000×2000); abajo, el mismo recorte central."""
    from PIL import Image
    original = RAIZ / "datos" / "imagen_2000.pgm"
    paneles = [("Original", original)] + [(n, corrida / "evidencia" / a) for n, a in
                                          (("Media 9×9", "media9.png"), ("Nitidez 3×3", "nitidez3.png"),
                                           ("Laplaciano 3×3", "laplaciano3.png"))]
    if faltan := [str(r.name) for _, r in paneles if not r.is_file()]:
        raise FaltanDatos("faltan " + ", ".join(faltan))
    fig, ejes = plt.subplots(2, 4, figsize=(9, 4.9))
    for col, (nombre, archivo) in enumerate(paneles):
        with Image.open(archivo) as imagen:
            gris = imagen.convert("L")
            ancho, alto = gris.size
            x, y = (ancho - recorte) // 2, (alto - recorte) // 2
            ejes[0, col].imshow(gris, cmap="gray", vmin=0, vmax=255)
            ejes[0, col].add_patch(plt.Rectangle((x, y), recorte, recorte, fill=False, edgecolor="#C44E52", lw=1.5))
            ejes[1, col].imshow(gris.crop((x, y, x + recorte, y + recorte)), cmap="gray", vmin=0, vmax=255,
                                interpolation="nearest")
        ejes[0, col].set_title(nombre, loc="left")
        for fila in (0, 1):
            ejes[fila, col].set_axis_off()
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


def escribir_resumen(filas, ruta: Path):
    grupos = defaultdict(list)
    for f in filas:
        if f["estado"] == "ok" and f["calentamiento"] == "no":
            grupos[(f["experimento"], f["modo"], f["filtro"], f["mascara"], f["resolucion"],
                    f["concurrencia"])].append(f)
    with open(ruta, "w", newline="", encoding="utf-8") as salida:
        w = csv.writer(salida)
        w.writerow(["experimento", "modo", "filtro", "mascara", "resolucion", "concurrencia", "metrica_ms",
                    "n", "mediana", "p25", "p75", "cv_pct", "min", "max"])
        for clave, g in grupos.items():
            for m in METRICAS:
                e = est(f["t"][m] for f in g)
                if e:
                    w.writerow([*clave, m, e["n"], *(round(e[k], 4) for k in
                                                       ("mediana", "p25", "p75", "cv", "min", "max"))])


# ── Gráficas ────────────────────────────────────────────────────────────────────────────────────

def _desglose(ax, grupos: list[tuple[str, list[dict]]], titulo: str):
    """Barras horizontales apiladas: mediana de cada componente; total del cliente rotulado al final."""
    usados = set()
    for y, (_, g) in enumerate(grupos):
        izquierda = 0.0
        for clave, nombre, color, trama in SEGMENTOS:
            v = mediana(g, clave)
            if not v:
                continue
            ax.barh(y, v, left=izquierda, height=0.6, color=color, hatch=trama, edgecolor="white",
                    linewidth=1.0, label=nombre if clave not in usados else None)
            usados.add(clave)
            izquierda += v
        total = mediana(g, "cliente") or mediana(g, "gpu_total")
        ax.text(izquierda, y, f"  {ms(total)} ms", va="center", fontsize=8)
    ax.set_yticks(range(len(grupos)), [n for n, _ in grupos])
    ax.invert_yaxis()
    ax.set_xlabel("Tiempo (ms, mediana de cada componente)")
    ax.set_title(titulo, loc="left")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.15)


def _leyenda(fig, ax):
    """Leyenda en el orden del procesamiento, solo con los componentes que aparecen."""
    presentes = dict(zip(*reversed(ax.get_legend_handles_labels())))
    nombres = [n for _, n, _, _ in SEGMENTOS if n in presentes]
    fig.legend([presentes[n] for n in nombres], nombres, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))


def grafica_desglose(filas, ruta, experimento, grupos, titulo):
    datos = [(n, estable(filas, experimento=experimento, **filtro)) for n, filtro in grupos]
    datos = [(n, g) for n, g in datos if g]
    if not datos:
        raise FaltanDatos(f"no hay filas de {experimento}")
    fig, ax = plt.subplots(figsize=(7.5, 1.0 + 0.55 * len(datos) + 0.8))
    _desglose(ax, datos, titulo)
    _leyenda(fig, ax)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


def grafica_concurrencia(filas, rondas, ruta):
    datos = [(f"{c} cliente{'s' if c > 1 else ''}", estable(filas, experimento="concurrencia", concurrencia=c))
             for c in (1, 2)]
    datos = [(n, g) for n, g in datos if g]
    if not datos or not rondas:
        raise FaltanDatos("no hay filas de concurrencia")
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.5, 3.2), gridspec_kw={"width_ratios": [2.2, 1]})
    _desglose(a, datos, "Latencia por solicitud")
    for i, c in enumerate((1, 2)):
        tasas = [numero(r["exitosas_por_s"]) for r in rondas if r["concurrencia"] == str(c)]
        if tasas:
            b.scatter([i] * len(tasas), tasas, s=40, color="#4C72B0", edgecolor="white", linewidth=1.5, zorder=3)
            b.hlines(statistics.median(tasas), i - 0.25, i + 0.25, color="#1F1F1F", linewidth=2)
    b.set_xticks([0, 1], ["1 cliente", "2 clientes"])
    b.set_xlim(-0.6, 1.6)
    b.set_ylim(bottom=0)
    b.set_ylabel("Solicitudes exitosas por segundo")
    b.set_title("Rendimiento por ronda (línea = mediana)", loc="left")
    _leyenda(fig, a)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


# ── Principal ───────────────────────────────────────────────────────────────────────────────────

def ultima_corrida() -> Path:
    corridas = sorted(p for p in (RAIZ / "resultados").glob("2*") if (p / "mediciones.csv").is_file())
    if not corridas:
        sys.exit("No hay corridas en resultados/. Pasa la carpeta como argumento.")
    return corridas[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("corrida", nargs="?", type=Path)
    p.add_argument("--salida", type=Path, default=RAIZ / "resultados")
    args = p.parse_args()
    corrida = args.corrida or ultima_corrida()
    filas = [preparar(f) for f in leer_csv(corrida / "mediciones.csv")]
    if not filas:
        sys.exit(f"No hay mediciones en {corrida}")
    rondas = leer_csv(corrida / "rondas.csv")
    entorno = json.loads((corrida / "entorno.json").read_text(encoding="utf-8"))
    tablas, graficas = args.salida / "tablas", args.salida / "graficas"
    tablas.mkdir(parents=True, exist_ok=True)
    graficas.mkdir(parents=True, exist_ok=True)

    modos = [(NOMBRE_MODO[m], {"modo": m}) for m in NOMBRE_MODO]
    salidas = [
        (tablas / "tabla_entorno.tex", lambda r: tabla_entorno(entorno, r)),
        (tablas / "tabla_estado.tex", lambda r: tabla_estado(filas, r)),
        (tablas / "tabla_validacion.tex", lambda r: tabla_validacion(corrida, r)),
        (graficas / "evidencia.png", lambda r: grafica_evidencia(corrida, r)),
        (tablas / "tabla_frio.tex", lambda r: tabla_frio(filas, r)),
        (tablas / "tabla_integracion.tex", lambda r: tabla_integracion(filas, r)),
        (tablas / "tabla_resolucion.tex",
         lambda r: _tabla_servicio(filas, r, "resolucion", grupos_resolucion(), "Resolución")),
        (tablas / "tabla_filtros.tex",
         lambda r: _tabla_servicio(filas, r, "filtros", grupos_filtros(), "Filtro")),
        (tablas / "tabla_concurrencia.tex", lambda r: tabla_concurrencia(filas, rondas, r)),
        (graficas / "integracion.png", lambda r: grafica_desglose(
            filas, r, "integracion", modos, "Media 9×9, 4000×4000, 256 hilos")),
        (graficas / "resolucion.png", lambda r: grafica_desglose(
            filas, r, "resolucion", grupos_resolucion(), "Servicio, media 9×9, 256 hilos")),
        (graficas / "filtros.png", lambda r: grafica_desglose(
            filas, r, "filtros", grupos_filtros(), "Servicio, 4000×4000, 256 hilos")),
        (graficas / "concurrencia.png", lambda r: grafica_concurrencia(filas, rondas, r)),
    ]
    escribir_resumen(filas, corrida / "resumen.csv")
    print(f"Corrida: {corrida}\n  {corrida / 'resumen.csv'}")
    for ruta, generar in salidas:
        try:
            generar(ruta)
            print(f"  {ruta.relative_to(args.salida)}")
        except FaltanDatos as motivo:
            ruta.unlink(missing_ok=True)
            print(f"  (omitida) {ruta.name}: {motivo}")


if __name__ == "__main__":
    main()
