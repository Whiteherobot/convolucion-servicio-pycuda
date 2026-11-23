"""Prueba del cliente y del CSV contra un servicio FALSO (sin GPU). Bloque 3 - Andrés Morocho.

Solo comprueba la plomería de los scripts (contrato, CSV, rondas, errores, lectura de app.directo).
No mide nada real: el servicio falso existe únicamente en esta prueba.

    pytest experimentos/test_experimentos.py
"""
import csv
import hashlib
import io
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from flask import Flask, jsonify, request
from werkzeug.serving import make_server

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cliente  # noqa: E402
import correr  # noqa: E402
import entorno  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32
DIRECTO_FALSO = '''import json, sys
from pathlib import Path
import numpy as np
np.save(Path(sys.argv[2]).with_suffix(".npy"), np.full((2, 3), 7.0, dtype=np.float32))
print(json.dumps({"contexto_ms": 50.0, "compilacion_ms": 900.0}))
print(json.dumps({"calentamiento": True, "h2d": 9.0, "kernel": 3.0, "d2h": 8.0, "gpu_total": 20.0}))
for r in range(1, int(sys.argv[sys.argv.index("--repeticiones") + 1]) + 1):
    print(json.dumps({"repeticion": r, "h2d": 9.0, "kernel": 3.0, "d2h": 8.0, "gpu_total": 20.0}))
'''


@pytest.fixture()
def servicio_falso():
    app = Flask(__name__)
    contador = [0]

    @app.get("/api/salud")
    def salud():
        return jsonify(gpu="GPU falsa", error=None, contexto_ms=1.0, compilacion_ms=2.0)

    @app.post("/api/procesar")
    def procesar():
        if request.form["filtro"] not in ("media", "nitidez", "laplaciano"):
            return jsonify(error="Filtro desconocido"), 400
        contador[0] += 1
        i = f"{contador[0]:032x}"
        tiempos = dict.fromkeys(cliente.CAMPOS_SERVIDOR, 1.5)
        return jsonify(id=i, bloques=62500, tiempos_ms=tiempos, url_resultado=f"/api/resultado/{i}.png")

    @app.get("/api/resultado/<i>.npy")
    def crudo(i):
        salida = io.BytesIO()
        np.save(salida, np.full((2, 3), 7.0, dtype=np.float32))
        return salida.getvalue(), 200, {"Content-Type": "application/octet-stream"}

    @app.get("/api/resultado/<i>.png")
    def resultado(i):
        return PNG, 200, {"Content-Type": "image/png"}

    servidor = make_server("127.0.0.1", 0, app, threaded=True)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{servidor.server_port}"
    servidor.shutdown()


@pytest.fixture()
def raiz_falsa(tmp_path, monkeypatch):
    """Repositorio mínimo: .pgm pequeños con su SHA256SUMS y un app/directo.py falso."""
    (tmp_path / "datos").mkdir()
    lineas = []
    for lado in (2000, 4000, 6000):
        contenido = f"P5 {lado}".encode()
        (tmp_path / "datos" / f"imagen_{lado}.pgm").write_bytes(contenido)
        lineas.append(f"{hashlib.sha256(contenido).hexdigest()}  datos/imagen_{lado}.pgm")
    (tmp_path / "datos" / "SHA256SUMS").write_text("\n".join(lineas) + "\n")
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("")
    (tmp_path / "app" / "directo.py").write_text(DIRECTO_FALSO)
    monkeypatch.setattr(entorno, "RAIZ", tmp_path)
    monkeypatch.setattr(correr, "RAIZ", tmp_path)
    return tmp_path


def leer(ruta):
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_cliente_ok_y_error(servicio_falso):
    fila = cliente.enviar(servicio_falso, b"x", "a.pgm", "media", 9)
    assert fila["estado"] == "ok" and fila["kernel_ms"] == 1.5 and fila["cliente_ms"] > 0
    fila = cliente.enviar(servicio_falso, b"x", "a.pgm", "desconocido", 3)
    assert fila["estado"] == "error" and fila["codigo_http"] == 400 and "Filtro" in fila["error"]


def test_sin_servicio_no_mide(raiz_falsa, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["correr.py", "--url", "http://127.0.0.1:9", "--imagen", "",
                                      "--contenedor", ""])
    with pytest.raises(SystemExit) as salida:
        correr.main()
    assert "No se mide" in str(salida.value)


def test_hash_distinto_es_problema(raiz_falsa):
    (raiz_falsa / "datos" / "imagen_4000.pgm").write_bytes(b"otra")
    problemas = []
    entorno.verificar_datos(problemas)
    assert any("imagen_4000" in p for p in problemas)


def test_corrida_completa(servicio_falso, raiz_falsa, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "correr.py", "--url", servicio_falso, "--imagen", "", "--contenedor", "",
        "--omitir-modos", "directo_contenedor", "--repeticiones", "3", "--rondas", "2",
        "--experimentos", "validacion,integracion,resolucion,filtros,concurrencia"])
    correr.main()
    carpeta = next((raiz_falsa / "resultados").iterdir())
    filas = leer(carpeta / "mediciones.csv")
    rondas = leer(carpeta / "rondas.csv")
    assert json.loads((carpeta / "entorno.json").read_text())["salud"]["gpu"] == "GPU falsa"

    directas = [f for f in filas if f["modo"] == "directo_host"]
    assert [f["repeticion"] for f in directas] == ["0", "1", "2", "3"]
    assert directas[0]["calentamiento"] == "si" and directas[1]["kernel_ms"] == "3.0"
    assert directas[1]["compilacion_ms"] == "900.0" and directas[1]["bloques"] == "62500"

    por_exp = {e: [f for f in filas if f["experimento"] == e] for e in ("integracion", "resolucion", "filtros")}
    assert len(por_exp["integracion"]) == 4 + 4          # directo_host + servicio, 1 calentamiento + 3
    assert len(por_exp["resolucion"]) == 3 * 4
    assert len(por_exp["filtros"]) == 5 * 4
    assert all(f["estado"] == "ok" and f["entorno"] for f in filas)

    concurrentes = [f for f in filas if f["experimento"] == "concurrencia" and f["ronda"]]
    assert len(concurrentes) == 2 * 1 + 2 * 2           # 2 rondas con 1 cliente y 2 rondas con 2
    assert [(r["concurrencia"], r["exitosas"]) for r in rondas] == [("1", "1"), ("1", "1"), ("2", "2"), ("2", "2")]
    assert all(float(r["exitosas_por_s"]) > 0 for r in rondas)
    assert not (carpeta / "errores.log").exists()

    validacion = leer(carpeta / "validacion.csv")
    assert [(v["filtro"], v["estado"], v["cumple"], v["error_max"]) for v in validacion] == [
        ("media", "ok", "si", "0.0"), ("nitidez", "ok", "si", "0.0"), ("laplaciano", "ok", "si", "0.0")]
    assert {p.name for p in (carpeta / "evidencia").iterdir()} == {"media9.png", "nitidez3.png", "laplaciano3.png"}


def test_arranque_en_frio(servicio_falso, raiz_falsa, monkeypatch):
    """Con contenedor: reinicia (simulado) y registra contexto, compilación y tiempo hasta estar listo."""
    monkeypatch.setattr(correr.Experimentos, "reiniciar", lambda self: 4321.0)
    monkeypatch.setattr(sys, "argv", ["correr.py", "--url", servicio_falso, "--imagen", "",
                                      "--contenedor", "conv-falso", "--experimentos", "frio", "--reinicios", "2"])
    correr.main()
    filas = leer(next((raiz_falsa / "resultados").iterdir()) / "mediciones.csv")
    assert [(f["experimento"], f["repeticion"], f["arranque_ms"], f["compilacion_ms"]) for f in filas] == [
        ("frio", "1", "4321.0", "2.0"), ("frio", "2", "4321.0", "2.0")]


def test_graficas_de_una_corrida(servicio_falso, raiz_falsa, monkeypatch, tmp_path):
    """graficas.py sobre una corrida real del servicio falso: tablas y PNG sin errores; frío se omite."""
    pytest.importorskip("matplotlib")
    import graficas
    monkeypatch.setattr(sys, "argv", ["correr.py", "--url", servicio_falso, "--imagen", "", "--contenedor", "",
                                      "--omitir-modos", "directo_contenedor", "--repeticiones", "2",
                                      "--rondas", "2", "--experimentos", "integracion,filtros,concurrencia"])
    correr.main()
    corrida = next((raiz_falsa / "resultados").iterdir())
    salida = tmp_path / "salida"
    monkeypatch.setattr(sys, "argv", ["graficas.py", str(corrida), "--salida", str(salida)])
    graficas.main()
    tablas = {p.name for p in (salida / "tablas").iterdir()}
    assert {"tabla_entorno.tex", "tabla_integracion.tex", "tabla_filtros.tex", "tabla_concurrencia.tex"} <= tablas
    assert "tabla_frio.tex" not in tablas and "tabla_resolucion.tex" not in tablas
    assert {"integracion.png", "filtros.png", "concurrencia.png"} <= {p.name for p in (salida / "graficas").iterdir()}
    assert (corrida / "resumen.csv").is_file()
