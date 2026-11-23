"""Rutas HTTP del servicio. Bloque 1 - Michael Lata."""
import time

from flask import Blueprint, current_app, jsonify, render_template, request, send_file

from app.modelo.solicitudes import ErrorSolicitud, validar


rutas = Blueprint("rutas", __name__)


@rutas.route("/", methods=["GET"])
def inicio():
    return render_template("index.html")


@rutas.route("/api/procesar", methods=["POST"])
def procesar_solicitud():
    cola = current_app.extensions["cola_gpu"]
    inicio = time.perf_counter()
    cola.limpiar_viejos()

    archivo = request.files.get("imagen")
    if archivo is None:
        raise ErrorSolicitud(400, "Falta el archivo de imagen")
    datos = archivo.read()

    filtro, k, hilos = validar(
        datos,
        request.form.get("filtro"),
        request.form.get("mascara"),
        request.form.get("hilos"),
    )

    respuesta = cola.enviar(datos, filtro, k, hilos)
    respuesta["tiempos_ms"]["servidor"] = round(
        (time.perf_counter() - inicio) * 1000, 3)
    return jsonify(respuesta)


@rutas.route("/api/resultado/<id_texto>.png", methods=["GET"])
def resultado_png(id_texto):
    cola = current_app.extensions["cola_gpu"]
    cola.limpiar_viejos()
    ruta = cola.ruta_resultado(id_texto, "png")
    if ruta is None:
        return jsonify(error="Resultado no encontrado"), 404
    return send_file(ruta, mimetype="image/png",
                     download_name="resultado_" + id_texto + ".png")


@rutas.route("/api/resultado/<id_texto>.npy", methods=["GET"])
def resultado_npy(id_texto):
    cola = current_app.extensions["cola_gpu"]
    cola.limpiar_viejos()
    ruta = cola.ruta_resultado(id_texto, "npy")
    if ruta is None:
        return jsonify(error="Resultado no encontrado"), 404
    return send_file(ruta, mimetype="application/octet-stream",
                     download_name="resultado_" + id_texto + ".npy")


@rutas.route("/api/salud", methods=["GET"])
def salud():
    cola = current_app.extensions["cola_gpu"]
    cola.limpiar_viejos()
    return jsonify(cola.salud())
