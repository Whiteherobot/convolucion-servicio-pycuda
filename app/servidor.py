"""Fábrica de la aplicación Flask. Bloque 1 - Michael Lata."""
from flask import Flask, jsonify
from werkzeug.exceptions import RequestEntityTooLarge

from app.controlador.rutas import rutas
from app.modelo.solicitudes import ColaGPU, ErrorSolicitud, MAX_BYTES


def crear_app():
    """Crea una aplicación con una única cola GPU."""
    app = Flask(
        __name__,
        template_folder="vista/templates",
        static_folder="vista/static",
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES + 1024 * 1024
    cola = ColaGPU()
    app.extensions["cola_gpu"] = cola
    app.register_blueprint(rutas)

    @app.errorhandler(ErrorSolicitud)
    def manejar_error_solicitud(error):
        return jsonify(error=str(error)), error.codigo

    @app.errorhandler(RequestEntityTooLarge)
    def manejar_archivo_grande(error):
        return jsonify(error="Archivo demasiado grande"), 413

    return app


if __name__ == "__main__":
    crear_app().run(host="0.0.0.0", port=8000, debug=False)
