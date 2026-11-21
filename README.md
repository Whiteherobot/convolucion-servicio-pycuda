# Servicio web de convolución con PyCUDA y Docker

Práctica 03 · Computación Paralela · Universidad Politécnica Salesiana

Continuación de la práctica 02 ([convolucion-pycuda](https://github.com/Whiteherobot/convolucion-pycuda),
etiqueta `entrega`, commit `1c04426`; el código reutilizado se copió de `d5ab309`, que es el mismo código
reagrupado en carpetas). El kernel, las máscaras y el manejo de imágenes se reutilizan sin cambios en
`app/modelo/`; lo nuevo es el servicio web, la gestión de solicitudes, Docker, el despliegue y la evaluación.

**Integrantes:** Michael Lata, Jordy Romero y Andrés Morocho.

```text
Navegador -> Flask (app/controlador) -> cola GPU (app/modelo/solicitudes.py) -> PyCUDA -> GPU -> PNG -> navegador
```

## Reparto

| Bloque | Responsable | Rama | Archivos |
|---|---|---|---|
| 1. Aplicación web, servicio HTTP y gestión de solicitudes GPU | Michael Lata | `bloque-1-web` | `app/` |
| 2. Docker, despliegue y validación | Jordy Romero | `bloque-2-docker` | `Dockerfile`, `.dockerignore`, `pruebas/`, `despliegue/` |
| 3. Experimentos, CSV, gráficas, informe y presentación | Andrés Morocho | `bloque-3-experimentos` | `experimentos/`, `resultados/`, `informe/` |

Cada uno trabaja en su rama y abre un Pull Request hacia `main`. No cambiar el contrato de abajo sin avisar.

### Bloque 2 — Jordy Romero

- [ ] `Dockerfile`: imagen base CUDA con etiqueta fija (p. ej. `nvidia/cuda:12.x.y-devel-ubuntu24.04`, hace falta `nvcc`
      porque PyCUDA compila el kernel), Python, `requirements.txt`, y arranque con `gunicorn` (comando de abajo),
      sin depurador ni recarga. `.dockerignore` (sin `.venv`, `datos/*.pgm`, `resultados/`, `.git`).
- [ ] Distinguir en el informe el controlador del anfitrión (`nvidia-smi`) de la CUDA de la imagen (`nvcc --version`
      dentro del contenedor). Configurar NVIDIA Container Toolkit (`nvidia-ctk runtime configure --runtime=docker`).
- [ ] Comandos exactos: construir, comprobar GPU (`docker run --rm --gpus all <imagen> nvidia-smi`) e iniciar.
      Construir desde un clon limpio y anotar imagen base, versiones, `docker images --digests` y commit.
- [ ] Despliegue: nube con GPU NVIDIA si hay créditos; si no, la laptop de Michael (RTX 4060) documentando la
      limitación y entrando desde otro equipo de la red. Documentar CPU, RAM, GPU, SO, controlador, Docker,
      Python, PyCUDA y CUDA, puerto y forma de acceso. Evidencia: contenedor corriendo + solicitud que usa PyCUDA.
- [ ] `pruebas/` (pytest): matriz pequeña y límites (prueba manual de la P02 y una imagen 1001×777),
      equivalencia directa–servicio de los tres filtros (error máximo y medio con `/api/resultado/<id>.npy`,
      tolerancia `validacion.TOLERANCIA`), entradas inválidas (archivo no imagen, demasiado grande, filtro
      desconocido, máscara no permitida → código y mensaje, y que el servicio siga respondiendo) y aislamiento
      (dos solicitudes distintas a la vez reciben cada una su resultado; con la cola llena responde 503).

### Bloque 3 — Andrés Morocho

- [ ] `experimentos/cliente.py`: cliente HTTP que mide con `time.perf_counter()` desde el envío hasta tener el PNG
      completo (incluye la descarga de `url_resultado`), y guarda también los tiempos del servidor de la respuesta.
- [ ] Los 4 experimentos de la guía (un factor a la vez, 256 hilos fijos): integración (PyCUDA directo en el
      anfitrión con `python -m app.directo`, directo dentro del contenedor y por el servicio; media 9×9, 4000×4000),
      resolución (2000, 4000, 6000; media 9×9), filtros y máscara (4000; media 3, 9, 21; nitidez; laplaciano),
      concurrencia (4000, media 9×9; 1 y 2 clientes simultáneos, 5 rondas, solicitudes exitosas por segundo).
- [ ] Calentamiento + al menos 5 repeticiones; mediana y variabilidad. La primera solicitud (arranque en frío)
      va aparte: `/api/salud` da `contexto_ms` y `compilacion_ms`.
- [ ] CSV con: id de solicitud, modo (`directo_host`, `directo_contenedor`, `servicio`), filtro, resolución, máscara,
      hilos, bloques, repetición, concurrencia, todos los tiempos en ms, estado y entorno. Registro de errores.
- [ ] Gráficas, informe PDF (preguntas de análisis, firmas, enlace al repo y commit) y diapositivas.

## Contrato (bloque 1, para los bloques 2 y 3)

Arranque del servicio. Un solo proceso: el contexto CUDA vive en un único hilo trabajador que atiende la GPU
en orden de llegada (FIFO). Los 8 hilos de gunicorn reciben solicitudes; como máximo 1 está en la GPU y 4 esperan,
la siguiente recibe 503:

```bash
gunicorn -w 1 --threads 8 -b 0.0.0.0:8000 "app.servidor:crear_app()"
```

| Ruta | Qué hace |
|---|---|
| `GET /` | Interfaz: subir imagen, elegir filtro y máscara, ver original y resultado, descargar |
| `POST /api/procesar` | `multipart/form-data`: `imagen` (archivo), `filtro` (`media`, `nitidez`, `laplaciano`), `mascara` (3, 9 o 21; solo media; los otros son 3×3), `hilos` (opcional, 256) |
| `GET /api/resultado/<id>.png` | Resultado para ver o descargar (recortado a 0–255) |
| `GET /api/resultado/<id>.npy` | Resultado `float32` sin recortar, para comparar con PyCUDA directo |
| `GET /api/salud` | GPU, `contexto_ms`, `compilacion_ms`, solicitudes en cola y límites |

Respuesta correcta (200):

```json
{"id": "a1b2...", "filtro": "media", "mascara": 9, "ancho": 4000, "alto": 4000, "hilos": 256, "bloques": 62500,
 "tiempos_ms": {"espera_cola": 0.1, "decodificacion": 0.0, "h2d": 0.0, "kernel": 0.0, "d2h": 0.0,
                "gpu_total": 0.0, "codificacion": 0.0, "servidor": 0.0},
 "url_resultado": "/api/resultado/a1b2....png", "url_crudo": "/api/resultado/a1b2....npy"}
```

- `h2d`, `kernel`, `d2h`, `gpu_total`: eventos CUDA (memoria paginable; reserva y liberación fuera, como en la P02).
- `decodificacion`, `codificacion`, `espera_cola` y `servidor` (desde que llega la solicitud hasta tener el PNG):
  `time.perf_counter()`.
- Errores: `{"error": "mensaje"}` con 400 (entrada inválida), 413 (archivo o imagen demasiado grande),
  503 (cola llena o GPU no disponible), 500 (falla del procesamiento).
- Límites: archivo ≤ 40 MB, imagen ≤ 6000×6000, formatos reales PNG, JPEG, PGM/PPM, BMP o TIFF, cola de 4.
  Los resultados viven en un directorio temporal por solicitud y se borran a los 10 minutos.

PyCUDA directo, sin HTTP (mismo `procesar` que usa el servicio):

```bash
python -m app.directo datos/imagen_4000.pgm resultados/media9.png media 9 --hilos 256 --repeticiones 5
```

## Cómo ejecutarlo (sin Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# imágenes de prueba: ver datos/README.md
gunicorn -w 1 --threads 8 -b 0.0.0.0:8000 "app.servidor:crear_app()"
# http://localhost:8000
curl -F imagen=@datos/imagen_2000.pgm -F filtro=media -F mascara=9 http://localhost:8000/api/procesar
```

Docker, despliegue y experimentos: ver las secciones que agreguen los bloques 2 y 3.

## Licencia

MIT. Ver [LICENSE](LICENSE).
