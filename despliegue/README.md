# Docker y despliegue (Bloque 2 - Jordy Romero)

## Por qué esta imagen

| Decisión | Motivo |
|---|---|
| Base `nvidia/cuda:13.4.3-devel-ubuntu24.04` (etiqueta fija) | PyCUDA compila el kernel al arrancar y necesita `nvcc`; el `-runtime` no lo trae. |
| `python3-dev` y `build-essential` | `pip` compila PyCUDA al instalarlo. |
| Entorno virtual en `/opt/venv` | Ubuntu 24.04 no permite `pip install` fuera de un entorno virtual. |
| Usuario `servicio` (no root) | El servicio no necesita privilegios. |
| `gunicorn -w 1 --threads 8` | Un solo proceso: el contexto CUDA vive en un único hilo trabajador. Los 8 hilos son los del contrato del Bloque 1. |
| `HEALTHCHECK` sobre `/api/salud` | `healthy` solo si la respuesta trae `gpu` distinto de `null`: `/api/salud` responde 200 desde que arranca gunicorn, pero la GPU carga después. Periodo de gracia de 60 s. |
| Base con digest | La etiqueta de una imagen se puede mover; el digest no. |

La imagen pesa unos 14 GB: la base `-devel` ya son 13.4 GB. No se redujo con multi-etapa porque `nvcc`
tiene que estar en tiempo de ejecución.

## Controlador del anfitrión y CUDA de la imagen

Son dos cosas distintas y no tienen que coincidir:

- **Controlador** (`nvidia-smi`): vive en el anfitrión y el contenedor lo usa a través de NVIDIA Container
  Toolkit. Aquí: `620.73.04`.
- **CUDA de la imagen** (`nvcc --version` dentro del contenedor): `13.4`. El anfitrión tiene `nvcc` 13.5, que el
  contenedor no usa.

Funciona porque el controlador del anfitrión debe ser igual o más nuevo que la CUDA de la imagen.

Configurar el toolkit (una vez; en esta laptop ya estaba hecho):

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info | grep -i runtimes     # debe listar "nvidia"
```

## Comandos

Todos desde la raíz del repositorio.

```bash
# 1. Construir (tarda unos 4 minutos: compila PyCUDA)
docker build -t convolucion-servicio:p03 .

# 2. Comprobar que el contenedor ve la GPU
docker run --rm --gpus all convolucion-servicio:p03 nvidia-smi

# 3. Iniciar el servicio
docker run -d --name conv-p03 --gpus all --memory=4g -p 8000:8000 --restart unless-stopped convolucion-servicio:p03

# 4. Comprobar (healthy cuando termina de arrancar)
docker ps --filter name=conv-p03
curl http://localhost:8000/api/salud
curl -F imagen=@datos/imagen_2000.pgm -F filtro=media -F mascara=9 http://localhost:8000/api/procesar

# Detener y borrar
docker rm -f conv-p03
```

Entorno completo para el informe (anfitrión + imagen + commit): `despliegue/entorno.sh`.

## Despliegue

El servicio no tiene autenticación ni límite de solicitudes: cualquiera en la red puede usar la GPU. Se deja abierto
solo mientras dura la demostración (`--memory=4g` acota la memoria del contenedor).

Se despliega en la laptop de Jordy (sin costo de nube), con acceso desde otro equipo de la red.

| | |
|---|---|
| CPU | Intel Core i7-11800H @ 2.30 GHz |
| RAM | 15 GiB |
| GPU | NVIDIA GeForce RTX 3080 Laptop, 8 GiB |
| SO | Omarchy (Arch), kernel 7.3.4-1-cachyos |
| Controlador | 620.73.04 |
| Docker | 29.10.0 |
| NVIDIA Container Toolkit | 1.20.1 |
| Python (imagen) | 3.12.3 |
| PyCUDA | 2026.1 |
| CUDA (imagen) | 13.4 (nvcc 13.4.92) |
| Puerto | 8000 |
| Acceso | `http://<IP de la laptop>:8000` desde la misma red |

La limitación de usar una laptop: el servicio solo está disponible mientras el equipo esté encendido y en la
misma red, y la GPU se comparte con lo que Jordy tenga abierto.

Para entrar desde otro equipo hay que permitir el puerto en el cortafuegos del anfitrión y usar la IP de la
laptop (`ip -4 addr`). Se comprueba desde el otro equipo con `curl http://<IP>:8000/api/salud`.

## Evidencia

- Contenedor `conv-p03` en estado `healthy`, corriendo como usuario `servicio` (uid 999).
- Solicitud real por el servicio en contenedor: media 9×9 sobre `imagen_2000.pgm`, `kernel` ≈ 0.9 ms,
  `h2d` ≈ 2 ms, `d2h` ≈ 3.4 ms.
- `pytest` contra el contenedor: 25 pasan y 1 fallo esperado (ver `pruebas/README.md`).
- **Acceso desde otro equipo:** funciona desde un teléfono en la misma red Wi-Fi, con
  `http://<IP de la laptop>:8000/api/salud` y la interfaz en `http://<IP de la laptop>:8000/`.

## Construcción desde un clon limpio

Clon nuevo de la rama `bloque-2-docker`, construido con `docker build --no-cache` (sin reutilizar capas):

| | |
|---|---|
| Commit | `bb95932` |
| Imagen base | `nvidia/cuda:13.4.3-devel-ubuntu24.04` (`sha256:5626a7d0ecafe3e7c115a452f6b05f20f214f8509346196a7835b205dc5b7fd9`) |
| Imagen construida | `convolucion-servicio:p03-limpio`, `sha256:ea0ca126916cfb4f4bf183a18c54316a3e666174dadca3089ffa5dbd54a2de92` |
| Tamaño | 13.8 GB |
| nvcc / Python / PyCUDA | 13.4.92 / 3.12.3 / 2026.1 |
| Resultado | contenedor `healthy`; `pytest`: 25 pasan y 1 fallo esperado |

El digest cambia en cada construcción (las capas llevan marcas de tiempo), así que no sirve para comparar
dos builds: lo que identifica la versión es el commit y el digest de la imagen base.
