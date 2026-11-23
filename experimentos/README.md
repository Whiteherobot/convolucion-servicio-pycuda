# Experimentos (Bloque 3 - Andrés Morocho)

Se miden **en la laptop de Jordy** (RTX 3080 Laptop, controlador 620.73, imagen `cuda:13.4.3-devel`), la
misma del despliegue documentado en [`despliegue/README.md`](../despliegue/README.md). No se mezclan corridas
de otros equipos: cada fila del CSV lleva la etiqueta del entorno y cada corrida su `entorno.json`.

| Archivo | Qué hace |
|---|---|
| `entorno.py` | Verifica antes de medir (hash de las imágenes, servicio con GPU, versiones) y anota el entorno. |
| `cliente.py` | Una solicitud: `perf_counter` desde el envío hasta tener el PNG completo (incluye la descarga). |
| `correr.py` | Arranque en frío y los 4 experimentos. Escribe `resultados/<fecha-hora>/`. |
| `graficas.py` | Andrés, con la corrida: `resumen.csv`, `resultados/tablas/*.tex` y `resultados/graficas/*.png` para `informe/informe.tex`. Necesita `requirements-graficas.txt`. |
| `test_experimentos.py` | Prueba de los scripts contra un servicio **falso** (sin GPU): `pytest experimentos/test_experimentos.py`. |

## Procedimiento (Jordy): un solo comando

La rama `bloque-3-experimentos` ya trae los tres bloques juntos.

```bash
git fetch && git checkout bloque-3-experimentos && git pull
bash experimentos/ejecutar_todo.sh
git push
```

Hace todo en orden y se detiene con un mensaje en rojo si algo falla: imágenes de prueba (las genera si
faltan), `.venv`, `docker build`, servicio, pytest, validación y experimentos. Al final hace commit de
`resultados/<corrida>/` y solo queda `git push`. Si algo falla: no hacer push y enviar la salida en rojo tal
cual. Antes: cargador conectado, perfil de rendimiento y aplicaciones cerradas. Tarda unos 20–30 minutos
(la mitad es la primera construcción de la imagen).

### Lo mismo, paso a paso (solo si hace falta repetir una parte)

```bash
# 0. Código e imágenes de prueba (datos/README.md)
git fetch && git checkout bloque-3-experimentos && git pull
sha256sum -c datos/SHA256SUMS                       # las tres deben decir OK

# 1. Imagen y servicio (mismos comandos del Bloque 2)
docker build -t convolucion-servicio:p03 .
docker rm -f conv-p03 2>/dev/null
docker run -d --name conv-p03 --gpus all --memory=4g -p 8000:8000 convolucion-servicio:p03

# 2. Python del anfitrión, con PyCUDA (para el modo directo_host)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PATH=/opt/cuda/bin:$PATH                     # nvcc del anfitrión

# 3. Verificar: debe terminar en "Listo para medir."
python experimentos/entorno.py

# 4. Medir (unos 10 minutos; no usar la laptop mientras tanto)
python experimentos/correr.py
```

Antes del paso 4: **laptop conectada al cargador, perfil de energía en rendimiento, navegador, juegos y
videollamadas cerrados**. La GPU de una laptop se comparte con la pantalla y baja su velocidad con el calor;
eso va al informe como limitación, pero no debe empeorarse durante la corrida.

Si algo falla, no repetir con otros parámetros ni con `--forzar`: mandar `errores.log` y la salida tal cual.

Opciones útiles: `--experimentos integracion,resolucion` (solo algunos), `--omitir-modos directo_host`
(si PyCUDA no funciona en el anfitrión; se documenta y **no** se atribuyen diferencias a Docker).

## Qué se mide

256 hilos por bloque fijos. Por configuración: 1 calentamiento (`repeticion = 0`, `calentamiento = si`, se
guarda pero no entra en las estadísticas) y 5 repeticiones. Un factor a la vez.

| Experimento | Configuración | Modos |
|---|---|---|
| `frio` | Reinicia el contenedor 3 veces; primera solicitud (media 9×9, 4000²). Registra `contexto_ms`, `compilacion_ms` y `arranque_ms` (desde `docker restart` hasta que `/api/salud` ve la GPU) | servicio |
| `integracion` | Media 9×9, 4000² | `directo_host`, `directo_contenedor`, `servicio` |
| `resolucion` | Media 9×9; 2000², 4000², 6000² | servicio |
| `filtros` | 4000²; media 3, 9, 21; nitidez; laplaciano | servicio |
| `concurrencia` | Media 9×9, 4000²; 1 y 2 clientes que envían a la vez, 5 rondas | servicio |

- **Alcances:** `kernel_ms`, `h2d_ms`, `d2h_ms`, `gpu_total_ms` son eventos CUDA (memoria paginable);
  `servidor_ms` va desde que llega la solicitud hasta tener el PNG (reloj monotónico del servidor);
  `cliente_ms` va desde el envío hasta tener el PNG descargado (reloj del cliente). Los relojes de cliente y
  servidor no se restan entre sí.
- **Modos directos:** `python -m app.directo` en el anfitrión y en un contenedor nuevo de la misma imagen
  (con `datos/` montado de solo lectura). Solo traen tiempos de GPU: no decodifican por HTTP ni codifican PNG.
- **Concurrencia:** rendimiento = solicitudes exitosas / duración total de la ronda (`rondas.csv`). Antes se
  reinicia el contenedor para empezar con el directorio temporal vacío.
- **El cliente corre en la misma laptop** (`localhost`): la red no entra en `cliente_ms`. Se declara en el
  informe.
- Fallos e incompletas quedan en el CSV con `estado = error` y en `errores.log`; nunca se estiman.
