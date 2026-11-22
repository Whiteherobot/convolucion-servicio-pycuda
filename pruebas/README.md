# Pruebas (Bloque 2 - Jordy Romero)

Hablan con el servicio por HTTP, así que sirven contra el contenedor o contra gunicorn en el anfitrión.
Los `.pgm` de `datos/` se generan según `datos/README.md`.

```bash
docker run -d --name conv-p03 --gpus all -p 8000:8000 convolucion-servicio:p03
export PATH=/opt/cuda/bin:$PATH          # PyCUDA directo en este proceso necesita nvcc
pytest -v                                # SERVICIO_URL=http://otra-ip:8000 para otra dirección
```

| Archivo | Qué comprueba |
|---|---|
| `test_manual.py` | Matriz 5×5 de la figura 1 (laplaciano, valores a mano: 8 y -60) y bordes con ceros. |
| `test_limites.py` | Imagen 1001×777 con 128, 256 y 512 hilos: tamaño, bloques y cobertura hasta el último píxel. |
| `test_equivalencia.py` | PyCUDA directo contra el `.npy` del servicio, tres filtros, error máximo y medio bajo `TOLERANCIA`. |
| `test_entradas_invalidas.py` | No es imagen, formato no permitido, imagen o archivo demasiado grande, filtro, máscara e hilos inválidos: código, mensaje y servicio vivo. |
| `test_aislamiento.py` | Dos solicitudes a la vez reciben cada una su resultado; con 12 a la vez y la cola llena, 503 y el servicio sigue. |
| `test_limpieza.py` | Varios hilos limpiando resultados vencidos a la vez (hoy falla: error del Bloque 1). |

## Hallazgos para el Bloque 1

Las pruebas marcadas `xfail` documentan errores del Bloque 1 con su parche en el [README principal](../README.md):

1. `test_imagen_truncada_responde_400`: la imagen truncada responde 500 en vez de 400.
2. `test_varios_hilos_pueden_limpiar_a_la_vez`: `limpiar_viejos()` falla si dos hilos borran el mismo archivo.

Al corregirlos hay que quitar la marca `xfail`.
