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
| `test_aislamiento.py` | Dos solicitudes a la vez reciben cada una su resultado; con la cola llena, 503. |

## Hallazgos para el Bloque 1

1. **Imagen truncada responde 500, no 400.** `validar()` solo lee la cabecera; el error sale después, en la GPU
   (`image file is truncated`). La prueba `test_imagen_truncada_responde_400` está marcada `xfail(strict=True)`:
   cuando se corrija pasará a fallar y habrá que quitarle la marca.
2. **El 503 por cola llena no se alcanza con el arranque oficial.** La cola admite 4 en espera + 1 en la GPU,
   pero `gunicorn --threads 4` deja entrar solo 4 solicitudes a la vez; el resto espera en el socket. Medido:
   12 solicitudes simultáneas contra el contenedor dan 12 respuestas 200. Por eso la prueba usa la app en el
   mismo proceso con una GPU falsa que tarda 0.3 s.
