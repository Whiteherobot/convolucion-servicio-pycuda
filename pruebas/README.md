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
| `test_entradas_invalidas.py` | No es imagen, imagen truncada, formato no permitido, imagen o archivo demasiado grande, filtro, máscara e hilos inválidos: código, mensaje y servicio vivo. |
| `test_aislamiento.py` | Dos solicitudes a la vez reciben cada una su resultado; con 12 a la vez y la cola llena, 503 y el servicio sigue. |
| `test_limpieza.py` | Varios hilos limpiando resultados vencidos a la vez, sin fallar. |
