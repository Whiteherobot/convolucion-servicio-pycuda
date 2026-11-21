# Datos

Mismas entradas de la práctica 02 (y de la P01): `practica01/datos/original.jpg` recortada y reducida
con `practica01/scripts/preparar_imagenes.py` del repo
[convolucion-pycuda](https://github.com/Whiteherobot/convolucion-pycuda). Los `.pgm` no se suben (pesan
hasta 36 MB); se generan así y se comprueban con su hash:

```bash
git clone https://github.com/Whiteherobot/convolucion-pycuda.git /tmp/p02
(cd /tmp/p02 && python3 practica01/scripts/preparar_imagenes.py practica01/datos/original.jpg)
cp /tmp/p02/datos/imagen_*.pgm datos/
sha256sum -c datos/SHA256SUMS
```
