/* Interacción de la interfaz. Bloque 1 - Michael Lata. */
const formulario = document.getElementById("formulario");
const filtro = document.getElementById("filtro");
const mascara = document.getElementById("mascara");
const boton = document.getElementById("procesar");
const errorCaja = document.getElementById("error");

filtro.addEventListener("change", function () {
  if (filtro.value === "media") {
    mascara.disabled = false;
  } else {
    mascara.value = "3";
    mascara.disabled = true;
  }
});

formulario.addEventListener("submit", async function (evento) {
  evento.preventDefault();
  const archivo = document.getElementById("imagen").files[0];
  errorCaja.hidden = true;
  if (!archivo) {
    errorCaja.textContent = "Seleccione una imagen.";
    errorCaja.hidden = false;
    return;
  }
  if (archivo.size > 40 * 1024 * 1024) {
    errorCaja.textContent = "El archivo excede 40 MB.";
    errorCaja.hidden = false;
    return;
  }

  const inicioNavegador = performance.now();
  const datos = new FormData();
  datos.append("imagen", archivo);
  datos.append("filtro", filtro.value);
  datos.append("mascara", mascara.value);
  datos.append("hilos", document.getElementById("hilos").value);

  const original = document.getElementById("original");
  const sinVista = document.getElementById("sin-vista");
  const urlOriginal = URL.createObjectURL(archivo);
  original.onload = function () {
    original.hidden = false;
    sinVista.hidden = true;
  };
  original.onerror = function () {
    original.hidden = true;
    sinVista.hidden = false;
  };
  original.src = urlOriginal;

  boton.disabled = true;
  boton.textContent = "Procesando...";
  try {
    const respuesta = await fetch("/api/procesar", {
      method: "POST",
      body: datos
    });
    const cuerpo = await respuesta.json();
    if (!respuesta.ok) {
      throw new Error(cuerpo.error || "Error en la solicitud");
    }

    const imagenResultado = document.getElementById("resultado");
    await new Promise(function (resolver, rechazar) {
      imagenResultado.onload = resolver;
      imagenResultado.onerror = rechazar;
      imagenResultado.src = cuerpo.url_resultado;
    });
    imagenResultado.hidden = false;
    document.getElementById("detalles").hidden = false;
    document.getElementById("id").textContent = cuerpo.id;
    document.getElementById("dimensiones").textContent =
      cuerpo.ancho + " x " + cuerpo.alto;
    document.getElementById("hilos-detalle").textContent = cuerpo.hilos;
    document.getElementById("bloques").textContent = cuerpo.bloques;
    document.getElementById("navegador").textContent =
      (performance.now() - inicioNavegador).toFixed(3);

    const descarga = document.getElementById("descarga");
    descarga.href = cuerpo.url_resultado;
    descarga.download = "resultado_" + cuerpo.id + ".png";

    const alcances = {
      espera_cola: "cola (desde que se encola hasta tomarla)",
      decodificacion: "decodificación de la imagen",
      h2d: "CPU → GPU (eventos CUDA)",
      kernel: "solo el kernel (eventos CUDA)",
      d2h: "GPU → CPU (eventos CUDA)",
      gpu_total: "transferencias + kernel (eventos CUDA)",
      codificacion: "generación del PNG",
      servidor: "llegada de la solicitud → PNG listo"
    };
    const tabla = document.getElementById("tabla-tiempos");
    tabla.textContent = "";
    for (const nombre in cuerpo.tiempos_ms) {
      const fila = document.createElement("tr");
      const celdaNombre = document.createElement("td");
      const celdaAlcance = document.createElement("td");
      const celdaValor = document.createElement("td");
      celdaNombre.textContent = nombre;
      celdaAlcance.textContent = alcances[nombre];
      celdaValor.textContent = cuerpo.tiempos_ms[nombre].toFixed(3);
      fila.append(celdaNombre, celdaAlcance, celdaValor);
      tabla.appendChild(fila);
    }
  } catch (error) {
    errorCaja.textContent = error.message;
    errorCaja.hidden = false;
  } finally {
    boton.disabled = false;
    boton.textContent = "Procesar";
  }
});
