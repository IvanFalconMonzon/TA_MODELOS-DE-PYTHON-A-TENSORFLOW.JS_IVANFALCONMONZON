// ------------------- Sección 1: Definición de variables -------------------

// Variable global para almacenar el modelo cargado
let modelo;

// ------------------- Sección 2: Función para cargar el modelo -------------------

async function cargarModelo() {
  // Cargar el modelo exportado en formato TensorFlow.js
  modelo = await tf.loadLayersModel('modelo_flor_js/model.json');
  console.log("Modelo cargado correctamente");
}

// ------------------- Sección 3: Función para predecir la clase de la imagen -------------------

async function predecir() {
  // Verificar si el modelo ya está cargado
  if (!modelo) {
    document.getElementById("resultado").innerText = "Cargando el modelo, por favor espere...";
    return;
  }

  // Mostrar mensaje de carga mientras se realiza la predicción
  document.getElementById("resultado").innerText = "Cargando...";

  // Obtener la imagen cargada por el usuario
  const imagen = document.getElementById("imagenSeleccionada");

  // Preprocesar la imagen para que tenga el formato adecuado para el modelo
  const tensorImagen = tf.browser.fromPixels(imagen)
    .resizeNearestNeighbor([128, 128]) // Redimensionar a 128x128 píxeles
    .toFloat()                         // Convertir a tipo float
    .expandDims(0)                     // Añadir dimensión extra para el batch
    .div(tf.scalar(255));              // Normalizar valores de píxel entre 0 y 1

  // Realizar la predicción
  const prediccion = modelo.predict(tensorImagen);
  const prediccionArray = await prediccion.data();

  // Determinar la clase con mayor probabilidad
  const clasePredicha = prediccionArray.indexOf(Math.max(...prediccionArray));

  // Definición de las clases en el mismo orden en que el modelo fue entrenado
  const clases = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"];

  // Mostrar el resultado de la predicción en la página
  document.getElementById("resultado").innerText = `La flor es: ${clases[clasePredicha]}`;
}

// ------------------- Sección 4: Eventos de interacción con el usuario -------------------

// Asociar la función de predicción al botón
document.getElementById("predecirBtn").onclick = predecir;

// Manejar el evento de selección de imagen
document.getElementById("imagenInput").onchange = (e) => {
  const file = e.target.files[0];

  // Verificar que el archivo seleccionado sea una imagen
  const fileType = file.type;
  if (!fileType.startsWith("image/")) {
    alert("Por favor selecciona una imagen.");
    return;
  }

  // Cargar y mostrar la imagen seleccionada
  if (file) {
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = document.getElementById("imagenSeleccionada");
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  }
};

// ------------------- Sección 5: Llamada inicial para cargar el modelo -------------------

// Cargar el modelo automáticamente cuando se carga la página
cargarModelo();