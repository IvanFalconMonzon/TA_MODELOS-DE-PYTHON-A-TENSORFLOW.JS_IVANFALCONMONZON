// conversion.js

// Variable global que almacenará el modelo cargado
let modelo;

// Carga el modelo de TensorFlow.js desde una URL local (carpeta modelo_temperatura_js)
async function cargarModelo() {
  try {
    modelo = await tf.loadLayersModel('modelo_temperatura_js/model.json');
    console.log("Modelo cargado correctamente");
  } catch (error) {
    console.error("Error cargando modelo:", error);
    // Mostrar error en el elemento HTML con id 'resultado'
    document.getElementById("resultado").innerText = "Error cargando modelo: " + error.message;
  }
}

// Función que se ejecuta al hacer clic en el botón "Convertir"
async function predecir() {
  if (!modelo) {
    // Si el modelo aún no se ha cargado, mostrar mensaje de advertencia
    document.getElementById("resultado").innerText = "Modelo no cargado aún.";
    return;
  }

  // Obtener el valor introducido por el usuario
  const input = parseFloat(document.getElementById("fahrenheitInput").value);

  // Validar que el valor ingresado sea un número
  if (isNaN(input)) {
    document.getElementById("resultado").innerText = "Ingrese un valor numérico válido.";
    return;
  }

  // Crear un tensor a partir del valor ingresado
  const tensorEntrada = tf.tensor2d([input], [1, 1]);

  // Usar el modelo para hacer una predicción
  const tensorSalida = modelo.predict(tensorEntrada);

  // Extraer el resultado del tensor
  const resultado = (await tensorSalida.data())[0];

  // Mostrar el resultado en la página
  document.getElementById("resultado").innerText = `Celsius: ${resultado.toFixed(2)} °C`;
}

// Llamar automáticamente a la función para cargar el modelo al inicio
cargarModelo();