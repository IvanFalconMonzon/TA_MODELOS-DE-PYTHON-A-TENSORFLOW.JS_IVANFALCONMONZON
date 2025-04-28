# ------------------- Sección 1: Instalación de dependencias -------------------

import subprocess
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

# Función para instalar las dependencias listadas en 'requirements.txt'
def instalar_dependencias():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencias instaladas correctamente.")
    except subprocess.CalledProcessError:
        print("Error al instalar las dependencias.")

# Llamar a la función para instalar dependencias automáticamente al inicio
instalar_dependencias()

# ------------------- Sección 2: Importación de librerías adicionales -------------------

from tensorflow.keras.utils import image_dataset_from_directory
import tensorflowjs as tfjs

# ------------------- Sección 3: Configuración del dataset -------------------

# Ruta donde se encuentra el dataset de flores
DATASET_DIR = 'data/flowers'

# Parámetros de configuración
IMG_SIZE = (128, 128)    # Tamaño de las imágenes
BATCH_SIZE = 32          # Tamaño del batch

# ------------------- Sección 4: Carga de datasets de entrenamiento y validación -------------------

# Cargar el dataset de entrenamiento (80% del total)
dataset_entrenamiento = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,    # Reservar 20% para validación
    subset="training",       # Subconjunto de entrenamiento
    seed=123,                # Semilla para reproducibilidad
    image_size=IMG_SIZE,     # Redimensionar imágenes
    batch_size=BATCH_SIZE
)

# Cargar el dataset de validación (20% del total)
dataset_validacion = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ------------------- Sección 5: Normalización de imágenes -------------------

# Crear una capa de normalización para escalar los píxeles de [0,255] a [0,1]
normalizar = tf.keras.layers.Rescaling(1./255)

# Aplicar normalización a los conjuntos de datos
dataset_entrenamiento = dataset_entrenamiento.map(lambda x, y: (normalizar(x), y))
dataset_validacion = dataset_validacion.map(lambda x, y: (normalizar(x), y))

# ------------------- Sección 6: Definición del modelo CNN -------------------

# Crear el modelo de red neuronal convolucional
modelo = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),     # Capa de dropout para reducir overfitting
    tf.keras.layers.Dense(5, activation='softmax')  # Capa de salida para 5 clases
])

# ------------------- Sección 7: Compilación del modelo -------------------

# Compilar el modelo especificando el optimizador y la función de pérdida
modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------- Sección 8: Entrenamiento del modelo -------------------

# Entrenar el modelo utilizando los datasets de entrenamiento y validación
modelo.fit(
    dataset_entrenamiento,
    epochs=10,
    validation_data=dataset_validacion
)

# ------------------- Sección 9: Guardado del modelo entrenado -------------------

# Guardar el modelo entrenado en formato .h5
modelo.save('modelo_flor_cnn.h5')

# ------------------- Sección 10: Exportación del modelo a TensorFlow.js -------------------

# Exportar el modelo en formato compatible con TensorFlow.js
tfjs.converters.save_keras_model(modelo, 'modelo_flor_js')

# ------------------- Sección 11: Mensaje final -------------------

# Mostrar mensaje de éxito
print("Modelo entrenado y exportado a TensorFlow.js correctamente.")