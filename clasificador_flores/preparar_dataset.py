# ------------------- Sección 1: Instalación de dependencias -------------------

import subprocess
import sys

# Función para instalar las dependencias listadas en 'requirements.txt'
def instalar_dependencias():
    try:
        # Ejecutar el comando de instalación
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencias instaladas correctamente.")
    except subprocess.CalledProcessError:
        print("Error al instalar las dependencias.")

# Llamar a la función para instalar dependencias automáticamente al inicio
instalar_dependencias()

# ------------------- Sección 2: Importación de librerías -------------------

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory

# ------------------- Sección 3: Configuración inicial -------------------

# Ruta donde se encuentra el dataset de flores
DATASET_DIR = 'data/flowers'

# Parámetros de configuración
IMG_SIZE = (128, 128)    # Tamaño al que se redimensionarán las imágenes
BATCH_SIZE = 32          # Número de imágenes por lote (batch)
SEED = 123               # Semilla para reproducibilidad

# ------------------- Sección 4: Carga de los datasets de entrenamiento y validación -------------------

# Cargar el dataset de entrenamiento (80% del total)
dataset_entrenamiento = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,   # Reservar el 20% para validación
    subset="training",      # Subconjunto de entrenamiento
    seed=SEED,              # Semilla para reproducibilidad
    image_size=IMG_SIZE,    # Redimensionar imágenes
    batch_size=BATCH_SIZE   # Tamaño de batch
)

# Cargar el dataset de validación (20% del total)
dataset_validacion = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,   # Mismo porcentaje para validación
    subset="validation",    # Subconjunto de validación
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ------------------- Sección 5: Visualización de clases -------------------

# Obtener y mostrar los nombres de las clases detectadas en el dataset
clases = dataset_entrenamiento.class_names
print(f"Clases detectadas: {clases}")

# ------------------- Sección 6: Normalización de imágenes -------------------

# Crear una capa de normalización para escalar los píxeles de [0,255] a [0,1]
normalizar = tf.keras.layers.Rescaling(1./255)

# Aplicar la normalización tanto al conjunto de entrenamiento como de validación
dataset_entrenamiento = dataset_entrenamiento.map(lambda x, y: (normalizar(x), y))
dataset_validacion = dataset_validacion.map(lambda x, y: (normalizar(x), y))

# ------------------- Sección 7: Visualización de algunas muestras del dataset -------------------

# Función para mostrar 9 imágenes de muestra del dataset junto a su etiqueta
def mostrar_muestras(dataset, clases):
    plt.figure(figsize=(10, 8))
    for images, labels in dataset.take(1): # Tomar un batch de imágenes
        for i in range(9): # Mostrar las primeras 9 imágenes
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())    # Mostrar la imagen
            plt.title(clases[labels[i]])     # Mostrar la etiqueta correspondiente
            plt.axis("off")                  # Ocultar ejes
    plt.tight_layout()
    plt.show()

# Llamar a la función para visualizar ejemplos
mostrar_muestras(dataset_entrenamiento, clases)