import os
import subprocess
import sys

# ------------------- Sección 1: Instalación de paquetes requeridos -------------------

# Lista de paquetes necesarios para ejecutar el script
requerimientos = [
    "tensorflow==2.11.0",
    "tensorflowjs==3.18.0",
    "numpy==1.23.5",
    "ipykernel",
    "matplotlib",
    "pandas",
    "scikit-learn"
]

# Función para instalar los paquetes que no estén disponibles en el entorno
def instalar_paquetes():
    for paquete in requerimientos:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", paquete], check=True)
        except subprocess.CalledProcessError:
            print(f"Error instalando {paquete}")

# Ejecutar la función para instalar los paquetes
instalar_paquetes()

# ------------------- Sección 2: Importación de librerías -------------------

import tensorflow as tf
import numpy as np
import tensorflowjs as tfjs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ------------------- Sección 3: Generación de datos -------------------

# Crear una serie de valores en Fahrenheit entre -100 y 212 grados
fahrenheit = np.linspace(-100, 212, 1000, dtype=np.float32)

# Convertir esos valores a Celsius usando la fórmula de conversión
celsius = (fahrenheit - 32) * 5 / 9

# Crear un DataFrame y guardarlo como archivo CSV
data = pd.DataFrame({'Fahrenheit': fahrenheit, 'Celsius': celsius})
data.to_csv('dataset_temperaturas.csv', index=False)
print("Dataset guardado como 'dataset_temperaturas.csv'")

# ------------------- Sección 4: División de datos -------------------

# Dividir los datos en entrenamiento+validación (80%) y prueba (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    fahrenheit, celsius, test_size=0.2, random_state=42
)

# Dividir el conjunto de entrenamiento+validación en entrenamiento (95%) y validación (5%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.05, random_state=42
)

# ------------------- Sección 5: Creación y entrenamiento del modelo -------------------

# Crear un modelo secuencial con una capa oculta y una capa de salida
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),          # Entrada de un valor (temperatura)
    tf.keras.layers.Dense(64, activation='relu'), # Capa oculta con 64 neuronas
    tf.keras.layers.Dense(1)                     # Capa de salida con 1 neurona (temperatura en Celsius)
])

# Compilar el modelo especificando el optimizador y la función de pérdida
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar el modelo con los datos de entrenamiento y validación
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    verbose=1
)

# ------------------- Sección 6: Visualización de entrenamiento -------------------

plt.figure(figsize=(12, 5))

# Gráfico de la pérdida (MSE)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title("Pérdida (MSE)")
plt.xlabel("Épocas")
plt.ylabel("Error Cuadrático Medio")
plt.legend()

# Gráfico del error absoluto medio (MAE)
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Entrenamiento')
plt.plot(history.history['val_mae'], label='Validación')
plt.title("Error Absoluto Medio (MAE)")
plt.xlabel("Épocas")
plt.ylabel("MAE")
plt.legend()

# Mostrar los gráficos
plt.tight_layout()
plt.show()

# ------------------- Sección 7: Exportación del modelo a TensorFlow.js -------------------

# Ruta de salida para el modelo exportado
export_dir = 'modelo_temperatura_js'

# Eliminar carpeta existente si ya hay un modelo previo
if os.path.exists(export_dir):
    import shutil
    shutil.rmtree(export_dir)

# Exportar el modelo entrenado en formato compatible con TensorFlow.js
tfjs.converters.save_keras_model(model, export_dir)
print("Modelo exportado a TensorFlow.js en:", export_dir)