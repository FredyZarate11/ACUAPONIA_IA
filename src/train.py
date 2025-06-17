# Importa las funciones de tus otros archivos
import time
from data_processing import load_and_prepare_data
from evaluation import plot
from models import ann_simple
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- PASO 1: Cargar los datos ---
DATA_PATH = './data/Tank_combined_v3.csv'
X, Y, scaler_X, scaler_Y, Y = load_and_prepare_data(DATA_PATH)
X['Fish_Weight(g)'] = Y  # Aseguramos que Y esté en X para la consistencia
X.to_csv('./data/processed_data.csv', index=False)  # Guardar los datos procesados
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X.drop('Fish_Weight(g)', axis=1), Y, test_size=0.2, random_state=42)

# --- PASO 2: Entrenar y Evaluar el primer modelo ---
input_shape = X_train.shape[1]
model_1 = ann_simple(input_shape)
print("\n--- Entrenando Modelo 1: ANN Simple ---")
print(f"Arquitectura del modelo:\n{model_1.summary()}")
history_1 = model_1.fit(
    X_train, y_train, 
    epochs=200,
    batch_size=32,
    validation_split=0.2, 
    verbose='1',
    shuffle=True
    )

print("\n--- Evaluando Modelo ---")
y_pred_scaled = model_1.predict(X_test, verbose='0')

# Desnormalizar valores reales y predicciones
y_test_original = scaler_Y.inverse_transform(y_test.to_numpy().reshape(-1, 1)).flatten()
y_pred_original = scaler_Y.inverse_transform(y_pred_scaled).flatten()

# --- PASO 6: Calcular métricas en escala original ---
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

print(f"\n--- RESULTADOS EN ESCALA ORIGINAL ---")
print(f"MSE (Error Cuadrático Medio): {mse:.4f}")
print(f"RMSE (Raíz del Error Cuadrático): {rmse:.4f}")
print(f"MAE (Error Absoluto Medio): {mae:.4f}")
print(f"R² (Coeficiente de Determinación): {r2:.4f}")
print(f"Error relativo promedio: {(mae/y_test_original.mean())*100:.2f}%")

print(f"\nRango de valores reales: {y_test_original.min():.2f} a {y_test_original.max():.2f}")
print(f"Rango de predicciones: {y_pred_original.min():.2f} a {y_pred_original.max():.2f}")

plot(model_1, "ANN Simple", history_1, X_test, y_test, y_pred_original)
time.sleep(2000)  # Pausa para evitar problemas de visualización

print("\nEntrenamiento y evaluación completados.")
time.sleep(2000)  # Pausa para evitar problemas de visualización