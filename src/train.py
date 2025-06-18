import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from data_processing import process_data
from models import ann_simple
from evaluation import calculate_metrics, plot_evaluation_results
import tensorflow as tf
import numpy as np
import random
import os

def set_global_seeds(seed_value=42):
    """
    Fija las semillas aleatorias para Python, NumPy y TensorFlow 
    para garantizar la reproducibilidad.
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


# --- 1. CONFIGURACIÓN DEL EXPERIMENTO ---
DATA_PATH = './data/Tank_combined_v3.csv'
SEED = 42
set_global_seeds(SEED) # Puedes usar el mismo valor de RANDOM_STATE
TARGET_COLUMN = 'Fish_Weight(g)'
DATE_COLUMN = 'Datetime'
FEATURE_COLUMNS = [
    'Temperature(C)', 
    'Turbidity(NTU)', 
    'Dissolved_Oxygen(g/ml)', 
    'PH', 
    'Ammonia(g/ml)', 
    'Nitrate(g/ml)', 
    'Population'
]
TEST_SIZE = 0.2
RANDOM_STATE = SEED
EPOCHS = 1000
BATCH_SIZE = 32

def main():
    """Función principal para ejecutar el flujo de trabajo de ML."""
    
    # --- 2. PROCESAMIENTO DE DATOS ---
    try:
        processed_data = process_data(
            file_path=DATA_PATH,
            feature_cols=FEATURE_COLUMNS,
            target_col=TARGET_COLUMN,
            date_col=DATE_COLUMN
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error durante el preprocesamiento: {e}")
        return

    # Guardar datos procesados para análisis futuro (opcional)
    pd.concat([processed_data['X_original'], processed_data['y_original']], axis=1).to_csv('./data/processed_data.csv', index=False)

    # --- 3. DIVISIÓN DE DATOS ---
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data['X_scaled'], 
        processed_data['y_scaled'], 
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    # --- 4. DEFINICIÓN Y ENTRENAMIENTO DEL MODELO ---
    input_shape = X_train.shape[1]
    model = ann_simple(input_shape)
    
    print("\n--- Entrenando Modelo: ANN Simple ---")
    model.summary()
    
    # * Callbacks
    # Early stopping para evitar sobreajuste (para cuando la validación no mejora)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        min_delta=1e-4, # type: ignore (0.0001 es un valor común para tolerancia)
        verbose=1,
        restore_best_weights=True
    )



    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=2, # type: ignore
        shuffle=True,
        callbacks=[early_stopping]
    )

    # --- 5. EVALUACIÓN DEL MODELO ---
    print("\n--- Evaluando el Modelo en el Conjunto de Prueba ---")
    
    # Predecir con datos normalizados
    y_pred_scaled = model.predict(X_test, verbose='1')
    
    # Desnormalizar para interpretar resultados
    scaler_Y = processed_data['scaler_Y']
    y_test_original = scaler_Y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = scaler_Y.inverse_transform(y_pred_scaled).flatten()

    # Calcular métricas
    metrics = calculate_metrics(y_test_original, y_pred_original)
    
    # Imprimir resultados en consola
    print("\n--- RESULTADOS EN ESCALA ORIGINAL ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # --- 6. VISUALIZACIÓN DE RESULTADOS ---
    plot_evaluation_results(
        model_name="ANN Simple",
        history=history,
        y_test_original=y_test_original,
        y_pred_original=y_pred_original,
        metrics=metrics
    )

    print("\n--- Proceso de entrenamiento y evaluación completado. ---")

if __name__ == "__main__":
    main()