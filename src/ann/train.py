import keras
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
import time

from data_processing import process_data
from models import ann_simple
from evaluation import calculate_metrics, plot_evaluation_results, generate_report
import tensorflow as tf
import numpy as np
import random
import os

def set_global_seeds(seed_value=42):
    """Fija las semillas aleatorias para garantizar la reproducibilidad."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

# --- 1. CONFIGURACIÓN DEL EXPERIMENTO ---
SEED_VALUE = 42
set_global_seeds(SEED_VALUE)

config = {
    'DATA_PATH': './data/annData/Tank_combined_v3.csv',
    'TARGET_COLUMN': 'Fish_Weight(g)',
    'DATE_COLUMN': 'Datetime',
    'FEATURE_COLUMNS': [
        'Temperature(C)', 'Turbidity(NTU)', 'Dissolved_Oxygen(g/ml)', 
        'PH', 'Ammonia(g/ml)', 'Nitrate(g/ml)', 'Population'
    ],
    'TEST_SIZE': 0.2,
    'RANDOM_STATE': SEED_VALUE,
    'EPOCHS': 1000,
    'BATCH_SIZE': 32
}

def main():
    """Función principal para ejecutar el flujo de trabajo de ML."""
    
    # ... (El preprocesamiento y la división de datos no cambian) ...
    try:
        processed_data = process_data(
            file_path=config['DATA_PATH'],
            feature_cols=config['FEATURE_COLUMNS'],
            target_col=config['TARGET_COLUMN'],
            date_col=config['DATE_COLUMN']
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error durante el preprocesamiento: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        processed_data['X_scaled'], 
        processed_data['y_scaled'], 
        test_size=config['TEST_SIZE'],
        random_state=config['RANDOM_STATE']
    )
    
    input_shape = X_train.shape[1]
    model = ann_simple(input_shape)
    
    print("\n--- Entrenando Modelo: ANN Simple ---")
    model.summary()
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, min_delta=1e-4, verbose=1, restore_best_weights=True
    )
    
    # --- MEDICIÓN DE TIEMPO: INICIO ---
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        batch_size=config['BATCH_SIZE'],
        epochs=config['EPOCHS'],
        validation_split=0.2,
        verbose=2,
        shuffle=True,
        callbacks=[early_stopping]
    )
    
    # --- MEDICIÓN DE TIEMPO: FIN ---
    end_time = time.time()
    training_duration = end_time - start_time
    
    # Obtenemos el número real de épocas ejecutadas
    epochs_run = len(history.history['loss'])

    # ... (La evaluación del modelo no cambia) ...
    print("\n--- Evaluando el Modelo en el Conjunto de Prueba ---")
    
    y_pred_scaled = model.predict(X_test, verbose=0)
    scaler_Y = processed_data['scaler_Y']
    y_test_original = scaler_Y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = scaler_Y.inverse_transform(y_pred_scaled).flatten()
    metrics = calculate_metrics(y_test_original, y_pred_original)
    
    print("\n--- RESULTADOS EN ESCALA ORIGINAL ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # --- GENERACIÓN DE INFORME Y VISUALIZACIÓN ---
    print("\n--- Generando informe de resultados ---")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_folder = './results/ann_reports'
    image_path = os.path.join(report_folder, f'grafico_{timestamp}.png')
    report_path = os.path.join(report_folder, f'informe_{timestamp}.md')
    
    plot_evaluation_results(
        model_name="ANN Simple",
        history=history,
        y_test_original=y_test_original,
        y_pred_original=y_pred_original,
        metrics=metrics,
        save_path=image_path
    )

    # NUEVO: Se pasan la duración y las épocas al generador del informe
    generate_report(
        model=model,
        model_name="ANN Simple",
        metrics=metrics,
        config=config,
        image_path=image_path,
        report_path=report_path,
        epochs_run=epochs_run,
        training_duration=training_duration
    )

    print("\n--- Proceso de entrenamiento y evaluación completado. ---")

if __name__ == "__main__":
    main()
