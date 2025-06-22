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
from config import *

config = {
    'N_EXPERIMENTS': N_EXPERIMENTS,
    'SEED': SEED,
    'DATA_PATH': DATA_PATH,
    'APPLY_CLEANING': APPLY_CLEANING,
    'TARGET_COLUMN': TARGET_COLUMN,
    'FEATURE_COLUMNS': FEATURE_COLUMNS,
    'DATETIME_COLUMN': DATETIME_COLUMN,
    'MODEL_TO_USE': MODEL_TO_USE,
    'TEST_SIZE': TEST_SIZE,
    'VALIDATION_SIZE': VALIDATION_SIZE,
    'EPOCHS': EPOCHS,
    'BATCH_SIZE': BATCH_SIZE,
    'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
    'EARLY_STOPPING_DELTA': EARLY_STOPPING_DELTA
}

def set_global_seeds(config: dict):
    if config['SEED'] is not None:
        os.environ['PYTHONHASHSEED'] = str(config['SEED'])
        random.seed(config['SEED'])
        np.random.seed(config['SEED'])
        tf.random.set_seed(config['SEED'])
    else:
        seed = random.randint(0, 10000)
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        config['SEED'] = seed

def main():
    """Función principal para ejecutar el flujo de trabajo de ML."""
    set_global_seeds(config)
    try:
        processed_data = process_data(
            file_path=config['DATA_PATH'],
            feature_cols=config['FEATURE_COLUMNS'],
            target_col=config['TARGET_COLUMN'],
            date_col=config['DATETIME_COLUMN']
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error durante el preprocesamiento: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        processed_data['X_scaled'], 
        processed_data['y_scaled'], 
        test_size=config['TEST_SIZE'],
        random_state=config['SEED']
    )
    
    input_shape = X_train.shape[1]
    model = ann_simple(input_shape)
    
    print("\n--- Entrenando Modelo: ANN Simple ---")
    model.summary()
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=config['EARLY_STOPPING_PATIENCE'], min_delta=config['EARLY_STOPPING_DELTA'], verbose=1, restore_best_weights=True
    )
    
    # Time measurement for training
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        batch_size=config['BATCH_SIZE'],
        epochs=config['EPOCHS'],
        validation_split=config['VALIDATION_SIZE'],
        verbose=2,
        shuffle=True,
        callbacks=[early_stopping, keras.callbacks.TerminateOnNaN() if config['SEED'] is not None else None]
    )
    
    # Time ended
    end_time = time.time()
    training_duration = end_time - start_time
    
    epochs_run = len(history.history['loss'])

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
    report_folder = './results/annResults'
    image_path = os.path.join(report_folder, f'grafico_{timestamp}.png')
    report_path = os.path.join(report_folder, f'informe_{timestamp}.md')
    
    plot_evaluation_results(
        model_name=config['MODEL_TO_USE'],
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
    print("Semilla Usada:", config['SEED'])
    print("\n--- Proceso de entrenamiento y evaluación completado. ---")

if __name__ == "__main__":
    main()
