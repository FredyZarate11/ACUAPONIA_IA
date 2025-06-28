# train.py

import keras
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
import time
import os
import random
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression

from config import *
from data_cleaner import clean_dataset
from data_processing import process_data
from model import dnn
from evaluation import calculate_metrics, plot_evaluation_results, generate_report, evaluate_prediction

def set_global_seeds(seed_value):
    """Establece semillas aleatorias para reproducibilidad."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def main():
    """Función principal para ejecutar el flujo de trabajo de ML."""
    if SEED is not None:
        set_global_seeds(SEED)
    
    # --- 1. LIMPIEZA DE DATOS (Paso Opcional) ---
    if APPLY_CLEANING:
        all_columns_to_clean = list(set(FEATURE_COLUMNS + [TARGET_COLUMN]))
        success = clean_dataset(
            raw_path=RAW_DATA_PATH,
            cleaned_path=CLEANED_DATA_PATH,
            datetime_col=DATETIME_COLUMN,
            columns_to_clean=all_columns_to_clean,
            cleaning_bounds=CLEANING_BOUNDS
        )
        if not success:
            return
        data_path = CLEANED_DATA_PATH
    else:
        print("Omitiendo el paso de limpieza de datos.")
        data_path = RAW_DATA_PATH

    
    try:
        processed_data = process_data(
            file_path=data_path,
            feature_cols=FEATURE_COLUMNS,
            target_col=TARGET_COLUMN,
            date_col=DATETIME_COLUMN
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error durante el preprocesamiento: {e}")
        return

    # --- 3. PREPARACIÓN PARA EL ENTRENAMIENTO ---
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data['X_scaled'],
        processed_data['y_scaled'],
        test_size=TEST_SIZE,
        random_state=SEED
    )
    
    input_shape = X_train.shape[1]
    model = dnn(input_shape)
    
    print("\n--- Entrenando Modelo: ANN Simple ---")
    model.summary()
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_DELTA,  # type: ignore
        verbose=1,
        restore_best_weights=True
    )
    
    # --- 4. ENTRENAMIENTO DEL MODELO ---
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SIZE,
        verbose=2,
        shuffle=True,
        callbacks=[early_stopping, keras.callbacks.TerminateOnNaN()]
    )
    end_time = time.time()
    training_duration = end_time - start_time
    
    # --- 5. EVALUACIÓN DEL MODELO ---
    print("\n--- Evaluando el Modelo en el Conjunto de Prueba ---")
    y_pred_scaled = model.predict(X_test, verbose=0)
    scaler_Y = processed_data['scaler_Y']

    y_test_original = scaler_Y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = scaler_Y.inverse_transform(y_pred_scaled).flatten()

    metrics = calculate_metrics(y_test_original, y_pred_original)
    
    print("\n--- RESULTADOS EN ESCALA ORIGINAL ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # --- 6. GENERACIÓN DE INFORME Y VISUALIZACIÓN ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(REPORT_FOLDER, f'evaluacion_{timestamp}.png')
    report_path = os.path.join(REPORT_FOLDER, f'informe_{timestamp}.md')
    
    plot_evaluation_results(
        model_name=MODEL_TO_USE,
        history=history,
        y_test_original=y_test_original,
        y_pred_original=y_pred_original,
        metrics=metrics,
        save_path=image_path
    )

    generate_report(
        model=model,
        model_name=MODEL_TO_USE,
        metrics=metrics,
        config=globals(), # Pasamos todas las variables globales de config.py
        image_path=image_path,
        report_path=report_path,
        epochs_run=len(history.history['loss']),
        training_duration=training_duration
    )
    print("Semilla Usada:", SEED)
    
    # --- 7. SIMULACIÓN DE PREDICCIÓN A FUTURO ---
    print("\n\n--- Iniciando simulación de predicción a futuro ---")
    try:
        dias_a_predecir = int(input("Ingrese el número de días a predecir (ejemplo: 7): "))
        
        X_original = processed_data['X_original']
        ultimo_dia_registrado = X_original['Dia_Cultivo'].max()
        dia_objetivo = ultimo_dia_registrado + dias_a_predecir
        
        future_values = {'Dia_Cultivo': dia_objetivo}
        
        for col in FEATURE_COLUMNS:
            if col == 'Dia_Cultivo':
                continue
            
            # Asumimos que la población se mantiene constante
            if col == 'Population':
                last_population = X_original.sort_values(by='Dia_Cultivo')['Population'].iloc[-1]
                future_values[col] = last_population
                continue
            
            # Para las demás variables, usamos regresión lineal simple contra 'Dia_Cultivo'
            X_hist = X_original[['Dia_Cultivo']]
            y_hist = X_original[col]
            linear_model = LinearRegression().fit(X_hist, y_hist)
            predicted_val = linear_model.predict(np.array([[dia_objetivo]]))
            future_values[col] = predicted_val[0]

        future_data = pd.DataFrame([future_values])[FEATURE_COLUMNS] # Asegurar orden
        
        print("\nDatos de entrada (proyectados) para la predicción:")
        print(future_data)
        
        future_data_scaled = processed_data['scaler_X'].transform(future_data)
        predicted_weight_scaled = model.predict(future_data_scaled)
        predicted_weight_original = scaler_Y.inverse_transform(predicted_weight_scaled)
        
        print("\n--- RESULTADO DE LA PREDICCIÓN ---")
        print(f"El peso estimado para el día {dia_objetivo} es: {predicted_weight_original[0][0]:.2f} gramos.")
        print("------------------------------------")
        
        prediction_plot_path = os.path.join(REPORT_FOLDER, f'prediccion_{timestamp}.png')
        evaluate_prediction(processed_data, dia_objetivo, predicted_weight_original, save_path=prediction_plot_path)

    except (ValueError, IndexError) as e:
        print(f"\nNo se pudo completar la predicción a futuro. Error: {e}")

    print("\n--- Proceso completado. ---")

if __name__ == "__main__":
    main()