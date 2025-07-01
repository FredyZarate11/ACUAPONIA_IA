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

# Importaciones locales
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

def run_cleaning():
    """Ejecuta solo el proceso de limpieza de datos."""
    print("--- Opción seleccionada: Limpiar Dataset ---")
    clean_dataset()
    print("--- Limpieza finalizada. ---")

def run_training():
    """Ejecuta el flujo completo de entrenamiento y evaluación."""
    print("--- Opción seleccionada: Entrenar Modelo ---")
    
    # --- 1. LIMPIEZA Y PREPROCESAMIENTO ---
    if not clean_dataset():
        return # Detener si la limpieza falla
    
    try:
        processed_data = process_data()
    except (ValueError, FileNotFoundError) as e:
        print(f"Error durante el preprocesamiento: {e}")
        return

    # --- 2. PREPARACIÓN PARA EL ENTRENAMIENTO ---
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data['X_scaled'],
        processed_data['y_scaled'],
        test_size=TEST_SIZE,
        random_state=SEED
    )
    
    input_shape = X_train.shape[1]
    model = dnn(input_shape)
    
    print(f"\n--- Entrenando Modelo: {MODEL_NAME} ---")
    model.summary()
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_DELTA,  # type: ignore
        verbose=1,
        restore_best_weights=True
    )
    
    # --- 3. ENTRENAMIENTO DEL MODELO ---
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SIZE,
        verbose=2, # type: ignore
        shuffle=True,
        callbacks=[early_stopping, keras.callbacks.TerminateOnNaN()]
    )
    end_time = time.time()
    training_duration = end_time - start_time
    
    # --- 4. GUARDAR MODELO ENTRENADO ---
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Modelo guardado en: {MODEL_SAVE_PATH}")

    # --- 5. EVALUACIÓN DEL MODELO ---
    print("\n--- Evaluando el Modelo en el Conjunto de Prueba ---")
    y_pred_scaled = model.predict(X_test, verbose=0) # type: ignore
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
        model_name=MODEL_NAME,
        history=history,
        y_test_original=y_test_original,
        y_pred_original=y_pred_original,
        metrics=metrics,
        save_path=image_path
    )

    generate_report(
        model=model,
        model_name=MODEL_NAME,
        metrics=metrics,
        config=globals(),
        image_path=image_path,
        report_path=report_path,
        epochs_run=len(history.history['loss']),
        training_duration=training_duration
    )
    print("Semilla Usada:", SEED)
    print("--- Entrenamiento finalizado. ---")

def run_prediction():
    """Carga un modelo y realiza una predicción a futuro."""
    print("--- Opción seleccionada: Realizar Predicción ---")
    if not IS_TIME_SERIES:
        print("La predicción a futuro solo está implementada para series temporales.")
        return
    try:
        model = keras.models.load_model(MODEL_SAVE_PATH)
        print(f"Modelo cargado desde {MODEL_SAVE_PATH}")
    except (IOError, ImportError) as e:
        print(f"Error al cargar el modelo: {e}")
        print("Asegúrate de haber entrenado un modelo primero (Opción 2).")
        return
    try:
        processed_data = process_data()
        dias_a_predecir = int(input("Ingrese el número de días a predecir (ejemplo: 7): "))

        X_original = processed_data['X_original']
        y_original = processed_data['scaler_Y'].inverse_transform(processed_data['y_scaled'].reshape(-1, 1))

        ultimo_dia_registrado = X_original['Dia_Cultivo'].max()
        dia_objetivo = ultimo_dia_registrado + dias_a_predecir

        # --- Simulación de peso con regresión lineal simple ---
        X_hist_dias = X_original[['Dia_Cultivo']]
        y_hist_pesos = y_original
        weight_regressor = LinearRegression().fit(X_hist_dias, y_hist_pesos)
        simulated_weight = weight_regressor.predict(np.array([[dia_objetivo]]))
        simulated_data_point = {'dia': dia_objetivo, 'peso': simulated_weight[0][0]}
        # ----------------------------------------------------

        future_values = {'Dia_Cultivo': dia_objetivo}

        feature_cols_for_pred = FEATURE_COLUMNS.copy()
        if 'Dia_Cultivo' not in feature_cols_for_pred:
             feature_cols_for_pred.append('Dia_Cultivo')

        for col in feature_cols_for_pred:
            if col == 'Dia_Cultivo':
                continue

            if col == 'Population':
                last_val = X_original.sort_values(by='Dia_Cultivo')[col].iloc[-1]
                future_values[col] = last_val
                continue

            X_hist = X_original[['Dia_Cultivo']]
            y_hist = X_original[col]
            linear_model = LinearRegression().fit(X_hist, y_hist)
            predicted_val = linear_model.predict(np.array([[dia_objetivo]]))
            future_values[col] = predicted_val[0]

        future_data = pd.DataFrame([future_values])[feature_cols_for_pred]

        print("\nDatos de entrada (proyectados) para la predicción:")
        print(future_data)

        future_data_scaled = processed_data['scaler_X'].transform(future_data)
        predicted_weight_scaled = model.predict(future_data_scaled)
        predicted_weight_original = processed_data['scaler_Y'].inverse_transform(predicted_weight_scaled)

        print("\n--- RESULTADO DE LA PREDICCIÓN ---")
        print(f"El peso estimado (Regresión Lineal) para el día {dia_objetivo} es: {simulated_weight[0][0]:.2f} gramos.")
        print(f"El peso estimado (Red Neuronal) para el día {dia_objetivo} es: {predicted_weight_original[0][0]:.2f} gramos.")
        print("------------------------------------")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_plot_path = os.path.join(REPORT_FOLDER, f'prediccion_{timestamp}.png')
        evaluate_prediction(processed_data, dia_objetivo, predicted_weight_original, simulated_data=simulated_data_point, save_path=prediction_plot_path)

    except (ValueError, IndexError, FileNotFoundError) as e:
        print(f"\nNo se pudo completar la predicción. Error: {e}")

def main_menu():
    """Muestra el menú principal y maneja la selección del usuario."""
    if SEED is not None:
        set_global_seeds(SEED)

    while True:
        print("\n=========================================")
        print("  Menú Principal del Pipeline de DNN")
        print("=========================================")
        print("1. Limpiar Dataset")
        print("2. Entrenar Modelo")
        print("3. Realizar Predicción a Futuro")
        print("4. Salir")
        print("-----------------------------------------")
        
        choice = input("Seleccione una opción (1-4): ")
        
        if choice == '1':
            run_cleaning()
        elif choice == '2':
            run_training()
        elif choice == '3':
            run_prediction()
        elif choice == '4':
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Por favor, intente de nuevo.")

if __name__ == "__main__":
    main_menu()