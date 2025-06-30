import keras
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
import time

from data_processing import process_data
from model import ann_simple
from evaluation import calculate_metrics, plot_evaluation_results, generate_report, evaluate_prediction
import tensorflow as tf
import numpy as np
from sklearn.linear_model import LinearRegression
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
        verbose=2, # type: ignore
        shuffle=True,
        callbacks=[early_stopping, keras.callbacks.TerminateOnNaN() if config['SEED'] is not None else None]
    )
    
    # Time ended
    end_time = time.time()
    training_duration = end_time - start_time
    
    epochs_run = len(history.history['loss'])

    print("\n--- Evaluando el Modelo en el Conjunto de Prueba ---")
    
    y_pred_scaled = model.predict(X_test, verbose=0) # type: ignore
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

    print("\n\n--- Iniciando simulación de predicción a futuro ---")
    
    dias_a_predecir = int(input("Ingrese el número de días a predecir (ejemplo: 7): "))
    clean_data = processed_data['X_original'].copy()

    ultimo_dia_registrado = clean_data['Dia_Cultivo'].max()
    dia_objetivo = ultimo_dia_registrado + dias_a_predecir
    
    print(f"Último día en el dataset: {ultimo_dia_registrado}. Prediciendo para el día: {dia_objetivo}.")

    
    # 2. Preparamos el diccionario que contendrá los datos futuros
    future_values = {'Dia_Cultivo': dia_objetivo}
    X_hist = clean_data[['Dia_Cultivo']] # El tiempo es nuestra característica

    # 3. Iteramos sobre cada columna de característica para predecir su valor futuro
    for col in config['FEATURE_COLUMNS']:
        if col == 'Dia_Cultivo':
            continue # Ya lo tenemos

        if col == 'Population':
            # La población la mantenemos constante al último valor conocido
            last_population = clean_data.sort_values(by='Dia_Cultivo')['Population'].iloc[-1] # type: ignore
            future_values[col] = last_population
            continue
        
        # Entrenamos un modelo lineal simple para esta columna
        y_hist = clean_data[col]
        linear_model = LinearRegression()
        linear_model.fit(X_hist, y_hist)
        
        # Predecimos el valor de esta variable para el día objetivo
        predicted_val = linear_model.predict(np.array([[dia_objetivo]]))
        future_values[col] = predicted_val[0]

    future_data = pd.DataFrame([future_values])
    future_data = future_data[config['FEATURE_COLUMNS']] # Asegurar orden

    print("\nDatos de entrada (predichos por regresión lineal) para la predicción:")
    print(future_data)
    
    future_data_scaled = processed_data['scaler_X'].transform(future_data)
    
    predicted_weight_scaled = model.predict(future_data_scaled)
    
    predicted_weight_original = scaler_Y.inverse_transform(predicted_weight_scaled)
    
    print("\n--- RESULTADO DE LA PREDICCIÓN ---")
    print(f"El peso estimado del pez para el día {dia_objetivo} ({dias_a_predecir} días en el futuro) es: {predicted_weight_original[0][0]:.2f} gramos.")
    print("------------------------------------")

    evaluate_prediction(processed_data, dia_objetivo, predicted_weight_original, save_path=os.path.join(report_folder, f'grafico_Predicción{timestamp}.png'))

if __name__ == "__main__":
    main()
