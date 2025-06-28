# data_processing.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from config import MIN_CULTIVATION_DAY # Importamos el nuevo parámetro

def process_data(file_path: str, feature_cols: list, target_col: str, date_col: str):
    """
    Carga datos previamente limpiados, los procesa y los escala para el entrenamiento.
    Las responsabilidades de esta función son:
    1. Cargar el dataset.
    2. Convertir la columna de fecha y establecerla como índice.
    3. Agregar los datos a una frecuencia diaria.
    4. Crear la característica 'Dia_Cultivo'.
    5. Aplicar un filtro opcional por día de cultivo.
    6. Separar y escalar las características (X) y el objetivo (y).
    """
    print(f"--- Iniciando procesamiento de datos desde: {os.path.basename(file_path)} ---")

    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: No se encontró el archivo en la ruta: {file_path}")

    # 1. Manejo de la fecha y remuestreo diario
    data[date_col] = pd.to_datetime(data[date_col], utc=True)
    data = data.set_index(date_col)
    
    # Asegurarse de que solo las columnas numéricas se incluyen en el remuestreo
    all_cols = feature_cols + [target_col]
    numeric_cols_in_data = [col for col in all_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
    
    # Remuestrear a promedios diarios y eliminar días sin datos
    df_daily = data[numeric_cols_in_data].resample('D').mean().dropna(how='all')
    
    # 2. Creación de la característica 'Dia_Cultivo'
    df_daily = df_daily.reset_index()
    start_date = df_daily[date_col].min()
    df_daily['Dia_Cultivo'] = (df_daily[date_col] - start_date).dt.days
    
    # Rellenar cualquier NaN que pudiera surgir del remuestreo
    df_daily = df_daily.ffill()
    df_daily = df_daily.dropna()

    print(f"Dataset agregado por día. Forma inicial: {df_daily.shape}")

    # 3. Filtrado opcional por día de cultivo
    if MIN_CULTIVATION_DAY and MIN_CULTIVATION_DAY > 0:
        print(f"Filtrando datos para mantener solo registros desde el día {MIN_CULTIVATION_DAY}.")
        df_daily = df_daily[df_daily['Dia_Cultivo'] >= MIN_CULTIVATION_DAY]
        print(f"Forma del dataset después de filtrar: {df_daily.shape}")
        
    if df_daily.empty:
        raise ValueError("El filtrado ha resultado en un DataFrame vacío. Ajusta el valor de 'MIN_CULTIVATION_DAY' o revisa tus datos.")

    # 4. Separación y escalado de datos
    # Asegurarnos de que todas las columnas necesarias existen después del preprocesamiento
    final_feature_cols = [col for col in feature_cols if col in df_daily.columns]
    
    X_original = df_daily[final_feature_cols]
    y_original = df_daily[target_col]

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_original)

    scaler_Y = StandardScaler()
    y_scaled = scaler_Y.fit_transform(y_original.to_numpy().reshape(-1, 1)).flatten()

    processed_data = {
        'X_original': X_original,
        'X_scaled': X_scaled,
        'y_scaled': y_scaled,
        'scaler_X': scaler_X,
        'scaler_Y': scaler_Y
    }
    
    print("--- Procesamiento de datos completado. ---")
    return processed_data