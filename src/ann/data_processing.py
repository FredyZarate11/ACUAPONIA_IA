# data_processing.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def process_data(file_path: str, feature_cols: list, target_col: str, date_col: str = 'Datetime', apply_cleaning=True):
    """
    Carga, limpia, procesa y escala los datos, devolviendo un diccionario 
    con todos los componentes necesarios para el entrenamiento y la predicción.
    Incluye un filtro para usar solo datos de la fase de crecimiento acelerado.
    """
    if not apply_cleaning:
        print("Cargando datos crudos sin limpieza.")
        return pd.read_csv(file_path)

    print(f"--- Iniciando preprocesamiento para {os.path.basename(file_path)} ---")
    
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: No se encontró el archivo en la ruta: {file_path}")

    if 'created_at' in data.columns and date_col not in data.columns:
        data = data.rename(columns={'created_at': date_col})

    data[date_col] = pd.to_datetime(data[date_col], utc=True, errors='coerce')
    data.dropna(subset=[date_col], inplace=True)
    
    all_cols = feature_cols + [target_col]
    numeric_cols = [col for col in all_cols if col in data.columns]
    
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col].replace([np.inf, -np.inf], np.nan, inplace=True)
        if data[col].isnull().any():
            median_value = data[col].median()
            data[col].fillna(median_value, inplace=True)
            
    data.dropna(subset=numeric_cols, inplace=True)
    data = data.set_index(date_col)
    df_daily = data[numeric_cols].resample('D').mean().dropna()

    df_daily = df_daily.reset_index()
    start_date = df_daily[date_col].min()
    df_daily['Dia_Cultivo'] = (df_daily[date_col] - start_date).dt.days

    print(f"--- Preprocesamiento completado. ---")

    # --- Parte 2: Filtrado de Datos por Día de Cultivo ---
    print(f"Forma del dataset antes de filtrar: {df_daily.shape}")
    
    print(f"Forma del dataset después de filtrar (Día > 75): {df_daily.shape}")
    
    if df_daily.empty:
        raise ValueError("El filtrado ha resultado en un DataFrame vacío. Ajusta el valor del filtro o revisa tus datos.")

    # --- Parte 3: Separación, Escalado y Empaquetado ---
    
    # 1. Separar características (X) y objetivo (y)
    X_original = df_daily[feature_cols]
    y_original = df_daily[target_col]

    # 2. Escalar las características (X)
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_original)

    # 3. Escalar el objetivo (y)
    scaler_Y = StandardScaler()
    # Usamos .to_numpy().reshape(-1, 1) para asegurar la forma correcta para el scaler
    y_scaled = scaler_Y.fit_transform(y_original.to_numpy().reshape(-1, 1)).flatten() 

    # 4. Empaquetar todo en un diccionario para devolverlo
    processed_data = {
        'X_original': X_original,
        'X_scaled': X_scaled,
        'y_scaled': y_scaled,
        'scaler_X': scaler_X,
        'scaler_Y': scaler_Y
    }
    
    return processed_data