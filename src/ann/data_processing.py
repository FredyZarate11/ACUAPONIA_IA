import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

def process_data(file_path: str, feature_cols: list, target_col: str, date_col: str = 'Datetime'):
    """
    Carga, limpia, agrega y normaliza los datos de un archivo CSV para un modelo de ML.

    Args:
        file_path (str): Ruta al archivo CSV.
        feature_cols (list): Lista de nombres de las columnas de características.
        target_col (str): Nombre de la columna objetivo.
        date_col (str): Nombre de la columna de fecha.

    Returns:
        dict: Un diccionario con los datos procesados y los scalers.
    """
    print(f"--- Iniciando preprocesamiento para {os.path.basename(file_path)} ---")
    
    # 1. Carga de datos
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: No se encontró el archivo en la ruta: {file_path}")

    if date_col not in data.columns:
        raise ValueError(f"Error: La columna de fecha '{date_col}' no se encuentra en el archivo.")

    # 2. Columns validations
    data[date_col] = pd.to_datetime(data[date_col], utc=True, errors='coerce')
    all_cols = feature_cols + [target_col]
    
    for col in all_cols:
        if col not in data.columns:
            print(f"Advertencia: La columna '{col}' no se encontró y será ignorada.")
            continue
        data[col] = pd.to_numeric(data[col], errors='coerce')


    X = data[feature_cols]
    y = data[target_col]

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_Y.fit_transform(y.to_numpy().reshape(-1, 1)).flatten()

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print(f"--- Preprocesamiento completado. Forma de X: {X_scaled.shape}, Forma de y: {y_scaled.shape} ---")
    
    return {
        'X_scaled': X_scaled,
        'y_scaled': y_scaled,
        'X_original': X,
        'y_original': y,
        'scaler_X': scaler_X,
        'scaler_Y': scaler_Y,
    }