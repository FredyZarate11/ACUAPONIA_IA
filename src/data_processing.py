# data_processing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

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

    # Renombrar 'created_at' si existe
    if 'created_at' in data.columns and date_col not in data.columns:
        data = data.rename(columns={'created_at': date_col})

    if date_col not in data.columns:
        raise ValueError(f"Error: La columna de fecha '{date_col}' no se encuentra en el archivo.")

    # 2. Conversión y validación de columnas
    data[date_col] = pd.to_datetime(data[date_col], utc=True, errors='coerce')
    all_cols = feature_cols + [target_col]
    
    for col in all_cols:
        if col not in data.columns:
            print(f"Advertencia: La columna '{col}' no se encontró y será ignorada.")
            continue
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # 3. Limpieza de datos (Infinitos y Nulos)
    numeric_cols = [col for col in all_cols if col in data.columns and data[col].dtype in ['int64', 'float64']]
    
    for col in numeric_cols:
        # Reemplazar infinitos con NaN para tratarlos de una sola vez
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        
        # Rellenar valores nulos con la mediana de la columna
        if data[col].isnull().any():
            median_value = data[col].median()
            null_count = data[col].isnull().sum()
            print(f" -> Rellenando {null_count} valores nulos/infinitos en '{col}' con la mediana: {median_value:.4f}")
            data[col] = data[col].fillna(median_value)

    data = data.dropna(subset=[date_col] + numeric_cols)
    
    if data.empty:
        raise ValueError("No quedan datos después de la limpieza inicial.")

    # 4. Agregación diaria
    data = data.set_index(date_col)
    df_daily = data[numeric_cols].resample('D').mean()
    df_daily = df_daily.dropna()  # Eliminar días sin datos después del resampleo

    if df_daily.empty:
        raise ValueError("No quedan datos después de la agregación diaria.")

    # 5. Ingeniería de Características
    df_daily =df_daily.reset_index()
    start_date = df_daily[date_col].min()
    df_daily['Dia_Cultivo'] = (df_daily[date_col] - start_date).dt.days
    
    # Asegurarnos de que 'Dia_Cultivo' esté en las características si no estaba antes
    if 'Dia_Cultivo' not in feature_cols:
        feature_cols.append('Dia_Cultivo')
        
    final_feature_cols = [col for col in feature_cols if col in df_daily.columns]

    # 6. Separación y Normalización
    X = df_daily[final_feature_cols]
    y = df_daily[target_col]

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