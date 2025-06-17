import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

def load_and_prepare_data(file_path):
    """
    Carga, diagnostica, limpia, estandariza y divide los datos para el entrenamiento.
    """
    print(f"--- Iniciando preprocesamiento para {os.path.basename(file_path)} ---")
    data = pd.read_csv(file_path)
    
    # 1. Renombrar columna de fecha para consistencia
    if 'created_at' in data.columns:
        data.rename(columns={'created_at': 'Datetime'}, inplace=True)

    if 'Datetime' not in data.columns:
        raise ValueError(f"Error: No se encontró una columna de fecha ('Datetime' o 'created_at') en {file_path}")

    # 2. ESTANDARIZACIÓN DE FECHAS A UTC
    try:
        data['Datetime'] = pd.to_datetime(data['Datetime'], utc=True)
        
    except Exception as e:
        print(f"  -> ERROR al convertir fechas: {e}. Omitiendo este archivo.")
        raise ValueError(f"Error al procesar el archivo {file_path}: {e}")

    # 3. Verificar que existe la columna objetivo antes de continuar
    if 'Fish_Weight(g)' not in data.columns:
        raise ValueError(f"Error: No se encontró la columna objetivo 'Fish_Weight(g)' en {file_path}")

    # 4. Limpieza de columnas numéricas
    features = ['Temperature(C)', 'Turbidity(NTU)', 'Dissolved_Oxygen(g/ml)', 'PH', 'Ammonia(g/ml)', 'Nitrate(g/ml)', 'Population', 'Dia_Cultivo']
    # Lista real de columnas a procesar que existen en el dataframe
    numeric_features_to_process = [col for col in features if col in data.columns and col != 'Dia_Cultivo']

    for col in numeric_features_to_process:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Manejo de valores infinitos usando mediana (más robusta contra outliers)
        if np.isinf(data[col]).any():
            # Calcular la mediana excluyendo infinitos y NaN
            finite_values = data[col][np.isfinite(data[col])]
            if len(finite_values) > 0:
                median_value = finite_values.median()
                mean_value = finite_values.mean()
                inf_count = np.isinf(data[col]).sum()
                print(f"  -> Reemplazando {inf_count} valores infinitos en '{col}' con mediana: {median_value:.4f} (promedio era: {mean_value:.4f})")
                data[col] = data[col].replace([np.inf, -np.inf], median_value)
            else:
                print(f"  -> ADVERTENCIA: Columna '{col}' solo contiene infinitos/NaN. Se eliminarán estas filas.")
        
        # Interpolación para valores nulos
        if data[col].isnull().any():
            null_count = data[col].isnull().sum()
            print(f"  -> Interpolando {null_count} valores nulos en '{col}'")
            data[col] = data[col].interpolate(method='linear', limit_direction='both')
        
    # Eliminar filas donde las características principales aún sean nulas después de interpolar
    data.dropna(subset=numeric_features_to_process, inplace=True)
    
    # 5. Verificar que quedan datos después de la limpieza
    if data.empty:
        raise ValueError(f"Error: No quedan datos después de la limpieza en {file_path}")
    
    # 6. Agregación diaria
    data.set_index('Datetime', inplace=True)
    # Incluir todas las columnas necesarias en el resample
    columns_to_resample = numeric_features_to_process + ['Fish_Weight(g)']
    df_daily = data[columns_to_resample].resample('D').mean()
    
    # Verificar y manejar infinitos después del resample usando mediana
    for col in df_daily.columns:
        if np.isinf(df_daily[col]).any():
            finite_values = df_daily[col][np.isfinite(df_daily[col])]
            if len(finite_values) > 0:
                median_value = finite_values.median()
                mean_value = finite_values.mean()
                inf_count = np.isinf(df_daily[col]).sum()
                print(f"  -> Reemplazando {inf_count} valores infinitos post-resample en '{col}' con mediana: {median_value:.4f} (promedio era: {mean_value:.4f})")
                df_daily[col] = df_daily[col].replace([np.inf, -np.inf], median_value)
                print(f"  -> Rango de valores finitos en '{col}': {finite_values.min():.4f} a {finite_values.max():.4f}")
                print(f"  -> Percentiles: P25={finite_values.quantile(0.25):.4f}, P75={finite_values.quantile(0.75):.4f}")
    
    df_daily.dropna(inplace=True)

   

    # 7. Verificar que quedan datos después del resample
    if df_daily.empty:
        raise ValueError(f"Error: No quedan datos después de la agregación diaria en {file_path}")

    # 8. Ingeniería de Características
    df_daily.reset_index(inplace=True)
    start_date = df_daily['Datetime'].min()
    df_daily['Dia_Cultivo'] = (df_daily['Datetime'] - start_date).dt.days
    
    # 9. Seleccionar características finales (ahora incluye Dia_Cultivo que se calculó)
    final_features = [col for col in features if col in df_daily.columns]
    
    # 10. Verificar que tenemos características y variable objetivo
    if not final_features:
        raise ValueError(f"Error: No se encontraron características válidas en {file_path}")
    
    if 'Fish_Weight(g)' not in df_daily.columns:
        raise ValueError(f"Error: La variable objetivo 'Fish_Weight(g)' no está disponible después del procesamiento en {file_path}")
    
    X = df_daily[final_features]
    Y = df_daily['Fish_Weight(g)']
    
    print(f"  -> Características finales: {final_features}")
    print(f"  -> Forma de X: {X.shape}, Forma de Y: {Y.shape}")
    print(f"Procesamiento de {os.path.basename(file_path)} completado.")

    # 11. Normalización de datos
    print(f"--- NORMALIZANDO DATOS ---")
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    Y_scaled = scaler_Y.fit_transform(Y.to_numpy().reshape(-1, 1)).flatten()

    print(f"Rango original de Y: {Y.min():.2f} a {Y.max():.2f}")
    print(f"Rango normalizado de Y: {Y_scaled.min():.2f} a {Y_scaled.max():.2f}")

    return X_scaled, Y_scaled, scaler_X, scaler_Y, Y