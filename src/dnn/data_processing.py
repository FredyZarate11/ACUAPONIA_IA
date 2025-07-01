import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from config import (
    CLEANED_DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN, IS_TIME_SERIES, 
    DATE_COLUMN, MIN_CULTIVATION_DAY
)

def process_data():
    """
    Carga datos limpios y los prepara para el entrenamiento del modelo.
    Responsabilidades:
    1. Cargar el dataset limpio.
    2. Si es una serie temporal, remuestrear a frecuencia diaria y crear 'Dia_Cultivo'.
    3. Si no, usar los datos como están.
    4. Separar y escalar las características (X) y el objetivo (y).
    """
    print(f"--- Iniciando procesamiento de datos desde: {CLEANED_DATA_PATH} ---")

    try:
        data = pd.read_csv(CLEANED_DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: No se encontró el archivo en: {CLEANED_DATA_PATH}")

    # Copia de las columnas de características para poder modificarla si es necesario
    final_feature_cols = FEATURE_COLUMNS.copy()

    if IS_TIME_SERIES:
        print("Procesando como Serie Temporal...")
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], utc=True)
        data = data.set_index(DATE_COLUMN)
        
        all_cols = final_feature_cols + [TARGET_COLUMN]
        numeric_cols = [col for col in all_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
        
        df_daily = data[numeric_cols].resample('D').mean().dropna(how='all')
        
        df_daily = df_daily.reset_index()
        start_date = df_daily[DATE_COLUMN].min()
        df_daily['Dia_Cultivo'] = (df_daily[DATE_COLUMN] - start_date).dt.days
        
        # Añadir 'Dia_Cultivo' a las características si no está ya
        if 'Dia_Cultivo' not in final_feature_cols:
            final_feature_cols.append('Dia_Cultivo')

        df_daily = df_daily.ffill().dropna()
        print(f"Dataset agregado por día. Forma inicial: {df_daily.shape}")

        if MIN_CULTIVATION_DAY and MIN_CULTIVATION_DAY > 0:
            df_daily = df_daily[df_daily['Dia_Cultivo'] >= MIN_CULTIVATION_DAY]
            print(f"Forma del dataset tras filtrar por día de cultivo: {df_daily.shape}")
            
        if df_daily.empty:
            raise ValueError("El filtrado resultó en un DataFrame vacío.")
        
        processed_df = df_daily

    else:
        print("Procesando como dataset tabular estándar.")
        processed_df = data.dropna()

    # Asegurarse de que todas las columnas necesarias existen
    missing_cols = [col for col in final_feature_cols + [TARGET_COLUMN] if col not in processed_df.columns]
    if missing_cols:
        raise ValueError(f"Las siguientes columnas no se encontraron en el dataset: {missing_cols}")

    X_original = processed_df[final_feature_cols]
    y_original = processed_df[TARGET_COLUMN]

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

if __name__ == '__main__':
    # Permite probar el script de forma independiente
    try:
        process_data()
    except (FileNotFoundError, ValueError) as e:
        print(e)