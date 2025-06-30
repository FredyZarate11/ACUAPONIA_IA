import pandas as pd
import numpy as np
import os
from config import (
    RAW_DATA_PATH, CLEANED_DATA_PATH, DATE_COLUMN, IS_TIME_SERIES,
    FEATURE_COLUMNS, TARGET_COLUMN, CATEGORICAL_COLUMNS, CLEANING_BOUNDS
)

def convert_categorical_to_numeric(data: pd.DataFrame, cat_columns: list) -> pd.DataFrame:
    """
    Convierte las columnas categóricas especificadas a valores numéricos usando pd.factorize.
    """
    for col in cat_columns:
        if col in data.columns:
            data[col], _ = pd.factorize(data[col])
            print(f"Columna categórica '{col}' convertida a numérica.")
    return data

def clean_dataset():
    """
    Carga un dataset, lo limpia basándose en la configuración y lo guarda.
    Responsabilidades:
    1. Cargar los datos crudos.
    2. Convertir columnas categóricas a numéricas.
    3. Manejar la columna de fecha/hora si aplica (IS_TIME_SERIES).
    4. Limpiar valores atípicos y rellenar NaNs en columnas numéricas.
    5. Guardar el dataset limpio.
    """
    print(f"--- Iniciando limpieza de datos de: {RAW_DATA_PATH} ---")

    try:
        data = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: {RAW_DATA_PATH}")
        return False

    # 1. Convertir columnas categóricas a numéricas
    if CATEGORICAL_COLUMNS:
        data = convert_categorical_to_numeric(data, CATEGORICAL_COLUMNS)

    # 2. Manejo de la columna de fecha (si aplica)
    if IS_TIME_SERIES and DATE_COLUMN in data.columns:
        print(f"Procesando columna de fecha: {DATE_COLUMN}")
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors='coerce')
        data = data.dropna(subset=[DATE_COLUMN])
        if data[DATE_COLUMN].dt.tz is None:
            data[DATE_COLUMN] = data[DATE_COLUMN].dt.tz_localize('UTC')
        else:
            data[DATE_COLUMN] = data[DATE_COLUMN].dt.tz_convert('UTC')
    
    # 3. Limpieza de columnas numéricas
    # Incluye características, objetivo y cualquier otra columna definida en los límites.
    columns_to_process = list(set(FEATURE_COLUMNS + [TARGET_COLUMN] + list(CLEANING_BOUNDS.keys())))

    for col in columns_to_process:
        if col not in data.columns:
            # No es un error, puede que la columna solo esté en los bounds pero no en el dataset actual
            continue

        # Asegurarse de que la columna es numérica antes de procesar
        if pd.api.types.is_numeric_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data = data.replace([np.inf, -np.inf], np.nan)

            # Aplicar límites definidos en config
            lower_bound, upper_bound = CLEANING_BOUNDS.get(col, (None, None))
            if lower_bound is not None:
                data.loc[data[col] < lower_bound, col] = np.nan
            if upper_bound is not None:
                data.loc[data[col] > upper_bound, col] = np.nan

            # Interpolar valores NaN y luego rellenar los restantes
            data[col] = data[col].interpolate(method='linear')
            data[col] = data[col].bfill()
            data[col] = data[col].ffill()

    # 4. Eliminar filas donde la variable objetivo sigue siendo nula
    if TARGET_COLUMN in data.columns:
        data = data.dropna(subset=[TARGET_COLUMN])

    print("--- Limpieza completada. ---")

    # 5. Guardar el archivo limpio
    output_dir = os.path.dirname(CLEANED_DATA_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Dataset limpio guardado en: {CLEANED_DATA_PATH}")
    return True

if __name__ == '__main__':
    # Esto permite ejecutar el script de forma independiente para probar la limpieza
    clean_dataset()