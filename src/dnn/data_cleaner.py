# data_cleaner.py

import pandas as pd
import numpy as np
import os

def clean_dataset(raw_path: str,
                  cleaned_path: str,
                  datetime_col: str,
                  columns_to_clean: list,
                  cleaning_bounds: dict):
    """
    Carga un dataset, lo limpia basándose en límites y lo guarda.

    Args:
        raw_path (str): Ruta al archivo de datos crudos.
        cleaned_path (str): Ruta donde se guardará el archivo limpio.
        datetime_col (str): Nombre de la columna de fecha y hora.
        columns_to_clean (list): Lista de columnas numéricas a limpiar.
        cleaning_bounds (dict): Diccionario con los límites para cada columna.
    """
    print(f"--- Iniciando limpieza de datos de: {raw_path} ---")

    try:
        data = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: {raw_path}")
        return False

    # 1. Manejo de la columna de fecha
    data[datetime_col] = pd.to_datetime(data[datetime_col], errors='coerce')
    data = data.dropna(subset=[datetime_col])
    if data[datetime_col].dt.tz is None:
        data[datetime_col] = data[datetime_col].dt.tz_localize('UTC')
    else:
        data[datetime_col] = data[datetime_col].dt.tz_convert('UTC')

    # 2. Limpieza de columnas numéricas
    for col in columns_to_clean:
        if col not in data.columns:
            print(f"Advertencia: La columna '{col}' no se encontró en el dataset.")
            continue

        data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.replace([np.inf, -np.inf], np.nan)

        # Aplicar límites definidos en config
        lower_bound, upper_bound = cleaning_bounds.get(col, (None, None))
        if lower_bound is not None:
            data.loc[data[col] < lower_bound, col] = np.nan
        if upper_bound is not None:
            data.loc[data[col] > upper_bound, col] = np.nan

        # Interpolar valores NaN y luego rellenar los restantes
        data[col] = data[col].interpolate(method='linear')
        data[col] = data[col].bfill()
        data[col] = data[col].ffill()

    # 3. Eliminar filas donde la variable objetivo es nula (si existe)
    if 'Fish_Weight(g)' in data.columns:
        data = data.dropna(subset=['Fish_Weight(g)'])

    print("--- Limpieza completada. ---")

    # 4. Guardar el archivo limpio
    output_dir = os.path.dirname(cleaned_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data.to_csv(cleaned_path, index=False)
    print(f"Dataset limpio guardado en: {cleaned_path}")
    return True