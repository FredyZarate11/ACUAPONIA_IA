import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def clear(route: str):

    data = pd.read_csv(route)

    print("Tipos de datos iniciales:")
    print(data.info())


    # Datetime conversion and timezone handling
    data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')

    data['Datetime'] = data['Datetime'].dt.tz_convert('UTC')

    print("\nTipos de datos después de la conversión:")
    print(data.info())

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=['Datetime'], inplace=True)
    data.dropna(subset=['Fish_Weight(g)'], inplace=True)

    # Temperature

    Q1 = data['Temperature(C)'].quantile(0.25)
    Q3 = data['Temperature(C)'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR if Q1 - 1.5 * IQR >= -20 else -20
    upper_bound = Q3 + 1.5 * IQR

    print("Limites aceptables de temperatura:")
    print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    data.loc[(data['Temperature(C)'] < lower_bound) | (data['Temperature(C)'] > upper_bound), 'Temperature(C)'] = np.nan
    data['Temperature(C)'].interpolate(method='linear', inplace=True)
    data['Temperature(C)'].fillna(method='bfill', inplace=True)

    # Turbidity(NTU)

    Q1 = data['Turbidity(NTU)'].quantile(0.25)
    Q3 = data['Turbidity(NTU)'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR if Q1 - 1.5 * IQR >= 0 else 0
    upper_bound = Q3 + 1.5 * IQR if Q3 + 1.5 * IQR <= 100 else 100
    print("\nLimites aceptables de turbidez:")
    print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    data.loc[(data['Turbidity(NTU)'] < lower_bound) | (data['Turbidity(NTU)'] > upper_bound), 'Turbidity(NTU)'] = np.nan
    data['Turbidity(NTU)'].interpolate(method='linear', inplace=True)
    data['Turbidity(NTU)'].fillna(method='bfill', inplace=True)

    # Dissolved Oxygen (mg/L)
    Q1 = data['Dissolved_Oxygen(mg/L)'].quantile(0.25)
    Q3 = data['Dissolved_Oxygen(mg/L)'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR if Q1 - 1.5 * IQR >= 0 else 0
    upper_bound = Q3 + 1.5 * IQR if Q3 + 1.5 * IQR <= 10 else 10
    print("\nLimites aceptables de oxígeno disuelto:")
    print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    data.loc[(data['Dissolved_Oxygen(mg/L)'] < lower_bound) | (data['Dissolved_Oxygen(mg/L)'] > upper_bound), 'Dissolved_Oxygen(mg/L)'] = np.nan
    data['Dissolved_Oxygen(mg/L)'].interpolate(method='linear', inplace=True)
    data['Dissolved_Oxygen(mg/L)'].fillna(method='bfill', inplace=True)

    # PH
    Q1 = data['PH'].quantile(0.25)
    Q3 = data['PH'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR if Q1 - 1.5 * IQR >= 0 else 0
    upper_bound = Q3 + 1.5 * IQR if Q3 + 1.5 * IQR <= 14 else 14
    print("\nLimites aceptables de pH:")
    print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    data.loc[(data['PH'] < lower_bound) | (data['PH'] > upper_bound), 'PH'] = np.nan
    data['PH'].interpolate(method='linear', inplace=True)
    data['PH'].fillna(method='bfill', inplace=True)

    # Ammonia
    Q1 = data['Ammonia(mg/L)'].quantile(0.25)
    Q3 = data['Ammonia(mg/L)'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR if Q1 - 1.5 * IQR >= 0 else 0
    upper_bound = Q3 + 1.5 * IQR if Q3 + 1.5 * IQR <= 7 else 7
    print("\nLimites aceptables de amoníaco:")
    print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    data.loc[(data['Ammonia(mg/L)'] < lower_bound) | (data['Ammonia(mg/L)'] > upper_bound), 'Ammonia(mg/L)'] = np.nan
    data['Ammonia(mg/L)'].interpolate(method='linear', inplace=True)
    data['Ammonia(mg/L)'].fillna(method='bfill', inplace=True)

    # Nitrate
    Q1 = data['Nitrate(mg/L)'].quantile(0.25)
    Q3 = data['Nitrate(mg/L)'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR if Q1 - 1.5 * IQR >= 0 else 0
    upper_bound = Q3 + 1.5 * IQR if Q3 + 1.5 * IQR <= 400 else 400
    print("\nLimites aceptables de nitratos:")
    print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    data.loc[(data['Nitrate(mg/L)'] < lower_bound) | (data['Nitrate(mg/L)'] > upper_bound), 'Nitrate(mg/L)'] = np.nan
    data['Nitrate(mg/L)'].interpolate(method='linear', inplace=True)
    data['Nitrate(mg/L)'].fillna(method='bfill', inplace=True)

    # Fish Length
    data['Fish_Length(cm)'] = pd.to_numeric(data['Fish_Length(cm)'], errors='coerce')
    Q1 = data['Fish_Length(cm)'].quantile(0.25)
    Q3 = data['Fish_Length(cm)'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    print("\nLimites aceptables de longitud de peces:")
    print(f"Lower bound: 0, Upper bound: {upper_bound}")
    data.loc[(data['Fish_Length(cm)'] <= 0) | (data['Fish_Length(cm)'] > upper_bound), 'Fish_Length(cm)'] = np.nan
    data['Fish_Length(cm)'].interpolate(method='linear', inplace=True)

    # New columns
    data['Length_Squared'] = data['Fish_Length(cm)'] ** 2
    data['Length_Cubed'] = data['Fish_Length(cm)'] ** 3


    # first_date = data['Datetime'].min()
    # data['Crop_Day'] = (data['Datetime'] - first_date).dt.days

    print("Tratando de guardar el archivo")
    data.to_csv(route.replace('.csv', '_cleaned.csv'), index=False)


def unir():
    allData = []
    routes = [
        'Tank1_cleaned.csv',
        'Tank2_cleaned.csv',
        'Tank3_cleaned.csv',
        'Tank4_cleaned.csv',
        'Tank6_cleaned.csv',
        'Tank7_cleaned.csv',
        'Tank8_cleaned.csv',
        'Tank9_cleaned.csv',
        'Tank10_cleaned.csv',
        'Tank11_cleaned.csv',
        'Tank12_cleaned.csv'
    ]

    for route in routes:
        data = pd.read_csv(route)
        allData.append(data)

    combined_data = pd.concat(allData, ignore_index=True)
    combined_data.to_csv('All_Tanks_cleaned.csv', index=False)

def daily(route: str):
    data = pd.read_csv(route)
    data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
    data.set_index('Datetime', inplace=True)
    daily_data = data.resample('D').mean()
    daily_data.to_csv(route.replace('.csv', '_daily.csv'))

datas = [
        'Tank1.csv',
        'Tank2.csv',
        'Tank3.csv',
        'Tank4.csv',
        'Tank6.csv',
        'Tank7.csv',
        'Tank8.csv',
        'Tank9.csv',
        'Tank10.csv',
        'Tank11.csv',
        'Tank12.csv',
         ]
# unir()
data = pd.read_csv('data/annData/data.csv')
# data.info()

# data.drop(columns=['Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'entry_id'], inplace=True)

# data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
# data.set_index('Datetime', inplace=True)
# data = data.resample('D').mean()
# data.reset_index(inplace=True)
# data.to_csv('All_Tanks_daily.csv', index=False)

# Agregar columna 'Dia_Cultivo'
data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
data['Dia_Cultivo'] = (data['Datetime'] - data['Datetime'].min()).dt.days
data.to_csv('All_Tanks_daily.csv', index=False)

