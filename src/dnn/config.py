# --- GENERAL ---
N_EXPERIMENTS = 1
SEED = 42  # None para una semilla aleatoria

# --- RUTAS DE ARCHIVOS ---
RAW_DATA_PATH = './data/dnnData/data.csv'
CLEANED_DATA_PATH = './data/dnn/dnnData/processed/data.csv'
REPORT_FOLDER = './results/dnnResults'

# --- CONFIGURACIÓN DE DATOS ---
APPLY_CLEANING = True
TARGET_COLUMN = 'Fish_Weight(g)'
DATETIME_COLUMN = 'Datetime'

# Columnas que el modelo usará como características
FEATURE_COLUMNS = [
    'Temperature(C)',
    'Turbidity(NTU)',
    'Dissolved_Oxygen(mg/L)',
    'PH',
    'Ammonia(mg/L)',
    'Nitrate(mg/L)',
    'Dia_Cultivo',
    'Population',
]

# --- PARÁMETROS DE LIMPIEZA ---
# Define los límites físicos o esperados para cada columna.
# Formato: 'Nombre_Columna': (limite_inferior, limite_superior)
# Usa `None` si no quieres aplicar un límite.
CLEANING_BOUNDS = {
    'Temperature(C)': (-20, 40),
    'Turbidity(NTU)': (0, 100),
    'Dissolved_Oxygen(mg/L)': (0, 10),
    'PH': (0, 14),
    'Ammonia(mg/L)': (0, 7),
    'Nitrate(mg/L)': (0, 400),
    'Fish_Weight(g)': (0, None), # El peso no puede ser negativo
    'Fish_Length(cm)': (0, None) # La longitud no puede ser negativa
}

# --- PARÁMETROS DE PREPROCESAMIENTO ---
# Filtrar datos para empezar desde un día de cultivo específico (p. ej., para usar solo la fase de crecimiento)
# Poner 0 o None para no aplicar ningún filtro.
MIN_CULTIVATION_DAY = None

# --- CONFIGURACIÓN DEL MODELO Y ENTRENAMIENTO ---
MODEL_TO_USE = 'DNN'
TEST_SIZE = 0.2  # Usado para la prueba final del modelo
VALIDATION_SIZE = 0.2  # Usado para la validación durante el entrenamiento
EPOCHS = 300
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_DELTA = 0.001