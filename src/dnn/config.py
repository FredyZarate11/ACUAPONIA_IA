# --- GENERAL ---
SEED = 42  # Semilla para reproducibilidad. None para aleatorio.

# --- RUTAS DE ARCHIVOS ---
RAW_DATA_PATH = './data/dnnData/data.csv'
CLEANED_DATA_PATH = './data/dnnData/processed/data.csv'
MODEL_SAVE_PATH = './models/dnn_model.keras'
REPORT_FOLDER = './results/dnnResults'

# --- CONFIGURACIÓN DEL DATASET ---
# Define los roles de las columnas de tu dataset.
# El sistema usará estas listas para procesar cualquier archivo CSV.

TARGET_COLUMN = 'Fish_Weight(g)'  # Columna que quieres predecir.

# Columnas que se usarán como características de entrada para el modelo.
# Si IS_TIME_SERIES es True, se generará 'Dia_Cultivo' y se añadirá aquí.
FEATURE_COLUMNS = [
    'Temperature(C)',
    'Turbidity(NTU)',
    'Dissolved_Oxygen(mg/L)',
    'PH',
    'Ammonia(mg/L)',
    'Nitrate(mg/L)',
    'Population',
]

# Columnas con texto que necesitan ser convertidas a números (ej: 'Bajo', 'Medio', 'Alto').
# El sistema las convertirá a 0, 1, 2, etc.
CATEGORICAL_COLUMNS = []

# --- CONFIGURACIÓN DE PROCESAMIENTO ---
# ¿Es tu dataset una serie temporal con una columna de fecha?
IS_TIME_SERIES = True
# Si es True, especifica el nombre de la columna de fecha.
DATE_COLUMN = 'Datetime'
# Día de cultivo mínimo para filtrar los datos (si es una serie temporal).
# Útil para ignorar fases iniciales. Poner 0 o None para no filtrar.
MIN_CULTIVATION_DAY = None

# --- PARÁMETROS DE LIMPIEZA ---
# Define límites para eliminar valores atípicos.
# Formato: 'Nombre_Columna': (limite_inferior, limite_superior).
# Usa `None` si no quieres aplicar un límite en una dirección.
CLEANING_BOUNDS = {
    'Temperature(C)': (-20, 40),
    'Turbidity(NTU)': (0, 100),
    'Dissolved_Oxygen(mg/L)': (0, 10),
    'PH': (0, 14),
    'Ammonia(mg/L)': (0, 7),
    'Nitrate(mg/L)': (0, 400),
    'Fish_Weight(g)': (0, None),
    'Fish_Length(cm)': (0, None)
}

# --- CONFIGURACIÓN DEL MODELO Y ENTRENAMIENTO ---
MODEL_NAME = 'DNN'
TEST_SIZE = 0.2  # Proporción para el conjunto de prueba.
VALIDATION_SIZE = 0.2  # Proporción del set de entrenamiento para validación.
EPOCHS = 300
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15  # Épocas a esperar antes de detener el entrenamiento.
EARLY_STOPPING_DELTA = 0.001 # Mejora mínima para considerar.
_STOPPING_PATIENCE = 15
EARLY_STOPPING_DELTA = 0.001