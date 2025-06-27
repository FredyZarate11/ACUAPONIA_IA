# --- Configuración General del Experimento ---
N_EXPERIMENTS = 1
SEED = None      # None = random

# --- Dataset Configuration ---
DATA_PATH = './data/annData/data.csv'
APPLY_CLEANING = True

# --- Columns ---
TARGET_COLUMN = 'Fish_Weight(g)'
FEATURE_COLUMNS = [
    'Temperature(C)', 
    'Turbidity(NTU)', 
    'Dissolved_Oxygen(mg/L)',
    'PH', 'Ammonia(mg/L)', 
    'Nitrate(mg/L)', 
    'Dia_Cultivo',
    'Population',
    # 'Fish_Length(cm)', 
    # 'Length_Squared',
    # 'Length_Cubed'
]
DATETIME_COLUMN = 'Datetime'

# --- Model ---

MODEL_TO_USE = 'ANN' 

# --- Training ---
TEST_SIZE = 0.2 # Used for final model testing
VALIDATION_SIZE = 0.2  # Used for validation split during training
EPOCHS = 300
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_DELTA = 0.001
