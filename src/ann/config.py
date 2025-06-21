# --- Configuración General del Experimento ---
N_EXPERIMENTS = 1
SEED = None        # None = random

# --- Dataset Configuration ---
DATA_PATH = './data/annData/Tank_combined_v3.csv'
APPLY_CLEANING = True

# --- Columns ---
TARGET_COLUMN = 'Fish_Weight(g)'
FEATURE_COLUMNS = [
    'Temperature(C)', 
    'Turbidity(NTU)', 
    'Dissolved_Oxygen(g/ml)',
    'PH', 'Ammonia(g/ml)', 
    'Nitrate(g/ml)', 
    'Population', 
    'Dia_Cultivo'
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
EARLY_STOPPING_DELTA = 0.0001
