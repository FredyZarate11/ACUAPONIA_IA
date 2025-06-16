# Importa las funciones de tus otros archivos
from data_processing import load_and_prepare_data
from models import create_ann_simple, create_ann_with_dropout
from evaluation import evaluate_and_plot

# --- PASO 1: Cargar los datos ---
DATA_PATH = './data/Tank_combined_v1.csv'
X_train, X_test, y_train, y_test = load_and_prepare_data(DATA_PATH)

# --- PASO 2: Entrenar y Evaluar el primer modelo ---
input_shape = X_train.shape[1]
model_1 = create_ann_simple(input_shape)
print("\n--- Entrenando Modelo 1: ANN Simple ---")
history_1 = model_1.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose='1')
evaluate_and_plot(model_1, "ANN Simple", history_1, X_test, y_test)

# --- PASO 3: Entrenar y Evaluar el segundo modelo ---
model_2 = create_ann_with_dropout(input_shape)
print("\n--- Entrenando Modelo 2: ANN con Dropout ---")
history_2 = model_2.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose='1')
evaluate_and_plot(model_2, "ANN con Dropout", history_2, X_test, y_test)