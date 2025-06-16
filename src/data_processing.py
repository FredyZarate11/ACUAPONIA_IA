import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(file_path):
    """Carga, prepara y divide los datos para el entrenamiento."""
    data = pd.read_csv(file_path)
    print("Dataset cargado y listo.")

    features = ['Temperature(C)', 'Turbidity(NTU)', 'Dissolved_Oxygen(g/ml)', 'PH', 'Ammonia(g/ml)', 'Nitrate(g/ml)', 'Population', 'Dia_Cultivo']
    X = data[features]
    Y = data['Fish_Weight(g)']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Devolvemos tanto los datos escalados como los datos de prueba originales para la evaluación
    return X_train_scaled, X_test_scaled, y_train, y_test