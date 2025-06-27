import keras
from keras import layers

def ann_simple(input_shape):
    """Crea un modelo ANN simple."""
    model = keras.Sequential([
        keras.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu', name='Capa_Oculta_1'),
        layers.Dense(1, activation='linear', name='Capa_Salida')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model