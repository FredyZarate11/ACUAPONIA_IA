import keras
from keras import layers

def dnn(input_shape):
    """Crea un modelo DNN con varias capas ocultas."""
    model = keras.Sequential([
        keras.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu', name='Capa_Oculta_1'),
        layers.Dense(64, activation='relu', name='Capa_Oculta_2'),
        layers.Dense(32, activation='relu', name='Capa_Oculta_3'),
        layers.Dense(1, activation='linear', name='Capa_Salida')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model