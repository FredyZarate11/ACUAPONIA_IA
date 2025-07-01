import keras
from keras import layers

def dnn(input_shape):
    model = keras.Sequential([
        keras.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu', name='Capa_Oculta_2'),
        layers.Dense(1, activation='linear', name='Capa_Salida')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model