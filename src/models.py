import keras
from keras import layers

def create_ann_simple(input_shape):
    """Crea un modelo ANN simple."""
    model = keras.Sequential([
        keras.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_ann_with_dropout(input_shape):
    """Crea un modelo ANN con regularización Dropout."""
    model = keras.Sequential([
        keras.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3), # Capa de regularización
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3), # Capa de regularización
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model