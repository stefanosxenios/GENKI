import tensorflow as tf
import keras
from keras import layers
from .sampling import Sampling

def build_encoder(input_dim, latent_dim,
                  hidden_layers=[256,128], activation="gelu"):

    inputs = layers.Input(shape=(input_dim,))
    x = inputs

    for units in hidden_layers:
        x = layers.Dense(units, activation=activation)(x)

    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])

    return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")







