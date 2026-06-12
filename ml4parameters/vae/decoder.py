import tensorflow as tf
import keras
from keras import layers

def build_decoder(input_dim, latent_dim,
                  hidden_layers=[128,64], activation="gelu"):

    latent_inputs = layers.Input(shape=(latent_dim,))
    x = latent_inputs

    for units in hidden_layers:
        x = layers.Dense(units, activation=activation)(x)

    outputs = layers.Dense(input_dim, activation="linear")(x)

    return keras.Model(latent_inputs, outputs, name="decoder")






