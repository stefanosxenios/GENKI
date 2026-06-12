import tensorflow as tf
import keras
from keras import layers

def build_classifier(latent_dim, label_dim=1, mode="binary",
                     hidden_layers=[64, 32], activation="gelu"):

    inputs = layers.Input(shape=(latent_dim,))
    x = inputs

    for units in hidden_layers:
        x = layers.Dense(units, activation=activation)(x)

    if mode == "binary":
        outputs = layers.Dense(1, activation="sigmoid")(x)
    elif mode == "multilabel":
        outputs = layers.Dense(label_dim, activation="sigmoid")(x)
    else:
        raise ValueError("mode must be 'binary' or 'multilabel'")

    return keras.Model(inputs, outputs, name="latent_classifier")






