# cvae/encoder.py
import tensorflow as tf

def build_conditional_encoder(
    input_dim: int,
    label_dim: int,
    latent_dim: int,
    hidden_layers=(256, 128, 128),
    activation="elu",
    use_label_embedding: bool = False,
    label_embed_dim: int = 32,
):
    x_in = tf.keras.Input(shape=(input_dim,), name="x")
    y_in = tf.keras.Input(shape=(label_dim,), name="y")

    y_feat = y_in
    if use_label_embedding:
        y_feat = tf.keras.layers.Dense(
            label_embed_dim,
            activation=activation,
            name="y_embed"
        )(y_in)

    h = tf.keras.layers.Concatenate(name="enc_concat")([x_in, y_feat])

    for i, units in enumerate(hidden_layers):
        h = tf.keras.layers.Dense(
            units,
            activation=activation,
            name=f"enc_dense_{i}"
        )(h)

    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(h)
    z_logvar = tf.keras.layers.Dense(latent_dim, name="z_logvar")(h)

    return tf.keras.Model([x_in, y_in], [z_mean, z_logvar], name="cvae_encoder")








