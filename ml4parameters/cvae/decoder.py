# cvae/decoder.py
import tensorflow as tf

def build_conditional_decoder(
    input_dim: int,
    label_dim: int,
    latent_dim: int,
    hidden_layers=(128, 128, 256),
    activation="gelu",
    output_activation=None,
    use_label_embedding: bool = False,
    label_embed_dim: int = 32,
):
    z_in = tf.keras.Input(shape=(latent_dim,), name="z")
    y_in = tf.keras.Input(shape=(label_dim,), name="y")

    y_feat = y_in
    if use_label_embedding:
        y_feat = tf.keras.layers.Dense(
            label_embed_dim,
            activation=activation,
            name="y_embed"
        )(y_in)

    h = tf.keras.layers.Concatenate(name="dec_concat")([z_in, y_feat])

    for i, units in enumerate(hidden_layers):
        h = tf.keras.layers.Dense(
            units,
            activation=activation,
            name=f"dec_dense_{i}"
        )(h)

    x_out = tf.keras.layers.Dense(
        input_dim,
        activation=output_activation,
        name="x_hat"
    )(h)

    return tf.keras.Model([z_in, y_in], x_out, name="cvae_decoder")







