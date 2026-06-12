import tensorflow as tf
from keras import layers, ops

class Sampling(layers.Layer):
    """Sampling layer using reparameterization trick."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = tf.random.Generator.from_seed(2)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon




