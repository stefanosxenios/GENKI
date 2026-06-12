# cvae/sampling.py
import tensorflow as tf

def sampling(z_mean, z_logvar):
    """
    Reparameterization trick.
    """
    eps = tf.random.normal(tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_logvar) * eps





