# cvae/vae_model.py
import tensorflow as tf
from .sampling import sampling


class ConditionalVAE(tf.keras.Model):
    def __init__(
        self,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        beta: float = 0.1,
        max_beta: float | None = None,
        warmup_epochs: int = 0,
        recon_loss: str = "mse",
        name="ConditionalVAE",
    ):
        super().__init__(name=name)
        self.encoder = encoder
        self.decoder = decoder

        self.beta = float(beta)
        self.max_beta = float(max_beta) if max_beta is not None else None
        self.warmup_epochs = int(warmup_epochs)
        self.recon_loss = recon_loss.lower()

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_tracker = tf.keras.metrics.Mean(name="recon")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl")
        self.beta_tracker = tf.keras.metrics.Mean(name="beta")

        self._epoch_var = tf.Variable(0, dtype=tf.int32, trainable=False)

    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_tracker,
                self.kl_tracker, self.beta_tracker]

    def set_epoch(self, epoch: int):
        self._epoch_var.assign(int(epoch))

    def _current_beta(self):
        if self.warmup_epochs <= 0 or self.max_beta is None:
            return tf.constant(self.beta, dtype=tf.float32)

        e = tf.cast(self._epoch_var, tf.float32)
        w = tf.cast(self.warmup_epochs, tf.float32)
        frac = tf.clip_by_value(e / w, 0.0, 1.0)

        return tf.constant(self.beta, tf.float32) + \
               frac * (tf.constant(self.max_beta, tf.float32)
                       - tf.constant(self.beta, tf.float32))

    def _reconstruction_loss(self, x, x_hat):
        if self.recon_loss == "bce":
            per_dim = tf.keras.losses.binary_crossentropy(x, x_hat)
        else:
            per_dim = tf.square(x - x_hat)

        return tf.reduce_mean(tf.reduce_sum(per_dim, axis=1))

    def train_step(self, data):
        (x, y), _ = data

        with tf.GradientTape() as tape:
            z_mean, z_logvar = self.encoder([x, y], training=True)
            z = sampling(z_mean, z_logvar)
            x_hat = self.decoder([z, y], training=True)

            recon = self._reconstruction_loss(x, x_hat)

            kl = tf.reduce_mean(
                -0.5 * tf.reduce_sum(
                    1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar),
                    axis=1,
                )
            )

            beta = self._current_beta()
            loss = recon + beta * kl

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        self.beta_tracker.update_state(beta)

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False):
        x, y = inputs
        z_mean, z_logvar = self.encoder([x, y], training=training)
        z = sampling(z_mean, z_logvar)
        return self.decoder([z, y], training=training)


class EpochSetter(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.model, "set_epoch"):
            self.model.set_epoch(epoch)







