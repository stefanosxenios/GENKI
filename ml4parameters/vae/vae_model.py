import tensorflow as tf
from keras import Model
import keras
def build_vae(encoder, decoder, classifier,
              beta=0.1, max_beta=0.5, warmup_epochs=100,
              lambda_classifier=5.0, steps_per_epoch=200):

    class VAE(keras.Model):
        def __init__(self, encoder, decoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.classifier = classifier

            # Trackers
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.recon_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
            self.class_loss_tracker = keras.metrics.Mean(name="classifier_loss")

            # Hyperparams
            self.beta = beta
            self.lambda_classifier = lambda_classifier
            self.max_beta = max_beta
            self.warmup_epochs = warmup_epochs
            self.current_epoch = 0
            self.steps_per_epoch = steps_per_epoch

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.recon_loss_tracker,
                self.kl_loss_tracker,
                self.class_loss_tracker,
            ]

        # --------------------------------------------------
        # TRAIN STEP
        def train_step(self, data):
            n_labels = self.classifier.output_shape[-1]
            x = data[:, :-n_labels]
            y_true = data[:, -n_labels:]

            # KL annealing
            global_step = tf.cast(self.optimizer.iterations, tf.float32)
            epoch_est = global_step / self.steps_per_epoch
            self.beta = self.max_beta * tf.minimum(epoch_est / self.warmup_epochs, 1.0)

            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(x)
                reconstruction = self.decoder(z)

                # Reconstruction loss
                recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - reconstruction), axis=1))

                # KL loss
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
                )

                # Classifier loss
                y_pred = self.classifier(z)
                class_loss = tf.reduce_mean(
                    keras.losses.binary_crossentropy(y_true, y_pred)
                )

                total_loss = recon_loss + self.beta * kl_loss + self.lambda_classifier * class_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            # Update trackers
            self.total_loss_tracker.update_state(total_loss)
            self.recon_loss_tracker.update_state(recon_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            self.class_loss_tracker.update_state(class_loss)

            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.recon_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
                "classifier_loss": self.class_loss_tracker.result(),
            }

        # --------------------------------------------------
        # TEST STEP (for validation during fit)
        def test_step(self, data):
            n_labels = self.classifier.output_shape[-1]
            x = data[:, :-n_labels]
            y_true = data[:, -n_labels:]

            beta_now = (
                self.max_beta * (self.current_epoch / self.warmup_epochs)
                if self.current_epoch < self.warmup_epochs else
                self.max_beta
            )

            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)

            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - reconstruction), axis=1))
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            y_pred = self.classifier(z)
            class_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(y_true, y_pred)
            )

            total_loss = recon_loss + beta_now * kl_loss + self.lambda_classifier * class_loss

            return {
                "loss": total_loss,
                "reconstruction_loss": recon_loss,
                "kl_loss": kl_loss,
                "classifier_loss": class_loss,
            }

        # --------------------------------------------------
        def on_epoch_end(self, epoch, logs=None):
            self.current_epoch += 1

        def call(self, inputs):
            _, _, z = self.encoder(inputs)
            return self.decoder(z)

    return VAE(encoder, decoder, classifier)






