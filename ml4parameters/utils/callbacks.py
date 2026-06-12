
import tensorflow as tf
from tensorflow import keras

class LossHistory(keras.callbacks.Callback):
    """
    Logs train + validation losses and metrics automatically.
    Compatible with custom train_step and test_step VAE.
    """

    def on_train_begin(self, logs=None):
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)





