from utils.env import reset_environment
from utils.callbacks import LossHistory

import numpy as np
from sklearn.model_selection import train_test_split


def train_vae(X, y, encoder, decoder, classifier,
              beta=0.1, max_beta=0.5, warmup_epochs=100,
              lambda_classifier=5,
              batch_size=80, epochs=200, seed=42,learning_rate=5e-4,validation_split=0.1):
    """
    Trains a VAE with a classifier and returns:
    - trained VAE model
    - LossHistory callback
    - latent vectors z
    """

    reset_environment(seed)

    # Concatenate labels for classifier training
    X_with_labels = np.concatenate([X, y], axis=1)

    # Build VAE
    vae = build_vae(
        encoder, decoder, classifier,
        beta=beta, max_beta=max_beta,
        warmup_epochs=warmup_epochs,
        lambda_classifier=lambda_classifier
    )

    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    # Steps per epoch
    vae.steps_per_epoch = len(X_with_labels) // batch_size

    # Callbacks
    history_cb = LossHistory()

    # Train
    vae.fit(
        X_with_labels,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[history_cb],
        validation_split=validation_split,
        shuffle=True,
    )

    # Encode entire dataset
    z_mean, z_log_var, z = vae.encoder.predict(X)

    return vae, history_cb, (z_mean, z_log_var, z)






