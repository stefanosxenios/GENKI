# utils/generator.py

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


# =====================================================================
#  CORE GENERATOR LOGIC (used by both single-label and multi-label)
# =====================================================================

def _sample_latent_vectors(latent_dim, temperature=1.0, n=1):
    """Sample n latent vectors from N(0, temperature^2 * I)."""
    return np.random.normal(loc=0.0, scale=temperature, size=(n, latent_dim)).astype("float32")


def _decode_and_inverse_scale(x_gen, scaler=None, log=False):
    """
    Apply inverse scaling and/or exp-log reverse transform.
    x_gen: array shape (1, n_features)
    """
    x_out = x_gen[0]

    # Case 1: Scaled + log transform used → reverse both
    if scaler is not None and log:
        x_out = scaler.inverse_transform([x_out])[0]
        x_out = np.exp(x_out)

    # Case 2: Scaled but no log transform
    elif scaler is not None and not log:
        x_out = scaler.inverse_transform([x_out])[0]

    # Case 3: Unscaled but log transform used
    elif scaler is None and log:
        x_out = np.exp(x_out)

    return x_out


# =====================================================================
#  SINGLE-LABEL GENERATOR
# =====================================================================

def generate_class_1_samples(
    decoder,
    classifier,
    latent_dim,
    n_samples=100,
    max_tries=5000,
    threshold=0.5,
    scaler=None,
    log=False,
    column_names=None,
    temperature=1
):
    """
    Generate samples where classifier output >= threshold (single-label).

    Returns: DataFrame with n_samples rows.
    """

    valid = []
    tries = 0
    pbar_try = tqdm(total=max_tries, desc="Sampling attempts", position=0)
    pbar_valid = tqdm(total=n_samples, desc="Valid samples collected", position=1)
    while len(valid) < n_samples and tries < max_tries:
        z = _sample_latent_vectors(latent_dim,temperature=temperature)
        y_pred = classifier.predict(z,verbose=0)

        if y_pred >= threshold:
            x_gen = decoder.predict(z,verbose=0)
            x_out = _decode_and_inverse_scale(x_gen, scaler, log)
            valid.append(x_out)
            pbar_valid.update(1)

        tries += 1
        pbar_try.update(1)

    pbar_try.close()
    pbar_valid.close()
    df = pd.DataFrame(valid)
    if column_names is not None:
        df.columns = column_names

    return df


# =====================================================================
#  MULTI-LABEL GENERATOR
# =====================================================================

# def generate_multilabel_samples(
#     decoder,
#     classifier,
#     latent_dim,
#     desired_label_vector,
#     n_samples=100,
#     max_tries=10000,
#     thresholds=None,
#     scaler=None,
#     log=False,
#     column_names=None,
# ):
#     """
#     Generate samples matching a multi-label classifier condition.

#     Parameters
#     ----------
#     decoder : VAE decoder model
#     classifier : latent classifier model
#     latent_dim : int
#     desired_label_vector : list or array, e.g. [1,1] or [1,0]
#         Defines which class(es) must be satisfied.
#     n_samples : number of generated samples desired
#     max_tries : max candidate latent draws
#     thresholds : list or array of thresholds for each label dimension
#                  If None, defaults to 0.5 per label.
#     scaler : inverse scaler
#     log : bool, if log-transform was applied before training
#     column_names : output DataFrame columns

#     Returns
#     -------
#     DataFrame (n_samples, n_features)
#     """

#     desired_label_vector = np.array(desired_label_vector)
#     label_dim = len(desired_label_vector)

#     if thresholds is None:
#         thresholds = np.array([0.5] * label_dim)
#     else:
#         thresholds = np.array(thresholds)

#     valid = []
#     tries = 0

#     while len(valid) < n_samples and tries < max_tries:
#         z = _sample_latent_vectors(latent_dim)
#         y_pred = classifier.predict(z)[0]  # vector of shape (label_dim,)

#         # Condition: for each label i, require y_pred[i] >= thresholds[i] if desired=1,
#         # and y_pred[i] < thresholds[i] if desired=0.
#         satisfies = True
#         for i in range(label_dim):
#             if desired_label_vector[i] == 1 and y_pred[i] < thresholds[i]:
#                 satisfies = False
#                 break
#             if desired_label_vector[i] == 0 and y_pred[i] >= thresholds[i]:
#                 satisfies = False
#                 break

#         if satisfies:
#             x_gen = decoder.predict(z)
#             x_out = _decode_and_inverse_scale(x_gen, scaler, log)
#             valid.append(x_out)

#         tries += 1

#     df = pd.DataFrame(valid)
#     if column_names is not None:
#         df.columns = column_names

#     return df



def generate_multilabel_samples(
    decoder,
    classifier,
    latent_dim,
    desired_label_vector,
    n_samples=100,
    max_tries=10000,
    thresholds=None,
    scaler=None,
    log=False,
    column_names=None,
):
    """
    Generate samples matching multi-label classifier conditions
    WITH progress bars (tqdm) and WITHOUT keras.predict spam.
    """

    desired_label_vector = np.array(desired_label_vector)
    label_dim = len(desired_label_vector)

    # Default thresholds
    if thresholds is None:
        thresholds = np.array([0.5] * label_dim)
    else:
        thresholds = np.array(thresholds)

    valid = []
    tries = 0

    # TWO progress bars
    pbar_try = tqdm(total=max_tries, desc="Sampling attempts", position=0)
    pbar_valid = tqdm(total=n_samples, desc="Valid samples collected", position=1)

    while len(valid) < n_samples and tries < max_tries:

        # Disable printouts from keras.predict
        z = np.random.normal(size=(1, latent_dim)).astype("float32")
        y_pred = classifier.predict(z, verbose=0)[0]

        # Check criteria
        satisfies = True
        for i in range(label_dim):
            if desired_label_vector[i] == 1 and y_pred[i] < thresholds[i]:
                satisfies = False
                break
            if desired_label_vector[i] == 0 and y_pred[i] >= thresholds[i]:
                satisfies = False
                break

        # If sample satisfies → decode and store
        if satisfies:
            x_gen = decoder.predict(z, verbose=0)
            x_out = _decode_and_inverse_scale(x_gen, scaler, log)
            valid.append(x_out)
            pbar_valid.update(1)

        tries += 1
        pbar_try.update(1)

    pbar_try.close()
    pbar_valid.close()

    df = pd.DataFrame(valid)
    if column_names is not None:
        df.columns = column_names

    return df






