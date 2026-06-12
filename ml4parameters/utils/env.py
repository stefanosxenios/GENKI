import tensorflow as tf
import numpy as np
import random
from tensorflow.keras import backend as K

def reset_environment(seed=42):
    """
    Resets TF, NumPy, and Python random seeds,
    and clears the current Keras session to avoid stale graph issues.
    """
    K.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"[Environment reset] random seed = {seed}")






