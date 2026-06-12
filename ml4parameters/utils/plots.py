# utils/plots.py

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


# ============================================================
#  VAE TRAINING LOSS PLOTS
# ============================================================

def plot_vae_training(history_cb):
    """
    Plots classifier, reconstruction, and KL loss over training epochs.
    """
    history = history_cb.history
    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(10, 6))

    if "classifier_loss" in history:
        plt.plot(epochs, history["classifier_loss"], label="Classifier Loss", linewidth=2)

    if "recon" in history:
        plt.plot(epochs, history["recon"], label="Reconstruction Loss", linewidth=2)

    if "kl" in history:
        plt.plot(epochs, history["kl"], label="KL Loss", linewidth=2)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("VAE Training Loss Components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_total_loss(history_cb):
    """
    Plots total VAE training and validation loss.
    """
    history = history_cb.history
    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history["loss"], label="Training Loss", linewidth=2)

    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)

    plt.xlabel("Epochs")
    plt.ylabel("Total Loss")
    plt.title("Total VAE Loss: Training vs Validation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
#  PCA / UMAP / t-SNE VISUALIZATIONS
# ============================================================

def plot_pca_latent(z, y_color=None, title="PCA of Latent Space"):
    """
    Runs PCA on latent vectors and plots the first two components.
    """
    if not isinstance(z, np.ndarray):
        z = np.array(z)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(z)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_color,
                          cmap="plasma", alpha=0.75)
    if y_color is not None:
        plt.colorbar(scatter, label="Color Variable")

    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pca_original(z, y_color):
    """
    Specialized PCA plot with percentile color clipping.
    """
    y_color = np.array(y_color)
    vmax = np.percentile(y_color, 95)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(z)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=y_color, cmap="plasma_r",
        alpha=0.7, vmin=y_color.min(), vmax=vmax
    )
    plt.colorbar(scatter, label="Error Function")

    plt.title("PCA of Latent Vectors (Error Function Colored)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# NEW: General PCA plot
# ------------------------------------------------------------

def plot_pca(X, y_labels=None, title="PCA Plot", cmap='coolwarm', size=10):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                          c=y_labels, cmap=cmap, s=size)
    plt.title(title)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)")

    if y_labels is not None:
        plt.colorbar(scatter, label="Class/Value")

    plt.grid(True)
    plt.show()


# ------------------------------------------------------------
# NEW: UMAP plot
# ------------------------------------------------------------

def plot_umap(X, y_labels, title_prefix="Data", cmap='coolwarm', size=10):
    reducer_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer_umap.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1],
                          c=y_labels, cmap=cmap, s=size)
    plt.title(f"{title_prefix} - UMAP")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.colorbar(scatter, label="Class/Value")
    plt.grid(True)
    plt.show()


# ------------------------------------------------------------
# NEW: t-SNE plot
# ------------------------------------------------------------

def plot_tsne(X, y_labels, title_prefix="Data", cmap='coolwarm', size=10):
    X_tsne = TSNE(n_components=2, perplexity=30,
                  learning_rate=200, random_state=42).fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                          c=y_labels, cmap=cmap, s=size)
    plt.title(f"{title_prefix} - t-SNE")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.colorbar(scatter, label="Class/Value")
    plt.grid(True)
    plt.show()


# ============================================================
#  GENERIC ERROR PLOT (AE or VAE)
# ============================================================

def plot_error(history):
    """
    Plots training and validation MSE loss for autoencoders or VAEs
    using standard Keras history object.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label="Train Loss")

    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label="Val Loss")

    plt.title("Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()







