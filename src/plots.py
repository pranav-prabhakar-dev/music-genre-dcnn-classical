"""Figures used in the write-up and slides."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from . import config as C

COLORS = {"classical": "#1f77b4", "hiphop": "#ff7f0e", "metal": "#2ca02c",
          "pop": "#d62728", "blues": "#9467bd"}


def _save(fig, name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def mfcc_heatmaps(X, y, frames=300):
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True)
    for ax, genre in zip(axes, ["classical", "metal"]):
        m = X[np.where(y == C.GENRES.index(genre))[0][0]][:, :frames]
        im = ax.imshow(m, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"{genre.capitalize()} - MFCC (first {frames} frames)")
        ax.set_ylabel("MFCC #")
        fig.colorbar(im, ax=ax, pad=0.01)
    axes[-1].set_xlabel("Time frame")
    return _save(fig, "mfcc_heatmaps.png")


def training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ep = np.arange(1, len(history["train_loss"]) + 1)
    axes[0].plot(ep, history["train_loss"], label="train")
    axes[0].plot(ep, history["test_loss"], label="test")
    axes[0].set_title("Cross-entropy loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(ep, history["train_acc"], label="train")
    axes[1].plot(ep, history["test_acc"], label="test")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    return _save(fig, "dcnn_training_curves.png")


def confusion(y_true, y_pred, title, name):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(C.GENRES)))
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=C.GENRES, yticklabels=C.GENRES, ax=ax)
    ax.set_xlabel("Predicted genre"); ax.set_ylabel("True genre"); ax.set_title(title)
    return _save(fig, name)


def pca_3d(feature_sets, y, name="pca_features_3d.png", titles=None):
    """3D PCA scatter for each feature set (dict title -> array), coloured by genre."""
    n = len(feature_sets)
    fig = plt.figure(figsize=(5 * n, 4.6))
    for i, (title, F) in enumerate(feature_sets.items()):
        Z = PCA(n_components=3, random_state=C.SEED).fit_transform(StandardScaler().fit_transform(F))
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        for k, g in enumerate(C.GENRES):
            m = y == k
            ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], s=14, c=COLORS[g], label=g)
        ax.set_title(title)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    fig.axes[0].legend(loc="upper left", fontsize=8)
    return _save(fig, name)


def results_heatmap(df):
    """df: rows = feature sets, cols = algorithms, values = test accuracy."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.2, vmax=1.0, ax=ax,
                cbar_kws={"label": "Test accuracy"})
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_title("Classical algorithms: test accuracy by input feature")
    return _save(fig, "classical_results_heatmap.png")


def results_bars(df, dcnn_acc):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(df.columns))
    w = 0.8 / len(df.index)
    for i, feat in enumerate(df.index):
        ax.bar(x + i * w - 0.4 + w / 2, df.loc[feat].values, w, label=feat)
    ax.axhline(dcnn_acc, color="k", ls="--", lw=1, label=f"DCNN alone ({dcnn_acc:.2f})")
    ax.set_xticks(x, df.columns); ax.set_ylim(0, 1.05); ax.set_ylabel("Test accuracy")
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "classical_results_bars.png")
