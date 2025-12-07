import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
]

default_cmap = ListedColormap(colors)

color_map = {i: colors[i] for i in range(len(colors))}


def plot_drift(X_before, y_before, X_after, y_after, show=True, in_subplot=False):
    plt.figure(figsize=(12, 5))

    # Before drift
    plt.subplot(1, 2, 1)
    plt.scatter(X_before[:, 0], X_before[:, 1], c=y_before, cmap="coolwarm", s=20, alpha=0.7)
    plt.title('Before')
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # After drift
    plt.subplot(1, 2, 2)
    plt.scatter(X_after[:, 0], X_after[:, 1], c=y_after, cmap="coolwarm", s=20, alpha=0.7)
    plt.title('After')
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.tight_layout()
    plt.show()


def plot_clusters(X, labels, title, color_map=None):
    # plot clusters with different colors
    # Function used later
    unique_labels = sorted(np.unique(labels))
    if color_map is None:
        cmap = plt.cm.viridis
        color_map = {ul: cmap(i / len(unique_labels)) for i, ul in enumerate(unique_labels)}

    for ul in unique_labels:
        plt.scatter(X[labels == ul, 0], X[labels == ul, 1], color=color_map[ul], label=f'Cluster {ul}', s=30)

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    return color_map


def plot_drift_clustered(
    X_before, X_after, labels_before, labels_after, color_map=color_map, show=True, in_subplot=False
):
    # plot and color clusters before and after the drift
    if not in_subplot:
        plt.figure(figsize=(12, 5))

    # Before drift
    plt.subplot(1, 2, 1)
    plot_clusters(X_before, labels_before, "Before Drift", color_map=color_map)

    # After drift
    plt.subplot(1, 2, 2)
    plot_clusters(X_after, labels_after, "After Drift", color_map=color_map)

    if show:
        plt.tight_layout()
        plt.show()


def plot_final_comparison(X_before, X_after, y_before, y_after, labels_before, labels_after):
    # Plot 4 subplots: clustered data before, clustered data after, drift before, drift after
    # matplotlib does not support nested subplots, so we need to create a 2x2 grid and plot in each cell
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: before drift
    plt.sca(axes[0, 0])
    plot_drift(X_before, y_before, X_after, y_after, show=False)

    # Top-right: after drift
    plt.sca(axes[0, 1])
    plot_drift_clustered(X_before, X_after, labels_before, labels_after, color_map=color_map, in_subplot=True)

    # Bottom-left: clustered before only
    plt.sca(axes[1, 0])
    plot_clusters(X_before, labels_before, "Clusters Before", color_map)

    # Bottom-right: clustered after only
    plt.sca(axes[1, 1])
    plot_clusters(X_after, labels_after, "Clusters After", color_map)

    plt.tight_layout()
    plt.show()


def plot_clusters_by_class(X_before, X_after, y_before, y_after, labels_before, labels_after):
    # Plot clusters for each class separately before and after drift
    classes = sorted(set(y_before).union(set(y_after)))
    n_classes = len(classes)

    fig, axes = plt.subplots(n_classes, 2, figsize=(12, 5 * n_classes))

    # If there is only 1 class, axes is not 2D; this line fixes that
    if n_classes == 1:
        axes = np.array([axes])

    for row, cl in enumerate(classes):
        # before drift
        ax_before = axes[row, 0]
        plt.sca(ax_before)

        mask_before = np.array(y_before) == cl
        plot_clusters(
            X_before[mask_before], labels_before[mask_before], title=f"Before Drift – Class {cl}", color_map=color_map
        )

        # after drift
        ax_after = axes[row, 1]
        plt.sca(ax_after)

        mask_after = np.array(y_after) == cl
        plot_clusters(
            X_after[mask_after], labels_after[mask_after], title=f"After Drift – Class {cl}", color_map=color_map
        )

    plt.tight_layout()
    plt.show()


def visualize_centers_shift():
    pass
