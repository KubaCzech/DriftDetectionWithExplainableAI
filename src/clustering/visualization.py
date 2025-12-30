import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import wraps
from typing import Sequence, Union, Callable
from src.common import DataDimensionsReducer, ReducerType

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

color_map = {i: colors[i] for i in range(len(colors))}


def lol():
    """
    Reduce the dimensionality of the dataset to n_components.

    Parameters
    ----------
    X : pd.DataFrame
        The input data to be reduced.
    n_components : int, default=2
        The number of dimensions to reduce to.
    reducer : ReducerType, default=PCA
        The type of dimensionality reduction to apply.

    Returns
    -------
    pd.DataFrame
        The dimensionally reduced data.
    """
    pass


def reduce_dimensions(
    n_components: int = 2,
    reducer: ReducerType = ReducerType.PCA,
) -> Callable:
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ddr = None  # lazy init

            def reduce_df(X):
                nonlocal ddr
                if not isinstance(X, pd.DataFrame):
                    return X

                if X.shape[1] <= n_components:
                    return X

                if ddr is None:
                    ddr = DataDimensionsReducer(reducer_type=reducer, n_components=n_components)
                    return ddr.fit_transform(X, return_df=True)

                return ddr.transform(X, return_df=True)

            # positional args
            new_args = [reduce_df(arg) for arg in args]

            # keyword args
            for k, v in kwargs.items():
                kwargs[k] = reduce_df(v)

            return func(*new_args, **kwargs)

        return wrapper

    return decorator


def _plot_clusters(X: pd.DataFrame, labels: Sequence[Union[float, int]], title: str) -> None:
    """
    Plot clusters with different colors for 2D data.

    Parameters
    ----------
    X : pd.DataFrame
        Data points to plot.
    labels : Sequence[int or float]
        Cluster labels for each data point.
    title : str
        Title of the plot. Usually indicating before or after drift.

    Notes
    -----
    Global color_map is used to ensure consistent coloring across plots.
    Function used internally by other plotting functions.
    """
    if hasattr(X, "values"):
        X = X.values
    if hasattr(labels, "values"):
        labels = labels.values

    # Ensure labels are integers
    if labels.dtype.kind == 'f':
        labels = labels.astype(int)

    unique_labels = sorted(np.unique(labels))

    for ul in unique_labels:
        color = color_map.get(ul, "#333333")  # Default to dark gray if missing
        plt.scatter(X[labels == ul, 0], X[labels == ul, 1], color=color, label=f'Cluster {ul}', s=30)

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()


@reduce_dimensions()
def plot_drift_clustered(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    labels_before: Sequence[Union[int, float]],
    labels_after: Sequence[Union[int, float]],
    show: bool = True,
    in_subplot: bool = False,
) -> None:
    """
    Plot clusters from first and second data block to visualize drift for 2D data.

    Parameters
    ----------
    X_before: pd.DataFrame
        Feature values from first data block.
    X_after: pd.DataFrame
        Feature values from second data block.
    labels_before: Sequence[int or float]
        Cluster labels for first data block.
    labels_after: Sequence[int or float]
        Cluster labels for second data block.
    show: bool, default=True
        Whether to display the plot immediately.
    in_subplot: bool, default=False
        Whether the function is called within a subplot context.
    """
    if not in_subplot:
        plt.figure(figsize=(12, 5))

    # Normalize inputs
    if hasattr(labels_before, "values"):
        labels_before = labels_before.values
    if hasattr(labels_after, "values"):
        labels_after = labels_after.values

    if labels_before.dtype.kind == 'f':
        labels_before = labels_before.astype(int)
    if labels_after.dtype.kind == 'f':
        labels_after = labels_after.astype(int)

    # # Dynamic color map generation if not provided
    # if color_map is None:
    #     unique_before = np.unique(labels_before)
    #     unique_after = np.unique(labels_after)
    #     all_labels = sorted(list(set(unique_before) | set(unique_after)))

    #     if len(all_labels) <= len(colors):
    #         color_map = {label: colors[i] for i, label in enumerate(all_labels)}
    #     else:
    #         # Use tab20 for more distinct colors if many clusters
    #         cmap = plt.cm.get_cmap('tab20')
    #         color_map = {label: cmap(i / len(all_labels)) for i, label in enumerate(all_labels)}

    # First data block
    plt.subplot(1, 2, 1)
    _plot_clusters(X_before, labels_before, "Before Drift")

    # Second data block
    plt.subplot(1, 2, 2)
    _plot_clusters(X_after, labels_after, "After Drift")

    if show:
        plt.tight_layout()
        plt.show()


@reduce_dimensions()
def plot_clusters_by_class(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    y_before: Sequence[int],
    y_after: Sequence[int],
    cluster_labels_before: Sequence[int],
    cluster_labels_after: Sequence[int],
) -> None:
    """
    Plot clusters separated by class labels to visualize drift per class (not overall) for 2D data.

    Parameters
    ----------
    X_before: pd.DataFrame
        Feature values from first data block.
    X_after: pd.DataFrame
        Feature values from second data block.
    y_before: Sequence
        Class labels for first data block.
    y_after: Sequence
        Class labels for second data block.
    cluster_labels_before: Sequence[int]
        Cluster labels for first data block.
    cluster_labels_after: Sequence[int]
        Cluster labels for second data block.

    Notes
    -----
    Creates a subplot for each class, showing clusters before and after drift.
    """
    classes = sorted(set(y_before).union(set(y_after)))
    n_classes = len(classes)

    fig, axes = plt.subplots(n_classes, 2, figsize=(12, 5 * n_classes))

    # If there is only 1 class, axes is not 2D; this line fixes that
    if n_classes == 1:
        axes = np.array([axes])

    for row, cl in enumerate(classes):
        # first data block
        ax_before = axes[row, 0]
        plt.sca(ax_before)

        mask_before = np.array(y_before) == cl
        _plot_clusters(X_before[mask_before], cluster_labels_before[mask_before], title=f"Before Drift – Class {cl}")

        # second data block
        ax_after = axes[row, 1]
        plt.sca(ax_after)

        mask_after = np.array(y_after) == cl
        _plot_clusters(X_after[mask_after], cluster_labels_after[mask_after], title=f"After Drift – Class {cl}")

    plt.tight_layout()
    plt.show()


@reduce_dimensions()
def plot_centers_shift(
    X_before: pd.DataFrame, X_after: pd.DataFrame, cluster_labels_old: Sequence[int], cluster_labels_new: Sequence[int]
) -> None:
    """
    Plot shifts of cluster centroids between two data blocks for 2D data.

    Parameters
    ----------
    X_old: pd.DataFrame
        Feature values from first data block.
    X_new: pd.DataFrame
        Feature values from second data block.
    cluster_labels_old: Sequence[int]
        Cluster labels for first data block.
    cluster_labels_new: Sequence[int]
        Cluster labels for second data block.
    """
    unique_labels_old = set(cluster_labels_old)
    unique_labels_new = set(cluster_labels_new)

    all_labels = sorted(list(unique_labels_old.union(unique_labels_new)))

    plt.figure(figsize=(8, 6))

    plt.scatter([], [], marker='x', color='black', label='Center (before)')
    plt.scatter([], [], marker='o', color='black', label='Center (after)')

    for label in all_labels:
        plt.scatter([], [], marker="o", color=color_map[label], label=f"Cluster {label}")

    for label in all_labels:
        center_old = None
        center_new = None

        if label in unique_labels_old:
            center_old = X_before[cluster_labels_old == label].mean().values

        if label in unique_labels_new:
            center_new = X_after[cluster_labels_new == label].mean().values

        # cluster shifted
        if center_old is not None and center_new is not None:
            plt.plot(
                [center_old[0], center_new[0]],
                [center_old[1], center_new[1]],
                linewidth=2,
                linestyle='--',
                color=color_map[label],
            )
            plt.scatter(*center_old, marker="x", color=color_map[label], s=60)
            plt.scatter(*center_new, marker="o", color=color_map[label], s=60)

        # cluster disappeared (only old)
        elif center_old is not None:
            plt.scatter(*center_old, marker="x", color=color_map[label], s=60)

        # cluster appeared (only new)
        elif center_new is not None:
            plt.scatter(*center_new, marker="o", color=color_map[label], s=60)

    plt.title("Cluster centroid shifts")
    plt.xlabel(X_before.columns[0])
    plt.ylabel(X_before.columns[1])
    plt.grid(True)
    plt.legend()
    plt.show()
