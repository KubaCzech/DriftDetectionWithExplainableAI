import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from enum import Enum
from functools import wraps
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, probplot

# TODO 2: legendy w histogramie, kde, ecdf
# TODO 3: wspólna oś y, jesli nie to zrobic sharey
# TODO 4: nazwy osi


class PlotOptions(Enum):
    Mean = 'mean'
    Median = 'median'
    Both = 'both'


def plot(title: str, sharey: bool = True, palette: dict = None):
    """
    Decorator for grid-based drift plots.

    Parameters
    ----------
    title : str
        Figure title.
    sharey : bool, default=True
        Whether subplots share Y axis.
    """

    def decorator(plot_func):
        @wraps(plot_func)
        def wrapper(X_before, y_before, X_after, y_after, *args, **kwargs):
            if hasattr(y_before, "values"):
                y_before = y_before.values
            if hasattr(y_after, "values"):
                y_after = y_after.values

            if palette is None:
                default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            else:
                default_colors = palette
            kwargs['default_colors'] = default_colors

            features = X_before.columns
            classes = sorted(set(y_before).union(set(y_after)))

            fig, axes = plt.subplots(
                len(features), len(classes), figsize=(6 * len(classes), 5 * len(features)), sharey=sharey
            )

            # normalize axes shape
            if len(features) == 1:
                axes = axes[None, :]
            if len(classes) == 1:
                axes = axes[:, None]

            for i, feature in enumerate(features):
                for j, cl in enumerate(classes):
                    ax = axes[i, j]

                    old_vals = X_before.loc[y_before == cl, feature].dropna()
                    new_vals = X_after.loc[y_after == cl, feature].dropna()

                    if old_vals.empty or new_vals.empty:
                        ax.set_visible(False)
                        continue

                    plot_func(
                        ax=ax,
                        old_vals=old_vals,
                        new_vals=new_vals,
                        feature=feature,
                        cl=cl,
                        colors=default_colors,
                        *args,
                        **kwargs,
                    )

                    if i == 0:
                        ax.set_title(f'Class {cl}')
                    if j == 0:
                        ax.set_ylabel(feature)

            fig.suptitle(title, fontsize=14)
            plt.tight_layout()
            plt.show()

        return wrapper

    return decorator


# 1. Boxplot
@plot("Boxplot – Before vs After")
def _boxplot(ax, old_vals, new_vals, feature, show_, **_):
    median_color, mean_color = None, None
    legend_elements = []

    if show_ == PlotOptions.Median:
        bp = ax.boxplot([old_vals, new_vals], tick_labels=['Before', 'After'])
        median_color = bp['medians'][0].get_color()
    elif show_ == PlotOptions.Mean:
        bp = ax.boxplot(
            [old_vals, new_vals],
            tick_labels=['Before', 'After'],
            showmeans=True,
            meanline=True,
            medianprops=dict(visible=False),
        )
        mean_color = bp['means'][0].get_color()
    elif show_ == PlotOptions.Both:
        bp = ax.boxplot([old_vals, new_vals], tick_labels=['Before', 'After'], showmeans=True, meanline=True)
        median_color = bp['medians'][0].get_color()
        mean_color = bp['means'][0].get_color()
    else:
        raise ValueError("Unsupported Boxplot option")
    ax.set_xlabel(feature)

    if median_color is not None:
        legend_elements.append(Line2D([0], [0], color=median_color, lw=1, label='Median'))
    if mean_color is not None:
        legend_elements.append(Line2D([0], [0], color=mean_color, lw=1, linestyle='--', label='Mean'))
    ax.legend(handles=legend_elements)


# 2. Histogram
@plot("Histogram - Before vs After")
def _histogram(ax, old_vals, new_vals, colors, bins=30, **_):
    all_vals = np.concatenate([old_vals, new_vals])
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

    ax.hist(old_vals, bins=bin_edges, alpha=0.6, density=False, label='Before', color=colors[0])
    ax.hist(new_vals, bins=bin_edges, alpha=0.6, density=False, label='After', color=colors[1])
    both_color = np.mean([to_rgb(colors[0]), to_rgb(colors[1])], axis=0)

    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.6, label='Before'),
        Patch(facecolor=colors[1], alpha=0.6, label='After'),
        Patch(facecolor=both_color, alpha=0.6, label='Both'),
    ]
    ax.legend(handles=legend_elements)


# 3. Violin Plot
@plot("Violin Plot – Before vs After")
def _violin(ax, old_vals, new_vals, show_, **_):
    data = [old_vals, new_vals]
    if show_ == PlotOptions.Median:
        ax.violinplot(data, positions=[1, 2], showmeans=False, showmedians=True)
    elif show_ == PlotOptions.Mean:
        ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=False)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Before', 'After'])


# 4. QQ Plot
@plot("QQ Plot – Before vs After")
def _qq(ax, old_vals, new_vals, **_):
    ax.set_ylim(-0.05, 1.05)
    probplot(old_vals, dist="norm", plot=ax)
    probplot(new_vals, dist="norm", plot=ax)


# 5. KDE Plot
@plot("KDE Plot – Before vs After")
def _kde(ax, old_vals, new_vals, colors, **_):
    kde_old = gaussian_kde(old_vals)
    kde_new = gaussian_kde(new_vals)

    x_min = min(min(old_vals), min(new_vals))
    x_max = max(max(old_vals), max(new_vals))
    x_range = np.linspace(x_min, x_max, 1000)

    ax.plot(x_range, kde_old(x_range), label='Before', color=colors[0])
    ax.plot(x_range, kde_new(x_range), label='After', color=colors[1])
    ax.legend()


# 6. ECDF Plot
@plot("ECDF – Before vs After")
def _ecdf(ax, old_vals, new_vals, colors, **_):
    def ecdf(x):
        x = np.sort(x)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    ax.plot(*ecdf(old_vals), label='Before', color=colors[0])
    ax.plot(*ecdf(new_vals), label='After', color=colors[1])
    ax.legend()


def plot_boxplot(X_before, y_before, X_after, y_after, show_=PlotOptions.Median, **kwargs):
    """
    Plot boxplots of feature distributions between data blocks for each class label.

    Paramaters
    ----------
    X_before : pd.DataFrame
        Feature data from first data block.
    y_before : np.ndarray or pd.Series
        Class labels from first data block.
    X_after : pd.DataFrame
        Feature data from second data block.
    y_after : np.ndarray or pd.Series
        Class labels from second data block.
    """
    _boxplot(X_before, y_before, X_after, y_after, show_=show_, **kwargs)


def plot_histogram(X_before, y_before, X_after, y_after, **kwargs):
    """
    Plot histograms of feature distributions between data blocks for each class label.

    Paramaters
    ----------
    X_before : pd.DataFrame
        Feature data from first data block.
    y_before : np.ndarray or pd.Series
        Class labels from first data block.
    X_after : pd.DataFrame
        Feature data from second data block.
    y_after : np.ndarray or pd.Series
        Class labels from second data block.
    bins : int
        Number of bins for the histograms.
    """
    _histogram(X_before, y_before, X_after, y_after, **kwargs)


def plot_violin(X_before, y_before, X_after, y_after, show_=PlotOptions.Median):
    """
    Plot violin plots of feature distributions between data blocks for each class label.

    Paramaters
    ----------
    X_before : pd.DataFrame
        Feature data from first data block.
    y_before : np.ndarray or pd.Series
        Class labels from first data block.
    X_after : pd.DataFrame
        Feature data from second data block.
    y_after : np.ndarray or pd.Series
        Class labels from second data block.
    """
    _violin(X_before, y_before, X_after, y_after, show_=show_)


def plot_qq(X_before, y_before, X_after, y_after):
    """
    Plot QQ-plots of feature distributions between data blocks for each class label.

    Paramaters
    ----------
    X_before : pd.DataFrame
        Feature data from first data block.
    y_before : np.ndarray or pd.Series
        Class labels from first data block.
    X_after : pd.DataFrame
        Feature data from second data block.
    y_after : np.ndarray or pd.Series
        Class labels from second data block.
    """
    _qq(X_before, y_before, X_after, y_after)


def plot_kde(X_before, y_before, X_after, y_after):
    """
    Plot Kernel Distribution Estimation of feature distributions between data blocks for each class label.

    Paramaters
    ----------
    X_before : pd.DataFrame
        Feature data from first data block.
    y_before : np.ndarray or pd.Series
        Class labels from first data block.
    X_after : pd.DataFrame
        Feature data from second data block.
    y_after : np.ndarray or pd.Series
        Class labels from second data block.
    """
    _kde(X_before, y_before, X_after, y_after)


def plot_ecdf(X_before, y_before, X_after, y_after):
    """
    Plot boxplots of feature distributions between data blocks for each class label.

    Paramaters
    ----------
    X_before : pd.DataFrame
        Feature data from first data block.
    y_before : np.ndarray or pd.Series
        Class labels from first data block.
    X_after : pd.DataFrame
        Feature data from second data block.
    y_after : np.ndarray or pd.Series
        Class labels from second data block.
    """
    _ecdf(X_before, y_before, X_after, y_after)
