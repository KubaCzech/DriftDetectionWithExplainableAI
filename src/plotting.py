import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def plot_feature_distribution_over_time(X, n_features, feature_names,
                                        time_steps_before, time_steps_after,
                                        X_before, X_after,
                                        mask_before_c0, mask_before_c1,
                                        mask_after_c0, mask_after_c1,
                                        class_colors,
                                        title='Feature Distributions over Time'):
    """
    Creates a figure showing feature distributions over time.
    """
    fig, axes = plt.subplots(n_features, 2,
                             figsize=(12, 4 * n_features),
                             squeeze=False)
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.0)

    for i in range(n_features):
        ax_before = axes[i, 0]
        ax_after = axes[i, 1]
        feat_name = feature_names[i]

        # Before Drift (Window 1)
        ax_before.scatter(time_steps_before[mask_before_c0],
                          X_before[mask_before_c0, i],
                          alpha=0.5, s=20, label='Class 0',
                          color=class_colors[0])
        ax_before.scatter(time_steps_before[mask_before_c1],
                          X_before[mask_before_c1, i],
                          alpha=0.5, s=20, label='Class 1',
                          color=class_colors[1])
        ax_before.set_xlabel('Time')
        ax_before.set_ylabel(f'{feat_name} Value')
        ax_before.set_title(f'{feat_name} - Window 1')
        ax_before.legend()
        ax_before.grid(True, alpha=0.3)

        # After Drift (Window 2)
        ax_after.scatter(time_steps_after[mask_after_c0],
                         X_after[mask_after_c0, i],
                         alpha=0.5, s=20, label='Class 0',
                         color=class_colors[0])
        ax_after.scatter(time_steps_after[mask_after_c1],
                         X_after[mask_after_c1, i],
                         alpha=0.5, s=20, label='Class 1',
                         color=class_colors[1])
        ax_after.set_xlabel('Time')
        ax_after.set_ylabel(f'{feat_name} Value')
        ax_after.set_title(f'{feat_name} - Window 2')
        ax_after.legend()
        ax_after.grid(True, alpha=0.3)

    plt.tight_layout()
    # Increased top margin
    fig.subplots_adjust(top=0.92)
    return fig


def plot_feature_target_relationship(X, n_features, feature_names,
                                     X_before, X_after,
                                     mask_before_c0, mask_before_c1,
                                     mask_after_c0, mask_after_c1,
                                     class_colors,
                                     title='Feature vs Target Relationship'):
    """
    Creates a figure showing feature vs target relationship.
    """
    fig, axes = plt.subplots(n_features, 2,
                             figsize=(12, 4 * n_features),
                             squeeze=False)
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.0)

    for i in range(n_features):
        ax_before = axes[i, 0]
        ax_after = axes[i, 1]
        feat_name = feature_names[i]

        # Add jitter to Y for better visualization
        y_jitter_before_c0 = (np.random.normal(0, 0.02,
                                               np.sum(mask_before_c0)))
        y_jitter_before_c1 = (1 + np.random.normal(0, 0.02,
                                                   np.sum(mask_before_c1)))
        y_jitter_after_c0 = (np.random.normal(0, 0.02,
                                              np.sum(mask_after_c0)))
        y_jitter_after_c1 = (1 + np.random.normal(0, 0.02,
                                                  np.sum(mask_after_c1)))

        # Before Drift (Window 1)
        ax_before.scatter(X_before[mask_before_c0, i], y_jitter_before_c0,
                          alpha=0.5, s=20, label='Class 0',
                          color=class_colors[0])
        ax_before.scatter(X_before[mask_before_c1, i], y_jitter_before_c1,
                          alpha=0.5, s=20, label='Class 1',
                          color=class_colors[1])
        ax_before.set_xlabel(feat_name)
        ax_before.set_ylabel('Target Class (with jitter)')
        ax_before.set_title(f'{feat_name} vs Target - Window 1')
        ax_before.set_yticks([0, 1])
        ax_before.set_yticklabels(['Class 0', 'Class 1'])
        ax_before.legend()
        ax_before.grid(True, alpha=0.3)

        # After Drift (Window 2)
        ax_after.scatter(X_after[mask_after_c0, i], y_jitter_after_c0,
                         alpha=0.5, s=20, label='Class 0',
                         color=class_colors[0])
        ax_after.scatter(X_after[mask_after_c1, i], y_jitter_after_c1,
                         alpha=0.5, s=20, label='Class 1',
                         color=class_colors[1])
        ax_after.set_xlabel(feat_name)
        ax_after.set_ylabel('Target Class (with jitter)')
        ax_after.set_title(f'{feat_name} vs Target - Window 2')
        ax_after.set_yticks([0, 1])
        ax_after.set_yticklabels(['Class 0', 'Class 1'])
        ax_after.legend()
        ax_after.grid(True, alpha=0.3)

    plt.tight_layout()
    # Increased top margin
    fig.subplots_adjust(top=0.92)
    return fig


def plot_class_distribution(class_dist_before, class_dist_after, class_colors,
                            title='Class Distribution'):
    """
    Creates a figure showing class distribution.
    """
    fig, (ax_class_before, ax_class_after) = plt.subplots(1, 2, figsize=(12, 6))
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.0)

    # Class distributions - Before (Window 1)
    ax_class_before.bar(['Class 0', 'Class 1'], class_dist_before,
                        color=[class_colors[0], class_colors[1]],
                        alpha=0.7, edgecolor='black')
    ax_class_before.set_ylabel('Proportion')
    ax_class_before.set_title('Window 1')
    ax_class_before.set_ylim([0, 1])
    ax_class_before.grid(True, alpha=0.3, axis='y')

    # Class distributions - After (Window 2)
    ax_class_after.bar(['Class 0', 'Class 1'], class_dist_after,
                       color=[class_colors[0], class_colors[1]],
                       alpha=0.7, edgecolor='black')
    ax_class_after.set_ylabel('Proportion')
    ax_class_after.set_title('Window 2')
    ax_class_after.set_ylim([0, 1])
    ax_class_after.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    # Increased top margin
    fig.subplots_adjust(top=0.88)
    return fig


def plot_feature_space(n_features, feature_names, X_before, X_after,
                       y_before, y_after, class_colors,
                       title='Feature Space'):
    """
    Creates a figure showing feature space (1D, 2D, or PCA).
    """
    fig, (ax_fs_before, ax_fs_after) = plt.subplots(1, 2, figsize=(14, 7))
    fs_title_suffix = ""

    if n_features == 1:
        # 1D plot (Histogram)
        ax_fs_before.hist(X_before[y_before == 0], bins=30, alpha=0.5,
                          label='Class 0', color=class_colors[0])
        ax_fs_before.hist(X_before[y_before == 1], bins=30, alpha=0.5,
                          label='Class 1', color=class_colors[1])
        ax_fs_after.hist(X_after[y_after == 0], bins=30, alpha=0.5,
                         label='Class 0', color=class_colors[0])
        ax_fs_after.hist(X_after[y_after == 1], bins=30, alpha=0.5,
                         label='Class 1', color=class_colors[1])
        ax_fs_before.set_xlabel(feature_names[0])
        ax_fs_after.set_xlabel(feature_names[0])
        ax_fs_before.set_ylabel('Frequency')
        ax_fs_after.set_ylabel('Frequency')

    elif n_features == 2:
        # 2D plot
        ax_fs_before.scatter(X_before[y_before == 0, 0],
                             X_before[y_before == 0, 1],
                             alpha=0.5, s=20, label='Class 0',
                             color=class_colors[0])
        ax_fs_before.scatter(X_before[y_before == 1, 0],
                             X_before[y_before == 1, 1],
                             alpha=0.5, s=20, label='Class 1',
                             color=class_colors[1])
        ax_fs_after.scatter(X_after[y_after == 0, 0],
                            X_after[y_after == 0, 1],
                            alpha=0.5, s=20, label='Class 0',
                            color=class_colors[0])
        ax_fs_after.scatter(X_after[y_after == 1, 0],
                            X_after[y_after == 1, 1],
                            alpha=0.5, s=20, label='Class 1',
                            color=class_colors[1])
        ax_fs_before.set_xlabel(feature_names[0])
        ax_fs_before.set_ylabel(feature_names[1])
        ax_fs_after.set_xlabel(feature_names[0])
        ax_fs_after.set_ylabel(feature_names[1])

    else:
        # > 2D plot (Use PCA)
        pca = PCA(n_components=2, random_state=42)
        # Fit PCA on combined data to ensure same projection
        X_combined = np.concatenate([X_before, X_after])
        X_2d = pca.fit_transform(X_combined)

        X_2d_before = X_2d[:len(X_before)]
        X_2d_after = X_2d[len(X_before):]

        y_before_pca = y_before
        y_after_pca = y_after

        ax_fs_before.scatter(X_2d_before[y_before_pca == 0, 0],
                             X_2d_before[y_before_pca == 0, 1],
                             alpha=0.5, s=20, label='Class 0',
                             color=class_colors[0])
        ax_fs_before.scatter(X_2d_before[y_before_pca == 1, 0],
                             X_2d_before[y_before_pca == 1, 1],
                             alpha=0.5, s=20, label='Class 1',
                             color=class_colors[1])
        ax_fs_after.scatter(X_2d_after[y_after_pca == 0, 0],
                            X_2d_after[y_after_pca == 0, 1],
                            alpha=0.5, s=20, label='Class 0',
                            color=class_colors[0])
        ax_fs_after.scatter(X_2d_after[y_after_pca == 1, 0],
                            X_2d_after[y_after_pca == 1, 1],
                            alpha=0.5, s=20, label='Class 1',
                            color=class_colors[1])
        ax_fs_before.set_xlabel('Principal Component 1')
        ax_fs_before.set_ylabel('Principal Component 2')
        ax_fs_after.set_xlabel('Principal Component 1')
        ax_fs_after.set_ylabel('Principal Component 2')
        fs_title_suffix = " (PCA)"

    ax_fs_before.set_title('Window 1')
    ax_fs_before.legend()
    ax_fs_before.grid(True, alpha=0.3)

    ax_fs_after.set_title('Window 2')
    ax_fs_after.legend()
    ax_fs_after.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title + fs_title_suffix,
                     fontsize=16, fontweight='bold', y=1.0)

    plt.tight_layout()
    # Increased top margin
    fig.subplots_adjust(top=0.88)
    return fig


def visualize_data_stream(X, y, window_before_start, window_after_start,
                          window_length, feature_names,
                          title_feat_dist='Feature Distributions over Time',
                          title_feat_target='Feature vs Target Relationship',
                          title_class_dist='Class Distribution',
                          title_feat_space='Feature Space'):
    """
    Visualize the data stream for two specific windows.

    Creates four separate figures:
    1. Feature distributions over time
    2. Feature-target relationships
    3. Class distributions
    4. 2D (PCA) Feature Space

    Parameters
    ----------
    X : array-like (n_samples, n_features)
        Feature matrix
    y : array-like (n_samples,)
        Binary class labels
    window_before_start : int
        Start index for the first window
    window_after_start : int
        Start index for the second window
    window_length : int
        Length of the windows
    feature_names : list
        Names of features
    """
    # Convert to numpy if pandas
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    # Define windows
    start_before = window_before_start
    end_before = start_before + window_length

    start_after = window_after_start
    end_after = start_after + window_length

    # Slice data for analysis
    X_before = X[start_before:end_before]
    y_before = y[start_before:end_before]

    X_after = X[start_after:end_after]
    y_after = y[start_after:end_after]

    n_features = X.shape[1]

    # Create time steps for plotting
    time_steps_before = np.arange(start_before, start_before + len(X_before))
    time_steps_after = np.arange(start_after, start_after + len(X_after))

    # Calculate class distributions
    class_dist_before = [np.mean(y_before == 0), np.mean(y_before == 1)]
    class_dist_after = [np.mean(y_after == 0), np.mean(y_after == 1)]

    # Print distribution info (kept for backward compatibility with dashboard capture)
    print("=" * 70)
    print("TARGET VARIABLE (Y) DISTRIBUTION")
    print("=" * 70)
    print(f"Window 1 ({start_before}-{end_before}): "
          f"Class 0: {class_dist_before[0]:.2%}, "
          f"Class 1: {class_dist_before[1]:.2%}")
    print(f"Window 2 ({start_after}-{end_after}):  "
          f"Class 0: {class_dist_after[0]:.2%}, "
          f"Class 1: {class_dist_after[1]:.2%}")
    print("=" * 70)

    # Colors for the two classes
    class_colors = {0: '#FF6B6B', 1: '#4ECDC4'}

    # Create masks for different periods and classes
    mask_before_c0 = (y_before == 0)
    mask_before_c1 = (y_before == 1)
    mask_after_c0 = (y_after == 0)
    mask_after_c1 = (y_after == 1)

    figs = []

    # 1. Feature Distributions over Time
    figs.append(plot_feature_distribution_over_time(
        X, n_features, feature_names, time_steps_before, time_steps_after,
        X_before, X_after, mask_before_c0, mask_before_c1,
        mask_after_c0, mask_after_c1, class_colors,
        title=title_feat_dist
    ))

    # 2. Feature vs Target Relationship
    figs.append(plot_feature_target_relationship(
         X, n_features, feature_names, X_before, X_after,
         mask_before_c0, mask_before_c1, mask_after_c0, mask_after_c1,
         class_colors,
         title=title_feat_target
    ))

    # 3. Class Distribution
    figs.append(plot_class_distribution(class_dist_before, class_dist_after,
                                        class_colors,
                                        title=title_class_dist))

    # 4. Feature Space
    figs.append(plot_feature_space(n_features, feature_names, X_before,
                                   X_after, y_before, y_after, class_colors,
                                   title=title_feat_space))

    return figs
