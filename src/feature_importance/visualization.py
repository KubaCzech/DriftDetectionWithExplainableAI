import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def visualize_data_stream(X, y, window_before_start, window_after_start, window_length, feature_names):
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
    
    # Create time steps for plotting (relative to the window start or absolute?)
    # Let's use absolute indices for x-axis to show where they are in the stream
    time_steps_before = np.arange(start_before, end_before)
    time_steps_after = np.arange(start_after, end_after)

    # Calculate class distributions
    class_dist_before = [np.mean(y_before == 0), np.mean(y_before == 1)]
    class_dist_after = [np.mean(y_after == 0), np.mean(y_after == 1)]

    print("=" * 70)
    print("TARGET VARIABLE (Y) DISTRIBUTION")
    print("=" * 70)
    print(f"Window 1 ({start_before}-{end_before}): Class 0: {class_dist_before[0]:.2%}, "
          f"Class 1: {class_dist_before[1]:.2%}")
    print(f"Window 2 ({start_after}-{end_after}):  Class 0: {class_dist_after[0]:.2%}, "
          f"Class 1: {class_dist_after[1]:.2%}")
    print("=" * 70)

    # Colors for the two classes
    class_colors = {0: '#FF6B6B', 1: '#4ECDC4'}

    # Create masks for different periods and classes (local to the sliced data)
    mask_before_c0 = (y_before == 0)
    mask_before_c1 = (y_before == 1)
    mask_after_c0 = (y_after == 0)
    mask_after_c1 = (y_after == 1)

    # 🚀 --- Figure: Feature Distributions over Time --- 🚀
    fig1, axes1 = plt.subplots(n_features, 2,
                               figsize=(12, 4 * n_features),
                               squeeze=False)
    fig1.suptitle('Feature Distributions over Time',
                  fontsize=16, fontweight='bold', y=1.0)

    for i in range(n_features):
        ax_before = axes1[i, 0]
        ax_after = axes1[i, 1]
        feat_name = feature_names[i]
        feat_data = X[:, i]

        # Before Drift (Window 1)
        ax_before.scatter(time_steps_before[mask_before_c0], X_before[mask_before_c0, i],
                          alpha=0.5, s=20, label='Class 0',
                          color=class_colors[0])
        ax_before.scatter(time_steps_before[mask_before_c1], X_before[mask_before_c1, i],
                          alpha=0.5, s=20, label='Class 1',
                          color=class_colors[1])
        ax_before.set_xlabel('Time')
        ax_before.set_ylabel(f'{feat_name} Value')
        ax_before.set_title(f'{feat_name} - Window 1')
        ax_before.legend()
        ax_before.grid(True, alpha=0.3)

        # After Drift (Window 2)
        ax_after.scatter(time_steps_after[mask_after_c0], X_after[mask_after_c0, i],
                         alpha=0.5, s=20, label='Class 0',
                         color=class_colors[0])
        ax_after.scatter(time_steps_after[mask_after_c1], X_after[mask_after_c1, i],
                         alpha=0.5, s=20, label='Class 1',
                         color=class_colors[1])
        ax_after.set_xlabel('Time')
        ax_after.set_ylabel(f'{feat_name} Value')
        ax_after.set_title(f'{feat_name} - Window 2')
        ax_after.legend()
        ax_after.grid(True, alpha=0.3)

    plt.tight_layout()
    # Increased top margin
    fig1.subplots_adjust(top=0.92)
    plt.show()

    # 🔗 --- Figure: Feature vs Target Relationship --- 🔗
    fig2, axes2 = plt.subplots(n_features, 2,
                               figsize=(12, 4 * n_features),
                               squeeze=False)
    fig2.suptitle('Feature vs Target Relationship',
                  fontsize=16, fontweight='bold', y=1.0)

    for i in range(n_features):
        ax_before = axes2[i, 0]
        ax_after = axes2[i, 1]
        feat_name = feature_names[i]
        feat_data = X[:, i]

        # Add jitter to Y for better visualization
        y_jitter_before_c0 = (np.random.normal(0, 0.02, np.sum(mask_before_c0)))
        y_jitter_before_c1 = (1 + np.random.normal(0, 0.02, np.sum(mask_before_c1)))
        y_jitter_after_c0 = (np.random.normal(0, 0.02, np.sum(mask_after_c0)))
        y_jitter_after_c1 = (1 + np.random.normal(0, 0.02, np.sum(mask_after_c1)))

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
    fig2.subplots_adjust(top=0.92)
    plt.show()

    # 📊 --- Figure: Class Distribution --- 📊
    fig3, (ax_class_before, ax_class_after) = plt.subplots(1, 2, figsize=(12, 6))
    fig3.suptitle('Class Distribution',
                  fontsize=16, fontweight='bold', y=1.0)

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
    fig3.subplots_adjust(top=0.88)
    plt.show()

    # 🗺️ --- Figure: Feature Space --- 🗺️
    fig4, (ax_fs_before, ax_fs_after) = plt.subplots(1, 2, figsize=(14, 7))
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

    fig4.suptitle('Feature Space' + fs_title_suffix,
                  fontsize=16, fontweight='bold', y=1.0)

    plt.tight_layout()
    # Increased top margin
    fig4.subplots_adjust(top=0.88)
    plt.show()


def visualize_data_drift_analysis(analysis_result, feature_names,
                                  show_boxplot=True):
    """
    Visualize the results of data drift analysis.

    Parameters
    ----------
    analysis_result : dict
        Result dictionary from compute_data_drift_analysis
    feature_names : list
        Names of features
    show_boxplot : bool, default=True
        Whether to display the boxplot of importance score distributions.
    """
    fi_result = analysis_result['importance_result']
    importance_mean = analysis_result['importance_mean']
    importance_std = analysis_result['importance_std']
    n_features = len(feature_names)

    # Determine the number of subplots
    n_plots = 2 if show_boxplot else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    fig.suptitle(f'Feature Importance for Data Drift (X-only Classification) '
                 f'- {fi_result["method"]}',
                 fontsize=14, fontweight='bold')

    # Ensure axes is an array even for 1 plot
    if n_plots == 1:
        axes = [axes]

    # Plot 1: Bar plot
    ax = axes[0]
    x_pos = np.arange(n_features)
    ax.bar(x_pos, importance_mean, yerr=importance_std,
           color='#e74c3c', alpha=0.8, edgecolor='black', capsize=5)
    ax.set_ylabel(f'Importance Score ({fi_result["method"]})')
    ax.set_title('Feature Importance for Detecting Time-Period '
                 '(Data Drift)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Box plot (Conditional)
    if show_boxplot:
        ax = axes[1]
        importances = fi_result['importances']

        # Handle cases where importances might not be (n_features, n_samples)
        if (importances.ndim == 2 and importances.shape[0] != n_features and
                importances.shape[1] == n_features):
            importances = importances.T

        # Ensure importances is (n_features, n_samples) before boxplot
        if importances.shape[0] == n_features:
            bp = ax.boxplot([importances[i] for i in range(n_features)],
                            tick_labels=feature_names, patch_artist=True,
                            notch=True, showmeans=True)

            for patch in bp['boxes']:
                patch.set_facecolor('#e74c3c')
                patch.set_alpha(0.7)
            ax.set_ylabel(f'{fi_result["method"]} Score')
            ax.set_xlabel('Features')
            ax.set_title(f'Distribution of {fi_result["method"]} Scores')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, "Boxplot not available (SHAP/LIME 1D result)",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            ax.set_axis_off()

    plt.tight_layout()
    plt.show()


def visualize_concept_drift_analysis(analysis_result, feature_names,
                                     show_boxplot=True):
    """
    Visualize the results of concept drift analysis.

    Parameters
    ----------
    analysis_result : dict
        Result dictionary from compute_concept_drift_analysis
    feature_names : list
        Names of features (e.g., ['X1', 'X2', 'Y'])
    show_boxplot : bool, default=True
        Whether to display the boxplot of importance score distributions.
    """
    fi_result = analysis_result['importance_result']
    importance_mean = analysis_result['importance_mean']
    importance_std = analysis_result['importance_std']
    n_features = len(feature_names)

    # Determine the number of subplots
    n_plots = 2 if show_boxplot else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    fig.suptitle(f'Feature Importance for Concept Drift (X+Y Classification) '
                 f'- {fi_result["method"]}',
                 fontsize=14, fontweight='bold')

    # Ensure axes is an array even for 1 plot
    if n_plots == 1:
        axes = [axes]

    # Plot 1: Bar plot
    ax = axes[0]
    x_pos = np.arange(n_features)
    ax.bar(x_pos, importance_mean, yerr=importance_std,
           color='#e74c3c', alpha=0.8, edgecolor='black', capsize=5)
    ax.set_ylabel(f'Importance Score ({fi_result["method"]})')
    ax.set_title(f'{fi_result["method"]} for Detecting '
                 'Time-Period (Concept Drift)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Box plot (Conditional)
    if show_boxplot:
        ax = axes[1]
        importances = fi_result['importances']

        # Handle cases where importances might not be (n_features, n_samples)
        if (importances.ndim == 2 and importances.shape[0] != n_features and
                importances.shape[1] == n_features):
            importances = importances.T

        # Ensure importances is (n_features, n_samples) before boxplot
        if importances.shape[0] == n_features:
            bp = ax.boxplot([importances[i] for i in range(n_features)],
                            tick_labels=feature_names, patch_artist=True,
                            notch=True, showmeans=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#e74c3c')
                patch.set_alpha(0.7)
            ax.set_ylabel(f'{fi_result["method"]} Score')
            ax.set_xlabel('Features')
            ax.set_title(f'Distribution of {fi_result["method"]} Scores')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, "Boxplot not available (SHAP/LIME 1D result)",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            ax.set_axis_off()

    plt.tight_layout()
    plt.show()


def visualize_predictive_importance_shift(analysis_result, feature_names,
                                          show_boxplot=True):
    """
    Visualize the results of predictive importance shift analysis.

    Parameters
    ----------
    analysis_result : dict
        Result dictionary from compute_predictive_importance_shift
    feature_names : list
        Names of features
    show_boxplot : bool, default=True
        Whether to display the boxplot of importance score distributions.
    """
    fi_before = analysis_result['fi_before']
    fi_after = analysis_result['fi_after']
    n_features = len(feature_names)

    # Determine the number of subplots
    n_plots = 2 if show_boxplot else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    fig.suptitle(f'Predictive Feature Importance (NN Before vs After Drift) '
                 f'- {fi_before["method"]}',
                 fontsize=14, fontweight='bold')

    # Ensure axes is an array even for 1 plot
    if n_plots == 1:
        axes = [axes]

    # Plot 1: Bar comparison
    ax = axes[0]
    x_pos = np.arange(n_features)
    width = 0.35
    ax.bar(x_pos - width/2, fi_before['importances_mean'], width,
           yerr=fi_before['importances_std'],
           label='NN (Trained BEFORE Drift)',
           color='#1abc9c', alpha=0.8, edgecolor='black', capsize=5)
    ax.bar(x_pos + width/2, fi_after['importances_mean'], width,
           yerr=fi_after['importances_std'],
           label='NN (Trained AFTER Drift)',
           color='#f39c12', alpha=0.8, edgecolor='black', capsize=5)
    ax.set_ylabel(f'Importance Score ({fi_before["method"]})')
    ax.set_title('Feature Importance for Predicting Target Class')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Side-by-side box plots (Conditional)
    if show_boxplot:
        ax = axes[1]

        # Handle cases where importances might not be (n_features, n_samples)
        importances_before = fi_before['importances']
        if (
            importances_before.ndim == 2 and
            importances_before.shape[0] != n_features and
            importances_before.shape[1] == n_features
        ):
            importances_before = importances_before.T

        importances_after = fi_after['importances']
        if (
            importances_after.ndim == 2 and
            importances_after.shape[0] != n_features and
            importances_after.shape[1] == n_features
        ):
            importances_after = importances_after.T

        # Calculate positions for side-by-side boxplots
        positions_before = np.arange(n_features) * 2 - 0.2
        positions_after = np.arange(n_features) * 2 + 0.2

        # Check if boxplot can be drawn (importances is 2D and aligned)
        if (importances_before.shape[0] == n_features and
                importances_after.shape[0] == n_features):
            bp1 = ax.boxplot([importances_before[i] for i in range(n_features)],
                             positions=positions_before, widths=0.3,
                             patch_artist=True, showmeans=True, notch=True)
            bp2 = ax.boxplot([importances_after[i] for i in range(n_features)],
                             positions=positions_after, widths=0.3,
                             patch_artist=True, showmeans=True, notch=True)

            for patch in bp1['boxes']:
                patch.set_facecolor('#1abc9c')
                patch.set_alpha(0.7)
            for patch in bp2['boxes']:
                patch.set_facecolor('#f39c12')
                patch.set_alpha(0.7)

            ax.set_ylabel(f'{fi_before["method"]} Score')
            ax.set_xlabel('Features')
            ax.set_title(f'Distribution of {fi_before["method"]} Scores')
            ax.set_xticks(np.arange(n_features) * 2)
            ax.set_xticklabels(feature_names)
            ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
                      ['Before Drift', 'After Drift'], loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, "Boxplot not available (SHAP/LIME 1D result)",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            ax.set_axis_off()

    plt.tight_layout()
    plt.show()
