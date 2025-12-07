import numpy as np
import matplotlib.pyplot as plt



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
