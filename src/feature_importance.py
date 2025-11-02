import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance


def generate_drift_data(n_samples_before=1000, n_samples_after=1000,
                        random_seed=42):
    """
    Generate synthetic data stream with concept drift.

    This function creates a binary classification dataset where both the
    feature distributions P(X) and the conditional distribution P(Y|X) change
    between two time periods, simulating concept drift.

    Before drift:
        - X1 ~ N(0, 1), X2 ~ N(0, 1)
        - P(y=1) ~ 0.7 (70% class 1, 30% class 0)
        - Decision boundary: 2*X1 + 0.5*X2 > threshold

    After drift:
        - X1 ~ N(2, 1.5) - MAJOR SHIFT in mean and variance
        - X2 ~ N(0.2, 1.05) - minor shift in mean and variance
        - P(y=1) ~ 0.3 (30% class 1, 70% class 0)
        - Decision boundary: -1.5*X1 + 0.6*X2 > threshold

    Parameters
    ----------
    n_samples_before : int, default=1000
        Number of samples to generate before drift
    n_samples_after : int, default=1000
        Number of samples to generate after drift
    random_seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    tuple
        (X1, X2, y, drift_point) where:
        - X1: array of feature 1 values
        - X2: array of feature 2 values
        - y: array of binary class labels (0 or 1)
        - drift_point: index where drift occurs
    """
    np.random.seed(random_seed)

    # Before drift
    X1_before = np.random.normal(0, 1, n_samples_before)
    X2_before = np.random.normal(0, 1, n_samples_before)
    scores_before = (2 * X1_before + 0.5 * X2_before +
                     np.random.normal(0, 0.5, n_samples_before))
    threshold_before = np.percentile(scores_before, 30)
    y_before = (scores_before > threshold_before).astype(int)

    # After drift - X1 distribution changes DRAMATICALLY, X2 changes slightly
    X1_after = np.random.normal(2, 1.5, n_samples_after)
    X2_after = np.random.normal(0.2, 1.05, n_samples_after)
    scores_after = (-1.5 * X1_after + 0.6 * X2_after +
                    np.random.normal(0, 0.5, n_samples_after))
    threshold_after = np.percentile(scores_after, 70)
    y_after = (scores_after > threshold_after).astype(int)

    # Combine data
    X1 = np.concatenate([X1_before, X1_after])
    X2 = np.concatenate([X2_before, X2_after])
    y = np.concatenate([y_before, y_after])

    drift_point = n_samples_before

    return X1, X2, y, drift_point


def visualize_data_stream(X1, X2, y, drift_point):
    """
    Visualize the data stream before and after concept drift.

    Creates a comprehensive 3x4 grid of plots showing:
    - Feature distributions over time (X1 and X2)
    - Class distributions before and after drift
    - Feature space visualizations
    - Feature-target relationships

    Parameters
    ----------
    X1 : array-like
        Feature 1 values
    X2 : array-like
        Feature 2 values
    y : array-like
        Binary class labels
    drift_point : int
        Index where drift occurs
    """
    time_steps = np.arange(len(X1))
    y_before = y[:drift_point]
    y_after = y[drift_point:]

    # Calculate class distributions
    class_dist_before = [np.mean(y_before == 0), np.mean(y_before == 1)]
    class_dist_after = [np.mean(y_after == 0), np.mean(y_after == 1)]

    print("=" * 70)
    print("TARGET VARIABLE (Y) DISTRIBUTION")
    print("=" * 70)
    print(f"Before Drift: Class 0: {class_dist_before[0]:.2%}, "
          f"Class 1: {class_dist_before[1]:.2%}")
    print(f"After Drift:  Class 0: {class_dist_after[0]:.2%}, "
          f"Class 1: {class_dist_after[1]:.2%}")
    print("=" * 70)

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Data Stream Visualization: Before vs After Concept Drift',
                 fontsize=16, fontweight='bold')

    # Colors for the two classes
    class_colors = {0: '#FF6B6B', 1: '#4ECDC4'}

    # Create masks for different periods and classes
    mask_before_c0 = (time_steps < drift_point) & (y == 0)
    mask_before_c1 = (time_steps < drift_point) & (y == 1)
    mask_after_c0 = (time_steps >= drift_point) & (y == 0)
    mask_after_c1 = (time_steps >= drift_point) & (y == 1)

    # Row 1: X1 distributions
    ax1 = plt.subplot(3, 4, 1)
    ax1.scatter(time_steps[mask_before_c0], X1[mask_before_c0], alpha=0.5,
                s=20, label='Class 0', color=class_colors[0])
    ax1.scatter(time_steps[mask_before_c1], X1[mask_before_c1], alpha=0.5,
                s=20, label='Class 1', color=class_colors[1])
    ax1.set_xlabel('Time')
    ax1.set_ylabel('X1 Value')
    ax1.set_title('X1 - Before Drift')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 4, 2)
    ax2.scatter(time_steps[mask_after_c0], X1[mask_after_c0], alpha=0.5,
                s=20, label='Class 0', color=class_colors[0])
    ax2.scatter(time_steps[mask_after_c1], X1[mask_after_c1], alpha=0.5,
                s=20, label='Class 1', color=class_colors[1])
    ax2.set_xlabel('Time')
    ax2.set_ylabel('X1 Value')
    ax2.set_title('X1 - After Drift')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Row 1: X2 distributions
    ax3 = plt.subplot(3, 4, 3)
    ax3.scatter(time_steps[mask_before_c0], X2[mask_before_c0], alpha=0.5,
                s=20, label='Class 0', color=class_colors[0])
    ax3.scatter(time_steps[mask_before_c1], X2[mask_before_c1], alpha=0.5,
                s=20, label='Class 1', color=class_colors[1])
    ax3.set_xlabel('Time')
    ax3.set_ylabel('X2 Value')
    ax3.set_title('X2 - Before Drift')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(3, 4, 4)
    ax4.scatter(time_steps[mask_after_c0], X2[mask_after_c0], alpha=0.5,
                s=20, label='Class 0', color=class_colors[0])
    ax4.scatter(time_steps[mask_after_c1], X2[mask_after_c1], alpha=0.5,
                s=20, label='Class 1', color=class_colors[1])
    ax4.set_xlabel('Time')
    ax4.set_ylabel('X2 Value')
    ax4.set_title('X2 - After Drift')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Row 2: Class distributions
    ax5 = plt.subplot(3, 4, 5)
    ax5.bar(['Class 0', 'Class 1'], class_dist_before,
            color=[class_colors[0], class_colors[1]],
            alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Proportion')
    ax5.set_title('Class Distribution - Before Drift')
    ax5.set_ylim([0, 1])
    ax5.grid(True, alpha=0.3, axis='y')

    ax6 = plt.subplot(3, 4, 6)
    ax6.bar(['Class 0', 'Class 1'], class_dist_after,
            color=[class_colors[0], class_colors[1]],
            alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Proportion')
    ax6.set_title('Class Distribution - After Drift')
    ax6.set_ylim([0, 1])
    ax6.grid(True, alpha=0.3, axis='y')

    # Row 2: X1 vs X2 feature space
    ax7 = plt.subplot(3, 4, 7)
    ax7.scatter(X1[mask_before_c0], X2[mask_before_c0], alpha=0.5, s=20,
                label='Class 0', color=class_colors[0])
    ax7.scatter(X1[mask_before_c1], X2[mask_before_c1], alpha=0.5, s=20,
                label='Class 1', color=class_colors[1])
    ax7.set_xlabel('X1')
    ax7.set_ylabel('X2')
    ax7.set_title('Feature Space - Before Drift')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    ax8 = plt.subplot(3, 4, 8)
    ax8.scatter(X1[mask_after_c0], X2[mask_after_c0], alpha=0.5, s=20,
                label='Class 0', color=class_colors[0])
    ax8.scatter(X1[mask_after_c1], X2[mask_after_c1], alpha=0.5, s=20,
                label='Class 1', color=class_colors[1])
    ax8.set_xlabel('X1')
    ax8.set_ylabel('X2')
    ax8.set_title('Feature Space - After Drift')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Row 3: X1 vs Y relationship
    ax9 = plt.subplot(3, 4, 9)
    ax9.scatter(X1[mask_before_c0], [0]*np.sum(mask_before_c0), alpha=0.5,
                s=20, label='Class 0', color=class_colors[0])
    ax9.scatter(X1[mask_before_c1], [1]*np.sum(mask_before_c1), alpha=0.5,
                s=20, label='Class 1', color=class_colors[1])
    ax9.set_xlabel('X1')
    ax9.set_ylabel('Target Class')
    ax9.set_title('X1 vs Target - Before Drift')
    ax9.set_yticks([0, 1])
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    ax10 = plt.subplot(3, 4, 10)
    ax10.scatter(X1[mask_after_c0], [0]*np.sum(mask_after_c0), alpha=0.5,
                 s=20, label='Class 0', color=class_colors[0])
    ax10.scatter(X1[mask_after_c1], [1]*np.sum(mask_after_c1), alpha=0.5,
                 s=20, label='Class 1', color=class_colors[1])
    ax10.set_xlabel('X1')
    ax10.set_ylabel('Target Class')
    ax10.set_title('X1 vs Target - After Drift')
    ax10.set_yticks([0, 1])
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    # Row 3: X2 vs Y relationship
    ax11 = plt.subplot(3, 4, 11)
    ax11.scatter(X2[mask_before_c0], [0]*np.sum(mask_before_c0), alpha=0.5,
                 s=20, label='Class 0', color=class_colors[0])
    ax11.scatter(X2[mask_before_c1], [1]*np.sum(mask_before_c1), alpha=0.5,
                 s=20, label='Class 1', color=class_colors[1])
    ax11.set_xlabel('X2')
    ax11.set_ylabel('Target Class')
    ax11.set_title('X2 vs Target - Before Drift')
    ax11.set_yticks([0, 1])
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    ax12 = plt.subplot(3, 4, 12)
    ax12.scatter(X2[mask_after_c0], [0]*np.sum(mask_after_c0), alpha=0.5,
                 s=20, label='Class 0', color=class_colors[0])
    ax12.scatter(X2[mask_after_c1], [1]*np.sum(mask_after_c1), alpha=0.5,
                 s=20, label='Class 1', color=class_colors[1])
    ax12.set_xlabel('X2')
    ax12.set_ylabel('Target Class')
    ax12.set_title('X2 vs Target - After Drift')
    ax12.set_yticks([0, 1])
    ax12.legend()
    ax12.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_data_drift(X1, X2, y, drift_point):
    """
    Analyze data drift by classifying time periods using only features (X).

    This function trains a neural network to distinguish between before-drift
    and after-drift data points based solely on feature distributions P(X).
    It then uses Permutation Feature Importance (PFI) to determine which
    features are most important for detecting the data drift.

    Parameters
    ----------
    X1 : array-like
        Feature 1 values
    X2 : array-like
        Feature 2 values
    y : array-like
        Binary class labels (not used in this analysis)
    drift_point : int
        Index where drift occurs

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': trained MLPClassifier
        - 'accuracy': model accuracy
        - 'pfi_result': permutation importance result object
        - 'importance_mean': mean importance scores
        - 'importance_std': standard deviation of importance scores
    """
    print("\n" + "=" * 70)
    print("STEP 1: DATA DRIFT DETECTION (Classification using X features only)")
    print("Goal: Classify if data point is BEFORE (0) or AFTER (1) drift, "
          "based on P(X).")
    print("=" * 70)

    # Prepare features and labels
    X_features = np.column_stack([X1, X2])
    n_samples_before = drift_point
    n_samples_after = len(X1) - drift_point
    time_labels = np.array([0] * n_samples_before + [1] * n_samples_after)

    # Train Neural Network
    nn_model = MLPClassifier(
        hidden_layer_sizes=(10, 10),
        max_iter=500,
        random_state=42,
        solver='adam',
        alpha=1e-5
    )
    nn_model.fit(X_features, time_labels)
    nn_accuracy = nn_model.score(X_features, time_labels)

    print("\nModel Accuracy (X features only):")
    print(f"  Neural Network (MLP) Accuracy: {nn_accuracy:.4f}")
    print("(Higher accuracy = stronger P(X) drift, Random guess = 0.50)")

    # Calculate Permutation Feature Importance
    nn_pfi_result = permutation_importance(
        nn_model, X_features, time_labels,
        n_repeats=30, random_state=42, n_jobs=-1
    )
    nn_importance_mean = nn_pfi_result.importances_mean
    nn_importance_std = nn_pfi_result.importances_std

    print("\n" + "-" * 70)
    print("PFI (Feature importance for detecting time-period using X only)")
    print("-" * 70)
    print("\nNeural Network (MLP) PFI:")
    print(f"  X1: {nn_importance_mean[0]:.4f} ± {nn_importance_std[0]:.4f}")
    print(f"  X2: {nn_importance_mean[1]:.4f} ± {nn_importance_std[1]:.4f}")

    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Feature Importance for Data Drift (X-only Classification)',
                 fontsize=14, fontweight='bold')

    features_x_only = ['X1', 'X2']

    # Plot 1: Bar plot
    ax = axes[0]
    x_pos = np.arange(len(features_x_only))
    ax.bar(x_pos, nn_importance_mean, yerr=nn_importance_std,
           color='#e74c3c', alpha=0.8, edgecolor='black', capsize=5)
    ax.set_ylabel('Mean Accuracy Drop (Importance Score)')
    ax.set_title('PFI for Detecting Time-Period (Data Drift)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(features_x_only)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Box plot
    ax = axes[1]
    pfi_values = nn_pfi_result.importances
    bp = ax.boxplot([pfi_values[0], pfi_values[1]], labels=features_x_only,
                    patch_artist=True, notch=True, showmeans=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#e74c3c')
        patch.set_alpha(0.7)
    ax.set_ylabel('Permutation Importance Score')
    ax.set_xlabel('Features')
    ax.set_title('Distribution of PFI Scores')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    return {
        'model': nn_model,
        'accuracy': nn_accuracy,
        'pfi_result': nn_pfi_result,
        'importance_mean': nn_importance_mean,
        'importance_std': nn_importance_std
    }


def analyze_concept_drift(X1, X2, y, drift_point):
    """
    Analyze concept drift by classifying time periods using features and target
    (X, Y).

    This function trains a neural network to distinguish between before-drift
    and after-drift data points using both features and target P(X, Y).
    This is sensitive to changes in P(Y|X), which indicates concept drift.

    Parameters
    ----------
    X1 : array-like
        Feature 1 values
    X2 : array-like
        Feature 2 values
    y : array-like
        Binary class labels
    drift_point : int
        Index where drift occurs

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': trained MLPClassifier
        - 'accuracy': model accuracy
        - 'pfi_result': permutation importance result object
        - 'importance_mean': mean importance scores
        - 'importance_std': standard deviation of importance scores
    """
    print("\n" + "=" * 70)
    print("STEP 2: CONCEPT DRIFT DETECTION (Classification using X and Y "
          "features)")
    print("Goal: Classify data point based on P(Period|X, Y). "
          "Highly sensitive to P(Y|X) changes.")
    print("=" * 70)

    # Prepare features and labels
    X_features_with_y = np.column_stack([X1, X2, y])
    n_samples_before = drift_point
    n_samples_after = len(X1) - drift_point
    time_labels = np.array([0] * n_samples_before + [1] * n_samples_after)

    # Train Neural Network
    nn_model_xy = MLPClassifier(
        hidden_layer_sizes=(10, 10),
        max_iter=500,
        random_state=42,
        solver='adam',
        alpha=1e-5
    )
    nn_model_xy.fit(X_features_with_y, time_labels)
    nn_accuracy_xy = nn_model_xy.score(X_features_with_y, time_labels)

    print("\nModel Accuracy (X and Y features):")
    print(f"  Neural Network (MLP) Accuracy: {nn_accuracy_xy:.4f}")

    # Calculate Permutation Feature Importance
    nn_pfi_result_xy = permutation_importance(
        nn_model_xy, X_features_with_y, time_labels,
        n_repeats=30, random_state=42, n_jobs=-1
    )
    nn_importance_mean_xy = nn_pfi_result_xy.importances_mean
    nn_importance_std_xy = nn_pfi_result_xy.importances_std

    print("\n" + "-" * 70)
    print("PFI (Feature importance for detecting time-period using X and Y)")
    print("-" * 70)
    print("\nNeural Network (MLP) PFI:")
    print(f"  X1: {nn_importance_mean_xy[0]:.4f} ± "
          f"{nn_importance_std_xy[0]:.4f}")
    print(f"  X2: {nn_importance_mean_xy[1]:.4f} ± "
          f"{nn_importance_std_xy[1]:.4f}")
    print(f"  Y:  {nn_importance_mean_xy[2]:.4f} ± "
          f"{nn_importance_std_xy[2]:.4f}")
    print("=" * 70)

    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Feature Importance for Concept Drift (X+Y Classification)',
                 fontsize=14, fontweight='bold')

    features_x_y = ['X1', 'X2', 'Y']

    # Plot 1: Bar plot
    ax = axes[0]
    x_pos = np.arange(len(features_x_y))
    ax.bar(x_pos, nn_importance_mean_xy, yerr=nn_importance_std_xy,
           color='#e74c3c', alpha=0.8, edgecolor='black', capsize=5)
    ax.set_ylabel('Mean Accuracy Drop (Importance Score)')
    ax.set_title('PFI for Detecting Time-Period (Concept Drift)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(features_x_y)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Box plot
    ax = axes[1]
    pfi_values_xy = nn_pfi_result_xy.importances
    bp = ax.boxplot([pfi_values_xy[0], pfi_values_xy[1], pfi_values_xy[2]],
                    labels=features_x_y, patch_artist=True, notch=True,
                    showmeans=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#e74c3c')
        patch.set_alpha(0.7)
    ax.set_ylabel('Permutation Importance Score')
    ax.set_xlabel('Features')
    ax.set_title('Distribution of PFI Scores')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    return {
        'model': nn_model_xy,
        'accuracy': nn_accuracy_xy,
        'pfi_result': nn_pfi_result_xy,
        'importance_mean': nn_importance_mean_xy,
        'importance_std': nn_importance_std_xy
    }


def analyze_predictive_importance_shift(X1, X2, y, drift_point):
    """
    Analyze how predictive feature importance shifts before and after drift.

    This function trains separate neural networks to predict the target
    variable before and after the drift, then compares the feature importance
    using PFI. This reveals how the relationship between features and target
    changes.

    Parameters
    ----------
    X1 : array-like
        Feature 1 values
    X2 : array-like
        Feature 2 values
    y : array-like
        Binary class labels
    drift_point : int
        Index where drift occurs

    Returns
    -------
    dict
        Dictionary containing:
        - 'model_before': trained MLPClassifier on before-drift data
        - 'model_after': trained MLPClassifier on after-drift data
        - 'accuracy_before': accuracy on before-drift data
        - 'accuracy_after': accuracy on after-drift data
        - 'pfi_before': PFI results for before-drift model
        - 'pfi_after': PFI results for after-drift model
    """
    print("\n" + "=" * 70)
    print("STEP 3: PREDICTIVE POWER SHIFT (Classification)")
    print("Goal: Compare feature importance for predicting target 'y' "
          "BEFORE vs AFTER drift.")
    print("=" * 70)

    # Split data into before and after drift
    X_features = np.column_stack([X1, X2])
    X_features_before = X_features[:drift_point]
    y_before = y[:drift_point]
    X_features_after = X_features[drift_point:]
    y_after = y[drift_point:]

    # Configuration for the Neural Network Classifiers
    mlp_config = {
        'hidden_layer_sizes': (5, 5),
        'max_iter': 500,
        'random_state': 42,
        'solver': 'adam',
        'alpha': 1e-5
    }

    # Neural Network trained BEFORE drift
    mlp_before = MLPClassifier(**mlp_config)
    mlp_before.fit(X_features_before, y_before)
    acc_before = mlp_before.score(X_features_before, y_before)

    # Neural Network trained AFTER drift
    mlp_after = MLPClassifier(**mlp_config)
    mlp_after.fit(X_features_after, y_after)
    acc_after = mlp_after.score(X_features_after, y_after)

    print("\nModel Accuracy Scores (on training data):")
    print(f"  NN Model (Before Drift) Accuracy: {acc_before:.4f}")
    print(f"  NN Model (After Drift) Accuracy: {acc_after:.4f}")

    # PFI for NN BEFORE drift
    nn_before_pfi = permutation_importance(
        mlp_before, X_features_before, y_before,
        n_repeats=30, random_state=42, n_jobs=-1
    )
    pfi_before_mean = nn_before_pfi.importances_mean
    pfi_before_std = nn_before_pfi.importances_std

    # PFI for NN AFTER drift
    nn_after_pfi = permutation_importance(
        mlp_after, X_features_after, y_after,
        n_repeats=30, random_state=42, n_jobs=-1
    )
    pfi_after_mean = nn_after_pfi.importances_mean
    pfi_after_std = nn_after_pfi.importances_std

    print("\nPredictive PFI (using Accuracy change):")
    print("Before Drift (X1, X2):")
    print(f"  X1: {pfi_before_mean[0]:.4f} ± {pfi_before_std[0]:.4f}")
    print(f"  X2: {pfi_before_mean[1]:.4f} ± {pfi_before_std[1]:.4f}")
    print("\nAfter Drift (X1, X2):")
    print(f"  X1: {pfi_after_mean[0]:.4f} ± {pfi_after_std[0]:.4f}")
    print(f"  X2: {pfi_after_mean[1]:.4f} ± {pfi_after_std[1]:.4f}")
    print("=" * 70)

    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Predictive Feature Importance (NN Before vs After Drift)',
                 fontsize=14, fontweight='bold')

    features_pred = ['X1', 'X2']

    # Plot 1: Bar comparison
    ax = axes[0]
    x_pos = np.arange(len(features_pred))
    width = 0.35
    ax.bar(x_pos - width/2, pfi_before_mean, width,
           yerr=pfi_before_std, label='NN (Trained BEFORE Drift)',
           color='#1abc9c', alpha=0.8, edgecolor='black', capsize=5)
    ax.bar(x_pos + width/2, pfi_after_mean, width,
           yerr=pfi_after_std, label='NN (Trained AFTER Drift)',
           color='#f39c12', alpha=0.8, edgecolor='black', capsize=5)
    ax.set_ylabel('Mean Accuracy Drop (Importance Score)')
    ax.set_title('PFI for Predicting Target Class')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(features_pred)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Side-by-side box plots
    ax = axes[1]
    positions_before = [0.8, 2.8]
    positions_after = [1.2, 3.2]
    bp1 = ax.boxplot([nn_before_pfi.importances[0],
                      nn_before_pfi.importances[1]],
                     positions=positions_before, widths=0.3,
                     patch_artist=True, showmeans=True)
    bp2 = ax.boxplot([nn_after_pfi.importances[0],
                      nn_after_pfi.importances[1]],
                     positions=positions_after, widths=0.3,
                     patch_artist=True, showmeans=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('#1abc9c')
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor('#f39c12')
        patch.set_alpha(0.7)
    ax.set_ylabel('Permutation Importance Score')
    ax.set_xlabel('Features')
    ax.set_title('Distribution of PFI Scores')
    ax.set_xticks([1, 3])
    ax.set_xticklabels(features_pred)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], 
              ['Before Drift', 'After Drift'], loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    return {
        'model_before': mlp_before,
        'model_after': mlp_after,
        'accuracy_before': acc_before,
        'accuracy_after': acc_after,
        'pfi_before': nn_before_pfi,
        'pfi_after': nn_after_pfi
    }


def main():
    """
    Main function to run the complete concept drift analysis pipeline.

    This function orchestrates the entire analysis by:
    1. Generating synthetic data with concept drift
    2. Visualizing the data stream
    3. Analyzing data drift (changes in P(X))
    4. Analyzing concept drift (changes in P(Y|X))
    5. Analyzing predictive importance shift
    (how feature-target relationships change)
    """
    # Step 1: Generate data
    print("Generating synthetic data with concept drift...")
    X1, X2, y, drift_point = generate_drift_data(
        n_samples_before=1000,
        n_samples_after=1000,
        random_seed=42
    )

    # Step 2: Visualize data stream
    print("\nVisualizing data stream...")
    visualize_data_stream(X1, X2, y, drift_point)

    # Step 3: Analyze data drift (P(X) changes)
    analyze_data_drift(X1, X2, y, drift_point)

    # Step 4: Analyze concept drift (P(Y|X) changes)
    analyze_concept_drift(X1, X2, y, drift_point)

    # Step 5: Analyze predictive importance shift
    analyze_predictive_importance_shift(X1, X2, y, drift_point)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nAll visualizations have been displayed.")
    print("The analysis demonstrates:")
    print("  1. Data Drift: How P(X) changes between periods")
    print("  2. Concept Drift: How P(Y|X) changes between periods")
    print("  3. Predictive Shift: How feature importance for predicting Y "
          "changes")


if __name__ == "__main__":
    main()
