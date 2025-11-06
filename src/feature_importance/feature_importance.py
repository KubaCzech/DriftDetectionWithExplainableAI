import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
import shap
from lime.lime_tabular import LimeTabularExplainer
from src.datasets import (
    DatasetName,
    generate_custom_3d_drift_data,
    generate_custom_normal_data,
    generate_hyperplane_data,
    generate_sea_drift_data
)


class FeatureImportanceMethod:
    """Enum-like class for feature importance methods."""
    PFI = "permutation"
    SHAP = "shap"
    LIME = "lime"

    @classmethod
    def all_available(cls):
        """Return list of all available methods."""
        return [cls.PFI, cls.SHAP, cls.LIME]


def calculate_feature_importance(
    model, X, y, method="permutation",
    feature_names=None, n_repeats=30,
    random_state=42
):
    """
    Calculate feature importance using specified method.

    Parameters
    ----------
    model : estimator
        Trained model
    X : array-like
        Feature matrix
    y : array-like
        Target variable
    method : str, default="permutation"
        Method to use: "permutation", "shap", or "lime"
    feature_names : list, optional
        Names of features
    n_repeats : int, default=30
        Number of repeats for permutation importance
    random_state : int, default=42
        Random state for reproducibility

    Returns
    -------
    dict
        Dictionary with keys:
        - 'importances_mean': mean importance scores
        - 'importances_std': standard deviation of importance scores
        - 'importances': full array of importance values (for boxplots)
        - 'method': method used
    """
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    if method == FeatureImportanceMethod.PFI:
        return _calculate_pfi(model, X, y, n_repeats, random_state)

    elif method == FeatureImportanceMethod.SHAP:
        return _calculate_shap(model, X, feature_names)

    elif method == FeatureImportanceMethod.LIME:
        return _calculate_lime(model, X, y, feature_names, random_state)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'permutation', "
                         "'shap', or 'lime'")


def _calculate_pfi(model, X, y, n_repeats=30, random_state=42):
    """Calculate Permutation Feature Importance."""
    pfi_result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    return {
        'importances_mean': pfi_result.importances_mean,
        'importances_std': pfi_result.importances_std,
        'importances': pfi_result.importances,
        'method': 'PFI'
    }


def _calculate_shap(model, X, feature_names):
    """Calculate SHAP values."""
    # Use a subset for efficiency if dataset is large
    background_size = min(100, len(X))
    background = shap.sample(X, background_size)

    # Create explainer
    explainer = shap.KernelExplainer(model.predict_proba, background)

    # Calculate SHAP values (use subset if too large)
    sample_size = min(500, len(X))
    sample_indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[sample_indices]

    shap_values = explainer.shap_values(X_sample)

    # For binary classification, SHAP returns a list
    # [class_0_values, class_1_values]
    # We use class 1 (positive class) for binary classification
    # For multiclass, this logic would need to adapt
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    # Handle possible 3D array for some model types
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # Ensure shap_values is 2D: (n_samples, n_features)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    # Calculate mean absolute SHAP values
    abs_shap_values = np.abs(shap_values)
    importances_mean = np.mean(abs_shap_values, axis=0)
    importances_std = np.std(abs_shap_values, axis=0)

    # Ensure arrays are 1D with correct length (n_features)
    importances_mean = np.atleast_1d(importances_mean).flatten()
    importances_std = np.atleast_1d(importances_std).flatten()

    # Ensure importances array is (n_features, n_samples)
    # for consistency with PFI
    if abs_shap_values.shape[1] == len(feature_names):
        importances_array = abs_shap_values.T
    else:
        importances_array = abs_shap_values

    return {
        'importances_mean': importances_mean,
        'importances_std': importances_std,
        'importances': importances_array,
        'method': 'SHAP'
    }


def _calculate_lime(model, X, y, feature_names, random_state=42):
    """Calculate LIME feature importance."""
    np.random.seed(random_state)

    # Create LIME explainer
    explainer = LimeTabularExplainer(
        X,
        feature_names=feature_names,
        class_names=['Class 0', 'Class 1'],
        mode='classification',
        random_state=random_state
    )

    # Calculate LIME explanations for a sample of instances
    sample_size = min(100, len(X))
    sample_indices = np.random.choice(len(X), sample_size, replace=False)

    all_importances = []

    for idx in sample_indices:
        exp = explainer.explain_instance(
            X[idx],
            model.predict_proba,
            num_features=len(feature_names)
        )

        # Extract feature importances (absolute values)
        importance_dict = dict(exp.as_list())
        importances = []
        for feat_name in feature_names:
            # Find matching feature (LIME may modify feature names)
            matching_imp = 0
            for key, val in importance_dict.items():
                if feat_name in key or key.startswith(feat_name):
                    matching_imp = abs(val)
                    break
            importances.append(matching_imp)

        all_importances.append(importances)

    all_importances = np.array(all_importances)

    return {
        'importances_mean': np.mean(all_importances, axis=0),
        'importances_std': np.std(all_importances, axis=0),
        'importances': all_importances.T,
        'method': 'LIME'
    }


def visualize_data_stream(X, y, drift_point, feature_names):
    """
    Visualize the data stream before and after concept drift for N features.

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
    drift_point : int
        Index where drift occurs
    feature_names : list
        Names of features
    """
    n_features = X.shape[1]
    time_steps = np.arange(len(y))
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

    # Colors for the two classes
    class_colors = {0: '#FF6B6B', 1: '#4ECDC4'}

    # Create masks for different periods and classes
    mask_before = (time_steps < drift_point)
    mask_after = (time_steps >= drift_point)
    mask_before_c0 = mask_before & (y == 0)
    mask_before_c1 = mask_before & (y == 1)
    mask_after_c0 = mask_after & (y == 0)
    mask_after_c1 = mask_after & (y == 1)

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

        # Before Drift
        ax_before.scatter(time_steps[mask_before_c0], feat_data[mask_before_c0],
                          alpha=0.5, s=20, label='Class 0',
                          color=class_colors[0])
        ax_before.scatter(time_steps[mask_before_c1], feat_data[mask_before_c1],
                          alpha=0.5, s=20, label='Class 1',
                          color=class_colors[1])
        ax_before.set_xlabel('Time')
        ax_before.set_ylabel(f'{feat_name} Value')
        ax_before.set_title(f'{feat_name} - Before Drift')
        ax_before.legend()
        ax_before.grid(True, alpha=0.3)

        # After Drift
        ax_after.scatter(time_steps[mask_after_c0], feat_data[mask_after_c0],
                         alpha=0.5, s=20, label='Class 0',
                         color=class_colors[0])
        ax_after.scatter(time_steps[mask_after_c1], feat_data[mask_after_c1],
                         alpha=0.5, s=20, label='Class 1',
                         color=class_colors[1])
        ax_after.set_xlabel('Time')
        ax_after.set_ylabel(f'{feat_name} Value')
        ax_after.set_title(f'{feat_name} - After Drift')
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

        # Before Drift
        ax_before.scatter(feat_data[mask_before_c0], y_jitter_before_c0,
                          alpha=0.5, s=20, label='Class 0',
                          color=class_colors[0])
        ax_before.scatter(feat_data[mask_before_c1], y_jitter_before_c1,
                          alpha=0.5, s=20, label='Class 1',
                          color=class_colors[1])
        ax_before.set_xlabel(feat_name)
        ax_before.set_ylabel('Target Class (with jitter)')
        ax_before.set_title(f'{feat_name} vs Target - Before Drift')
        ax_before.set_yticks([0, 1])
        ax_before.set_yticklabels(['Class 0', 'Class 1'])
        ax_before.legend()
        ax_before.grid(True, alpha=0.3)

        # After Drift
        ax_after.scatter(feat_data[mask_after_c0], y_jitter_after_c0,
                         alpha=0.5, s=20, label='Class 0',
                         color=class_colors[0])
        ax_after.scatter(feat_data[mask_after_c1], y_jitter_after_c1,
                         alpha=0.5, s=20, label='Class 1',
                         color=class_colors[1])
        ax_after.set_xlabel(feat_name)
        ax_after.set_ylabel('Target Class (with jitter)')
        ax_after.set_title(f'{feat_name} vs Target - After Drift')
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

    # Class distributions - Before
    ax_class_before.bar(['Class 0', 'Class 1'], class_dist_before,
                        color=[class_colors[0], class_colors[1]],
                        alpha=0.7, edgecolor='black')
    ax_class_before.set_ylabel('Proportion')
    ax_class_before.set_title('Before Drift')
    ax_class_before.set_ylim([0, 1])
    ax_class_before.grid(True, alpha=0.3, axis='y')

    # Class distributions - After
    ax_class_after.bar(['Class 0', 'Class 1'], class_dist_after,
                       color=[class_colors[0], class_colors[1]],
                       alpha=0.7, edgecolor='black')
    ax_class_after.set_ylabel('Proportion')
    ax_class_after.set_title('After Drift')
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
        X_before = X[mask_before]
        X_after = X[mask_after]
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
        X_before = X[mask_before]
        X_after = X[mask_after]
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
        X_2d = pca.fit_transform(X)
        X_2d_before = X_2d[mask_before]
        X_2d_after = X_2d[mask_after]
        y_before_pca = y[mask_before]
        y_after_pca = y[mask_after]

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

    ax_fs_before.set_title('Before Drift')
    ax_fs_before.legend()
    ax_fs_before.grid(True, alpha=0.3)

    ax_fs_after.set_title('After Drift')
    ax_fs_after.legend()
    ax_fs_after.grid(True, alpha=0.3)

    fig4.suptitle('Feature Space' + fs_title_suffix,
                  fontsize=16, fontweight='bold', y=1.0)

    plt.tight_layout()
    # Increased top margin
    fig4.subplots_adjust(top=0.88)
    plt.show()


def compute_data_drift_analysis(X, y, drift_point, feature_names,
                                importance_method="permutation"):
    """
    Compute data drift analysis by classifying time periods using only
    features (X).

    Parameters
    ----------
    X : array-like (n_samples, n_features)
        Feature matrix
    y : array-like (n_samples,)
        Binary class labels (not used in this analysis)
    drift_point : int
        Index where drift occurs
    feature_names : list
        Names of features
    importance_method : str, default="permutation"
        Method for feature importance: "permutation", "shap", or "lime"

    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    print("\n" + "=" * 70)
    print("STEP 1: DATA DRIFT DETECTION "
          "(Classification using X features only)")
    print(f"Feature Importance Method: {importance_method.upper()}")
    print("Goal: Classify if data point is BEFORE (0) or AFTER (1) drift, "
          "based on P(X).")
    print("=" * 70)

    # Prepare features and labels
    X_features = X
    n_samples_before = drift_point
    n_samples_after = len(y) - drift_point
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

    # Calculate Feature Importance
    fi_result = calculate_feature_importance(
        nn_model, X_features, time_labels,
        method=importance_method,
        feature_names=feature_names
    )

    importance_mean = fi_result['importances_mean']
    importance_std = fi_result['importances_std']

    print("\n" + "-" * 70)
    print(f"{fi_result['method']} (Feature importance for detecting "
          "time-period using X only)")
    print("-" * 70)
    print(f"\nNeural Network (MLP) {fi_result['method']}:")
    for i, name in enumerate(feature_names):
        print(f"  {name}: {importance_mean[i]:.4f} ± {importance_std[i]:.4f}")

    return {
        'model': nn_model,
        'accuracy': nn_accuracy,
        'importance_result': fi_result,
        'importance_mean': importance_mean,
        'importance_std': importance_std
    }


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


def analyze_data_drift(X, y, drift_point, feature_names,
                       importance_method="permutation",
                       show_importance_boxplot=True):
    """
    Analyze and visualize data drift by classifying time periods using only
    features (X).

    Parameters
    ----------
    X : array-like (n_samples, n_features)
        Feature matrix
    y : array-like (n_samples,)
        Binary class labels (not used in this analysis)
    drift_point : int
        Index where drift occurs
    feature_names : list
        Names of features
    importance_method : str, default="permutation"
        Method for feature importance: "permutation", "shap", or "lime"
    show_importance_boxplot : bool, default=True
        Whether to display the boxplot of importance score distributions.

    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    result = compute_data_drift_analysis(X, y, drift_point, feature_names,
                                         importance_method)
    visualize_data_drift_analysis(result, feature_names,
                                  show_boxplot=show_importance_boxplot)
    return result


def compute_concept_drift_analysis(X, y, drift_point, feature_names,
                                   importance_method="permutation"):
    """
    Compute concept drift analysis by classifying time periods using features
    and target (X, Y).

    Parameters
    ----------
    X : array-like (n_samples, n_features)
        Feature matrix
    y : array-like (n_samples,)
        Binary class labels
    drift_point : int
        Index where drift occurs
    feature_names : list
        Names of features
    importance_method : str, default="permutation"
        Method for feature importance: "permutation", "shap", or "lime"

    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    print("\n" + "=" * 70)
    print("STEP 2: CONCEPT DRIFT DETECTION (Classification using X and Y "
          "features)")
    print(f"Feature Importance Method: {importance_method.upper()}")
    print("Goal: Classify data point based on P(Period|X, Y). "
          "Highly sensitive to P(Y|X) changes.")
    print("=" * 70)

    # Prepare features and labels
    X_features_with_y = np.column_stack([X, y])
    n_samples_before = drift_point
    n_samples_after = len(y) - drift_point
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

    # Calculate Feature Importance
    feature_names_with_y = feature_names + ['Y']
    fi_result = calculate_feature_importance(
        nn_model_xy, X_features_with_y, time_labels,
        method=importance_method,
        feature_names=feature_names_with_y
    )

    importance_mean = fi_result['importances_mean']
    importance_std = fi_result['importances_std']

    print("\n" + "-" * 70)
    print(f"{fi_result['method']} (Feature importance for detecting "
          "time-period using X and Y)")
    print("-" * 70)
    print(f"\nNeural Network (MLP) {fi_result['method']}:")
    for i, name in enumerate(feature_names_with_y):
        print(f"  {name}: {importance_mean[i]:.4f} ± {importance_std[i]:.4f}")
    print("=" * 70)

    return {
        'model': nn_model_xy,
        'accuracy': nn_accuracy_xy,
        'importance_result': fi_result,
        'importance_mean': importance_mean,
        'importance_std': importance_std,
        'feature_names_with_y': feature_names_with_y  # <-- NEW: Pass this back
    }


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


def analyze_concept_drift(X, y, drift_point, feature_names,
                          importance_method="permutation",
                          show_importance_boxplot=True):
    """
    Analyze and visualize concept drift by classifying time periods using
    features and target (X, Y).

    Parameters
    ----------
    X : array-like (n_samples, n_features)
        Feature matrix
    y : array-like (n_samples,)
        Binary class labels
    drift_point : int
        Index where drift occurs
    feature_names : list
        Names of features
    importance_method : str, default="permutation"
        Method for feature importance: "permutation", "shap", or "lime"
    show_importance_boxplot : bool, default=True
        Whether to display the boxplot of importance score distributions.

    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    result = compute_concept_drift_analysis(X, y, drift_point, feature_names,
                                            importance_method)
    # Get the feature names list that includes 'Y' from the result
    feature_names_with_y = result['feature_names_with_y']
    visualize_concept_drift_analysis(result, feature_names_with_y,
                                     show_boxplot=show_importance_boxplot)
    return result


def compute_predictive_importance_shift(X, y, drift_point, feature_names,
                                        importance_method="permutation"):
    """
    Compute how predictive feature importance shifts before and after drift.

    Parameters
    ----------
    X : array-like (n_samples, n_features)
        Feature matrix
    y : array-like (n_samples,)
        Binary class labels
    drift_point : int
        Index where drift occurs
    feature_names : list
        Names of features
    importance_method : str, default="permutation"
        Method for feature importance: "permutation", "shap", or "lime"

    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    print("\n" + "=" * 70)
    print("STEP 3: PREDICTIVE POWER SHIFT (Classification)")
    print(f"Feature Importance Method: {importance_method.upper()}")
    print("Goal: Compare feature importance for predicting target 'y' "
          "BEFORE vs AFTER drift.")
    print("=" * 70)

    # Split data into before and after drift
    X_features = X
    X_features_before = X_features[:drift_point]
    y_before = y[:drift_point]
    X_features_after = X_features[drift_point:]
    y_after = y[drift_point:]

    # Configuration for the Neural Network Classifiers
    mlp_config = {
        'hidden_layer_sizes': (5, 5),
        'max_iter': 2000,
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

    # Feature Importance for BEFORE drift
    fi_before = calculate_feature_importance(
        mlp_before, X_features_before, y_before,
        method=importance_method,
        feature_names=feature_names
    )

    # Feature Importance for AFTER drift
    fi_after = calculate_feature_importance(
        mlp_after, X_features_after, y_after,
        method=importance_method,
        feature_names=feature_names
    )

    print(f"\nPredictive {fi_before['method']}:")
    print("Before Drift:")
    for i, name in enumerate(feature_names):
        print(f"  {name}: {fi_before['importances_mean'][i]:.4f} ± "
              f"{fi_before['importances_std'][i]:.4f}")

    print("\nAfter Drift:")
    for i, name in enumerate(feature_names):
        print(f"  {name}: {fi_after['importances_mean'][i]:.4f} ± "
              f"{fi_after['importances_std'][i]:.4f}")
    print("=" * 70)

    return {
        'model_before': mlp_before,
        'model_after': mlp_after,
        'accuracy_before': acc_before,
        'accuracy_after': acc_after,
        'fi_before': fi_before,
        'fi_after': fi_after
    }


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


def analyze_predictive_importance_shift(X, y, drift_point, feature_names,
                                        importance_method="permutation",
                                        show_importance_boxplot=True):
    """
    Analyze and visualize how predictive feature importance shifts before
    and after drift.

    Parameters
    ----------
    X : array-like (n_samples, n_features)
        Feature matrix
    y : array-like (n_samples,)
        Binary class labels
    drift_point : int
        Index where drift occurs
    feature_names : list
        Names of features
    importance_method : str, default="permutation"
        Method for feature importance: "permutation", "shap", or "lime"
    show_importance_boxplot : bool, default=True
        Whether to display the boxplot of importance score distributions.

    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    result = compute_predictive_importance_shift(X, y, drift_point,
                                                 feature_names,
                                                 importance_method)
    visualize_predictive_importance_shift(result, feature_names,
                                          show_boxplot=show_importance_boxplot)
    return result


# --- MODIFIED: main ---
def main(dataset_name, importance_method="permutation",
         show_importance_boxplot=True):
    """
    Main function to run the complete concept drift analysis pipeline.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to generate (from DatasetName class)
    importance_method : str, default="permutation"
        Method for feature importance calculation.
        Options: "permutation", "shap", "lime"
    show_importance_boxplot : bool, default=True
        Whether to display the boxplots for feature importance distributions
        in the analysis steps.
    """
    # Validate method
    available_methods = FeatureImportanceMethod.all_available()
    if importance_method not in available_methods:
        print(f"Warning: '{importance_method}' not available.")
        print(f"Available methods: {available_methods}")
        print("Falling back to 'permutation'")
        importance_method = "permutation"

    # Validate dataset
    available_datasets = DatasetName.all_available()
    if dataset_name not in available_datasets:
        print(f"Error: Dataset '{dataset_name}' is not available.")
        print(f"Available datasets: {available_datasets}")
        return

    print("=" * 70)
    print("CONCEPT DRIFT ANALYSIS")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Feature Importance Method: {importance_method.upper()}")
    print(f"Show Importance Boxplots: {show_importance_boxplot}")
    print("=" * 70)

    # Step 1: Generate data
    print(f"\nGenerating synthetic data for: {dataset_name}...")

    gen_params = {
        "n_samples_before": 1000,
        "n_samples_after": 1000,
        "random_seed": 42
    }

    if dataset_name == DatasetName.CUSTOM_NORMAL:
        X, y, drift_point, feature_names = \
            generate_custom_normal_data(**gen_params)
    elif dataset_name == DatasetName.CUSTOM_3D_DRIFT:
        X, y, drift_point, feature_names = \
            generate_custom_3d_drift_data(**gen_params)
    elif dataset_name == DatasetName.SEA_DRIFT:
        X, y, drift_point, feature_names = \
            generate_sea_drift_data(**gen_params)
    elif dataset_name == DatasetName.HYPERPLANE_DRIFT:
        X, y, drift_point, feature_names = \
            generate_hyperplane_data(**gen_params)
    else:
        # This case should be caught by the validation above, but as a fallback
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    print(f"Data generated with {X.shape[1]} features: {feature_names}")

    # Step 2: Visualize data stream
    print("\nVisualizing data stream...")
    visualize_data_stream(X, y, drift_point, feature_names)

    # Step 3: Analyze data drift (P(X) changes)
    analyze_data_drift(X, y, drift_point, feature_names,
                       importance_method=importance_method,
                       show_importance_boxplot=show_importance_boxplot)

    # Step 4: Analyze concept drift (P(Y|X) changes)
    analyze_concept_drift(X, y, drift_point, feature_names,
                          importance_method=importance_method,
                          show_importance_boxplot=show_importance_boxplot)

    # Step 5: Analyze predictive importance shift
    analyze_predictive_importance_shift(
        X, y, drift_point, feature_names,
        importance_method=importance_method,
        show_importance_boxplot=show_importance_boxplot
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nAll visualizations have been displayed.")
    print("The analysis demonstrates:")
    print("  1. Data Drift: How P(X) changes between periods")
    print("  2. Concept Drift: How P(Y|X) changes between periods")
    print("  3. Predictive Shift: How feature importance for predicting Y "
          "changes")
    print(f"\nDataset Used: {dataset_name.upper()}")
    print(f"Feature Importance Method Used: {importance_method.upper()}")


if __name__ == "__main__":

    # --- CHOOSE YOUR SETTINGS HERE ---

    # 1. Choose the dataset to run:
    # Options:
    # - DatasetName.CUSTOM_NORMAL (Original 2-feature dataset)
    # - DatasetName.CUSTOM_3D_DRIFT (NEW 3-feature dataset)
    # - DatasetName.SEA_DRIFT (2-feature dataset, requires 'river')
    # - DatasetName.HYPERPLANE_DRIFT (2-feature dataset, requires 'river')

    DATASET_TO_RUN = DatasetName.CUSTOM_NORMAL

    # 2. Choose the feature importance method:
    # Options: "permutation", "shap", "lime"
    # (Note: "shap" and "lime" require installation)

    IMPORTANCE_METHOD_TO_RUN = "permutation"

    # 3. Toggle for showing importance score boxplots:
    # Options: True or False
    SHOW_IMPORTANCE_BOXPLOT = False

    # --- END SETTINGS ---

    # Run the main analysis
    main(dataset_name=DATASET_TO_RUN,
         importance_method=IMPORTANCE_METHOD_TO_RUN,
         show_importance_boxplot=SHOW_IMPORTANCE_BOXPLOT)

    # --- EXAMPLES OF OTHER RUNS (uncomment to try) ---

    # Example 2: Run SEA_DRIFT (2 features) with permutation (No Boxplots)
    # print("\n\n" + "*"*80 + "\nRUNNING SEA_DRIFT ANALYSIS (NO BOXPLOT)\n" + "*"*80)
    # main(dataset_name=DatasetName.SEA_DRIFT,
    #      importance_method="permutation",
    #      show_importance_boxplot=False)

    # Example 3: Run HYPERPLANE_DRIFT (2 features) with permutation
    # print("\n\n" + "*"*80 + "\nRUNNING HYPERPLANE_DRIFT ANALYSIS\n" + "*"*80)
    # main(dataset_name=DatasetName.HYPERPLANE_DRIFT,
    #      importance_method="permutation")

    # Example 4: Run CUSTOM_3D_DRIFT with SHAP (if installed)
    # print("\n\n" + "*"*80 + "\nRUNNING CUSTOM_3D_DRIFT (SHAP) ANALYSIS\n" + "*"*80)
    # main(dataset_name=DatasetName.CUSTOM_3D_DRIFT,
    #      importance_method="shap")
