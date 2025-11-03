import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
import warnings
import itertools

# --- NEW IMPORTS ---
# Added 'river' imports for synthetic dataset generation
try:
    from river.datasets import synth
    from river import stream
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    warnings.warn(
        "River not available. Install with: pip install river. "
        "SEA and Hyperplane datasets will not be available."
    )
# --- END NEW IMPORTS ---

# Optional imports for SHAP and LIME
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Install with: pip install lime")


class FeatureImportanceMethod:
    """Enum-like class for feature importance methods."""
    PFI = "permutation"
    SHAP = "shap"
    LIME = "lime"

    @classmethod
    def all_available(cls):
        """Return list of all available methods."""
        methods = [cls.PFI]  # PFI always available
        if SHAP_AVAILABLE:
            methods.append(cls.SHAP)
        if LIME_AVAILABLE:
            methods.append(cls.LIME)
        return methods


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
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. "
                              "Install with: pip install shap")
        return _calculate_shap(model, X, feature_names)

    elif method == FeatureImportanceMethod.LIME:
        if not LIME_AVAILABLE:
            raise ImportError("LIME not available. "
                              "Install with: pip install lime")
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


# --- NEW: DATASET NAME ENUM ---
class DatasetName:
    """Enum-like class for selecting datasets."""
    CUSTOM_NORMAL = "custom_normal"
    SEA_DRIFT = "sea_drift"
    HYPERPLANE_DRIFT = "hyperplane_drift"

    @classmethod
    def all_available(cls):
        methods = [cls.CUSTOM_NORMAL]
        if RIVER_AVAILABLE:
            methods.extend([cls.SEA_DRIFT, cls.HYPERPLANE_DRIFT])
        return methods


# --- MODIFIED: Renamed function ---
def generate_custom_normal_data(n_samples_before=1000, n_samples_after=1000,
                                random_seed=42):
    """
    Generate synthetic data stream with concept drift (Original Function).

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
    
    Returns
    -------
    tuple
        (X1, X2, y, drift_point)
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


# --- NEW: RIVER-BASED DATA GENERATORS ---

def _generate_river_data(river_stream, n_samples_before, n_samples_after):
    """Helper function to generate data from a river stream."""
    if not RIVER_AVAILABLE:
        raise ImportError("River library not installed.")
        
    X_all = []
    y_all = []
    total_samples = n_samples_before + n_samples_after

    for x, y_val in itertools.islice(river_stream, total_samples):
        # Ensure we only take 2 features for compatibility
        # SEA has 3 features, but only first 2 are relevant
        X_all.append([x[0], x[1]])
        y_all.append(y_val)
    
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    X1 = X_all[:, 0]
    X2 = X_all[:, 1]
    y = y_all
    drift_point = n_samples_before
    
    return X1, X2, y, drift_point

def generate_sea_drift_data(n_samples_before=1000, n_samples_after=1000,
                            random_seed=42):
    """
    Generate synthetic data stream using River's SEA generator.
    
    Drifts from variant 0 to variant 3 at the drift point.
    The SEA generator has 3 features; we use the first 2.

    Returns
    -------
    tuple
        (X1, X2, y, drift_point)
    """
    stream_SEA = synth.ConceptDriftStream(
        stream=synth.SEA(seed=random_seed, variant=0),
        drift_stream=synth.SEA(seed=random_seed, variant=3),
        position=n_samples_before,
        width=400,  # Gradual drift
        seed=random_seed
    )
    return _generate_river_data(stream_SEA, n_samples_before, n_samples_after)

def generate_hyperplane_data(n_samples_before=1000, n_samples_after=1000,
                             random_seed=42):
    """
    Generate synthetic data stream using River's Hyperplane generator.
    
    Drifts from a stable hyperplane to one with magnitude change.
    Uses n_features=2 for compatibility with analysis functions.

    Returns
    -------
    tuple
        (X1, X2, y, drift_point)
    """
    stream_HP = synth.ConceptDriftStream(
        stream=synth.Hyperplane(n_features=2, seed=random_seed,
                                noise_percentage=0.05),
        drift_stream=synth.Hyperplane(n_features=2, seed=random_seed,
                                      mag_change=0.2, noise_percentage=0.1),
        position=n_samples_before,
        width=400,  # Gradual drift
        seed=random_seed
    )
    return _generate_river_data(stream_HP, n_samples_before, n_samples_after)

# --- END NEW DATA GENERATORS ---


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


def compute_data_drift_analysis(X1, X2, y, drift_point,
                                importance_method="permutation"):
    """
    Compute data drift analysis by classifying time periods using only
    features (X).

    This function trains a neural network to distinguish between before-drift
    and after-drift data points based solely on feature distributions P(X).
    It then uses the specified feature importance method to determine which
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
    importance_method : str, default="permutation"
        Method for feature importance: "permutation", "shap", or "lime"

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': trained MLPClassifier
        - 'accuracy': model accuracy
        - 'importance_result': feature importance result
        - 'importance_mean': mean importance scores
        - 'importance_std': standard deviation of importance scores
    """
    print("\n" + "=" * 70)
    print("STEP 1: DATA DRIFT DETECTION "
          "(Classification using X features only)")
    print(f"Feature Importance Method: {importance_method.upper()}")
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

    # Calculate Feature Importance
    feature_names = ['X1', 'X2']
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
    print(f"  X1: {importance_mean[0]:.4f} ± {importance_std[0]:.4f}")
    print(f"  X2: {importance_mean[1]:.4f} ± {importance_std[1]:.4f}")

    return {
        'model': nn_model,
        'accuracy': nn_accuracy,
        'importance_result': fi_result,
        'importance_mean': importance_mean,
        'importance_std': importance_std
    }


def visualize_data_drift_analysis(analysis_result, feature_names=['X1', 'X2']):
    """
    Visualize the results of data drift analysis.

    Parameters
    ----------
    analysis_result : dict
        Result dictionary from compute_data_drift_analysis
    feature_names : list, default=['X1', 'X2']
        Names of features
    """
    fi_result = analysis_result['importance_result']
    importance_mean = analysis_result['importance_mean']
    importance_std = analysis_result['importance_std']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Feature Importance for Data Drift (X-only Classification) '
                 f'- {fi_result["method"]}',
                 fontsize=14, fontweight='bold')

    # Plot 1: Bar plot
    ax = axes[0]
    x_pos = np.arange(len(feature_names))
    ax.bar(x_pos, importance_mean, yerr=importance_std,
           color='#e74c3c', alpha=0.8, edgecolor='black', capsize=5)
    ax.set_ylabel(f'Importance Score ({fi_result["method"]})')
    ax.set_title('Feature Importance for Detecting Time-Period '
                 '(Concept Drift)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Box plot
    ax = axes[1]
    importances = fi_result['importances']
    
    # Handle cases where importances might not be (n_features, n_samples)
    if importances.shape[0] != len(feature_names) and importances.shape[1] == len(feature_names):
        importances = importances.T
        
    bp = ax.boxplot([importances[i] for i in range(len(feature_names))],
                    tick_labels=feature_names, patch_artist=True, notch=True,
                    showmeans=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('#e74c3c')
        patch.set_alpha(0.7)
    ax.set_ylabel(f'{fi_result["method"]} Score')
    ax.set_xlabel('Features')
    ax.set_title(f'Distribution of {fi_result["method"]} Scores')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def analyze_data_drift(X1, X2, y, drift_point,
                       importance_method="permutation"):
    """
    Analyze and visualize data drift by classifying time periods using only
    features (X).

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
    importance_method : str, default="permutation"
        Method for feature importance: "permutation", "shap", or "lime"

    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    result = compute_data_drift_analysis(X1, X2, y, drift_point,
                                         importance_method)
    visualize_data_drift_analysis(result)
    return result


def compute_concept_drift_analysis(X1, X2, y, drift_point,
                                   importance_method="permutation"):
    """
    Compute concept drift analysis by classifying time periods using features
    and target (X, Y).

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
    importance_method : str, default="permutation"
        Method for feature importance: "permutation", "shap", or "lime"

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': trained MLPClassifier
        - 'accuracy': model accuracy
        - 'importance_result': feature importance result
        - 'importance_mean': mean importance scores
        - 'importance_std': standard deviation of importance scores
    """
    print("\n" + "=" * 70)
    print("STEP 2: CONCEPT DRIFT DETECTION (Classification using X and Y "
          "features)")
    print(f"Feature Importance Method: {importance_method.upper()}")
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

    # Calculate Feature Importance
    feature_names = ['X1', 'X2', 'Y']
    fi_result = calculate_feature_importance(
        nn_model_xy, X_features_with_y, time_labels,
        method=importance_method,
        feature_names=feature_names
    )

    importance_mean = fi_result['importances_mean']
    importance_std = fi_result['importances_std']

    print("\n" + "-" * 70)
    print(f"{fi_result['method']} (Feature importance for detecting "
          "time-period using X and Y)")
    print("-" * 70)
    print(f"\nNeural Network (MLP) {fi_result['method']}:")
    print(f"  X1: {importance_mean[0]:.4f} ± {importance_std[0]:.4f}")
    print(f"  X2: {importance_mean[1]:.4f} ± {importance_std[1]:.4f}")
    print(f"  Y:  {importance_mean[2]:.4f} ± {importance_std[2]:.4f}")
    print("=" * 70)

    return {
        'model': nn_model_xy,
        'accuracy': nn_accuracy_xy,
        'importance_result': fi_result,
        'importance_mean': importance_mean,
        'importance_std': importance_std
    }


def visualize_concept_drift_analysis(analysis_result,
                                     feature_names=['X1', 'X2', 'Y']):
    """
    Visualize the results of concept drift analysis.

    Parameters
    ----------
    analysis_result : dict
        Result dictionary from compute_concept_drift_analysis
    feature_names : list, default=['X1', 'X2', 'Y']
        Names of features
    """
    fi_result = analysis_result['importance_result']
    importance_mean = analysis_result['importance_mean']
    importance_std = analysis_result['importance_std']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Feature Importance for Concept Drift (X+Y Classification) '
                 f'- {fi_result["method"]}',
                 fontsize=14, fontweight='bold')

    # Plot 1: Bar plot
    ax = axes[0]
    x_pos = np.arange(len(feature_names))
    ax.bar(x_pos, importance_mean, yerr=importance_std,
           color='#e74c3c', alpha=0.8, edgecolor='black', capsize=5)
    ax.set_ylabel(f'Importance Score ({fi_result["method"]})')
    ax.set_title(f'{fi_result["method"]} for Detecting '
                 'Time-Period (Concept Drift)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Box plot
    ax = axes[1]
    importances = fi_result['importances']
    
    # Handle cases where importances might not be (n_features, n_samples)
    if importances.shape[0] != len(feature_names) and importances.shape[1] == len(feature_names):
        importances = importances.T

    bp = ax.boxplot([importances[i] for i in range(len(feature_names))],
                    tick_labels=feature_names, patch_artist=True, notch=True,
                    showmeans=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#e74c3c')
        patch.set_alpha(0.7)
    ax.set_ylabel(f'{fi_result["method"]} Score')
    ax.set_xlabel('Features')
    ax.set_title(f'Distribution of {fi_result["method"]} Scores')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def analyze_concept_drift(X1, X2, y, drift_point,
                          importance_method="permutation"):
    """
    Analyze and visualize concept drift by classifying time periods using
    features and target (X, Y).

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
    importance_method : str, default="permutation"
        Method for feature importance: "permutation", "shap", or "lime"

    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    result = compute_concept_drift_analysis(X1, X2, y, drift_point,
                                            importance_method)
    visualize_concept_drift_analysis(result)
    return result


def compute_predictive_importance_shift(X1, X2, y, drift_point,
                                        importance_method="permutation"):
    """
    Compute how predictive feature importance shifts before and after drift.

    This function trains separate neural networks to predict the target
    variable before and after the drift, then compares the feature importance
    using the specified method. This reveals how the relationship between
    features and target changes.

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
    importance_method : str, default="permutation"
        Method for feature importance: "permutation", "shap", or "lime"

    Returns
    -------
    dict
        Dictionary containing:
        - 'model_before': trained MLPClassifier on before-drift data
        - 'model_after': trained MLPClassifier on after-drift data
        - 'accuracy_before': accuracy on before-drift data
        - 'accuracy_after': accuracy on after-drift data
        - 'fi_before': feature importance results for before-drift model
        - 'fi_after': feature importance results for after-drift model
    """
    print("\n" + "=" * 70)
    print("STEP 3: PREDICTIVE POWER SHIFT (Classification)")
    print(f"Feature Importance Method: {importance_method.upper()}")
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
    feature_names = ['X1', 'X2']
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
    print("Before Drift (X1, X2):")
    print(f"  X1: {fi_before['importances_mean'][0]:.4f} ± "
          f"{fi_before['importances_std'][0]:.4f}")
    print(f"  X2: {fi_before['importances_mean'][1]:.4f} ± "
          f"{fi_before['importances_std'][1]:.4f}")
    print("\nAfter Drift (X1, X2):")
    print(f"  X1: {fi_after['importances_mean'][0]:.4f} ± "
          f"{fi_after['importances_std'][0]:.4f}")
    print(f"  X2: {fi_after['importances_mean'][1]:.4f} ± "
          f"{fi_after['importances_std'][1]:.4f}")
    print("=" * 70)

    return {
        'model_before': mlp_before,
        'model_after': mlp_after,
        'accuracy_before': acc_before,
        'accuracy_after': acc_after,
        'fi_before': fi_before,
        'fi_after': fi_after
    }


def visualize_predictive_importance_shift(
    analysis_result,
    feature_names=['X1', 'X2']
):
    """
    Visualize the results of predictive importance shift analysis.

    Parameters
    ----------
    analysis_result : dict
        Result dictionary from compute_predictive_importance_shift
    feature_names : list, default=['X1', 'X2']
        Names of features
    """
    fi_before = analysis_result['fi_before']
    fi_after = analysis_result['fi_after']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Predictive Feature Importance (NN Before vs After Drift) '
                 f'- {fi_before["method"]}',
                 fontsize=14, fontweight='bold')

    # Plot 1: Bar comparison
    ax = axes[0]
    x_pos = np.arange(len(feature_names))
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

    # Plot 2: Side-by-side box plots
    ax = axes[1]
    
    # Handle cases where importances might not be (n_features, n_samples)
    importances_before = fi_before['importances']
    if importances_before.shape[0] != len(feature_names) and importances_before.shape[1] == len(feature_names):
        importances_before = importances_before.T
        
    importances_after = fi_after['importances']
    if importances_after.shape[0] != len(feature_names) and importances_after.shape[1] == len(feature_names):
        importances_after = importances_after.T

    positions_before = [0.8, 2.8]
    positions_after = [1.2, 3.2]
    bp1 = ax.boxplot([importances_before[0],
                      importances_before[1]],
                     positions=positions_before, widths=0.3,
                     patch_artist=True, showmeans=True)
    bp2 = ax.boxplot([importances_after[0], importances_after[1]],
                     positions=positions_after, widths=0.3,
                     patch_artist=True, showmeans=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('#1abc9c')
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor('#f39c12')
        patch.set_alpha(0.7)
    ax.set_ylabel(f'{fi_before["method"]} Score')
    ax.set_xlabel('Features')
    ax.set_title(f'Distribution of {fi_before["method"]} Scores')
    ax.set_xticks([1, 3])
    ax.set_xticklabels(feature_names)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
              ['Before Drift', 'After Drift'], loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def analyze_predictive_importance_shift(X1, X2, y, drift_point,
                                        importance_method="permutation"):
    """
    Analyze and visualize how predictive feature importance shifts before
    and after drift.

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
    importance_method : str, default="permutation"
        Method for feature importance: "permutation", "shap", or "lime"

    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    result = compute_predictive_importance_shift(X1, X2, y, drift_point,
                                                 importance_method)
    visualize_predictive_importance_shift(result)
    return result


def main(dataset_name, importance_method="permutation"):
    """
    Main function to run the complete concept drift analysis pipeline.

    This function orchestrates the entire analysis by:
    1. Generating synthetic data with concept drift
    2. Visualizing the data stream
    3. Analyzing data drift (changes in P(X))
    4. Analyzing concept drift (changes in P(Y|X))
    5. Analyzing predictive importance shift
       (how feature-target relationships change)

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to generate (from DatasetName class)
    importance_method : str, default="permutation"
        Method for feature importance calculation.
        Options: "permutation", "shap", "lime"
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
        if not RIVER_AVAILABLE and dataset_name in [DatasetName.SEA_DRIFT, DatasetName.HYPERPLANE_DRIFT]:
             print("The 'river' library is not installed. "
                   "Please install it with: pip install river")
        print(f"Available datasets: {available_datasets}")
        return

    print("=" * 70)
    print("CONCEPT DRIFT ANALYSIS")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Feature Importance Method: {importance_method.upper()}")
    print("=" * 70)

    # Step 1: Generate data
    print(f"\nGenerating synthetic data for: {dataset_name}...")
    
    gen_params = {
        "n_samples_before": 1000,
        "n_samples_after": 1000,
        "random_seed": 42
    }
    
    if dataset_name == DatasetName.CUSTOM_NORMAL:
        X1, X2, y, drift_point = generate_custom_normal_data(**gen_params)
    elif dataset_name == DatasetName.SEA_DRIFT:
        X1, X2, y, drift_point = generate_sea_drift_data(**gen_params)
    elif dataset_name == DatasetName.HYPERPLANE_DRIFT:
        X1, X2, y, drift_point = generate_hyperplane_data(**gen_params)
    else:
        # This case should be caught by the validation above, but as a fallback
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Step 2: Visualize data stream
    print("\nVisualizing data stream...")
    visualize_data_stream(X1, X2, y, drift_point)

    # Step 3: Analyze data drift (P(X) changes)
    analyze_data_drift(X1, X2, y, drift_point,
                       importance_method=importance_method)

    # Step 4: Analyze concept drift (P(Y|X) changes)
    analyze_concept_drift(X1, X2, y, drift_point,
                          importance_method=importance_method)

    # Step 5: Analyze predictive importance shift
    analyze_predictive_importance_shift(
        X1, X2, y, drift_point,
        importance_method=importance_method
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
    # - DatasetName.CUSTOM_NORMAL (Your original dataset)
    # - DatasetName.SEA_DRIFT (From the notebook, requires 'river')
    # - DatasetName.HYPERPLANE_DRIFT (From the notebook, requires 'river')

    DATASET_TO_RUN = DatasetName.HYPERPLANE_DRIFT

    # 2. Choose the feature importance method:
    # Options: "permutation", "shap", "lime"
    # (Note: "shap" and "lime" require installation)

    IMPORTANCE_METHOD_TO_RUN = "permutation"

    # --- END SETTINGS ---

    # Run the main analysis
    main(dataset_name=DATASET_TO_RUN,
         importance_method=IMPORTANCE_METHOD_TO_RUN)

    # --- EXAMPLES OF OTHER RUNS (uncomment to try) ---

    # Example 2: Run SEA_DRIFT with permutation
    # print("\n\n" + "*"*80 + "\nRUNNING SEA_DRIFT ANALYSIS\n" + "*"*80)
    # main(dataset_name=DatasetName.SEA_DRIFT,
    #      importance_method="permutation")

    # Example 3: Run HYPERPLANE_DRIFT with permutation
    # print("\n\n" + "*"*80 + "\nRUNNING HYPERPLANE_DRIFT ANALYSIS\n" + "*"*80)
    # main(dataset_name=DatasetName.HYPERPLANE_DRIFT,
    #      importance_method="permutation")

    # Example 4: Run CUSTOM_NORMAL with SHAP (if installed)
    # print("\n\n" + "*"*80 + "\nRUNNING CUSTOM_NORMAL (SHAP) ANALYSIS\n" + "*"*80)
    # main(dataset_name=DatasetName.CUSTOM_NORMAL,
    #      importance_method="shap")
