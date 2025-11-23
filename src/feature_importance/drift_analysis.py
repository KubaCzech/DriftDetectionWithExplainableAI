import numpy as np
from sklearn.neural_network import MLPClassifier
from .methods import calculate_feature_importance
from .visualization import (
    visualize_data_drift_analysis,
    visualize_concept_drift_analysis,
    visualize_predictive_importance_shift
)


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
        'feature_names_with_y': feature_names_with_y
    }


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
