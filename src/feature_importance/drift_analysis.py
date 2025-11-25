import numpy as np
from sklearn.neural_network import MLPClassifier
from .methods import calculate_feature_importance
from .visualization import (
    visualize_data_drift_analysis,
    visualize_concept_drift_analysis,
    visualize_predictive_importance_shift
)


def compute_data_drift_analysis(X, y, drift_point, feature_names=None,
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
    # Prepare features and labels
    if feature_names is None and hasattr(X, 'columns'):
        feature_names = X.columns.tolist()
    
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

    # Calculate Feature Importance
    fi_result = calculate_feature_importance(
        nn_model, X_features, time_labels,
        method=importance_method,
        feature_names=feature_names
    )

    importance_mean = fi_result['importances_mean']
    importance_std = fi_result['importances_std']

    return {
        'model': nn_model,
        'accuracy': nn_accuracy,
        'importance_result': fi_result,
        'importance_mean': importance_mean,
        'importance_std': importance_std
    }


def analyze_data_drift(X, y, drift_point, feature_names=None,
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


def compute_concept_drift_analysis(X, y, drift_point, feature_names=None,
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
    # Prepare features and labels
    if feature_names is None and hasattr(X, 'columns'):
        feature_names = X.columns.tolist()

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

    # Calculate Feature Importance
    feature_names_with_y = feature_names + ['Y']
    fi_result = calculate_feature_importance(
        nn_model_xy, X_features_with_y, time_labels,
        method=importance_method,
        feature_names=feature_names_with_y
    )

    importance_mean = fi_result['importances_mean']
    importance_std = fi_result['importances_std']

    return {
        'model': nn_model_xy,
        'accuracy': nn_accuracy_xy,
        'importance_result': fi_result,
        'importance_mean': importance_mean,
        'importance_std': importance_std,
        'feature_names_with_y': feature_names_with_y
    }


def analyze_concept_drift(X, y, drift_point, feature_names=None,
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


def compute_predictive_importance_shift(X, y, drift_point, feature_names=None,
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
    # Split data into before and after drift
    if feature_names is None and hasattr(X, 'columns'):
        feature_names = X.columns.tolist()

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

    return {
        'model_before': mlp_before,
        'model_after': mlp_after,
        'accuracy_before': acc_before,
        'accuracy_after': acc_after,
        'fi_before': fi_before,
        'fi_after': fi_after
    }


def analyze_predictive_importance_shift(X, y, drift_point, feature_names=None,
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
