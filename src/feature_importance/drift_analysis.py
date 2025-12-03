import numpy as np
from sklearn.neural_network import MLPClassifier
from .methods import calculate_feature_importance
from .visualization import (
    visualize_data_drift_analysis,
    visualize_concept_drift_analysis,
    visualize_predictive_importance_shift
)


def compute_data_drift_analysis(X, y, feature_names=None,
                                importance_method="permutation",
                                window_before_start=0,
                                window_after_start=0,
                                window_length=1000,
                                model_class=None,
                                model_params=None):
    """
    Compute data drift analysis by classifying time periods using only
    features (X).

    Parameters
    ----------
    X : array-like (n_samples, n_features)
        Feature matrix
    y : array-like (n_samples,)
        Binary class labels (not used in this analysis)
    feature_names : list
        Names of features
    importance_method : str, default="permutation"
        Method for feature importance: "permutation", "shap", or "lime"
    window_before_start : int
        Start index for window before drift
    window_after_start : int
        Start index for window after drift (relative to drift point)
    window_length : int
        Length of the analysis window

    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    # Prepare features and labels
    if feature_names is None and hasattr(X, 'columns'):
        feature_names = X.columns.tolist()
    
    # Define windows
    start_before = window_before_start
    end_before = start_before + window_length
    
    start_after = window_after_start
    end_after = start_after + window_length
    
    # Slice data
    X_before = X[start_before:end_before]
    X_after = X[start_after:end_after]
    
    X_features = np.concatenate([X_before, X_after])
    
    n_samples_before = len(X_before)
    n_samples_after = len(X_after)
    time_labels = np.array([0] * n_samples_before + [1] * n_samples_after)

    # Train Neural Network
    # Train Model
    if model_class is None:
        from src.models.mlp import MLPModel
        model_class = MLPModel
    
    if model_params is None:
        model_params = {}

    model = model_class(**model_params)
    model.fit(X_features, time_labels)
    nn_accuracy = model.score(X_features, time_labels)

    # Calculate Feature Importance
    fi_result = calculate_feature_importance(
        model, X_features, time_labels,
        method=importance_method,
        feature_names=feature_names
    )

    importance_mean = fi_result['importances_mean']
    importance_std = fi_result['importances_std']

    return {
        'model': model,
        'accuracy': nn_accuracy,
        'importance_result': fi_result,
        'importance_mean': importance_mean,
        'importance_std': importance_std
    }


def analyze_data_drift(X, y, drift_point, feature_names=None,
                       importance_method="permutation",
                       show_importance_boxplot=True,
                       model_class=None,
                       model_params=None):
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
                                         importance_method,
                                         model_class=model_class,
                                         model_params=model_params)
    visualize_data_drift_analysis(result, feature_names,
                                  show_boxplot=show_importance_boxplot)
    return result


def compute_concept_drift_analysis(X, y, feature_names=None,
                                   importance_method="permutation",
                                   window_before_start=0,
                                   window_after_start=0,
                                   window_length=1000,
                                   model_class=None,
                                   model_params=None):
    """
    Compute concept drift analysis by classifying time periods using features
    and target (X, Y).

    Parameters
    ----------
    X : array-like (n_samples, n_features)
        Feature matrix
    y : array-like (n_samples,)
        Binary class labels
    feature_names : list
        Names of features
    importance_method : str, default="permutation"
        Method for feature importance: "permutation", "shap", or "lime"
    window_before_start : int
        Start index for window before drift
    window_after_start : int
        Start index for window after drift (relative to drift point)
    window_length : int
        Length of the analysis window

    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    # Prepare features and labels
    if feature_names is None and hasattr(X, 'columns'):
        feature_names = X.columns.tolist()

    # Define windows
    start_before = window_before_start
    end_before = start_before + window_length
    
    start_after = window_after_start
    end_after = start_after + window_length
    
    # Slice data
    X_before = X[start_before:end_before]
    y_before = y[start_before:end_before]
    
    X_after = X[start_after:end_after]
    y_after = y[start_after:end_after]
    
    X_combined = np.concatenate([X_before, X_after])
    y_combined = np.concatenate([y_before, y_after])

    X_features_with_y = np.column_stack([X_combined, y_combined])
    
    n_samples_before = len(X_before)
    n_samples_after = len(X_after)
    time_labels = np.array([0] * n_samples_before + [1] * n_samples_after)

    # Train Neural Network
    # Train Model
    if model_class is None:
        from src.models.mlp import MLPModel
        model_class = MLPModel
    
    if model_params is None:
        model_params = {}

    model_xy = model_class(**model_params)
    model_xy.fit(X_features_with_y, time_labels)
    nn_accuracy_xy = model_xy.score(X_features_with_y, time_labels)

    # Calculate Feature Importance
    feature_names_with_y = feature_names + ['Y']
    fi_result = calculate_feature_importance(
        model_xy, X_features_with_y, time_labels,
        method=importance_method,
        feature_names=feature_names_with_y
    )

    importance_mean = fi_result['importances_mean']
    importance_std = fi_result['importances_std']

    return {
        'model': model_xy,
        'accuracy': nn_accuracy_xy,
        'importance_result': fi_result,
        'importance_mean': importance_mean,
        'importance_std': importance_std,
        'feature_names_with_y': feature_names_with_y
    }


def analyze_concept_drift(X, y, drift_point, feature_names=None,
                          importance_method="permutation",
                          show_importance_boxplot=True,
                          model_class=None,
                          model_params=None):
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
                                            importance_method,
                                            model_class=model_class,
                                            model_params=model_params)
    # Get the feature names list that includes 'Y' from the result
    feature_names_with_y = result['feature_names_with_y']
    visualize_concept_drift_analysis(result, feature_names_with_y,
                                     show_boxplot=show_importance_boxplot)
    return result


def compute_predictive_importance_shift(X, y, feature_names=None,
                                        importance_method="permutation",
                                        window_before_start=0,
                                        window_after_start=0,
                                        window_length=1000,
                                        model_class=None,
                                        model_params=None):
    """
    Compute how predictive feature importance shifts before and after drift.

    Parameters
    ----------
    X : array-like (n_samples, n_features)
        Feature matrix
    y : array-like (n_samples,)
        Binary class labels
    feature_names : list
        Names of features
    importance_method : str, default="permutation"
        Method for feature importance: "permutation", "shap", or "lime"
    window_before_start : int
        Start index for window before drift
    window_after_start : int
        Start index for window after drift (relative to drift point)
    window_length : int
        Length of the analysis window

    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    # Split data into before and after drift
    if feature_names is None and hasattr(X, 'columns'):
        feature_names = X.columns.tolist()

    # Define windows
    start_before = window_before_start
    end_before = start_before + window_length
    
    start_after = window_after_start
    end_after = start_after + window_length

    X_features_before = X[start_before:end_before]
    y_before = y[start_before:end_before]
    
    X_features_after = X[start_after:end_after]
    y_after = y[start_after:end_after]

    # Train Models
    if model_class is None:
        from src.models.mlp import MLPModel
        model_class = MLPModel
    
    if model_params is None:
        model_params = {}

    # Model trained BEFORE drift
    model_before = model_class(**model_params)
    model_before.fit(X_features_before, y_before)
    acc_before = model_before.score(X_features_before, y_before)

    # Model trained AFTER drift
    model_after = model_class(**model_params)
    model_after.fit(X_features_after, y_after)
    acc_after = model_after.score(X_features_after, y_after)

    # Feature Importance for BEFORE drift
    fi_before = calculate_feature_importance(
        model_before, X_features_before, y_before,
        method=importance_method,
        feature_names=feature_names
    )

    # Feature Importance for AFTER drift
    fi_after = calculate_feature_importance(
        model_after, X_features_after, y_after,
        method=importance_method,
        feature_names=feature_names
    )

    return {
        'model_before': model_before,
        'model_after': model_after,
        'accuracy_before': acc_before,
        'accuracy_after': acc_after,
        'fi_before': fi_before,
        'fi_after': fi_after
    }


def analyze_predictive_importance_shift(X, y, drift_point, feature_names=None,
                                        importance_method="permutation",
                                        show_importance_boxplot=True,
                                        model_class=None,
                                        model_params=None):
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
                                                 importance_method,
                                                 model_class=model_class,
                                                 model_params=model_params)
    visualize_predictive_importance_shift(result, feature_names,
                                          show_boxplot=show_importance_boxplot)
    return result
