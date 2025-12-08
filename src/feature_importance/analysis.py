import numpy as np
from .methods import calculate_feature_importance


def compute_data_drift_analysis(X_before, y_before, X_after, y_after,
                                feature_names=None,
                                importance_method="permutation",
                                model_class=None,
                                model_params=None):
    """
    Compute data drift analysis by classifying time periods using only
    features (X).

    Parameters
    ----------
    X_before : array-like (n_samples, n_features)
        Feature matrix for 'before' window
    y_before : array-like (n_samples,)
        Labels for 'before' window
    X_after : array-like (n_samples, n_features)
        Feature matrix for 'after' window
    y_after : array-like (n_samples,)
        Labels for 'after' window
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
    if feature_names is None and hasattr(X_before, 'columns'):
        feature_names = X_before.columns.tolist()
    
    X_features = np.concatenate([X_before, X_after])
    
    n_samples_before = len(X_before)
    n_samples_after = len(X_after)
    time_labels = np.array([0] * n_samples_before + [1] * n_samples_after)

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


def compute_concept_drift_analysis(X_before, y_before, X_after, y_after,
                                   feature_names=None,
                                   importance_method="permutation",
                                   model_class=None,
                                   model_params=None):
    """
    Compute concept drift analysis by classifying time periods using features
    and target (X, Y).

    Parameters
    ----------
    X_before : array-like (n_samples, n_features)
        Feature matrix for 'before' window
    y_before : array-like (n_samples,)
        Labels for 'before' window
    X_after : array-like (n_samples, n_features)
        Feature matrix for 'after' window
    y_after : array-like (n_samples,)
        Labels for 'after' window
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
    if feature_names is None and hasattr(X_before, 'columns'):
        feature_names = X_before.columns.tolist()
    
    X_combined = np.concatenate([X_before, X_after])
    y_combined = np.concatenate([y_before, y_after])

    X_features_with_y = np.column_stack([X_combined, y_combined])
    
    n_samples_before = len(X_before)
    n_samples_after = len(X_after)
    time_labels = np.array([0] * n_samples_before + [1] * n_samples_after)

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


def compute_predictive_importance_shift(X_before, y_before, X_after, y_after,
                                        feature_names=None,
                                        importance_method="permutation",
                                        model_class=None,
                                        model_params=None):
    """
    Compute how predictive feature importance shifts before and after drift.

    Parameters
    ----------
    X_before : array-like (n_samples, n_features)
        Feature matrix for 'before' window
    y_before : array-like (n_samples,)
        Labels for 'before' window
    X_after : array-like (n_samples, n_features)
        Feature matrix for 'after' window
    y_after : array-like (n_samples,)
        Labels for 'after' window
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
    if feature_names is None and hasattr(X_before, 'columns'):
        feature_names = X_before.columns.tolist()

    X_features_before = X_before
    X_features_after = X_after

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
