import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def _get_feature_bounds(feature_names, X_data=None):
    feature_mins = {}
    feature_maxs = {}
    if X_data is not None:
        if isinstance(X_data, pd.DataFrame):
            for col in feature_names:
                if col in X_data.columns:
                    feature_mins[col] = X_data[col].min()
                    feature_maxs[col] = X_data[col].max()
        else:
            # If numpy array, assume order matches feature_names
            for i, col in enumerate(feature_names):
                feature_mins[col] = X_data[:, i].min()
                feature_maxs[col] = X_data[:, i].max()
    return feature_mins, feature_maxs


def _format_rule_bounds(path_tuples, feature_mins, feature_maxs):
    feature_bounds = {}
    for name, op, threshold in path_tuples:
        if name not in feature_bounds:
            mn = feature_mins.get(name, -np.inf)
            mx = feature_maxs.get(name, np.inf)
            feature_bounds[name] = [mn, mx]
        if op == "<=":
            feature_bounds[name][1] = min(feature_bounds[name][1], threshold)
        elif op == ">":
            feature_bounds[name][0] = max(feature_bounds[name][0], threshold)

    range_strings = []
    for name, bounds in feature_bounds.items():
        lower, upper = bounds
        lower_str = "(-∞" if lower == -np.inf else f"[{lower:.2f}"
        upper_str = "+∞)" if upper == np.inf else f"{upper:.2f}]"
        range_strings.append(f"{name} ∈ {lower_str}, {upper_str}")
    return ",  ".join(range_strings)


def _recurse_rules(
    node, path_tuples, tree_, tree_classes, feature_names, rules_data, drift_leaves, feature_mins, feature_maxs
):
    # Check if leaf node
    if tree_.feature[node] == -2:
        val = tree_.value[node][0]
        total = val.sum()

        # Identify class 1 probability (Drift)
        target_class = 1
        class_prob = 0.0
        if len(tree_classes) > 1:
            if target_class in tree_classes:
                idx_1 = np.where(tree_classes == target_class)[0][0]
                class_prob = val[idx_1] / total
        else:
            # If only one class exists
            class_prob = 1.0 if tree_classes[0] == target_class else 0.0

        if class_prob > 0.5:
            readable_rule = _format_rule_bounds(path_tuples, feature_mins, feature_maxs)
            rules_data.append({
                'Rule': readable_rule,
                'Drift Conf.': f"{class_prob:.1%}",
                'Samples': int(tree_.n_node_samples[node]),
                'Coverage': tree_.n_node_samples[node] / tree_.n_node_samples[0],
                'Leaf_ID': node
            })
            drift_leaves.append(node)
        return

    name = feature_names[tree_.feature[node]]
    threshold = tree_.threshold[node]
    _recurse_rules(
        tree_.children_left[node], path_tuples + [(name, "<=", threshold)],
        tree_, tree_classes, feature_names, rules_data, drift_leaves, feature_mins, feature_maxs
    )
    _recurse_rules(
        tree_.children_right[node], path_tuples + [(name, ">", threshold)],
        tree_, tree_classes, feature_names, rules_data, drift_leaves, feature_mins, feature_maxs
    )


def get_disagreement_table(tree, feature_names, X_data=None):
    """
    Extracts rules from the tree for printing.
    Returns the dataframe and the raw leaf indices corresponding to 'Drift' rules.
    """
    tree_ = tree.tree_
    tree_classes = tree.classes_
    rules_data = []

    # Store leaf indices that classify as Drift (Class 1)
    drift_leaves = []

    feature_mins, feature_maxs = _get_feature_bounds(feature_names, X_data)

    _recurse_rules(0, [], tree_, tree_classes, feature_names, rules_data, drift_leaves, feature_mins, feature_maxs)

    if not rules_data:
        return pd.DataFrame(columns=["No Drift Regions Found"]), []

    df = pd.DataFrame(rules_data).sort_values(by='Coverage', ascending=False)
    # Format coverage
    df['Coverage'] = df['Coverage'].apply(lambda x: f"{x:.1%}")
    return df, df['Leaf_ID'].tolist()


def compute_disagreement_analysis(clf_pre, clf_post, X_eval_raw, X_eval_scaled, feature_names=None):
    """
    Computes disagreement (drift) between two classifiers on the evaluation set.

    Parameters
    ----------
    clf_pre : sklearn-like classifier
        Model trained on pre-drift data (expects scaled input).
    clf_post : sklearn-like classifier
        Model trained on post-drift data (expects scaled input).
    X_eval_raw : np.ndarray or pd.DataFrame
        Raw data (for interpreting rules).
    X_eval_scaled : np.ndarray
        Scaled data (for model prediction and visualization tree).
    feature_names : list, optional
        Names of the features.

    Returns
    -------
    dict
        Results including disagreement table, trees, and leaf IDs.
    """
    if feature_names is None:
        if hasattr(X_eval_raw, "columns"):
            feature_names = list(X_eval_raw.columns)
        else:
            feature_names = [f"Feature {i}" for i in range(X_eval_raw.shape[1])]

    # 1. Detect Drift Points (Disagreement)
    # Both models predict on scaled data
    pred_pre = clf_pre.predict(X_eval_scaled)
    pred_post = clf_post.predict(X_eval_scaled)
    y_delta = (pred_pre != pred_post).astype(int)

    drift_rate = np.mean(y_delta)

    if drift_rate == 0:
        return {
            'drift_rate': 0.0,
            'disagreement_table': None,
            'drift_leaf_ids': [],
            'viz_tree': None
        }

    # 2. Train Rule Surrogate (Interpretability)
    # Train on RAW data so rules are readable (e.g., Age > 25)
    rule_tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=0.01,
        class_weight='balanced',
        random_state=42
    )
    rule_tree.fit(X_eval_raw, y_delta)

    df_rules, _ = get_disagreement_table(rule_tree, feature_names, X_data=X_eval_raw)

    # 3. Train Visualization Tree (Plotting)
    # Train on SCALED data because SSNP inverse transform provides scaled data
    viz_tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=0.01,
        class_weight='balanced',
        random_state=42
    )
    viz_tree.fit(X_eval_scaled, y_delta)

    # We get leaf IDs from the viz_tree (for colormapping the plot)
    # We pass dummy feature names because internal split names don't matter for the map,
    # but we need consistent indexing.
    dummy_names = [f"f{i}" for i in range(X_eval_scaled.shape[1])]
    _, drift_leaf_ids = get_disagreement_table(viz_tree, dummy_names)

    return {
        'drift_rate': drift_rate,
        'disagreement_table': df_rules,
        'drift_leaf_ids': drift_leaf_ids,
        'viz_tree': viz_tree
    }
