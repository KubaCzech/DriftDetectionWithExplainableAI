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
            for i, col in enumerate(feature_names):
                feature_mins[col] = X_data[:, i].min()
                feature_maxs[col] = X_data[:, i].max()
    return feature_mins, feature_maxs

def _get_unscaler_map(feature_names, X_raw, X_scaled):
    unscalers = {}
    def get_col(X, idx, name):
        if isinstance(X, pd.DataFrame):
            return X[name].values if name in X.columns else None
        else:
            return X[:, idx] if idx < X.shape[1] else None

    for i, name in enumerate(feature_names):
        r_vals = get_col(X_raw, i, name)
        s_vals = get_col(X_scaled, i, name)
        if r_vals is not None and s_vals is not None and len(s_vals) > 1:
            try:
                if np.max(s_vals) != np.min(s_vals):
                    m, c = np.polyfit(s_vals, r_vals, 1)
                    unscalers[name] = (m, c)
                else:
                    unscalers[name] = (0, r_vals[0]) 
            except Exception:
                unscalers[name] = (1, 0)
    return unscalers

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
    node, path_tuples, tree_, tree_classes, feature_names, rules_data, drift_leaves, 
    feature_mins, feature_maxs, unscalers, total_samples
):
    # Check if leaf node
    if tree_.feature[node] == -2:
        val = tree_.value[node][0]
        node_total = val.sum()

        # Identify class 1 probability (Drift)
        target_class = 1
        class_prob = 0.0
        if len(tree_classes) > 1:
            if target_class in tree_classes:
                idx_1 = np.where(tree_classes == target_class)[0][0]
                class_prob = val[idx_1] / node_total
        else:
            class_prob = 1.0 if tree_classes[0] == target_class else 0.0

        # FILTER: Only show leaves that are predominantly Drift (>50%)
        if class_prob > 0.5:
            readable_rule = _format_rule_bounds(path_tuples, feature_mins, feature_maxs)
            
            # Coverage is relative to the Manifold (Grid), which represents the Visual Area
            coverage = node_total / total_samples

            rules_data.append({
                'Rule': readable_rule,
                'Drift Conf.': f"{class_prob:.1%}",
                'Samples (Grid)': int(node_total),
                'Coverage (Visual)': coverage,
                'Leaf_ID': node
            })
            drift_leaves.append(node)
        return

    # Extract feature name and threshold
    feat_idx = tree_.feature[node]
    name = feature_names[feat_idx]
    threshold_scaled = tree_.threshold[node]
    
    # Unscale threshold
    threshold_raw = threshold_scaled
    if name in unscalers:
        m, c = unscalers[name]
        threshold_raw = threshold_scaled * m + c

    _recurse_rules(
        tree_.children_left[node], path_tuples + [(name, "<=", threshold_raw)],
        tree_, tree_classes, feature_names, rules_data, drift_leaves, 
        feature_mins, feature_maxs, unscalers, total_samples
    )
    _recurse_rules(
        tree_.children_right[node], path_tuples + [(name, ">", threshold_raw)],
        tree_, tree_classes, feature_names, rules_data, drift_leaves, 
        feature_mins, feature_maxs, unscalers, total_samples
    )


def get_disagreement_table(tree, feature_names, X_raw=None, X_scaled=None):
    tree_ = tree.tree_
    tree_classes = tree.classes_
    rules_data = []
    drift_leaves = []

    feature_mins, feature_maxs = _get_feature_bounds(feature_names, X_raw)
    unscalers = {}
    if X_raw is not None and X_scaled is not None:
        unscalers = _get_unscaler_map(feature_names, X_raw, X_scaled)

    total_samples = tree_.n_node_samples[0]

    _recurse_rules(
        0, [], tree_, tree_classes, feature_names, 
        rules_data, drift_leaves, feature_mins, feature_maxs, unscalers, total_samples
    )

    if not rules_data:
        return pd.DataFrame(columns=["No Drift Regions Found"]), []

    df = pd.DataFrame(rules_data)

    # --- ASSIGN CONSISTENT RULE LABELS ---
    unique_leaves = sorted(df['Leaf_ID'].unique())
    leaf_to_label = {leaf: f"Rule {i+1}" for i, leaf in enumerate(unique_leaves)}
    df['Rule Label'] = df['Leaf_ID'].map(leaf_to_label)
    
    # Sort by Coverage
    df = df.sort_values(by='Coverage (Visual)', ascending=False)
    
    # Format coverage
    df['Coverage'] = df['Coverage (Visual)'].apply(lambda x: f"{x:.1%}")
    
    # Reorder columns
    cols = ['Rule Label', 'Rule', 'Drift Conf.', 'Samples (Grid)', 'Coverage', 'Leaf_ID']
    df = df[cols]
    
    return df, df['Leaf_ID'].tolist()


def compute_disagreement_analysis(clf_pre, clf_post, X_eval_raw, X_eval_scaled, X_grid_high_scaled=None, feature_names=None):
    """
    Computes disagreement using the Inverse-Projected Grid (SDBM approach) to ensure
    visual consistency between the table and the plot.
    """
    if feature_names is None:
        if hasattr(X_eval_raw, "columns"):
            feature_names = list(X_eval_raw.columns)
        else:
            feature_names = [f"Feature {i}" for i in range(X_eval_raw.shape[1])]

    # 1. Real Drift Rate (based on actual data samples)
    pred_pre = clf_pre.predict(X_eval_scaled)
    pred_post = clf_post.predict(X_eval_scaled)
    y_delta_real = (pred_pre != pred_post).astype(int)
    drift_rate_real = np.mean(y_delta_real)

    if drift_rate_real == 0:
        return {
            'drift_rate': 0.0,
            'disagreement_table': None,
            'drift_leaf_ids': [],
            'viz_tree': None
        }
        
    # 2. Train Visualization Tree on the GRID (Manifold) if available
    # This ensures the tree explains the PLOT, not just the data.
    if X_grid_high_scaled is not None:
        grid_pred_pre = clf_pre.predict(X_grid_high_scaled)
        grid_pred_post = clf_post.predict(X_grid_high_scaled)
        y_delta_grid = (grid_pred_pre != grid_pred_post).astype(int)
        
        X_train_tree = X_grid_high_scaled
        y_train_tree = y_delta_grid
    else:
        # Fallback to data if grid not provided
        X_train_tree = X_eval_scaled
        y_train_tree = y_delta_real

    # Train Tree
    viz_tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=0.01,
        class_weight='balanced',
        random_state=42
    )
    viz_tree.fit(X_train_tree, y_train_tree)

    # 3. Generate Rules
    # We pass X_eval_raw and X_eval_scaled ONLY to calculate the unscaling map 
    # (so the text rules match the raw data units).
    df_rules, drift_leaf_ids = get_disagreement_table(
        viz_tree, 
        feature_names, 
        X_raw=X_eval_raw, 
        X_scaled=X_eval_scaled
    )

    return {
        'drift_rate': drift_rate_real, # Metric from real data
        'disagreement_table': df_rules,
        'drift_leaf_ids': drift_leaf_ids,
        'viz_tree': viz_tree
    }