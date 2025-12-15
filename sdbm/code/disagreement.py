import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.extmath import cartesian
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import ssnp  # Requires your ssnp.py

# ===========================================================
# Configuration
# ===========================================================
OUTPUT_DIR = "./results_hyperplane"
GRID_SIZE = 300

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

    feature_mins = {}
    feature_maxs = {}
    if X_data is not None:
        for col in feature_names:
            feature_mins[col] = X_data[col].min()
            feature_maxs[col] = X_data[col].max()

    def format_bounds(path_tuples):
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

    def recurse(node, path_tuples):
        # Check if leaf node
        if tree_.feature[node] == -2:
            val = tree_.value[node][0]
            total = val.sum()
            class_prob = val[1] / total if len(tree_classes) > 1 else (1.0 if tree_classes[0] == 1 else 0.0)
            
            if class_prob > 0.5: 
                readable_rule = format_bounds(path_tuples)
                rules_data.append({
                    'Rule': readable_rule,
                    'Drift Conf.': f"{class_prob:.1%}",
                    'Samples': tree_.n_node_samples[node],
                    'Coverage': tree_.n_node_samples[node] / tree_.n_node_samples[0], # Keep as float for sorting
                    'Leaf_ID': node  # Store internal ID
                })
                drift_leaves.append(node)
            return

        name = feature_names[tree_.feature[node]]
        threshold = tree_.threshold[node]
        recurse(tree_.children_left[node], path_tuples + [(name, "<=", threshold)])
        recurse(tree_.children_right[node], path_tuples + [(name, ">", threshold)])

    recurse(0, [])
    
    if not rules_data: 
        return pd.DataFrame(columns=["No Drift Regions Found"]), []
    
    df = pd.DataFrame(rules_data).sort_values(by='Coverage', ascending=False)
    # Reformat coverage for display
    df['Coverage'] = df['Coverage'].apply(lambda x: f"{x:.1%}")
    return df, df['Leaf_ID'].tolist()

# ===========================================================
# Visualization: Categorical Drift Map (New Function)
# ===========================================================
def plot_drift_map_rules(viz_tree, ssnpgt, arch_name, drift_leaf_ids, bounds):
    print(f"[INFO] Generating Categorical Drift Map for {arch_name}...")

    # Unpack exact bounds passed from main (matching the DBM script)
    # Note: DBM script maps x_intrvls (Comp 0) to Rows (Vertical)
    # and y_intrvls (Comp 1) to Cols (Horizontal)
    xmin, xmax, ymin, ymax = bounds

    # Create Grid Arrays (Exact match to experiments.py)
    x_intrvls = np.linspace(xmin, xmax, num=GRID_SIZE)
    y_intrvls = np.linspace(ymin, ymax, num=GRID_SIZE)
    
    # Indices for matrix assignment
    x_grid_indices = np.linspace(0, GRID_SIZE - 1, num=GRID_SIZE)
    y_grid_indices = np.linspace(0, GRID_SIZE - 1, num=GRID_SIZE)

    # Create coordinate pairs
    pts = cartesian((x_intrvls, y_intrvls))
    pts_grid = cartesian((x_grid_indices, y_grid_indices)).astype(int)

    # Fill Matrix
    rule_map_grid = np.zeros((GRID_SIZE, GRID_SIZE))
    
    batch_size = 100000
    for i in range(0, len(pts), batch_size):
        pts_batch = pts[i:i + batch_size]
        pts_grid_batch = pts_grid[i:i + batch_size]
        
        high_dim_batch = ssnpgt.inverse_transform(pts_batch)
        leaves = viz_tree.apply(high_dim_batch)
        
        rows = pts_grid_batch[:, 0]
        cols = pts_grid_batch[:, 1]
        rule_map_grid[rows, cols] = leaves

    # Flip to match standard image orientation
    rule_map_grid = np.flipud(rule_map_grid)

    # ... [Rest of categorical mapping logic stays the same] ...
    # Map raw leaf IDs to plot indices
    final_grid = np.zeros_like(rule_map_grid)
    leaf_to_plot_idx = {leaf: i+1 for i, leaf in enumerate(drift_leaf_ids)}
    unique_leaves = np.unique(rule_map_grid)
    for leaf in unique_leaves:
        if leaf in leaf_to_plot_idx:
            final_grid[rule_map_grid == leaf] = leaf_to_plot_idx[leaf]
        else:
            final_grid[rule_map_grid == leaf] = 0 

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10)) # Square figure to match
    
    num_rules = len(drift_leaf_ids)
    cmap_base = plt.get_cmap('tab10', num_rules)
    colors = ['#f0f0f0'] + [mcolors.rgb2hex(cmap_base(i)) for i in range(num_rules)]
    cmap = mcolors.ListedColormap(colors)
    bounds_norm = np.arange(-0.5, num_rules + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds_norm, cmap.N)

    # Use explicit extent matching bounds
    # imshow extent order: [left, right, bottom, top]
    # ymin/ymax = Component 2 (Horizontal/Cols)
    # xmin/xmax = Component 1 (Vertical/Rows)
    img = ax.imshow(final_grid, cmap=cmap, norm=norm, 
                    extent=[ymin, ymax, xmin, xmax], 
                    aspect='auto', interpolation='nearest')

    ax.set_title(f"{arch_name.upper()} Disagreement Map")
    ax.set_ylabel("SSNP Component 1")
    ax.set_xlabel("SSNP Component 2")

    cbar = plt.colorbar(img, ticks=np.arange(0, num_rules + 1), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['Stable'] + [f'Rule {i+1}' for i in range(num_rules)])
    
    save_path = os.path.join(OUTPUT_DIR, f"drift_map_rules_{arch_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {save_path}")

# ===========================================================
# Core Logic
# ===========================================================
def analyze_drift(arch_name, X_eval_norm, X_eval_raw, feature_names, projector, bounds):
    print(f"\n--- Analyzing {arch_name.upper()} ---")

    path_pre = os.path.join(OUTPUT_DIR, f"hyperplane4d_pre_drift_{arch_name}.pkl")
    path_post = os.path.join(OUTPUT_DIR, f"hyperplane4d_post_drift_{arch_name}.pkl")
    
    if not os.path.exists(path_pre) or not os.path.exists(path_post):
        print(f"[WARN] Models missing for {arch_name}. Skipping.")
        return

    clf_pre = pickle.load(open(path_pre, "rb"))
    clf_post = pickle.load(open(path_post, "rb"))

    # 1. Detect Drift Points
    pred_pre = clf_pre.predict(X_eval_norm)
    pred_post = clf_post.predict(X_eval_norm)
    y_delta = (pred_pre != pred_post).astype(int)
    
    if np.mean(y_delta) == 0:
        print("[INFO] No drift detected.")
        return

    # 2. Train Explainable Surrogate (Visualization Tree)
    surrogate = DecisionTreeClassifier(
        max_depth=4, 
        min_samples_leaf=0.01, 
        class_weight='balanced', 
        random_state=42
    )
    X_df_raw = pd.DataFrame(X_eval_raw, columns=feature_names)
    surrogate.fit(X_df_raw, y_delta)
    
    # 3. Get Rules and Leaf IDs
    viz_tree = DecisionTreeClassifier(
        max_depth=4, 
        min_samples_leaf=0.01, 
        class_weight='balanced', 
        random_state=42
    )
    viz_tree.fit(X_eval_norm, y_delta)
    
    df_rules, _ = get_disagreement_table(surrogate, feature_names, X_data=X_df_raw)
    print(f"\n[DRIFT RULES] {arch_name.upper()}")
    print(df_rules.to_string(index=False))

    _, drift_leaf_ids = get_disagreement_table(viz_tree, [f"f{i}" for i in range(X_eval_norm.shape[1])])

    if projector is not None:
        plot_drift_map_rules(viz_tree, projector, arch_name, drift_leaf_ids, bounds)
        
if __name__ == "__main__":
    print("[INFO] Loading datasets...")
    
    X_pre_norm = np.load("X_pre_drift.npy")
    X_post_norm = np.load("X_post_drift.npy") 
    
    if os.path.exists("X_pre_drift_raw.npy") and os.path.exists("X_post_drift_raw.npy"):
        X_pre_raw = np.load("X_pre_drift_raw.npy")
        X_post_raw = np.load("X_post_drift_raw.npy")
    else:
        raise FileNotFoundError("Missing raw .npy files.")

    X_combined_norm = np.concatenate([X_pre_norm, X_post_norm], axis=0)
    X_combined_raw = np.concatenate([X_pre_raw, X_post_raw], axis=0)
    
    # Subsample for speed if dataset is huge
    if len(X_combined_norm) > 20000:
        np.random.seed(42)
        idx = np.random.choice(len(X_combined_norm), 10000, replace=False)
        X_combined_norm = X_combined_norm[idx]
        X_combined_raw = X_combined_raw[idx]
    
    feature_names = [f"Feature_{i}" for i in range(X_combined_raw.shape[1])]

    n_features = X_pre_norm.shape[1]
    projector_path = os.path.join(OUTPUT_DIR, f"hyperplane{n_features}d_pre_drift_ssnp")
    
    ssnp_model = None
    ref_bounds = None

    if os.path.exists(projector_path):
        print("[INFO] Loading SSNP Projector...")
        ssnp_model = ssnp.SSNP(epochs=1, opt="adam", bottleneck_activation="linear")
        ssnp_model.load_model(projector_path)
        
        # --- CHANGE START: Use Post-Drift Data ---
        print("[INFO] Calculating Bounds from Post-Drift Data (for exact DBM alignment)...")
        
        # 1. Transform Post-Drift Data
        X_ssnpgt_post = ssnp_model.transform(X_post_norm) 
        
        # 2. Extract EXACT min/max (No margins, matching experiments.py)
        xmin = np.min(X_ssnpgt_post[:, 0])
        xmax = np.max(X_ssnpgt_post[:, 0])
        ymin = np.min(X_ssnpgt_post[:, 1])
        ymax = np.max(X_ssnpgt_post[:, 1])
        
        ref_bounds = (xmin, xmax, ymin, ymax)
        # --- CHANGE END ---

    analyze_drift("rf", X_combined_norm, X_combined_raw, feature_names, ssnp_model, ref_bounds)
    analyze_drift("mlp", X_combined_norm, X_combined_raw, feature_names, ssnp_model, ref_bounds)
    
    print("\n[DONE]")