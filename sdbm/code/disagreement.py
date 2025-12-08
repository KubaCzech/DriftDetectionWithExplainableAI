import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.extmath import cartesian
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from skimage.color import rgb2hsv, hsv2rgb
import ssnp  # Requires your ssnp.py

# ===========================================================
# Configuration
# ===========================================================
OUTPUT_DIR = "./results_hyperplane"
GRID_SIZE = 300

def get_disagreement_table(tree, feature_names, X_data=None):
    tree_ = tree.tree_
    tree_classes = tree.classes_
    rules_data = []

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
                    'Coverage': f"{tree_.n_node_samples[node] / tree_.n_node_samples[0]:.1%}"
                })
            return
        name = feature_names[tree_.feature[node]]
        threshold = tree_.threshold[node]
        recurse(tree_.children_left[node], path_tuples + [(name, "<=", threshold)])
        recurse(tree_.children_right[node], path_tuples + [(name, ">", threshold)])

    recurse(0, [])
    if not rules_data: 
        return pd.DataFrame(columns=["No Drift Regions Found"])
    return pd.DataFrame(rules_data).sort_values(by='Coverage', ascending=False)

# ===========================================================
# Visualization: Exact Square Replica of Experiments.py
# ===========================================================
def plot_drift_overlay_exact(clf_post, ssnpgt, arch_name, X_norm, y_delta, X_ssnpgt_ref):
    print(f"[INFO] Generating Exact SDBM + Rule Overlay for {arch_name}...")

    # 1. Train Visualization Tree (High Dim -> Rule ID)
    viz_tree = DecisionTreeClassifier(
        max_depth=4, 
        min_samples_leaf=0.01, 
        class_weight='balanced', 
        random_state=42
    )
    viz_tree.fit(X_norm, y_delta)

    # 2. Determine Bounds EXACTLY like experiments.py
    # Component 0 = Rows (Vertical), Component 1 = Cols (Horizontal)
    xmin, xmax = np.min(X_ssnpgt_ref[:, 0]), np.max(X_ssnpgt_ref[:, 0])
    ymin, ymax = np.min(X_ssnpgt_ref[:, 1]), np.max(X_ssnpgt_ref[:, 1])

    # 3. Create Grid Arrays
    x_intrvls = np.linspace(xmin, xmax, num=GRID_SIZE)
    y_intrvls = np.linspace(ymin, ymax, num=GRID_SIZE)
    
    # Indices for matrix assignment
    x_grid_indices = np.linspace(0, GRID_SIZE - 1, num=GRID_SIZE)
    y_grid_indices = np.linspace(0, GRID_SIZE - 1, num=GRID_SIZE)

    # Create coordinate pairs
    pts = cartesian((x_intrvls, y_intrvls))
    pts_grid = cartesian((x_grid_indices, y_grid_indices)).astype(int)

    # 4. Fill Matrices
    img_grid = np.zeros((GRID_SIZE, GRID_SIZE))         
    prob_grid = np.zeros((GRID_SIZE, GRID_SIZE))        
    rule_grid = np.zeros((GRID_SIZE, GRID_SIZE))        
    drift_mask_grid = np.zeros((GRID_SIZE, GRID_SIZE))  

    batch_size = 100000
    
    for i in range(0, len(pts), batch_size):
        pts_batch = pts[i:i + batch_size]
        pts_grid_batch = pts_grid[i:i + batch_size]
        
        high_dim_batch = ssnpgt.inverse_transform(pts_batch)

        # A. Background
        if hasattr(clf_post, "predict_proba"):
            probs = clf_post.predict_proba(high_dim_batch)
            alpha = np.amax(probs, axis=1)
            labels = probs.argmax(axis=1)
        else:
            labels = clf_post.predict(high_dim_batch)
            alpha = np.ones(len(labels))

        # B. Foreground
        rule_leaves = viz_tree.apply(high_dim_batch)
        is_drift = viz_tree.predict(high_dim_batch)

        # C. Assign
        rows = pts_grid_batch[:, 0]
        cols = pts_grid_batch[:, 1]
        
        img_grid[rows, cols] = labels
        prob_grid[rows, cols] = alpha
        rule_grid[rows, cols] = rule_leaves
        drift_mask_grid[rows, cols] = is_drift

    # 5. Flip Vertical Axis (Standard orientation from experiments.py)
    img_grid = np.flipud(img_grid)
    prob_grid = np.flipud(prob_grid)
    rule_grid = np.flipud(rule_grid)
    drift_mask_grid = np.flipud(drift_mask_grid)

    # 6. Construct Background Image Manually (Tab20 + HSV)
    n_classes = len(np.unique(img_grid))
    if n_classes < 2: n_classes = 2 
    
    norm_classes = img_grid / n_classes
    rgba_base = cm.tab20(norm_classes)
    
    hsv_base = rgb2hsv(rgba_base[:, :, :3])
    hsv_base[:, :, 2] = prob_grid 
    final_rgb = hsv2rgb(hsv_base)

    # 7. Plotting - FORCE SQUARE FIGURE
    # We use (10, 10) to force a square canvas
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # We allow aspect='auto' so the rectangular data range fills the square axes
    # This matches the behavior of generating a 300x300 pixel image regardless of data range
    ax.imshow(final_rgb, extent=[ymin, ymax, xmin, xmax], aspect='auto')
    
    # Overlay Rules
    drift_leaf_ids = np.unique(rule_grid[drift_mask_grid == 1])
    rule_colors = ['#FF0000', '#FF00FF', '#000000', '#00FFFF', '#FFFFFF']
    legend_patches = []

    for i, leaf_id in enumerate(drift_leaf_ids):
        color = rule_colors[i % len(rule_colors)]
        mask = (rule_grid == leaf_id) & (drift_mask_grid == 1)
        mask = mask.astype(float)
        
        if np.sum(mask) == 0: continue

        ax.contour(mask, levels=[0.5], extent=[ymin, ymax, xmin, xmax], 
                   colors=[color], linewidths=3, origin='upper')
        ax.contourf(mask, levels=[0.5, 1.5], extent=[ymin, ymax, xmin, xmax], 
                    colors='none', hatches=['//'], origin='upper', extend='max')
        
        patch = mpatches.Patch(facecolor='none', hatch='//', edgecolor=color, label=f'Rule {i+1}')
        legend_patches.append(patch)

    ax.set_title(f"{arch_name.upper()}: Post-Drift SDBM + Rule Regions")
    ax.set_xlabel("SSNP Component 2")
    ax.set_ylabel("SSNP Component 1")

    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right')

    save_path = os.path.join(OUTPUT_DIR, f"drift_overlay_rules_{arch_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {save_path}")

# ===========================================================
# Core Logic
# ===========================================================
def analyze_drift(arch_name, X_eval_norm, X_eval_raw, feature_names, projector, X_ssnpgt_ref):
    print(f"\n--- Analyzing {arch_name.upper()} ---")

    path_pre = os.path.join(OUTPUT_DIR, f"hyperplane4d_pre_drift_{arch_name}.pkl")
    path_post = os.path.join(OUTPUT_DIR, f"hyperplane4d_post_drift_{arch_name}.pkl")
    
    if not os.path.exists(path_pre) or not os.path.exists(path_post):
        print(f"[WARN] Models missing for {arch_name}. Skipping.")
        return

    clf_pre = pickle.load(open(path_pre, "rb"))
    clf_post = pickle.load(open(path_post, "rb"))

    pred_pre = clf_pre.predict(X_eval_norm)
    pred_post = clf_post.predict(X_eval_norm)
    y_delta = (pred_pre != pred_post).astype(int)
    
    if np.mean(y_delta) == 0:
        print("[INFO] No drift detected.")
        return

    surrogate = DecisionTreeClassifier(
        max_depth=4, 
        min_samples_leaf=0.01, 
        class_weight='balanced', 
        random_state=42
    )
    X_df_raw = pd.DataFrame(X_eval_raw, columns=feature_names)
    surrogate.fit(X_df_raw, y_delta)
    
    df_rules = get_disagreement_table(surrogate, feature_names, X_data=X_df_raw)
    print(f"\n[DRIFT RULES] {arch_name.upper()}")
    print(df_rules.to_string(index=False))

    if projector is not None:
        plot_drift_overlay_exact(clf_post, projector, arch_name, X_eval_norm, y_delta, X_ssnpgt_ref)

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
    
    if len(X_combined_norm) > 20000:
        np.random.seed(42)
        idx = np.random.choice(len(X_combined_norm), 10000, replace=False)
        X_combined_norm = X_combined_norm[idx]
        X_combined_raw = X_combined_raw[idx]
    
    feature_names = [f"Feature_{i}" for i in range(X_combined_raw.shape[1])]

    n_features = X_pre_norm.shape[1]
    projector_path = os.path.join(OUTPUT_DIR, f"hyperplane{n_features}d_pre_drift_ssnp")
    
    ssnp_model = None
    X_ssnpgt_ref = None

    if os.path.exists(projector_path):
        print("[INFO] Loading SSNP Projector...")
        ssnp_model = ssnp.SSNP(epochs=1, opt="adam", bottleneck_activation="linear")
        ssnp_model.load_model(projector_path)
        
        # Calculate Reference Bounds from Post-Drift Data
        print("[INFO] Calculating Bounds from Post-Drift Data...")
        X_ssnpgt_ref = ssnp_model.transform(X_post_norm) 

    analyze_drift("rf", X_combined_norm, X_combined_raw, feature_names, ssnp_model, X_ssnpgt_ref)
    analyze_drift("mlp", X_combined_norm, X_combined_raw, feature_names, ssnp_model, X_ssnpgt_ref)
    
    print("\n[DONE]")