import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.extmath import cartesian
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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

    # 1. Pre-calculate Min/Max for every feature from the raw data
    feature_mins = {}
    feature_maxs = {}
    
    if X_data is not None:
        # Calculate bounds from the actual dataframe passed in
        for col in feature_names:
            feature_mins[col] = X_data[col].min()
            feature_maxs[col] = X_data[col].max()

    def format_bounds(path_tuples):
        feature_bounds = {}
        
        for name, op, threshold in path_tuples:
            # Initialize with Data Min/Max instead of Infinity
            if name not in feature_bounds:
                # Default to -inf/inf only if we didn't get X_data
                mn = feature_mins.get(name, -np.inf)
                mx = feature_maxs.get(name, np.inf)
                feature_bounds[name] = [mn, mx]
            
            if op == "<=":
                # Tighter upper bound
                feature_bounds[name][1] = min(feature_bounds[name][1], threshold)
            elif op == ">":
                # Tighter lower bound
                feature_bounds[name][0] = max(feature_bounds[name][0], threshold)

        range_strings = []
        for name, bounds in feature_bounds.items():
            lower, upper = bounds
            
            # Formatting Logic:
            # If bound matches data min/max, use '[' (Inclusive)
            # If bound comes from a '>' split, use '(' (Exclusive)
            
            # LOWER BOUND
            if lower == -np.inf:
                lower_str = "(-∞"
            else:
                # Check if this is the Data Min (inclusive) or a Split Threshold (exclusive)
                data_min = feature_mins.get(name, -np.inf)
                if lower == data_min:
                    lower_str = f"[{lower:.2f}" # Data start
                else:
                    lower_str = f"({lower:.2f}" # Split boundary (>)

            # UPPER BOUND
            if upper == np.inf:
                upper_str = "+∞)"
            else:
                # Upper bound from '<=' is inclusive, Data Max is inclusive
                upper_str = f"{upper:.2f}]"
            
            range_strings.append(f"{name} ∈ {lower_str}, {upper_str}")
            
        return ",  ".join(range_strings)

    def recurse(node, path_tuples):
        if tree_.feature[node] == -2:
            val = tree_.value[node][0]
            total = val.sum()
            
            if len(tree_classes) == 1:
                class_prob = 1.0 if tree_classes[0] == 1 else 0.0
            else:
                class_prob = val[1] / total

            samples = tree_.n_node_samples[node]
            coverage = samples / tree_.n_node_samples[0]
            
            if class_prob > 0.5: 
                readable_rule = format_bounds(path_tuples)
                rules_data.append({
                    'Rule': readable_rule,
                    'Drift Conf.': f"{class_prob:.1%}",
                    'Samples': samples,
                    'Coverage': f"{coverage:.1%}"
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
# Core Logic: Compare Pre vs Post Models
# ===========================================================
# [MODIFIED] Now accepts X_eval_norm (for models) AND X_eval_raw (for tree rules)
def analyze_drift(arch_name, X_eval_norm, X_eval_raw, feature_names, projector=None):
    print(f"\n" + "="*60)
    print(f" ANALYZING DRIFT FOR ARCHITECTURE: {arch_name.upper()}")
    print("="*60)

    # 1. Load Pre and Post Models
    path_pre = os.path.join(OUTPUT_DIR, f"hyperplane4d_pre_drift_{arch_name}.pkl")
    path_post = os.path.join(OUTPUT_DIR, f"hyperplane4d_post_drift_{arch_name}.pkl")
    
    if not os.path.exists(path_pre) or not os.path.exists(path_post):
        print(f"[WARN] Could not find models for {arch_name}. Skipping.")
        return

    clf_pre = pickle.load(open(path_pre, "rb"))
    clf_post = pickle.load(open(path_post, "rb"))

    # 2. Predict using NORMALIZED data (What the models expect)
    pred_pre = clf_pre.predict(X_eval_norm)
    pred_post = clf_post.predict(X_eval_norm)
    
    # 3. Create Disagreement Labels
    y_delta = (pred_pre != pred_post).astype(int)
    drift_rate = np.mean(y_delta)
    print(f"[INFO] {arch_name.upper()} Prediction Change Rate: {drift_rate:.2%}")
    
    if drift_rate == 0:
        print("[INFO] No drift detected.")
        return

    # 4. Generate DeltaXplainer Rules using RAW data
    # We train the tree on RAW features to predict the disagreement found above
    surrogate = DecisionTreeClassifier(
        max_depth=4, 
        min_samples_leaf=0.01, 
        class_weight='balanced',
        random_state=42
    )
    
    # Create DataFrame from Raw Data
    X_df_raw = pd.DataFrame(X_eval_raw, columns=feature_names)
    
    # Fit tree on Raw Data
    surrogate.fit(X_df_raw, y_delta)
    
    # PASS X_df_raw to the table generator to get Min/Max bounds
    df_rules = get_disagreement_table(surrogate, feature_names, X_data=X_df_raw)
    
    pd.set_option('display.max_colwidth', None)
    print(f"\n--- {arch_name.upper()} Drift Regions (Raw Values) ---")
    print(df_rules.to_string(index=False))

    # 5. Visualize (Visualization still uses Normalized data/SSNP logic)
    if projector:
        plot_drift_map(clf_pre, clf_post, projector, arch_name)

def plot_drift_map(clf_pre, clf_post, ssnpgt, arch_name):
    # Determine bounds based on projection
    # (Simplified: assume standard range or load projection file)
    xmin, xmax, ymin, ymax = -4, 4, -4, 4
    
    x_grid = np.linspace(xmin, xmax, GRID_SIZE)
    y_grid = np.linspace(ymin, ymax, GRID_SIZE)
    pts = cartesian((x_grid, y_grid))
    
    # Inverse transform to get high-dim points
    # NOTE: We map 2D latent -> High Dim -> Predict with both models
    print(f"[INFO] Generating drift map for {arch_name}...")
    
    batch_size = 50000
    disagreement_grid = np.zeros(len(pts))
    
    for i in range(0, len(pts), batch_size):
        batch_pts = pts[i:i+batch_size]
        batch_high_dim = ssnpgt.inverse_transform(batch_pts)
        
        p1 = clf_pre.predict(batch_high_dim)
        p2 = clf_post.predict(batch_high_dim)
        disagreement_grid[i:i+batch_size] = (p1 != p2).astype(int)

    disagreement_grid = disagreement_grid.reshape(GRID_SIZE, GRID_SIZE)
    disagreement_grid = np.flipud(disagreement_grid)

    plt.figure(figsize=(8, 8))
    plt.title(f"{arch_name.upper()}: Pre-Drift vs Post-Drift Disagreement\nRed Areas = Decision Boundary Shifted")
    
    # Custom cmap: Green (Stable), Red (Drifted)
    cmap = ListedColormap(['#e6ffe6', '#ff4d4d']) 
    
    plt.imshow(disagreement_grid, extent=[xmin, xmax, ymin, ymax], cmap=cmap, alpha=0.8)
    plt.colorbar(label="Prediction Change")
    plt.xlabel("SSNP Component 2")
    plt.ylabel("SSNP Component 1")
    
    save_path = os.path.join(OUTPUT_DIR, f"drift_map_{arch_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Map saved to {save_path}")



if __name__ == "__main__":
    print("[INFO] Loading datasets...")
    
    # 1. Load Normalized Data (For Models)
    X_pre_norm = np.load("X_pre_drift.npy")
    X_post_norm = np.load("X_post_drift.npy")
    
    # 2. Load Raw Data (For Explanation)
    # Assuming you have saved these files previously
    if os.path.exists("X_pre_drift_raw.npy") and os.path.exists("X_post_drift_raw.npy"):
        X_pre_raw = np.load("X_pre_drift_raw.npy")
        X_post_raw = np.load("X_post_drift_raw.npy")
        print("[INFO] Raw data loaded successfully.")
    else:
        raise FileNotFoundError("Please ensure X_pre_drift_raw.npy and X_post_drift_raw.npy exist.")

    # 3. Combine Datasets (Pre U Post)
    X_combined_norm = np.concatenate([X_pre_norm, X_post_norm], axis=0)
    X_combined_raw = np.concatenate([X_pre_raw, X_post_raw], axis=0)
    
    # 4. Subsample (Must use same indices for both!)
    if len(X_combined_norm) > 20000:
        # Fix seed for reproducibility
        np.random.seed(42)
        indices = np.random.choice(len(X_combined_norm), 10000, replace=False)
        
        X_combined_norm = X_combined_norm[indices]
        X_combined_raw = X_combined_raw[indices]
    
    # Create meaningful feature names if available, otherwise generic
    # If raw data has columns like "Salary", "Age", put them here list(df.columns)
    feature_names = [f"Feature_{i}" for i in range(X_combined_raw.shape[1])]

    # 5. Load SSNP (Stays on Normalized Logic)
    n_features = X_pre_norm.shape[1]
    projector_path = os.path.join(OUTPUT_DIR, f"hyperplane{n_features}d_pre_drift_ssnp")
    
    ssnp_model = None
    if os.path.exists(projector_path):
        print("[INFO] Loading SSNP Projector...")
        ssnp_model = ssnp.SSNP(epochs=1, opt="adam", bottleneck_activation="linear")
        ssnp_model.load_model(projector_path)

    # 6. Analyze
    # Pass BOTH datasets
    analyze_drift("rf", X_combined_norm, X_combined_raw, feature_names, ssnp_model)
    analyze_drift("mlp", X_combined_norm, X_combined_raw, feature_names, ssnp_model)
    
    print("\n[DONE] Check ./results_hyperplane/ for images.")