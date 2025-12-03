# import numpy as np
# import matplotlib.pyplot as plt
# from river.datasets import synth

# # ===========================================================
# # Generate Hyperplane data with drift
# # ===========================================================
# def generate_hyperplane_data(n_samples=2000, n_features=4, seed=42, mag_change=0.01):
#     print(f"[INFO] Generating {n_features}D Hyperplane stream data (mag_change={mag_change})...")
#     stream = synth.Hyperplane(
#         n_features=n_features,
#         noise_percentage=0.0,
#         mag_change=mag_change,
#         seed=seed
#     )

#     X, y = [], []
#     for x, label in stream.take(n_samples):
#         X.append([x[i] for i in range(n_features)])
#         y.append(label)

#     X = np.array(X)
#     y = np.array(y)

#     print(f"[INFO] Generated {len(X)} samples with {X.shape[1]} features.")
#     return X, y


# # ===========================================================
# # Visualization of drift (first 2 dimensions only)
# # ===========================================================
# def visualize_drift(X_pre, y_pre, X_post, y_post, n_samples):
#     plt.figure(figsize=(12, 5))

#     plt.subplot(1, 2, 1)
#     plt.scatter(X_pre[:, 0], X_pre[:, 1],
#                 c=y_pre, cmap="coolwarm", s=20, alpha=0.7)
#     plt.title("Pre-Drift (dims 0 vs 1)")
#     plt.xlabel("att0")
#     plt.ylabel("att1")

#     plt.subplot(1, 2, 2)
#     plt.scatter(X_post[:, 0], X_post[:, 1],
#                 c=y_post, cmap="coolwarm", s=20, alpha=0.7)
#     plt.title("Post-Drift (dims 0 vs 1)")
#     plt.xlabel("att0")
#     plt.ylabel("att1")

#     plt.tight_layout()
#     plt.show()


# # ===========================================================
# # Main runner
# # ===========================================================
# if __name__ == "__main__":
#     n_samples = 2000
#     n_features = 4

#     # -------------------------------------------------------
#     # Generate pre-drift data (smaller mag_change = stable)
#     # -------------------------------------------------------
#     X_pre, y_pre = generate_hyperplane_data(
#         n_samples=n_samples,
#         n_features=n_features,
#         seed=42,
#         mag_change=0.0
#     )

#     # -------------------------------------------------------
#     # Generate post-drift data (higher mag_change = drift)
#     # -------------------------------------------------------
#     X_post, y_post = generate_hyperplane_data(
#         n_samples=n_samples,
#         n_features=n_features,
#         seed=84,         # different seed to simulate shift
#         mag_change=0.05  # larger change to create concept drift
#     )

#     # -------------------------------------------------------
#     # Visualize drift using first two features
#     # -------------------------------------------------------
#     visualize_drift(X_pre, y_pre, X_post, y_post, n_samples)

#     # -------------------------------------------------------
#     # Save both datasets for later experiments
#     # -------------------------------------------------------
#     np.save("X_pre_drift.npy", X_pre)
#     np.save("y_pre_drift.npy", y_pre)
#     np.save("X_post_drift.npy", X_post)
#     np.save("y_post_drift.npy", y_post)

#     print("[INFO] Saved:")
#     print("  → X_pre_drift.npy, y_pre_drift.npy")
#     print("  → X_post_drift.npy, y_post_drift.npy")

import numpy as np
import matplotlib.pyplot as plt
from river.datasets import synth
import os

# ===========================================================
# Helper: Scale Normalized Data to "Raw" Ranges
# ===========================================================
def scale_to_raw(X):
    """
    Transforms normalized [0,1] data into simulated 'raw' feature space.
    We apply arbitrary min/max ranges to create realistic-looking values.
    """
    X_raw = X.copy()
    n_features = X.shape[1]
    
    # Define arbitrary ranges for 4 dimensions: (min, max)
    # Dim 0: Age-like (18 to 90)
    # Dim 1: Income-like (20,000 to 150,000)
    # Dim 2: Credit Score-like (300 to 850)
    # Dim 3: Years Employed-like (0 to 40)
    ranges = [
        (18, 90),       
        (20000, 150000),
        (300, 850),     
        (0, 40)         
    ]
    
    for i in range(min(n_features, len(ranges))):
        min_val, max_val = ranges[i]
        # Transformation: scaled = norm * (max - min) + min
        X_raw[:, i] = X[:, i] * (max_val - min_val) + min_val
        
    return X_raw

# ===========================================================
# Generate Hyperplane data with drift
# ===========================================================
def generate_hyperplane_data(n_samples=2000, n_features=4, seed=42, 
                             mag_change=0.0, noise_percentage=0.0, 
                             normal_vector=None):
    """
    Generates hyperplane data (Normalized) AND creates a Raw version.
    """
    print(f"[INFO] Generating {n_features}D Hyperplane: mag_change={mag_change}, noise={noise_percentage}...")
    stream = synth.Hyperplane(
        n_features=n_features,
        noise_percentage=noise_percentage,
        mag_change=mag_change,
        seed=seed
    )

    # Manually override the generator's internal normal vector
    if normal_vector is not None:
        if len(normal_vector) == n_features:
            print(f"[INFO] Manually setting normal vector to: {normal_vector}")
            vec = np.array(normal_vector, dtype=float)
            stream._normal_vector = vec / np.linalg.norm(vec)
        else:
            print(f"[WARNING] 'normal_vector' length mismatch. Ignoring.")

    X_norm, y = [], []
    for x, label in stream.take(n_samples):
        X_norm.append([x[i] for i in range(n_features)])
        y.append(label)

    X_norm = np.array(X_norm)
    y = np.array(y)

    # --- NEW: Generate Raw Data ---
    # We maintain the exact same rows/labels, just scale the features
    X_raw = scale_to_raw(X_norm)

    print(f"[INFO] Generated {len(X_norm)} samples.")
    return X_norm, X_raw, y


# ===========================================================
# Visualization of drift
# ===========================================================
def visualize_drift(X_pre, y_pre, X_post, y_post):
    print("[INFO] Visualizing data drift (dims 0 vs 1)...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_pre[:, 0], X_pre[:, 1], c=y_pre, cmap="coolwarm", s=20, alpha=0.7)
    plt.title("Pre-Drift (Normalized)")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")

    plt.subplot(1, 2, 2)
    plt.scatter(X_post[:, 0], X_post[:, 1], c=y_post, cmap="coolwarm", s=20, alpha=0.7)
    plt.title("Post-Drift (Normalized)")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")

    plt.tight_layout()
    plt.savefig("data_drift_visualization.png", dpi=150)
    print(f"[INFO] Saved drift visualization to data_drift_visualization.png")


# ===========================================================
# Main runner
# ===========================================================
if __name__ == "__main__":
    n_samples = 2000
    n_features = 4

    # Define normal vectors (Drift Logic)
    vec_pre = [0.0] * n_features
    vec_pre[1] = 1.0  # Separate on Feature 1 (Income-like)
    
    vec_post = [0.0] * n_features
    vec_post[0] = 1.0 # Separate on Feature 0 (Age-like)

    # 1. Generate Pre-Drift
    X_pre, X_pre_raw, y_pre = generate_hyperplane_data(
        n_samples=n_samples, n_features=n_features, seed=42,
        normal_vector=vec_pre
    )

    # 2. Generate Post-Drift
    X_post, X_post_raw, y_post = generate_hyperplane_data(
        n_samples=n_samples, n_features=n_features, seed=84,
        normal_vector=vec_post
    )

    # 3. Visualize (Normalized)
    visualize_drift(X_pre, y_pre, X_post, y_post)

    # 4. Save Everything
    # Normalized (For Model Training)
    np.save("X_pre_drift.npy", X_pre)
    np.save("y_pre_drift.npy", y_pre)
    np.save("X_post_drift.npy", X_post)
    np.save("y_post_drift.npy", y_post)

    # Raw (For Explanation Rules)
    np.save("X_pre_drift_raw.npy", X_pre_raw)
    np.save("X_post_drift_raw.npy", X_post_raw)

    print("[INFO] Saved all datasets:")
    print("   -> X_pre_drift.npy / .raw.npy")
    print("   -> X_post_drift.npy / .raw.npy")