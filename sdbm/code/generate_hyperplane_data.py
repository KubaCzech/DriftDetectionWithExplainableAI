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
# Generate Hyperplane data with drift
# ===========================================================
def generate_hyperplane_data(n_samples=2000, n_features=4, seed=42, 
                             mag_change=0.0, noise_percentage=0.0, 
                             normal_vector=None):
    """
    Generates hyperplane data, with an option to manually set the
    hyperplane's normal vector.
    """
    print(f"[INFO] Generating {n_features}D Hyperplane: mag_change={mag_change}, noise={noise_percentage}...")
    stream = synth.Hyperplane(
        n_features=n_features,
        noise_percentage=noise_percentage,
        mag_change=mag_change,
        seed=seed
    )

    # --- THIS IS THE NEW PART ---
    # We manually override the generator's internal normal vector
    if normal_vector is not None:
        if len(normal_vector) == n_features:
            print(f"[INFO] Manually setting normal vector to: {normal_vector}")
            # This accesses the internal state of the River generator
            # We ensure it's a normalized numpy array
            vec = np.array(normal_vector, dtype=float)
            stream._normal_vector = vec / np.linalg.norm(vec)
        else:
            print(f"[WARNING] 'normal_vector' length ({len(normal_vector)}) "
                  f"does not match 'n_features' ({n_features}). Ignoring.")
    # --- END NEW PART ---

    X, y = [], []
    for x, label in stream.take(n_samples):
        X.append([x[i] for i in range(n_features)])
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"[INFO] Generated {len(X)} samples with {X.shape[1]} features.")
    return X, y


# ===========================================================
# Visualization of drift (first 2 dimensions only)
# ===========================================================
def visualize_drift(X_pre, y_pre, X_post, y_post, n_samples):
    print("[INFO] Visualizing data drift (dims 0 vs 1)...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_pre[:, 0], X_pre[:, 1],
                c=y_pre, cmap="coolwarm", s=20, alpha=0.7)
    plt.title("Pre-Drift (Horizontal Boundary)")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")

    plt.subplot(1, 2, 2)
    plt.scatter(X_post[:, 0], X_post[:, 1],
                c=y_post, cmap="coolwarm", s=20, alpha=0.7)
    plt.title("Post-Drift (Vertical Boundary)")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")

    plt.tight_layout()
    
    # --- CHANGE: Save plot instead of showing ---
    save_path = "data_drift_visualization.png"
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] Saved drift visualization to {save_path}")
    # plt.show() # <-- This would block your pipeline


# ===========================================================
# Main runner
# ===========================================================
if __name__ == "__main__":
    n_samples = 2000
    n_features = 4

    # --- Define the normal vectors for our controlled drift ---
    
    # Horizontal boundary (separates on feature 1)
    vec_pre = [0.0] * n_features
    vec_pre[1] = 1.0 
    
    # Vertical boundary (separates on feature 0)
    vec_post = [0.0] * n_features
    vec_post[0] = 1.0

    # -------------------------------------------------------
    # Generate pre-drift data (HORIZONTAL)
    # -------------------------------------------------------
    X_pre, y_pre = generate_hyperplane_data(
        n_samples=n_samples,
        n_features=n_features,
        seed=42,
        mag_change=0.0,      # No random drift
        noise_percentage=0.0, # Clean data
        normal_vector=vec_pre # Set the horizontal vector
    )

    # -------------------------------------------------------
    # Generate post-drift data (VERTICAL)
    # -------------------------------------------------------
    X_post, y_post = generate_hyperplane_data(
        n_samples=n_samples,
        n_features=n_features,
        seed=84, 
        mag_change=0.0,      # No random drift
        noise_percentage=0.0, # Clean data
        normal_vector=vec_post # Set the vertical vector
    )

    # -------------------------------------------------------
    # Visualize drift using first two features
    # -------------------------------------------------------
    visualize_drift(X_pre, y_pre, X_post, y_post, n_samples)

    # -------------------------------------------------------
    # Save both datasets for later experiments
    # -------------------------------------------------------
    np.save("X_pre_drift.npy", X_pre)
    np.save("y_pre_drift.npy", y_pre)
    np.save("X_post_drift.npy", X_post)
    np.save("y_post_drift.npy", y_post)

    print("[INFO] Saved:")
    print("   → X_pre_drift.npy, y_pre_drift.npy")
    print("   → X_post_drift.npy, y_post_drift.npy")