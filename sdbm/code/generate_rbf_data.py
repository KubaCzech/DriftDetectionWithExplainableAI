#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ===========================================================
# Configuration
# ===========================================================
n_samples = 2000
n_features = 4
gamma = 30.0
noise = 0
cluster_std = 0.05
prefix = "rbf_swap_4D_gaussian_fixed"
# ===========================================================

# Define 4 cluster centers in 4D → 2 per class
centers_class0 = np.array([
    [0.2, 0.8, 0.2, 0.8],
    [0.8, 0.2, 0.8, 0.2]
])

centers_class1 = np.array([
    [0.2, 0.2, 0.8, 0.8],
    [0.8, 0.8, 0.2, 0.2]
])

def generate_cluster_data(centers, total_samples, std):
    """Generate Gaussian clusters with guaranteed samples per cluster."""
    n_centers = len(centers)
    base = total_samples // n_centers
    remainder = total_samples % n_centers

    X_list = []
    for i, c in enumerate(centers):
        n_cluster = base + (1 if i < remainder else 0)
        X_list.append(np.random.normal(loc=c, scale=std, size=(n_cluster, len(c))))
    return np.vstack(X_list)

def generate_labels(X, centers_0, centers_1, gamma):
    """Assign class based on stronger RBF response."""
    K0 = np.max(rbf_kernel(X, centers_0, gamma=gamma), axis=1)
    K1 = np.max(rbf_kernel(X, centers_1, gamma=gamma), axis=1)
    return (K1 > K0).astype(int)

def add_label_noise(y, ratio):
    n_flip = int(ratio * len(y))
    idx = np.random.choice(len(y), n_flip, replace=False)
    y[idx] = 1 - y[idx]
    return y

# ===========================================================
# Generate PRE and POST drift data
# ===========================================================
half = n_samples // 2
X_pre = generate_cluster_data(np.vstack([centers_class0, centers_class1]), half, cluster_std)
X_post = generate_cluster_data(np.vstack([centers_class0, centers_class1]), half, cluster_std)

y_pre = generate_labels(X_pre, centers_class0, centers_class1, gamma)
y_post = generate_labels(X_post, centers_class1, centers_class0, gamma)  # swapped

y_pre = add_label_noise(y_pre, noise)
y_post = add_label_noise(y_post, noise)

# ===========================================================
# Save to files
# ===========================================================
np.save(f"X_pre_drift.npy", X_pre)
np.save(f"y_pre_drift.npy", y_pre)
np.save(f"X_post_drift.npy", X_post)
np.save(f"y_post_drift.npy", y_post)

print(f"[INFO] Saved Gaussian drift data with compact 4D clusters and swapped classes.")
print(f"[INFO] Shapes: X_pre={X_pre.shape}, X_post={X_post.shape}")

# ===========================================================
# Visualization (PCA projection)
# ===========================================================
pca = PCA(n_components=2)
X_pre_2D = pca.fit_transform(X_pre)
X_post_2D = pca.transform(X_post)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_pre_2D[:, 0], X_pre_2D[:, 1], c=y_pre, cmap="coolwarm", alpha=0.7)
plt.title("Pre-Drift Concept (Compact Gaussian Clusters)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.subplot(1, 2, 2)
plt.scatter(X_post_2D[:, 0], X_post_2D[:, 1], c=y_post, cmap="coolwarm", alpha=0.7)
plt.title("Post-Drift Concept (Swapped Gaussian Clusters)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()
plt.show()
