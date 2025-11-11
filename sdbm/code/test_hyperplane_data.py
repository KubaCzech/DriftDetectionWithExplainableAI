"""
Stable SDBM for 4D River Hyperplane (Inspired by SDBM paper)
Author: ChatGPT (GPT-5)
"""

import os
import numpy as np
import matplotlib.cm as cm
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import cartesian
from sklearn.neighbors import KNeighborsRegressor
from skimage.color import rgb2hsv, hsv2rgb
from ssnp import SSNP

# ===========================================================
# Settings
# ===========================================================
output_dir = "./sdbm_hyperplane_rf4d_paperstyle"
os.makedirs(output_dir, exist_ok=True)
grid_size = 300

# ===========================================================
# Visualization helper
# ===========================================================
def results_to_png(np_matrix, prob_matrix, grid_size, n_classes,
                   dataset_name, classifier_name, real_points=None,
                   max_value_hsv=None, suffix=None):

    if suffix is not None:
        suffix = f"_{suffix}"
    else:
        suffix = ""

    data = cm.tab20(np_matrix / n_classes)
    data_vanilla = data[:, :, :3].copy()

    if max_value_hsv is not None:
        data_vanilla = rgb2hsv(data_vanilla)
        data_vanilla[:, :, 2] = max_value_hsv
        data_vanilla = hsv2rgb(data_vanilla)

    if real_points is not None:
        data_vanilla = rgb2hsv(data_vanilla)
        data_vanilla[real_points[:, 0], real_points[:, 1], 2] = 1
        data_vanilla = hsv2rgb(data_vanilla)

    data_hsv = data[:, :, :3].copy()
    data_hsv = rgb2hsv(data_hsv)
    data_hsv[:, :, 2] = prob_matrix
    data_hsv = hsv2rgb(data_hsv)

    for name, img_data in zip(
        ["vanilla", "hsv"],
        [data_vanilla, data_hsv]
    ):
        rescaled = (img_data * 255.0).astype(np.uint8)
        im = Image.fromarray(rescaled)
        filename = f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_{name}{suffix}.png"
        path = os.path.join(output_dir, filename)
        im.save(path)
        print(f"[SAVED] {path}")

# ===========================================================
# Load Data
# ===========================================================
print("[INFO] Loading 4D hyperplane data...")
X = np.load("X_hyperplane_4d.npy")
y = np.load("y_hyperplane_4d.npy")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

print(f"[INFO] Train shape: {X_train.shape}")
print(f"[INFO] Classes: {np.unique(y)}")

# ===========================================================
# Train classifier
# ===========================================================
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
print(f"[INFO] Test accuracy: {clf.score(X_test, y_test):.4f}")
n_classes = len(np.unique(y_train))

# ===========================================================
# Train SSNP Projection (4D → 2D)
# ===========================================================
proj = SSNP(epochs=20, patience=5, opt="adam", bottleneck_activation="relu", verbose=False)
proj.fit(X_train, y_train)
Z_train = proj.transform(X_train)

print("[INFO] Projection complete. Shape:", Z_train.shape)
np.save(os.path.join(output_dir, "X_SSNP_hyperplane_4d.npy"), Z_train)

# ===========================================================
# Build smooth decision boundary using latent-space interpolation
# ===========================================================
print("[INFO] Interpolating decision boundaries in latent space...")

# Predict probabilities in original 4D space
probs_train = clf.predict_proba(X_train)
y_pred_train = np.argmax(probs_train, axis=1)
confidence_train = np.max(probs_train, axis=1)

# Interpolate label and confidence fields in latent space
label_interp = KNeighborsRegressor(n_neighbors=10, weights='distance')
label_interp.fit(Z_train, y_pred_train)

conf_interp = KNeighborsRegressor(n_neighbors=10, weights='distance')
conf_interp.fit(Z_train, confidence_train)

# Generate 2D latent grid
xmin, xmax = np.percentile(Z_train[:, 0], [1, 99])
ymin, ymax = np.percentile(Z_train[:, 1], [1, 99])

x_intrvls = np.linspace(xmin, xmax, grid_size)
y_intrvls = np.linspace(ymin, ymax, grid_size)
xx, yy = np.meshgrid(x_intrvls, y_intrvls)
pts = np.c_[xx.ravel(), yy.ravel()]

# Predict interpolated values
print("[INFO] Predicting interpolated class and confidence fields...")
labels_interp = label_interp.predict(pts)
conf_interp_vals = conf_interp.predict(pts)

# Reshape to grid
img_grid = labels_interp.reshape(grid_size, grid_size)
prob_grid = conf_interp_vals.reshape(grid_size, grid_size)

# Smooth slightly
img_grid = gaussian_filter(img_grid, sigma=0.6)
prob_grid = gaussian_filter(prob_grid, sigma=0.6)
prob_grid = np.clip(prob_grid, 0, 1)

# Normalize projected points for overlay
mms = MinMaxScaler()
norm_pts = mms.fit_transform(Z_train)
norm_pts = (norm_pts * (grid_size - 1)).astype(int)

# Save visualization
results_to_png(
    np_matrix=img_grid,
    prob_matrix=prob_grid,
    grid_size=grid_size,
    n_classes=n_classes,
    dataset_name="hyperplane_4d",
    classifier_name="rf_interp",
    real_points=norm_pts,
    max_value_hsv=0.8,
    suffix="paperstyle"
)

print("[DONE] Stable SDBM (inspired by original paper) created successfully!")
