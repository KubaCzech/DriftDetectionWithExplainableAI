"""
SDBM-style Decision Boundary Map (MNIST)
Faithful recreation of the original paper result.
"""

import os
import numpy as np
import matplotlib.cm as cm
from tqdm import tqdm
from PIL import Image
from skimage.color import rgb2hsv, hsv2rgb
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.extmath import cartesian
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.datasets import mnist
from ssnp import SSNP  # ensure ssnp.py is in same folder

# ===========================================================
# Output setup
# ===========================================================
output_dir = "./sdbm_mnist_results"
os.makedirs(output_dir, exist_ok=True)
grid_size = 300
batch_size = 10000


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

    data_alpha = data.copy()
    data_hsv = data[:, :, :3].copy()
    data_alpha[:, :, 3] = prob_matrix

    data_hsv = rgb2hsv(data_hsv)
    data_hsv[:, :, 2] = prob_matrix
    data_hsv = hsv2rgb(data_hsv)

    for name, img_data in zip(
        ["vanilla", "alpha", "hsv"],
        [data_vanilla, data_alpha, data_hsv]
    ):
        rescaled = (img_data * 255.0).astype(np.uint8)
        im = Image.fromarray(rescaled)
        filename = f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_{name}{suffix}.png"
        path = os.path.join(output_dir, filename)
        im.save(path)
        print(f"[SAVED] {path}")


# ===========================================================
# 1️⃣ Load MNIST dataset
# ===========================================================
print("[INFO] Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten and normalize
X_train = X_train.reshape((len(X_train), -1)) / 255.0
X_test = X_test.reshape((len(X_test), -1)) / 255.0

# Reduce for faster runtime
X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=8000, stratify=y_train, random_state=42)
X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=2000, stratify=y_test, random_state=42)

n_classes = len(np.unique(y_train))
print(f"[INFO] Using {len(X_train)} training samples, {len(X_test)} testing samples")

# ===========================================================
# 2️⃣ Train Logistic Regression (same as paper)
# ===========================================================
print("[INFO] Training Logistic Regression classifier...")
clf = LogisticRegression(
    max_iter=500,
    solver='lbfgs',
    multi_class='multinomial',
    n_jobs=-1
)
clf.fit(X_train, y_train)
print(f"[INFO] Test accuracy: {clf.score(X_test, y_test):.4f}")

# ===========================================================
# 3️⃣ Train SSNP projection
# ===========================================================
print("[INFO] Training SSNP projection...")
proj = SSNP(
    epochs=20,  # 400 works too, but 20 is usually enough
    patience=5,
    verbose=False,
    opt="adam",
    bottleneck_activation="linear",
)
proj.fit(X_train, y_train)
Z_train = proj.transform(X_train)
np.save(os.path.join(output_dir, "X_SSNP_mnist.npy"), Z_train)
print("[INFO] Projection complete. Shape:", Z_train.shape)

# ===========================================================
# 4️⃣ Create 2D decision boundary grid
# ===========================================================
xmin, xmax = np.percentile(Z_train[:, 0], [1, 99])
ymin, ymax = np.percentile(Z_train[:, 1], [1, 99])

x_intrvls = np.linspace(xmin, xmax, num=grid_size)
y_intrvls = np.linspace(ymin, ymax, num=grid_size)
x_grid = np.linspace(0, grid_size - 1, num=grid_size)
y_grid = np.linspace(0, grid_size - 1, num=grid_size)

pts = cartesian((x_intrvls, y_intrvls))
pts_grid = cartesian((x_grid, y_grid)).astype(int)

img_grid = np.zeros((grid_size, grid_size))
prob_grid = np.zeros((grid_size, grid_size))

print("[INFO] Building decision map...")
pbar = tqdm(total=len(pts))

position = 0
while position < len(pts):
    pts_batch = pts[position:position + batch_size]
    inv_batch = proj.inverse_transform(pts_batch)

    probs = clf.predict_proba(inv_batch)
    alpha = np.amax(probs, axis=1)
    labels = probs.argmax(axis=1)

    pts_grid_batch = pts_grid[position:position + batch_size]
    img_grid[pts_grid_batch[:, 0], pts_grid_batch[:, 1]] = labels
    prob_grid[pts_grid_batch[:, 0], pts_grid_batch[:, 1]] = alpha

    position += batch_size
    pbar.update(batch_size)

pbar.close()

# ===========================================================
# 5️⃣ Smooth and save results
# ===========================================================
print("[INFO] Post-processing decision maps...")
img_grid = gaussian_filter(img_grid, sigma=0.7)
prob_grid = gaussian_filter(prob_grid, sigma=0.7)
prob_grid = np.clip(prob_grid, 0, 1)

# Normalize projected points for overlay
scaler = MinMaxScaler()
scaler.fit(Z_train)
normalized = scaler.transform(Z_train)
normalized = (normalized * (grid_size - 1)).astype(int)

# Save in 3 styles
results_to_png(
    np_matrix=img_grid,
    prob_matrix=prob_grid,
    grid_size=grid_size,
    n_classes=n_classes,
    dataset_name="mnist",
    classifier_name="lr",  # ✅ Correct name now
    real_points=normalized,
    max_value_hsv=0.8,
    suffix="ssnp_w_real"
)

print("[DONE] 🧠 MNIST SDBM (Logistic Regression) created successfully!")
