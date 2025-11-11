import os
import numpy as np
import matplotlib.cm as cm
from tqdm import tqdm
from PIL import Image
from skimage.color import rgb2hsv, hsv2rgb
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import cartesian
import seaborn as sns
from ssnp import SSNP  # ensure ssnp.py is in same folder

# ===========================================================
# Output setup
# ===========================================================
output_dir = "./sdbm_titanic_results_rf"
os.makedirs(output_dir, exist_ok=True)
grid_size = 300
batch_size = 5000

# ===========================================================
# Utility: save visualization
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
# 1️⃣ Load Titanic dataset
# ===========================================================
print("[INFO] Loading Titanic dataset...")
titanic = sns.load_dataset("titanic").dropna(subset=["age", "sex", "pclass", "fare", "survived"])

# Select relevant features
features = ["pclass", "sex", "age", "fare"]
target = "survived"

X = titanic[features]
y = titanic[target]

# Encode categorical data and scale numeric
categorical_features = ["sex"]
numeric_features = ["pclass", "age", "fare"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first"), categorical_features)
])

# Preprocess and split
X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, train_size=0.8, stratify=y, random_state=42
)

n_classes = len(np.unique(y_train))
print(f"[INFO] Using {len(X_train)} training samples, {len(X_test)} testing samples")

# ===========================================================
# 2️⃣ Train Random Forest classifier
# ===========================================================
print("[INFO] Training Random Forest classifier...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)
print(f"[INFO] Test accuracy: {clf.score(X_test, y_test):.4f}")

# ===========================================================
# 3️⃣ Train SSNP projection
# ===========================================================
print("[INFO] Training SSNP projection...")
proj = SSNP(
    epochs=20,
    patience=5,
    verbose=False,
    opt="adam",
    bottleneck_activation="linear",
)
proj.fit(X_train, y_train)
Z_train = proj.transform(X_train)
np.save(os.path.join(output_dir, "X_SSNP_titanic_rf.npy"), Z_train)
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

# Save results
results_to_png(
    np_matrix=img_grid,
    prob_matrix=prob_grid,
    grid_size=grid_size,
    n_classes=n_classes,
    dataset_name="titanic",
    classifier_name="rf",  # 👈 changed
    real_points=normalized,
    max_value_hsv=0.8,
    suffix="ssnp_w_real"
)

print("[DONE] 🚢 Titanic SDBM (Random Forest) created successfully!")
