import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
FILE_X = "X_digits.npy"
FILE_Y = "y_digits.npy"
# ---------------------

print("[INFO] Loading Digits dataset from sklearn...")
# 1. Load the dataset
digits = load_digits()
X = digits.data
y = digits.target

print(f"[INFO] Loaded {X.shape[0]} samples with {X.shape[1]} features.")
print(f"[INFO] Found {len(np.unique(y))} unique classes.")

# 2. Preprocess the data (Standard Scaling)
print("[INFO] Preprocessing data with StandardScaler...")
scaler = StandardScaler()
X_processed = scaler.fit_transform(X)

# 3. Save to .npy files
print(f"[INFO] Saving processed data to {FILE_X} and {FILE_Y}...")
np.save(FILE_X, X_processed)
np.save(FILE_Y, y)

# 4. Print confirmation in your requested format
print("\n[INFO] Saved:")
print(f"   → {FILE_X}")
print(f"   → {FILE_Y}")