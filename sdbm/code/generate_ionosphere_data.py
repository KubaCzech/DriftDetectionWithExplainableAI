import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

# --- Configuration ---
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
OUTPUT_DIR = "../data/ionosphere"
# ----------------------

print("[INFO] Loading Ionosphere dataset from UCI repository...")

# The dataset has 34 continuous features and a binary label ('g' or 'b')
columns = [f"feature_{i+1}" for i in range(34)] + ["class"]
df = pd.read_csv(DATA_URL, header=None, names=columns)

# Split features and labels
X = df.drop(columns=["class"]).values.astype("float32")
y = LabelEncoder().fit_transform(df["class"].values)  # 'g' → 1, 'b' → 0

print(f"[INFO] Loaded {X.shape[0]} samples with {X.shape[1]} features.")
print(f"[INFO] Found {len(np.unique(y))} unique classes.")

# Scale features to [0, 1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

np.save("X_ionosphere.npy", X)
np.save("y_ionosphere.npy", y)
