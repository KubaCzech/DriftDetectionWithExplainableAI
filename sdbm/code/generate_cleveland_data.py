import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

# --- Configuration ---
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
OUTPUT_DIR = "../data/cleveland"
# ----------------------

print("[INFO] Loading Cleveland Heart Disease dataset from UCI repository...")

# Define column names based on UCI documentation
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

# Load dataset (missing values are '?')
df = pd.read_csv(DATA_URL, header=None, names=columns, na_values="?")

# Drop rows with missing values (6 rows contain '?')
df = df.dropna()

# Split features and target
X = df.drop(columns=["num"]).astype("float32").values
y = df["num"].astype(int).values

# Convert target: 0 = no disease, 1–4 = heart disease
y = (y > 0).astype(int)  # binary classification

print(f"[INFO] Loaded {X.shape[0]} samples with {X.shape[1]} features.")
print(f"[INFO] Found {len(np.unique(y))} unique classes.")

# Scale features to [0, 1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
np.save("X_cleveland.npy", X)
np.save("y_cleveland.npy", y)