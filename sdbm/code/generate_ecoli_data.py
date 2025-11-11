import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

# --- Configuration ---
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
OUTPUT_DIR = "../data/ecoli"
# ----------------------

print("[INFO] Loading E. coli dataset from UCI repository...")

# Load dataset directly from the URL
columns = [
    "sequence_name", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class"
]
df = pd.read_csv(DATA_URL, delim_whitespace=True, names=columns)

# Drop the sequence name column
df = df.drop(columns=["sequence_name"])

# Split features and labels
X = df.drop(columns=["class"]).values.astype("float32")
y = LabelEncoder().fit_transform(df["class"].values)

print(f"[INFO] Loaded {X.shape[0]} samples with {X.shape[1]} features.")
print(f"[INFO] Found {len(np.unique(y))} unique classes.")

# Scale features to [0, 1] range (required for SSNP pipeline)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


np.save("X_ecoli.npy", X)
np.save("y_ecoli.npy", y)
