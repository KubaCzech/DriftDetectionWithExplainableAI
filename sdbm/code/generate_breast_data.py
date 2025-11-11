from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Scale to [0,1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X.astype('float32'))

# Save as .npy files (compatible with your script)
np.save("X_breast.npy", X)
np.save("y_breast.npy", y)

print("Saved X_breast-w.npy and y_breast-w.npy")
print("X shape:", X.shape)
print("y shape:", y.shape)
