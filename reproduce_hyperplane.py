from src.datasets.hyperplane_drift import HyperplaneDriftDataset
import pandas as pd
import numpy as np

def analyze_drift():
    dataset = HyperplaneDriftDataset()
    print("Generating data...")
    # Generate chunks
    # Default: 1000 before, 1000 after.
    # Drift width 400.
    X, y = dataset.generate(n_samples_before=1000, n_samples_after=1000, random_seed=42)
    feature_names = X.columns.tolist()
    
    y = pd.Series(y)
    
    # Split
    y_before = y.iloc[:1000]
    y_after = y.iloc[1000:]
    
    print("Class Balance Before:")
    print(y_before.value_counts(normalize=True))
    
    print("\nClass Balance After:")
    print(y_after.value_counts(normalize=True))
    
    # Check if 'After' is stable or not?
    # Split After into two halves
    y_after_1 = y_after.iloc[:500]
    y_after_2 = y_after.iloc[500:]
    
    print("\nClass Balance After (First Half):")
    print(y_after_1.value_counts(normalize=True))
    
    print("\nClass Balance After (Second Half):")
    print(y_after_2.value_counts(normalize=True))

if __name__ == "__main__":
    analyze_drift()
