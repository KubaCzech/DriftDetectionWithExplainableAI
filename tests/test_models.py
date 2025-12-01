import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import MODELS
from src.feature_importance.drift_analysis import compute_data_drift_analysis

def test_models():
    print("Testing Models...")
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    for name, model_class in MODELS.items():
        print(f"Testing {name}...")
        model = model_class()
        model.fit(X, y)
        score = model.score(X, y)
        print(f"{name} score: {score}")
        assert score >= 0.0 and score <= 1.0

def test_drift_analysis():
    print("\nTesting Drift Analysis...")
    X = pd.DataFrame(np.random.rand(200, 5), columns=[f"f{i}" for i in range(5)])
    y = np.random.randint(0, 2, 200)

    for name, model_class in MODELS.items():
        print(f"Testing drift analysis with {name}...")
        result = compute_data_drift_analysis(
            X, y, 
            window_before_start=0, 
            window_after_start=100, 
            window_length=50,
            model_class=model_class,
            model_params={}
        )
        print(f"Result keys: {result.keys()}")
        assert 'model' in result
        assert 'importance_result' in result

if __name__ == "__main__":
    test_models()
    test_drift_analysis()
    print("\nAll tests passed!")
