import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import MODELS  # noqa: E402
from src.feature_importance.analysis import FeatureImportanceDriftAnalyzer  # noqa: E402


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
        # Use new generic analyzer
        analyzer = FeatureImportanceDriftAnalyzer(
            X_before=X.iloc[:100], y_before=y[:100],
            X_after=X.iloc[100:], y_after=y[100:]
        )
        result = analyzer.compute_data_drift(model_class=model_class)

        print(f"Result keys: {result.keys()}")
        assert 'model' in result
        assert 'importance_result' in result


if __name__ == "__main__":
    test_models()
    test_drift_analysis()
    print("\nAll tests passed!")
