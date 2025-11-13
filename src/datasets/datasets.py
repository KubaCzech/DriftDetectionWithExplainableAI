import numpy as np
import itertools
from river.datasets import synth


class DatasetName:
    """Enum-like class for selecting datasets."""
    CUSTOM_NORMAL = "custom_normal"
    CUSTOM_3D_DRIFT = "custom_3d_drift"
    SEA_DRIFT = "sea_drift"
    HYPERPLANE_DRIFT = "hyperplane_drift"

    @classmethod
    def all_available(cls):
        return [cls.CUSTOM_NORMAL, cls.CUSTOM_3D_DRIFT,
                cls.SEA_DRIFT, cls.HYPERPLANE_DRIFT]


def generate_custom_normal_data(n_samples_before=1000, n_samples_after=1000,
                                random_seed=42):
    """
    Generate synthetic data stream with concept drift (Original Function, 2D).

    Returns
    -------
    tuple
        (X, y, drift_point, feature_names)
    """
    np.random.seed(random_seed)

    # Before drift
    X1_before = np.random.normal(0, 1, n_samples_before)
    X2_before = np.random.normal(0, 1, n_samples_before)
    scores_before = (2 * X1_before + 0.5 * X2_before +
                     np.random.normal(0, 0.5, n_samples_before))
    threshold_before = np.percentile(scores_before, 30)
    y_before = (scores_before > threshold_before).astype(int)

    # After drift - X1 distribution changes DRAMATICALLY, X2 changes slightly
    X1_after = np.random.normal(2, 1.5, n_samples_after)
    X2_after = np.random.normal(0.2, 1.05, n_samples_after)
    scores_after = (-1.5 * X1_after + 0.6 * X2_after +
                    np.random.normal(0, 0.5, n_samples_after))
    threshold_after = np.percentile(scores_after, 70)
    y_after = (scores_after > threshold_after).astype(int)

    # Combine data
    X1 = np.concatenate([X1_before, X1_after])
    X2 = np.concatenate([X2_before, X2_after])
    X = np.column_stack([X1, X2])
    y = np.concatenate([y_before, y_after])
    drift_point = n_samples_before
    feature_names = ['X1', 'X2']

    return X, y, drift_point, feature_names


def generate_custom_3d_drift_data(n_samples_before=1000, n_samples_after=1000,
                                  random_seed=42):
    """
    Generate synthetic 3D data stream with concept drift.

    Before drift:
        - X1, X2, X3 ~ N(0, 1)
        - Decision boundary: X1 + X2 + X3 > threshold

    After drift:
        - X1, X2 ~ N(0, 1)
        - X3 ~ N(3, 1.5)  (Data Drift on X3)
        - Decision boundary: X1 + X2 - 2*X3 > threshold (Concept Drift)

    Returns
    -------
    tuple
        (X, y, drift_point, feature_names)
    """
    np.random.seed(random_seed)

    # Before drift
    X_before = np.random.normal(0, 1, (n_samples_before, 3))
    scores_before = (X_before[:, 0] + X_before[:, 1] + X_before[:, 2] +
                     np.random.normal(0, 0.5, n_samples_before))
    threshold_before = np.percentile(scores_before, 40)
    y_before = (scores_before > threshold_before).astype(int)

    # After drift
    X1_after = np.random.normal(0, 1, n_samples_after)
    X2_after = np.random.normal(0, 1, n_samples_after)
    X3_after = np.random.normal(3, 1.5, n_samples_after)  # Data drift on X3
    X_after = np.column_stack([X1_after, X2_after, X3_after])

    # Concept drift: X3 becomes more important and inverts relationship
    scores_after = (X_after[:, 0] + X_after[:, 1] - 2 * X_after[:, 2] +
                    np.random.normal(0, 0.5, n_samples_after))
    threshold_after = np.percentile(scores_after, 60)
    y_after = (scores_after > threshold_after).astype(int)

    # Combine data
    X = np.concatenate([X_before, X_after])
    y = np.concatenate([y_before, y_after])
    drift_point = n_samples_before
    feature_names = ['X1', 'X2', 'X3']

    return X, y, drift_point, feature_names


def _generate_river_data(river_stream, n_samples_before,
                         n_samples_after, n_features=2):
    """
    Helper function to generate data from a river stream.

    Parameters
    ----------
    river_stream : river stream object
        The stream generator
    n_samples_before : int
        Samples before drift point
    n_samples_after : int
        Samples after drift point
    n_features : int, default=2
        Number of features to extract

    Returns
    -------
    tuple
        (X, y, drift_point, feature_names)
    """
    X_all = []
    y_all = []
    total_samples = n_samples_before + n_samples_after

    try:
        for x, y_val in itertools.islice(river_stream, total_samples):
            # Extract the specified number of features
            features = [x[i] for i in range(n_features)]
            X_all.append(features)
            y_all.append(y_val)

    except (StopIteration, KeyError) as e:
        print(f"Warning: Stream generation failed: {e}")
        return np.array([]), np.array([]), 0, []

    X = np.array(X_all)
    y = np.array(y_all)
    drift_point = n_samples_before

    # Generate feature names
    feature_names = [f'X{i+1}' for i in range(n_features)]

    return X, y, drift_point, feature_names


def generate_sea_drift_data(n_samples_before=1000, n_samples_after=1000,
                            random_seed=42):
    """
    Generate synthetic data stream using River's SEA generator.
    Drifts from variant 0 to variant 3 at the drift point.
    The SEA generator has 3 features, but we only use the first 2.

    Returns
    -------
    tuple
        (X, y, drift_point, feature_names)
    """
    stream_SEA = synth.ConceptDriftStream(
        stream=synth.SEA(seed=random_seed, variant=0),
        drift_stream=synth.SEA(seed=random_seed, variant=3),
        position=n_samples_before,
        width=400,  # Gradual drift
        seed=random_seed
    )
    return _generate_river_data(stream_SEA, n_samples_before, n_samples_after)


def generate_hyperplane_data(n_samples_before=1000, n_samples_after=1000,
                             n_features=2, n_drift_features=2, random_seed=42):
    """
    Generate synthetic data stream using River's Hyperplane generator.
    Drifts from a stable hyperplane to one with magnitude change.

    Parameters
    ----------
    n_samples_before : int, default=1000
        Number of samples before drift point
    n_samples_after : int, default=1000
        Number of samples after drift point
    n_features : int, default=2
        The number of features to generate (must be >= 2)
    n_drift_features : int, default=2
        The number of features with drift (must be >= 2 and <= n_features)
    random_seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    tuple
        (X, y, drift_point, feature_names)
        - X: numpy array of shape
          (n_samples_before + n_samples_after, n_features)
        - y: numpy array of binary labels
        - drift_point: int, index where drift begins
        - feature_names: list of feature names
    """
    # Validation
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    if n_drift_features < 2:
        raise ValueError("n_drift_features must be at least 2")
    if n_drift_features > n_features:
        raise ValueError("n_drift_features cannot exceed n_features")

    stream_HP = synth.ConceptDriftStream(
        stream=synth.Hyperplane(
            n_features=n_features,
            n_drift_features=n_drift_features,
            seed=random_seed,
            noise_percentage=0.05
        ),
        drift_stream=synth.Hyperplane(
            n_features=n_features,
            n_drift_features=n_drift_features,
            seed=random_seed,
            mag_change=0.2,
            noise_percentage=0.1
        ),
        position=n_samples_before,
        width=400,  # Gradual drift
        seed=random_seed
    )
    return _generate_river_data(stream_HP, n_samples_before,
                                n_samples_after, n_features)
