import numpy as np
import itertools

def generate_river_data(river_stream, n_samples_before, n_samples_after, n_features=2):
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
