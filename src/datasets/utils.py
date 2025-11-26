import numpy as np
import itertools
import pandas as pd


def generate_river_data(river_stream, total_samples, n_features=2):
    # TODO: return feature names
    """
    Helper function to generate data from a river stream.

    Parameters
    ----------
    river_stream : river stream object
        The stream generator
    total_samples : int
        Total number of samples to generate
    n_features : int, default=2
        Number of features to extract

    Returns
    -------
    tuple
        (X, y)
    """
    X_all = []
    y_all = []

    try:
        for x, y_val in itertools.islice(river_stream, total_samples):
            # Extract the specified number of features
            features = [x[i] for i in range(n_features)]
            X_all.append(features)
            y_all.append(y_val)

    except (StopIteration, KeyError) as e:
        print(f"Warning: Stream generation failed: {e}")
        return np.array([]), np.array([])

    X = pd.DataFrame(X_all, columns=[f'X{i+1}' for i in range(n_features)])
    y = pd.Series(y_all, name='Y')
    
    return X, y
