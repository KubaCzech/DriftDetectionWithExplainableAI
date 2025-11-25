import numpy as np
import itertools
import pandas as pd


def generate_river_data(river_stream, n_samples_before, n_samples_after, n_features=2):
    # TODO: return feature names
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

    X = pd.DataFrame(X_all, columns=[f'X{i+1}' for i in range(n_features)])
    y = pd.Series(y_all, name='Y')
    drift_point = n_samples_before

    # return X, y, drift_point, X.columns.tolist()
    return X, y, drift_point


def generate_data_from_dataset_stream(stream, size_of_block):
    """
    Helper function to generate data from ready river dataset.

    Parameters
    ----------
    stream : river stream object
        The dataset in form of river stream
    size_of_block : int
        Number of samples to extract

    Returns
    -------
    tuple
        (X, y, drift_point, feature_names)
    """
    # TODO add number of features as parameter ???
    # TODO will this be compatible with generate_river_data ??
    X_all = []
    y_all = []

    try:
        for x, y_val in itertools.islice(stream, size_of_block):
            features = [i for i in x]
            X_all.append(features)
            y_all.append(y_val)

    except (StopIteration, KeyError) as e:
        print(f"Warning: Stream generation failed: {e}")
        return np.array([]), np.array([])

    n_features = len(features)
    X = pd.DataFrame(X_all, columns=[f'X{i+1}' for i in range(n_features)])
    y = pd.Series(y_all, name='Y')

    return X, y
