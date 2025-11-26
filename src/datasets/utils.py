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


def generate_river_data_with_selection(river_stream, total_samples, feature_names):
    """
    Helper function to generate data from a river stream with selected features.

    Parameters
    ----------
    river_stream : river stream object
        The stream generator
    total_samples : int
        Total number of samples to generate
    feature_names : list of str
        List of feature names to extract

    Returns
    -------
    tuple
        (X, y)
    """
    X_all = []
    y_all = []

    try:
        for x, y_val in itertools.islice(river_stream, total_samples):
            # Extract the specified features by name
            features = [x[name] for name in feature_names if name in x]
            # Check if all features were found
            if len(features) == len(feature_names):
                X_all.append(features)
                y_all.append(y_val)
            else:
                # Handle missing features if necessary, for now skip or warn
                # Printing warning might be too verbose for large streams
                pass

    except (StopIteration, KeyError) as e:
        print(f"Warning: Stream generation failed: {e}")
        return pd.DataFrame(), pd.Series()

    X = pd.DataFrame(X_all, columns=feature_names)
    y = pd.Series(y_all, name='Y')
    
    return X, y
