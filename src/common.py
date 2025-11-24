import pandas as pd


def transform_dict_to_list(stream, n_samples=1000):
    X, y = [], []
    for x, label in stream.take(n_samples):
        X.append(list(x.values()))
        y.append(label)
    return X, y


def transform_lists_to_df(X, y):
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)
    data = pd.concat([X_df, y_series], axis=1)
    data.columns = [f'feature{i}' for i in range(X_df.shape[1])] + ['label']
    return data
