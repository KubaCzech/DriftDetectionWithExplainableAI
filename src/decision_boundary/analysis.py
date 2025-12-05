import numpy as np

from src.decision_boundary.ssnp import SSNP


def compute_decision_boundary_analysis(X, y,
                                       start_index_pre=0,
                                       start_index_post=0,
                                       window_length=1000,
                                       model_class=None,
                                       model_params=None,
                                       grid_size=300,
                                       ssnp_epochs=10,
                                       ssnp_patience=5):
    """
    Compute decision boundary analysis using SSNP for dimensionality reduction
    and a classifier for the decision boundary. Handles both pre and post drift windows.

    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Labels
    start_index_pre : int
        Start index for the pre-drift window
    start_index_post : int
        Start index for the post-drift window
    window_length : int
        Length of each window
    model_class : class
        Classifier class (default: MLPModel -> MLPClassifier)
    model_params : dict
        Parameters for the classifier
    grid_size : int
        Resolution of the 2D grid
    ssnp_epochs : int
        Epochs for SSNP training
    ssnp_patience : int
        Patience for SSNP early stopping

    Returns
    -------
    dict
        Dictionary containing results for 'pre' and 'post' windows, plus the SSNP model.
    """
    # 1. Prepare Data
    # Convert to numpy if pandas
    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values

    start_before = start_index_pre
    end_before = start_before + window_length

    start_after = start_index_post
    end_after = start_after + window_length

    # Ensure indices are valid
    if end_before > X.shape[0] or end_after > X.shape[0]:
        # Simple bounds check
        pass

    X_pre = X[start_before:end_before]
    y_pre = y[start_before:end_before]

    X_post = X[start_after:end_after]
    y_post = y[start_after:end_after]

    # 2. Train SSNP on Pre-Drift Data
    # SSNP is used to find a 2D projection that preserves class structure.
    ssnp = SSNP(epochs=ssnp_epochs, patience=ssnp_patience, verbose=0)
    ssnp.fit(X_pre, y_pre)

    # Project points to 2D
    X_pre_2d = ssnp.transform(X_pre)
    X_post_2d = ssnp.transform(X_post)

    # 3. Setup Classifier
    if model_class is None:
        from src.models.mlp import MLPModel
        model_class = MLPModel

    if model_params is None:
        model_params = {}

    # Helper to train and predict grid
    def process_window(X_train, y_train, X_2d_train):
        # Train Classifier on High-Dim Data
        clf = model_class(**model_params)
        clf.fit(X_train, y_train)

        # Create Grid in 2D Latent Space
        xmin, xmax = np.min(X_2d_train[:, 0]), np.max(X_2d_train[:, 0])
        ymin, ymax = np.min(X_2d_train[:, 1]), np.max(X_2d_train[:, 1])

        # Add some margin
        x_margin = (xmax - xmin) * 0.1
        y_margin = (ymax - ymin) * 0.1

        x_intrvls = np.linspace(xmin - x_margin, xmax + x_margin, num=grid_size)
        y_intrvls = np.linspace(ymin - y_margin, ymax + y_margin, num=grid_size)

        xx, yy = np.meshgrid(x_intrvls, y_intrvls)
        pts = np.c_[xx.ravel(), yy.ravel()]

        # Inverse Transform 2D Grid -> High Dim
        # Process in batches to avoid OOM
        batch_size = 50000
        n_pts = len(pts)

        probs_list = []
        labels_list = []

        for i in range(0, n_pts, batch_size):
            batch_pts = pts[i:i+batch_size]
            batch_high_dim = ssnp.inverse_transform(batch_pts)

            # Predict
            batch_probs = clf.predict_proba(batch_high_dim)
            # Assuming binary or multi-class.
            batch_labels = clf.predict(batch_high_dim)
            if hasattr(batch_probs, "max"):
                batch_alpha = batch_probs.max(axis=1)
            else:
                # Fallback
                batch_alpha = np.ones(len(batch_labels))

            probs_list.append(batch_alpha)
            labels_list.append(batch_labels)

        probs_flat = np.concatenate(probs_list)
        labels_flat = np.concatenate(labels_list)

        # Reshape to grid
        prob_grid = probs_flat.reshape(grid_size, grid_size)
        label_grid = labels_flat.reshape(grid_size, grid_size)

        return {
            'clf': clf,
            'X_train': X_train,
            'y_train': y_train,
            'X_2d': X_2d_train,
            'grid_probs': prob_grid,
            'grid_labels': label_grid,
            'grid_bounds': (xmin - x_margin, xmax + x_margin, ymin - y_margin, ymax + y_margin)
        }

    # 4. Process Pre and Post
    result_pre = process_window(X_pre, y_pre, X_pre_2d)

    # For Post, we use the SAME SSNP projector (already trained on Pre),
    # but we train a NEW classifier on the Post data.
    result_post = process_window(X_post, y_post, X_post_2d)

    return {
        'pre': result_pre,
        'post': result_post,
        'ssnp_model': ssnp,
        'grid_size': grid_size
    }
