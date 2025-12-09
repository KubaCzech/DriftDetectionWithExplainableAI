import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from src.decision_boundary.ssnp import SSNP


class DecisionBoundaryDriftAnalyzer:
    def __init__(self, X_before, y_before, X_after, y_after, random_state=42):
        self.random_state = random_state
        # 0. Enforce Determinism
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        # 1. Prepare Data
        # Convert to numpy if pandas
        if hasattr(X_before, "values"):
            X_before = X_before.values
        if hasattr(y_before, "values"):
            y_before = y_before.values
        if hasattr(X_after, "values"):
            X_after = X_after.values
        if hasattr(y_after, "values"):
            y_after = y_after.values

        self.X_before = X_before
        self.y_before = y_before
        self.X_after = X_after
        self.y_after = y_after

    def analyze(self, model_class=None, model_params=None, grid_size=300, ssnp_epochs=10, ssnp_patience=5):
        """
        Compute decision boundary analysis using SSNP for dimensionality reduction
        and a classifier for the decision boundary. Handles both pre and post drift windows.

        Parameters
        ----------
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

        # Normalize Data (Fit on Pre, Transform both)
        scaler = MinMaxScaler()
        X_before_scaled = scaler.fit_transform(self.X_before)
        X_after_scaled = scaler.transform(self.X_after)

        # 2. Train SSNP on Pre-Drift Data
        # SSNP is used to find a 2D projection that preserves class structure.
        ssnp = SSNP(epochs=ssnp_epochs, patience=ssnp_patience, verbose=0)
        ssnp.fit(X_before_scaled, self.y_before)

        # Project points to 2D
        X_before_2d = ssnp.transform(X_before_scaled)
        X_after_2d = ssnp.transform(X_after_scaled)

        # 3. Setup Classifier
        if model_class is None:
            from src.models.mlp import MLPModel
            model_class = MLPModel

        if model_params is None:
            model_params = {}

        # Force random_state for classifier determinism
        model_params['random_state'] = self.random_state

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
        result_pre = process_window(X_before_scaled, self.y_before, X_before_2d)

        # For Post, we use the SAME SSNP projector (already trained on Pre),
        # but we train a NEW classifier on the Post data.
        result_post = process_window(X_after_scaled, self.y_after, X_after_2d)

        return {
            'pre': result_pre,
            'post': result_post,
            'ssnp_model': ssnp,
            'grid_size': grid_size
        }
