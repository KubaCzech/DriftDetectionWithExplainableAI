from .base import BaseDataset
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel


class SDBMRBFDriftDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "sdbm_rbf_drift"

    @property
    def display_name(self) -> str:
        return "SDBM RBF Drift"

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({
            "n_windows_before": 1,
            "n_windows_after": 1,
            "n_features": 4,
            "gamma": 30.0,
            "noise": 0.0,
            "gamma": 30.0,
            "noise": 0.0,
            "cluster_std": 0.05,
            "random_seed": 42
        })
        return params
        
    def get_settings_schema(self) -> list[dict]:
        return [
             {
                "name": "n_windows_before",
                "type": "int",
                "label": "Windows Before Drift",
                "default": 1,
                "min_value": 1,
                "step": 1,
                "help": "Number of windows before the specific drift point."
            },
            {
                "name": "n_windows_after",
                "type": "int",
                "label": "Windows After Drift",
                "default": 1,
                "min_value": 1,
                "step": 1,
                "help": "Number of windows after the specific drift point."
            },
             {
                "name": "gamma",
                "type": "float",
                "label": "Gamma",
                "default": 30.0,
                "min_value": 0.1,
                "step": 0.1,
                "help": "Gamma parameter for RBF kernel."
            },
            {
                "name": "noise",
                "type": "float",
                "label": "Label Noise",
                "default": 0.0,
                "min_value": 0.0,
                "max_value": 1.0,
                "step": 0.01,
                "help": "Ratio of labels to flip."
            },
            {
                "name": "cluster_std",
                "type": "float",
                "label": "Cluster Std Dev",
                "default": 0.05,
                "min_value": 0.01,
                "step": 0.01,
                "step": 0.01,
                "help": "Standard deviation of the Gaussian clusters."
            },
            {
                "name": "random_seed",
                "type": "int",
                "label": "Random Seed",
                "default": 42,
                "min_value": 0,
                "step": 1,
                "help": "Seed for reproducible random generation."
            }
        ]

    def _generate_cluster_data(self, centers, total_samples, std):
        """Generate Gaussian clusters with guaranteed samples per cluster."""
        n_centers = len(centers)
        if n_centers == 0:
            return np.zeros((total_samples, 0)) # Should not happen with default logic
        
        base = total_samples // n_centers
        remainder = total_samples % n_centers

        X_list = []
        for i, c in enumerate(centers):
            n_cluster = base + (1 if i < remainder else 0)
            X_list.append(np.random.normal(loc=c, scale=std, size=(n_cluster, len(c))))
        
        if not X_list:
             return np.zeros((total_samples, len(centers[0]) if len(centers)>0 else 0))

        return np.vstack(X_list)

    def _generate_labels(self, X, centers_0, centers_1, gamma):
        """Assign class based on stronger RBF response."""
        K0 = np.max(rbf_kernel(X, centers_0, gamma=gamma), axis=1)
        K1 = np.max(rbf_kernel(X, centers_1, gamma=gamma), axis=1)
        return (K1 > K0).astype(int)

    def _add_label_noise(self, y, ratio):
        n_flip = int(ratio * len(y))
        if n_flip > 0:
            idx = np.random.choice(len(y), n_flip, replace=False)
            y[idx] = 1 - y[idx]
        return y

    def generate(self, n_samples_before=2000, n_samples_after=2000,
                 gamma=30.0, noise=0.0, cluster_std=0.05,
                 random_seed=42, **kwargs) -> tuple[pd.DataFrame, pd.Series]:
        
        np.random.seed(random_seed)

        # Define 4 cluster centers in 4D → 2 per class
        # (Taken from reference implementation)
        centers_class0 = np.array([
            [0.2, 0.8, 0.2, 0.8],
            [0.8, 0.2, 0.8, 0.2]
        ])

        centers_class1 = np.array([
            [0.2, 0.2, 0.8, 0.8],
            [0.8, 0.8, 0.2, 0.2]
        ])

        # Generate PRE drift data
        X_pre = self._generate_cluster_data(
            np.vstack([centers_class0, centers_class1]), 
            n_samples_before, 
            cluster_std
        )
        # Generate POST drift data
        X_post = self._generate_cluster_data(
            np.vstack([centers_class0, centers_class1]), 
            n_samples_after, 
            cluster_std
        )

        # Generate labels (Drift: swapped centers for class definition)
        y_pre = self._generate_labels(X_pre, centers_class0, centers_class1, gamma)
        y_post = self._generate_labels(X_post, centers_class1, centers_class0, gamma) # Swapped

        # Add noise
        y_pre = self._add_label_noise(y_pre, noise)
        y_post = self._add_label_noise(y_post, noise)

        # Shuffle pre-drift
        perm_pre = np.random.permutation(len(y_pre))
        X_pre = X_pre[perm_pre]
        y_pre = y_pre[perm_pre]

        # Shuffle post-drift
        perm_post = np.random.permutation(len(y_post))
        X_post = X_post[perm_post]
        y_post = y_post[perm_post]

        # Concatenate
        X = np.vstack([X_pre, X_post])
        y = np.concatenate([y_pre, y_post])

        # Convert to Pandas
        feature_names = [f"x{i+1}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name="target")

        # Add drift_idx to kwags if needed for downstream (not standard in base, but useful context)
        # But this method returns only X, y. 
        # The calling code usually knows n_samples_before implies the drift index.
        
        return X_df, y_series
