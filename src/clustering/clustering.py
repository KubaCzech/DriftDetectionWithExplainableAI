import numpy as np
import pandas as pd
import warnings

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from typing import Optional, Union

from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer  # type: ignore

if not hasattr(np, "warnings"):
    np.warnings = warnings


class ClusterBasedDriftDetector:
    """
    Cluster-based data drift detector using X-Means clustering.

    The detector compares an old dataset with a new dataset by performing
    clustering separately for each class label and analyzing structural
    differences between the resulting clusters.

    Drift is detected based on three complementary criteria:

    1. Changes in the number of clusters within a class.
    2. Shifts of cluster centroids between the old and new datasets.
    3. Changes in descriptive statistics computed for each cluster.

    Clusters from the new dataset are matched to clusters from the old dataset
    using the Hungarian algorithm applied to pairwise centroid distances.

    Parameters
    ----------
    data_old : tuple of (pd.DataFrame, np.ndarray)
        Old dataset represented as a tuple (X_old, y_old), where X_old
        is the feature matrix and y_old contains class labels.
    data_new : tuple of (pd.DataFrame, np.ndarray)
        New dataset represented as a tuple (X_new, y_new), where X_new
        is the feature matrix and y_new contains class labels.
    k_init : int
        Minimal number of clusters that must be present in each iteration of X-means.
        By default set to 2.
    k_max : int
        Maximal number of clusters that can be present in each iteration of X-means.
        By default set to 10.
    thr_clusters : int
        Minimal difference in number of clusters between data blocks to acknowledge
        the drift.
    thr_centroid_shift : float
        Measure of how much the cluster must be shifted so it does not appear that the
        cluster was shifted, but instead that the old cluster disappeared and a new one
        appeared.
    thr_desc_stats : float
        Minimal relative change between corresponding clusters' descriptive statistics
        to acknowledge the drift.

    Attributes
    ----------
    X_old : pd.DataFrame
        Feature matrix of the old dataset.
    y_old : np.ndarray
        Class labels of the old dataset.
    X_new : pd.DataFrame
        Feature matrix of the new dataset.
    y_new : np.ndarray
        Class labels of the new dataset.

    k_init : int
        Minimal number of clusters that must be present in each iteration of X-means.
        By default set to 2.
    k_max : int
        Maximal number of clusters that can be present in each iteration of X-means.
        By default set to 10.

    thr_clusters : int
        Minimal difference in number of clusters between data blocks to acknowledge
        the drift.
    thr_centroid_shift : float
        Measure of how much the cluster must be shifted so it does not appear that the
        cluster was shifted, but instead that the old cluster disappeared and a new one
        appeared.
    thr_desc_stats : float
        Minimal relative change between corresponding clusters' descriptive statistics
        to acknowledge the drift.

    cluster_labels_old : np.ndarray or None
        Global cluster labels assigned to samples in the old dataset.
    cluster_labels_new : np.ndarray or None
        Global cluster labels assigned to samples in the new dataset.

    centers_old : dict or None
        Cluster centroids coordinates for the old dataset indexed by global cluster id.
    centers_new : dict or None
        Cluster centroids coordinates for the new dataset indexed by global cluster id.

    cluster_shifts : dict or None
        Euclidean distances between matched old and new cluster centroids.

    number_of_clusters_old : int or None
        Total number of clusters detected in the old dataset.
    number_of_clusters_new : int or None
        Total number of clusters detected in the new dataset.

    drift_flag : bool
        Indicates whether drift was detected.
    drift_details : dict or None
        Detailed drift information reported per class label.

    Notes
    -----
    This detector assumes that class labels are available for both datasets
    and that drift should be analyzed independently within each class.

    The X-Means algorithm is used to automatically determine the number of
    clusters within a predefined range.
    """

    X_old: pd.DataFrame
    y_old: np.ndarray
    X_new: pd.DataFrame
    y_new: np.ndarray

    k_init: int
    k_max: int

    thr_clusters: int
    thr_centroid_shift: float
    thr_desc_stats: float

    centers_old: Union[Optional[dict], np.ndarray]
    centers_new: Union[Optional[dict], np.ndarray]
    cluster_labels_old: Optional[np.ndarray]
    cluster_labels_new: Optional[np.ndarray]

    stats_combined: Optional[pd.DataFrame]
    stats_shifts: Optional[pd.DataFrame]
    cluster_shifts: Optional[dict]

    number_of_clusters_old: int
    number_of_clusters_new: int

    drift_flag: bool
    drift_details: dict

    def __init__(
        self,
        X_before: pd.DataFrame,
        y_before: np.ndarray,
        X_after: pd.DataFrame,
        y_after: np.ndarray,
        k_init: int = 2,
        k_max: int = 10,
        thr_clusters: int = 1,
        thr_centroid_shift: float = 0.3,
        thr_desc_stats: float = 0.2,
        random_state=None,
    ) -> None:  # Handle data conversion if pandas
        if hasattr(X_before, "values"):
            X_before = X_before.values
        if hasattr(y_before, "values"):
            y_before = y_before.values
        if hasattr(X_after, "values"):
            X_after = X_after.values
        if hasattr(y_after, "values"):
            y_after = y_after.values

        self.X_old = X_before
        self.y_old = y_before
        self.X_new = X_after
        self.y_new = y_after

        self.k_init = k_init
        self.k_max = k_max

        self.thr_clusters = thr_clusters
        self.thr_centroid_shift = thr_centroid_shift
        self.thr_desc_stats = thr_desc_stats

        self.centers_old, self.centers_new = (None, None)
        self.cluster_labels_old, self.cluster_labels_new = (None, None)

        self.stats_combined = None
        self.stats_shifts = None
        self.cluster_shifts = None

        self.number_of_clusters_old, self.number_of_clusters_new = (None, None)

        self.drift_flag = False
        self.drift_details = None

        self.random_state = random_state

    def _reshape_clusters(self, clusters: list) -> np.ndarray:
        """
        Reshape clusters from list of lists to a flat array of cluster labels.

        Parameters
        ----------
        clusters : list
            List of lists where each sublist consists of indexes of data forming a cluster, e. g. [[0, 2, 3], [1, 4]].

        Returns
        -------
        reshaped_clusters : np.ndarray
            Flat array of cluster labels assignment, e. g. np.array([0, 1, 0, 0, 1]).
        """
        n_samples = sum(len(c) for c in clusters)
        reshaped_clusters = np.empty(n_samples, dtype=int)
        for idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                reshaped_clusters[sample_index] = idx
        return reshaped_clusters

    def _transform_labels_with_mapping(self, labels: np.ndarray, mapp: dict) -> np.ndarray:
        """
        Transform cluster labels using a given mapping (i.e. when we use X-means
        twice centers before and after may have different class labels; this
        function handles the problem).

        Parameters
        ----------
        labels : np.ndarray
            Array of class labels to be mapped.
        mapp : dict
            Dictionary with mappings.

        Returns
        -------
        transformed_labels : np.ndarray
            Array with labels after transformation with mapping.
        """
        n_samples = len(labels)
        transformed_labels = np.empty(n_samples, dtype=object)
        for idx, cluster_number in enumerate(labels):
            transformed_labels[idx] = mapp[cluster_number]
        return transformed_labels

    def _get_final_labels(self, transformed_labels: list, maps: tuple[dict, list, list]) -> list:
        """
        Assign labels in such a way, that they are not repeated.
        Handles also the case of disappearing and appearing labels.

        Parameters
        ----------
        transformed_labels : list
            List of lists, where each sublist corresponds to cluster labels
            belonging to the subclass.

        maps : tuple(dict, list, list)
            Tuple with mapping of clusters (before/after), list of clusters
            that disappeared and list of clusters that appeared.

        Returns
        -------
        transformed_labels : list
            List of lists, where each sublist corresponds to cluster labels
            after transformation.
        """
        class_counter = 0

        for labels, mapp in zip(transformed_labels, maps):
            m, d, a = mapp
            for idx, l in enumerate(labels):
                if isinstance(l, np.int64) or isinstance(l, int):
                    labels[idx] = labels[idx] + class_counter
            class_counter += len(m) + len(d) - len(a)  # clusters + disappeared clusters - new clusters

        for labels, mapp in zip(transformed_labels, maps):
            m, d, a = mapp
            for idx, l in enumerate(labels):
                if type(l) is str:
                    labels[idx] = class_counter + a.index(int(l[4:]))
            class_counter += len(a)  # new clusters
        return transformed_labels

    def _merge_clusters(self, clusters: list, y: np.ndarray, maps: Optional[tuple[dict, list, list]] = None) -> list:
        """
        Transform list of lists into proper cluster labels (each sublist represents assignments
        produced by X-Means).

        Parameters
        ----------
        clusters : list
            3D list of cluster assignments produced by X-means (in form of list of lists with
            indexes) for each class

        y : np.ndarray
            class labels (≠ cluster labels)

        maps : tuple(dict, list, list) or None
            Tuple consisting of map (before/after), list of disappeared clusters, list of appeared
            clusters. If it is None, then the mapping won't be performed because it is not needed
            (in other words, there is no sense to map box A to box A).

        Returns
        -------
        final_labels : list
            Cluster assignments in form of flat list after all necessary transformations.
        """
        final_labels = np.zeros(len(y))
        transformed_labels = []
        if maps is None:
            maps = [({j: j for j in range(len(clusters[i]))}, [], []) for i in range(len(clusters))]

        for mapp, klass in zip(maps, clusters):
            mapping, _, _ = mapp
            cluster_labels = self._reshape_clusters(klass)
            unique_local_clusters = set(np.unique(cluster_labels))

            new_cluster_ids = unique_local_clusters.difference(set(mapping.keys()))  # appeared clusters
            for local_cluster_id in new_cluster_ids:
                mapping[local_cluster_id] = f"new_{local_cluster_id}"
            transformed_labels.append(self._transform_labels_with_mapping(cluster_labels, mapping))

        transformed_labels = self._get_final_labels(transformed_labels, maps)

        classes = sorted(list(set(y)))
        for klass, labels in zip(classes, transformed_labels):
            final_labels[np.array(y) == klass] = np.array(labels)

        return final_labels

    def _map_new_clusters_to_old(self) -> tuple[dict, list, list]:
        # TODO : implement self.thr_centr_shift
        """
        Map new clusters' assignment to old clusters' assignment based on centroid distances
        using the Hungarian algorithm (it may happen that clusters before and after have the
        same center coordinates but different labels; this function handles the problem by
        creating mapping).

        Returns
        -------
        mapping : dict
            Mapping created by running Hungarian algorithm.
        disappeared : list
            List of IDs of clusters that disappeared (those that were in first data block,
            but somehow were not present in second data block).
        appeared : list
            List of IDs of clusters that appeared (those that were not in first data block,
            but somehow were present in second data block)
        """

        n_old = len(self.centers_old)
        n_new = len(self.centers_new)

        dist = cdist(self.centers_old, self.centers_new)

        # padding, jeśli liczba klastrów się zmieniła
        size = max(n_old, n_new)
        padded = np.full((size, size), 1e9)
        padded[:n_old, :n_new] = dist

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(padded)

        mapping = {}  # new_cluster -> old_cluster
        disappeared = []  # cluster existed before, no match after
        appeared = []  # cluster new

        for r, c in zip(row_ind, col_ind):
            if r < n_old and c < n_new:
                mapping[c] = r
            elif r < n_old and c >= n_new:
                disappeared.append(r)
            elif r >= n_old and c < n_new:
                appeared.append(c)
        return mapping, disappeared, appeared

    def _xmeans(self, X: pd.DataFrame) -> tuple[np.ndarray, list]:
        """
        Perform X-Means clustering on the given data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature values (data to be clustered).

        Returns
        -------
        centers : np.ndarray
            Array of arrays that contains coordinates of each cluster.
        clusters : list
            List of lists where each sublist contains indexes of data points belonging to
            cluster at that position.

        Notes
        -----
        kmeans_plusplus_initializer from pyclustering can take random_state but it's not standard.
        However, looking at the library, it usually uses random module. But if the user says
        "ensure deterministic", passing random_state implies we should manage seeds. The standard
        pyclustering kmeans_plusplus_initializer might access global random state. But let's check
        if it accepts a random_state argument in initializer. Assuming standard signature:
        kmeans_plusplus_initializer(data, amount_centers, candidate_centers=None, random_state=None)
        If not, we rely on global seed which we can set here if self.random_state is set.

        """
        if self.random_state is not None:
            import random

            random.seed(self.random_state)
            np.random.seed(self.random_state)

        # Try to pass random_state if supported
        try:
            init_centers = kmeans_plusplus_initializer(X, self.k_init, random_state=self.random_state).initialize()
        except TypeError:
            # Fallback if random_state is not supported by this version of pyclustering
            init_centers = kmeans_plusplus_initializer(X, self.k_init).initialize()

        # xmeans also uses random splitting sometimes in various implementations, typically relies on global random
        # setting global seeds above handles it if pyclustering relies on random/np.random

        xm = xmeans(X, init_centers, kmax=self.k_max, ccore=False)
        xm.process()
        centers = np.array(xm.get_centers())
        clusters = xm.get_clusters()
        return centers, clusters

    def _detect(self, X_old: pd.DataFrame, X_new: pd.DataFrame) -> tuple[list, list, tuple[dict, list, list]]:
        """
        Detect drift between the same class for two data blocks using X-Means clustering.

        Parameters
        ----------
        X_old : pd.DataFrame
            Feature values for first (old) data block.
        X_new : pd.DataFrame
            Feature values for second (new) data block.

        Returns
        -------
        clusters_old : list
            List of lists where each sublist contains indexes of data points belonging to
            cluster at that position (for first data block).
        clusters_new : list
            List of lists where each sublist contains indexes of data points belonging to
            cluster at that position (for second data block).
        mapp : tuple(dict, list, list)
            Tuple that contains mapping (between clusters before/after), list of clusters
            that disappeared and list of clusters that appeared.
        """

        self.centers_old, clusters_old = self._xmeans(X_old)
        self.centers_new, clusters_new = self._xmeans(X_new)

        mapp = self._map_new_clusters_to_old()

        return clusters_old, clusters_new, mapp

    # TODO
    def detect(self) -> tuple[bool, dict]:
        """
        Detect drift between two data blocks using X-Means clustering for each class separately.

        Returns
        -------
        drift_flag : bool
            Flag that tells whether the drift occurred or not.
        details : dict
            Dictionary that explains why drift happened (each category marked as True).
        """

        classes = set(self.y_old).union(set(self.y_new))

        clusters_all = {'old': [], 'new': []}
        details = {cl: {} for cl in classes}
        maps = []

        # 1. number of clusters
        for cl in classes:
            X_old_cl = self.X_old[np.array(self.y_old) == cl]
            X_new_cl = self.X_new[np.array(self.y_new) == cl]
            clusters_old, clusters_new, mapp = self._detect(X_old_cl, X_new_cl)

            clusters_all['old'].append(clusters_old)
            clusters_all['new'].append(clusters_new)
            maps.append(mapp)

        self.cluster_labels_old = self._merge_clusters(clusters_all['old'], self.y_old)
        self.cluster_labels_new = self._merge_clusters(clusters_all['new'], self.y_new, maps=maps)

        self.number_of_clusters_old = len(set(self.cluster_labels_old))
        self.number_of_clusters_new = len(set(self.cluster_labels_new))

        # 2. centroid shifts
        self.calculate_centroid_shifts()
        self.stats_combined = self.compute_desc_stats_for_clusters()

        # 3. desc stats change
        self.stats_shifts = self.compare_desc_stats_for_clusters(self.stats_combined)
        details_stats_shifts = self.assess_statistics_shifts(self.stats_shifts)

        for cl in classes:
            mask_old = self.y_old == cl
            cl_old = set(self.cluster_labels_old[mask_old])

            mask_new = self.y_new == cl
            cl_new = set(self.cluster_labels_new[mask_new])

            details[cl]['nr_of_clusters'] = (
                abs(len(set(self.cluster_labels_new[mask_new])) - len(set(self.cluster_labels_old[mask_old])))
                >= self.thr_clusters
            )
            details[cl]['centroid_shift'] = any(
                v > self.thr_centroid_shift
                for v in {i: self.cluster_shifts[i] for i in cl_old.intersection(cl_new)}.values()
            )

            idx = set(self.cluster_labels_old[self.y_old == cl]).union(set(self.cluster_labels_new[self.y_new == cl]))
            details[cl]['desc_stats_changes'] = {k: details_stats_shifts[k] for k in idx if k in details_stats_shifts}

        drift_flag = any([details[cl]['nr_of_clusters'] or details[cl]['centroid_shift'] for cl in classes]) or any(
            stat is True
            for klass in details.values()
            for cluster in klass['desc_stats_changes'].values()
            for feature in cluster.values()
            for stat in feature.values()
        )

        self.drift_flag = drift_flag
        self.drift_details = details
        return drift_flag, details

    def calculate_centroid_shifts(self) -> None:
        """
        Calculate Euclidean centroid shifts between corresponding clusters across data blocks.
        """

        def calculate_cluster_centers(X, labels):
            unique_labels = np.unique(labels)
            centers = {}
            for label in unique_labels:
                cluster_points = X[labels == label]
                center = np.mean(cluster_points, axis=0)
                centers[label] = center
            return centers

        self.centers_old = calculate_cluster_centers(self.X_old, self.cluster_labels_old)
        self.centers_old.update({i: None for i in set(self.cluster_labels_new) - set(self.cluster_labels_old)})

        self.centers_new = calculate_cluster_centers(self.X_new, self.cluster_labels_new)
        self.centers_new.update({i: None for i in set(self.cluster_labels_old) - set(self.cluster_labels_new)})

        shifts = {}
        for i in self.centers_old.keys():
            center_old = self.centers_old[i]
            center_new = self.centers_new[i]

            if center_old is None:
                shift = 'appeared'
            elif center_new is None:
                shift = 'disappeared'
            else:
                shift = np.linalg.norm(center_new - center_old)
                shifts[i] = shift
        self.cluster_shifts = shifts

    def compute_desc_stats_for_clusters(self) -> pd.DataFrame:
        """
        Compute descriptive statistics (min, median, mean, max, std) for each cluster separately between
        data blocks. As a result, a pandas DataFrame with a 3-level MultiIndex on columns will be created
        and saved to self.stats_combined.

        Returns
        -------
        stats_combined : pd.DataFrame
            Pandas DataFrame with descriptive statistics.
        """

        def compute_cluster_stats(X, labels):
            df = pd.DataFrame(X)
            df['cluster'] = labels
            features = df.columns[:-1]
            clusters = sorted(df['cluster'].unique())

            records = []
            for cluster in clusters:
                cluster_df = df[df['cluster'] == cluster]
                stats_dict = {}
                for f in features:
                    stats_dict[(f, 'min')] = cluster_df[f].min()
                    stats_dict[(f, 'median')] = cluster_df[f].median()
                    stats_dict[(f, 'mean')] = cluster_df[f].mean()
                    stats_dict[(f, 'std')] = cluster_df[f].std()
                    stats_dict[(f, 'max')] = cluster_df[f].max()
                stats_dict[('cluster', 'id')] = cluster
                records.append(stats_dict)

            stats_df = pd.DataFrame(records)
            stats_df.set_index([('cluster', 'id')], inplace=True)
            stats_df.sort_index(inplace=True)
            return stats_df

        stats_old = compute_cluster_stats(self.X_old, self.cluster_labels_old)
        stats_new = compute_cluster_stats(self.X_new, self.cluster_labels_new)

        stats_old.columns = pd.MultiIndex.from_tuples([('old', f, stat) for f, stat in stats_old.columns])
        stats_new.columns = pd.MultiIndex.from_tuples([('new', f, stat) for f, stat in stats_new.columns])

        stats_combined = pd.concat([stats_old, stats_new], axis=1)

        stats_combined = stats_combined.sort_index(axis=1, level=[0, 1, 2])

        return stats_combined

    def compare_desc_stats_for_clusters(self, stats_combined: pd.DataFrame) -> dict:
        """
        Compare descriptive statistics and calculate the shift between them for corresponding clusters
        between data blocks.

        Parameters
        ----------
        stats_combined : pd.DataFrame
            Pandas DataFrame with a 3-level MultiIndex on columns with descriptive statistics
            of each feature between corresponding clusters and data blocks.

        Returns
        -------
        details : dict
            Relative changes between descriptive statistics across data blocks.
        """
        eps = 1e-10
        details = {}

        for cluster in stats_combined.index:
            row_df = stats_combined.loc[[cluster]]

            details[cluster] = {}
            features = row_df.columns.levels[1]

            for feature in features:
                details[cluster][feature] = {}

                stats_available = row_df.columns.levels[2]

                for stat in stats_available:
                    col_old = ('old', feature, stat)
                    col_new = ('new', feature, stat)

                    if col_old not in row_df.columns or col_new not in row_df.columns:
                        details[cluster][feature][stat] = 'N/A'
                        continue

                    old_value = row_df[col_old].iloc[0]
                    new_value = row_df[col_new].iloc[0]

                    if pd.isna(old_value) or pd.isna(new_value):
                        details[cluster][feature][stat] = 'N/A'
                        continue

                    denom = abs(old_value)
                    if denom < eps:
                        denom = 1.0

                    change = (new_value - old_value) / denom
                    details[cluster][feature][stat] = change
        return details

    def assess_statistics_shifts(self, stats_shifts: dict) -> dict:
        """
        Evaluate whether statistical shifts exceed a predefined threshold.

        Parameters
        ----------
        stats_shifts : dict
            Nested dictionary of statistical shift values in the form
            ``{class: {feature: {statistic: value | 'N/A'}}}``.

        Returns
        -------
        dict
            Nested dictionary with the same structure as `stats_shifts`, where each
            statistic is mapped to:
            - ``True`` if ``abs(value) > self.thr_desc_stats``,
            - ``False`` if not,
            - ``None`` if the input value was ``'N/A'``.
        """
        return {
            cl: {
                f: {s: abs(v) > self.thr_desc_stats if v != 'N/A' else True for s, v in stats.items()}
                for f, stats in features.items()
            }
            for cl, features in stats_shifts.items()
        }
