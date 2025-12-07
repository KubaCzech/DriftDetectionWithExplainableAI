import numpy as np
import pandas as pd
import warnings

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer  # type: ignore

if not hasattr(np, "warnings"):
    np.warnings = warnings


class ClusterBasedDriftDetector:
    def __init__(self, data_old, data_new, random_state=None):
        self.X_old, self.y_old = data_old
        self.X_new, self.y_new = data_new

        self.centers_old, self.centers_new = (None, None)
        self.cluster_labels_old, self.cluster_labels_new = (None, None)

        self.stats_combined = None
        self.stats_shifts = None
        self.cluster_shifts = None

        self.number_of_clusters_old, self.number_of_clusters_new = (None, None)

        self.drift_flag = False
        self.drift_details = None
        
        self.random_state = random_state

    # DONE
    def _reshape_clusters(self, clusters):
        """
        Function to reshape clusters from list of lists to a flat array of cluster labels.
        """
        n_samples = sum(len(c) for c in clusters)
        reshaped_clusters = np.empty(n_samples, dtype=int)
        for idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                reshaped_clusters[sample_index] = idx
        return reshaped_clusters

    # DONE
    def _transform_labels_with_mapping(self, labels, mapp):
        """
        Function to transform cluster labels using a given mapping.
        """
        n_samples = len(labels)
        transformed_labels = np.empty(n_samples, dtype=object)
        for idx, cluster_number in enumerate(labels):
            transformed_labels[idx] = mapp[cluster_number]
        return transformed_labels

    # DONE
    def _get_final_labels(self, transformed_labels, maps):
        """Function to get final labels after merging clusters."""
        class_counter = 0

        for labels, mapp in zip(transformed_labels, maps):
            m, d, a = mapp
            for idx, l in enumerate(labels):
                if isinstance(l, np.int64) or isinstance(l, int):
                    labels[idx] = labels[idx] + class_counter
            class_counter += len(m) + len(d)  # old clusters + disappeared clusters

        for labels, mapp in zip(transformed_labels, maps):
            m, d, a = mapp
            for idx, l in enumerate(labels):
                if type(l) is str:
                    labels[idx] = class_counter + a.index(int(l[4:]))
            class_counter += len(a)  # new clusters
        return transformed_labels

    # DONE
    def _merge_clusters(self, clusters, y, maps=None):
        final_labels = np.zeros(len(y))
        transformed_labels = []
        if maps is None:
            maps = [({j: j for j in range(len(clusters[i]))}, [], []) for i in range(len(clusters))]

        for mapp, klass in zip(maps, clusters):
            mapping, a, d = mapp
            cluster_labels = self._reshape_clusters(klass)
            unique_local_clusters = set(np.unique(cluster_labels))

            new_cluster_ids = unique_local_clusters.difference(set(mapping.keys()))  # appeared clusters
            for local_cluster_id in new_cluster_ids:
                mapping[local_cluster_id] = f"new_{local_cluster_id}"
            transformed_labels.append(self._transform_labels_with_mapping(cluster_labels, mapping))

        transformed_labels = self._get_final_labels(transformed_labels, maps)

        classes = set(y)
        for klass, labels in zip(classes, transformed_labels):
            final_labels[np.array(y) == klass] = np.array(labels)

        return final_labels

    # DONE
    def _map_new_clusters_to_old(self):
        """
        Function to map new clusters to old clusters based on centroid distances using the Hungarian algorithm.
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
                d = dist[r, c]
                mapping[c] = r
            elif r < n_old and c >= n_new:
                disappeared.append(r)
            elif r >= n_old and c < n_new:
                appeared.append(c)
        return mapping, disappeared, appeared

    # DONE
    def _xmeans(self, X, k_init=2, k_max=10):
        """
        Function to perform X-Means clustering on the given data
        """

        # kmeans_plusplus_initializer from pyclustering can take random_state but it's not standard
        # However, looking at the library, it usually uses random module.
        # But if the user says "ensure deterministic", passing random_state implies we should manage seeds.
        # The standard pyclustering kmeans_plusplus_initializer might access global random state.
        # But let's check if it accepts a random_state argument in initializer.
        # Assuming standard signature: kmeans_plusplus_initializer(data, amount_centers, candidate_centers=None, random_state=None)
        # If not, we rely on global seed which we can set here if self.random_state is set.
        
        if self.random_state is not None:
             import random
             random.seed(self.random_state)
             np.random.seed(self.random_state)

        # Try to pass random_state if supported
        try:
             init_centers = kmeans_plusplus_initializer(X, k_init, random_state=self.random_state).initialize()
        except TypeError:
             # Fallback if random_state is not supported by this version of pyclustering
             init_centers = kmeans_plusplus_initializer(X, k_init).initialize()
        
        # xmeans also uses random splitting sometimes in various implementations, typically relies on global random
        # setting global seeds above handles it if pyclustering relies on random/np.random
        
        xm = xmeans(X, init_centers, kmax=k_max, ccore=False)
        xm.process()
        centers = np.array(xm.get_centers())
        clusters = xm.get_clusters()
        return centers, clusters

    # DONE
    def _detect(self, X_old, X_new, k_init, k_max):
        """
        Detect drift between the same class for two datasets using X-Means clustering
        """

        self.centers_old, clusters_old = self._xmeans(X_old, k_init, k_max)
        self.centers_new, clusters_new = self._xmeans(X_new, k_init, k_max)

        mapp = self._map_new_clusters_to_old()

        return clusters_old, clusters_new, mapp

    # DONE
    def detect(self, k_init=2, k_max=10, thr_clusters=1, thr_centroid_shift=0.3, thr_desc_stats=0.2):
        """
        Detect drift between two datasets using X-Means clustering for each class separately
        """

        classes = set(self.y_old).union(set(self.y_new))

        clusters_all = {'old': [], 'new': []}

        details = {cl: {} for cl in classes}

        maps = []

        # 1. number of clusters
        for cl in classes:
            X_old_cl = self.X_old[np.array(self.y_old) == cl]
            X_new_cl = self.X_new[np.array(self.y_new) == cl]
            clusters_old, clusters_new, mapp = self._detect(X_old_cl, X_new_cl, k_init, k_max)

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

        details_3 = self.compare_desc_stats_for_clusters(self.stats_combined)

        # 3. desc stats changes
        for cl in classes:
            mask_old = self.y_old == cl
            cl_old = set(self.cluster_labels_old[mask_old])

            mask_new = self.y_new == cl
            cl_new = set(self.cluster_labels_new[mask_new])

            details[cl]['nr_of_clusters'] = len(set(self.cluster_labels_new[mask_new])) != len(
                set(self.cluster_labels_old[mask_old])
            )
            details[cl]['centroid_shift'] = any(
                v > thr_centroid_shift
                for v in {i: self.cluster_shifts[i] for i in cl_old.intersection(cl_new)}.values()
            )

            idx = set(self.cluster_labels_old[self.y_old == cl]).union(set(self.cluster_labels_new[self.y_new == cl]))
            # details[cl]['desc_stats_changes'] = {k: details_3[k] for k in idx if k in details_3}

        self.drift_flag = any([details[cl]['nr_of_clusters'] or details[cl]['centroid_shift'] for cl in classes])
        return self.drift_flag, details

    # DONE
    def calculate_centroid_shifts(self):
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

    # DONE
    def describe_clusters(self, X, labels):
        df = pd.DataFrame(X)
        df['cluster'] = labels
        desc = df.groupby('cluster').describe()
        return desc

    # DONE
    def compute_desc_stats_for_clusters(self):
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

        self.stats_combined = stats_combined
        return stats_combined

    # DONE
    def compare_desc_stats_for_clusters(self, stats_combined):
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
