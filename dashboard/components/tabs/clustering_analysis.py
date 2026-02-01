import traceback
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.clustering.clustering import ClusterBasedDriftDetector
from src.clustering.visualization import (
    plot_clusters_by_class,
    plot_drift_clustered,
    plot_centers_shift,
)


# DONE
def _display_intro():
    """
    Renders the introductory text for the Clustering Analysis tab.
    """

    st.header("Clustering Analysis")
    st.markdown(
        """
    This tab uses cluster-based methods to detect and analyze concept drift.
    It compares the data distribution and cluster structures between two windows:
    **Before Drift** and **After Drift**.
    """
    )


# DONE
def _display_drift_status(drift_flag):
    """
    Displays whether drift was detected.

    Parameters
    ----------
    drift_flag : bool
        Indicates whether drift was detected.
    """
    if drift_flag:
        st.success(f"**Drift Detected:** Yes")
    else:
        st.error(f"**Drift Detected:** No")


# DONE
def _display_drift_strength(drift_strength):
    """
    Displays strength of drift indicator.

    Parameters
    ----------
    drift_strength : bool
        Probability of drift occurring.
    """
    st.subheader("Drift Strength")
    st.write(f"The calculated drift strength is: **{drift_strength:.4f}**")


# TODO
def _config_details_dataframe(details):
    rows = []
    for class_label, class_details in details.items():
        row = {'Class': str(class_label)}
        row['Cluster Count Drift'] = class_details.get('nr_of_clusters')
        row['Centroid Shift Drift'] = class_details.get('centroid_shift')
        rows.append(row)
    return pd.DataFrame(rows)


# TODO
def _display_drift_details(details):
    st.subheader("Drift Details by Class")

    details_df = _config_details_dataframe(details)

    st.dataframe(
        details_df,
        column_config={
            "Class": st.column_config.TextColumn("Class"),
            "Cluster Count Drift": st.column_config.CheckboxColumn(
                "Cluster Count Drift", help="True if the number of clusters changed", disabled=True
            ),
            "Centroid Shift Drift": st.column_config.CheckboxColumn(
                "Centroid Shift Drift", help="True if cluster centroids shifted significantly", disabled=True
            ),
        },
        width="stretch",
        hide_index=True,
    )


# TODO
def _display_visualization(X_before, X_after, y_before, y_after, labels_old, labels_new, viz_type='cluster by class'):
    plot_func = CLUSTER_PLOT_FUNCTIONS.get(viz_type, plot_clusters_by_class)
    plot_func(X_before, X_after, y_before, y_after, labels_old, labels_new)


# TODO
def _display_table(detector, table_type):
    table_func = CLUSTER_TABLE_FUNCTIONS.get(table_type, _display_centroid_shift_table)
    table_func(detector)


CLUSTER_PLOT_FUNCTIONS = {
    'clusters by class': plot_clusters_by_class,
    'clusters overall': plot_drift_clustered,
    'centroid shifts (plot)': plot_centers_shift,
}


# TABULAR DISPLAY FUNCTIONS
# DONE
def _display_cluster_assignments_table(detector: ClusterBasedDriftDetector):
    """
    Displays a table showing cluster assignments for each class before and after drift.

    Parameters
    ----------
    detector : ClusterBasedDriftDetector
        The drift detector containing cluster labels and class labels.
    """
    y_before, y_after = detector.y_old, detector.y_new
    labels_before, labels_after = detector.cluster_labels_old, detector.cluster_labels_new

    classes_ids = sorted(set(y_before).union(set(y_after)))

    rows = []
    for class_id in classes_ids:
        before_clusters = sorted(set(labels_before[np.array(y_before) == class_id]))
        after_clusters = sorted(set(labels_after[np.array(y_after) == class_id]))

        rows.append(
            {
                "Class": int(class_id),
                "Before Drift": ", ".join(map(str, before_clusters)),
                "After Drift": ", ".join(map(str, after_clusters)),
            }
        )

    df_total = pd.DataFrame(rows)

    st.dataframe(
        df_total,
        column_config={
            "Class": st.column_config.TextColumn("Class", help="Class label"),
            "Before Drift": st.column_config.TextColumn(
                "Before Drift", help="Cluster IDs associated with this class before drift"
            ),
            "After Drift": st.column_config.TextColumn(
                "After Drift", help="Cluster IDs associated with this class after drift"
            ),
        },
        width="stretch",
        hide_index=True,
    )


# DONE
def _display_stats_shift_table(detector):
    """
    Displays a table showing statistical shifts for each cluster.

    Parameters
    ----------
    detector : ClusterBasedDriftDetector
        The drift detector containing statistical shift information.
    """

    def stats_dict_to_df(stats_dict: dict) -> pd.DataFrame:
        """
        Converts a nested dict {cluster -> feature -> stat -> value}
        into a flat pandas DataFrame.
        """
        records = []
        for cluster_id, features in stats_dict.items():
            row = {"Cluster": cluster_id}
            for feature_name, stats in features.items():
                for stat_name, value in stats.items():
                    row[f"{feature_name} | {stat_name}"] = f"{round(100*value, 2)}%" if value is not None else "N/A"
            records.append(row)

        df = pd.DataFrame(records)
        return df

    stats_shifts = stats_dict_to_df(detector.stats_shifts)
    st.dataframe(stats_shifts, use_container_width=True, hide_index=True)


# DONE
def _display_centroid_shift_table(detector: ClusterBasedDriftDetector):
    """
    Displays a table showing centroid shifts (in units) for each cluster.

    Parameters
    ----------
    detector : ClusterBasedDriftDetector
        The drift detector containing cluster shift information.
    """
    centroid_shifts = {
        int(i): (
            round(detector.cluster_shifts[i]['euclidean_distance'], 4)
            if isinstance(detector.cluster_shifts[i], dict)
            else detector.cluster_shifts[i]
        )
        for i in detector.cluster_shifts
    }
    centroid_shifts = pd.DataFrame.from_dict({'cluster_id': centroid_shifts.keys(), 'shift': centroid_shifts.values()})

    centroid_shifts["drifted"] = centroid_shifts["shift"].apply(
        lambda x: isinstance(x, float) and x > detector.thr_centroid_shift or x == 'appeared' or x == 'disappeared'
    )

    st.dataframe(
        centroid_shifts,
        column_config={
            "cluster_id": st.column_config.TextColumn("Cluster ID"),
            "shift": st.column_config.TextColumn("Cluster Shift", help="How much did the cluster shift"),
            "drifted": st.column_config.CheckboxColumn(
                "Drifted",
                help=f"True if the cluster centroid shifted above the threshold ({detector.thr_centroid_shift:.2f}) or the cluster appeared/disappeared",
                disabled=True,
            ),
        },
        width="stretch",
        hide_index=True,
    )


# DONE
def _display_centroids_table(detector: ClusterBasedDriftDetector):
    """
    Displays centroids in a format:
    Cluster | Data Block (Before/After) | x1 ... xn

    Parameters
    ----------
    detector : ClusterBasedDriftDetector
        The drift detector containing cluster centroids.
    """
    centroids_before = detector.centers_old
    centroids_after = detector.centers_new

    rows = []

    all_cluster_ids = sorted(set(centroids_before.keys()).union(centroids_after.keys()))

    # Infer feature count
    sample_centroid = next(c for c in list(centroids_before.values()) + list(centroids_after.values()) if c is not None)
    n_features = len(sample_centroid)
    feature_cols = [f"x{i+1}" for i in range(n_features)]

    for cluster_id in all_cluster_ids:
        for window, centroids in [
            ("Before", centroids_before),
            ("After", centroids_after),
        ]:
            centroid = centroids.get(cluster_id)

            row = {
                "Cluster": int(cluster_id),
                "Data Block": window,
            }

            if centroid is None:
                for f in feature_cols:
                    row[f] = "-"
            else:
                for i, f in enumerate(feature_cols):
                    row[f] = round(float(centroid[i]), 4)

            rows.append(row)

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        column_config={
            "Cluster": st.column_config.NumberColumn("Cluster"),
            "Data Block": st.column_config.TextColumn(
                "Data Block",
                help="Data Block Before or After drift",
            ),
            **{f: st.column_config.TextColumn(f) for f in feature_cols},
        },
        width="stretch",
        hide_index=True,
    )


# DONE
def _display_avg_distance_to_center_change(detector: ClusterBasedDriftDetector):
    """
    Displays a table showing average distance to cluster center changes.

    Parameters
    ----------
    detector : ClusterBasedDriftDetector
        The drift detector containing average distance metrics.
    """
    avg_distance = pd.DataFrame.from_dict(
        {
            'cluster_id': [int(i) for i in sorted(detector.avg_distance_old.keys())],
            'avg_distance_before': [i[1] for i in sorted(detector.avg_distance_old.items(), key=lambda x: x[0])],
            'avg_distance_after': [i[1] for i in sorted(detector.avg_distance_new.items(), key=lambda x: x[0])],
            'avg_distance_shift': [
                f"{'+' if i[1] > 0 else '-'}{abs(round(i[1]*100, 2))}%" if i[1] is not None else "-"
                for i in sorted(detector.avg_distance_shift.items(), key=lambda x: x[0])
            ],
        }
    )

    st.dataframe(
        avg_distance[['cluster_id', 'avg_distance_shift']],
        column_config={
            "cluster_id": st.column_config.TextColumn("Class", help="Class label"),
            "avg_distance_shift": st.column_config.TextColumn(
                "Average Distance to Center Relative Change",
                help="How much did the average distance between data blocks to center change",
            ),
        },
        width="stretch",
        hide_index=True,
    )


CLUSTER_TABLE_FUNCTIONS = {
    'cluster assignments (table)': _display_cluster_assignments_table,
    'centroid shifts (table)': _display_centroid_shift_table,
    'stats shifts': _display_stats_shift_table,
    'average distance to center change': _display_avg_distance_to_center_change,
    'centroids table': _display_centroids_table,
}


# TODO
def _display(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    y_before: np.ndarray,
    y_after: np.ndarray,
    detector: ClusterBasedDriftDetector,
):
    """
    Displays clustering analysis results including visualizations and tables.

    Parameters
    ----------
    X_before : pd.DataFrame
        Feature matrix for 'before' window
    X_after : pd.DataFrame
        Feature matrix for 'after' window
    y_before : np.ndarray
        Target variable for 'before' window
    y_after : np.ndarray
        Target variable for 'after' window
    detector : ClusterBasedDriftDetector
        The drift detector containing clustering results.
    """
    st.subheader("Clustering Visualization")

    labels_old = detector.cluster_labels_old
    labels_new = detector.cluster_labels_new

    if labels_old is not None and labels_new is not None:
        col_viz_options, _ = st.columns([1, 2])
        with col_viz_options:
            # TODO: wymyslic opisy
            _type = st.selectbox(
                "TODO",
                options=[
                    'clusters by class',
                    'clusters overall',
                    'cluster assignments (table)',
                    'stats shifts',
                    'centroid shifts (plot)',
                    'centroid shifts (table)',
                    'centroids table',
                    'average distance to center change',
                ],
                index=0,  # Default to 'cluster by class'
                key='viz_type_selector',
                help="TODO",
            )
        if _type in CLUSTER_PLOT_FUNCTIONS:
            _display_visualization(X_before, X_after, y_before, y_after, labels_old, labels_new, viz_type=_type)
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)
        elif _type in CLUSTER_TABLE_FUNCTIONS:
            _display_table(detector, table_type=_type)
    else:
        st.warning("Cluster labels not available for plotting.")


# DONE
def render_clustering_analysis_tab(
    X_before: pd.DataFrame, y_before: np.ndarray, X_after: pd.DataFrame, y_after: np.ndarray
):
    """
    Renders the Clustering Analysis tab.

    Parameters
    ----------
    X_before : pd.DataFrame
        Feature matrix for 'before' window
    y_before : np.ndarray
        Target variable for 'before' window
    X_after : pd.DataFrame
        Feature matrix for 'after' window
    y_after : np.ndarray
        Target variable for 'after' window
    """
    _display_intro()

    # Initialize Detector with random_state for determinism
    detector = ClusterBasedDriftDetector(X_before, y_before, X_after, y_after, random_state=42)

    with st.spinner("Running cluster-based drift detection..."):
        try:
            drift_flag, details = detector.detect()

            _display_drift_status(drift_flag)
            _display_drift_strength(detector.strength_of_drift)
            _display_drift_details(details)
            _display(X_before, X_after, y_before, y_after, detector)

        except Exception as e:
            st.error(f"An error occurred during drift detection: {e}")
            st.text(traceback.format_exc())
