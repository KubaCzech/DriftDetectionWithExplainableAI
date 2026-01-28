import traceback
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.clustering.clustering import ClusterBasedDriftDetector
from src.clustering.visualization import plot_clusters_by_class, plot_drift_clustered, plot_centers_shift


def _display_intro():
    st.header("Clustering Analysis")
    st.markdown(
        """
    This tab uses cluster-based methods to detect and analyze concept drift.
    It compares the data distribution and cluster structures between two windows:
    **Before Drift** and **After Drift**.
    """
    )


def _display_drift_status(drift_flag):
    if drift_flag:
        st.error(f"**Drift Detected:** {drift_flag}")
    else:
        st.success(f"**Drift Detected:** {drift_flag}")


def _display_drift_strength(drift_strength):
    st.subheader("Drift Strength")
    st.write(f"The calculated drift strength is: **{drift_strength:.4f}**")


def _config_details_dataframe(details):
    rows = []
    for class_label, class_details in details.items():
        row = {'Class': str(class_label)}
        row['Cluster Count Drift'] = class_details.get('nr_of_clusters')
        row['Centroid Shift Drift'] = class_details.get('centroid_shift')
        rows.append(row)
    return pd.DataFrame(rows)


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


def _display_visualization(X_before, X_after, y_before, y_after, labels_old, labels_new, viz_type='cluster by class'):
    plot_func = CLUSTER_PLOT_FUNCTIONS.get(viz_type, plot_clusters_by_class)
    plot_func(X_before, X_after, y_before, y_after, labels_old, labels_new)


def _display_centroid_shift_table(detector):
    # TODO: add docstring
    centroid_shifts = (
        pd.DataFrame.from_dict(detector.cluster_shifts, orient="index", columns=["Shift"])
        .reset_index()
        .rename(columns={"index": "Cluster ID"})
    )
    centroid_shifts["Drifted"] = centroid_shifts["Shift"].apply(
        lambda x: isinstance(x, float) and x > detector.thr_centroid_shift or x == 'appeared' or x == 'disappeared'
    )

    st.dataframe(
        centroid_shifts,
        column_config={
            "Cluster ID": st.column_config.TextColumn("Cluster ID"),
            "Shift": st.column_config.TextColumn("Cluster Shift", help="How much did the cluster shift"),
            "Drifted": st.column_config.CheckboxColumn(
                "Drifted",
                help=f"True if the cluster centroid shifted above the threshold ({detector.thr_centroid_shift})",
                disabled=True,
            ),
        },
        width="stretch",
        hide_index=True,
    )


def _display_stats_shift_table(detector):
    # TODO: add docstring
    # TODO: add better formatting
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
                    row[f"{feature_name} | {stat_name}"] = value
            records.append(row)

        df = pd.DataFrame(records)
        return df

    stats_shifts = stats_dict_to_df(detector.stats_shifts)
    st.dataframe(
        stats_shifts,
        use_container_width=True,
    )


def _display_stats_table(detector):
    # TODO: add docstring
    # TODO: add better formatting
    def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [f"{time} | {feature} | {stat}" for time, feature, stat in df.columns]
        return df

    stats = flatten_multiindex_columns(detector.stats_combined)
    st.subheader("Cluster Descriptive Statistics")
    st.dataframe(
        stats,
        use_container_width=True,
    )


def _display_table(detector, table_type):
    table_func = CLUSTER_TABLE_FUNCTIONS.get(table_type, _display_centroid_shift_table)
    table_func(detector)


CLUSTER_PLOT_FUNCTIONS = {
    'clusters by class': plot_clusters_by_class,
    'clusters overall': plot_drift_clustered,
    'centroid shifts (plot)': plot_centers_shift,
}

CLUSTER_TABLE_FUNCTIONS = {
    'centroid shifts (table)': _display_centroid_shift_table,
    'stats shifts': _display_stats_shift_table,
    'stats table': _display_stats_table,
}


def _display(X_before, X_after, y_before, y_after, detector):
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
                    'centroid shifts (plot)',
                    'centroid shifts (table)',
                    'stats table',
                    'stats shifts',
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


def render_clustering_analysis_tab(X_before, y_before, X_after, y_after):
    """
    Renders the Clustering Analysis tab.

    Parameters
    ----------
    X_before : array-like
        Feature matrix for 'before' window
    y_before : array-like
        Target variable for 'before' window
    X_after : array-like
        Feature matrix for 'after' window
    y_after : array-like
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
