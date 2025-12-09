import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from src.clustering.clustering import ClusterBasedDriftDetector
from src.clustering.visualization import plot_drift_clustered, color_map


def _display_intro():
    st.header("Clustering Analysis (Cluster-Based Drift Detection)")
    st.markdown("""
    This tab uses cluster-based methods to detect and analyze concept drift.
    It compares the data distribution and cluster structures between two windows:
    **Before Drift** and **After Drift**.
    """)


def _display_drift_status(drift_flag):
    if drift_flag:
        st.error(f"**Drift Detected:** {drift_flag}")
    else:
        st.success(f"**Drift Detected:** {drift_flag}")


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
                "Cluster Count Drift",
                help="True if the number of clusters changed",
                disabled=True
            ),
            "Centroid Shift Drift": st.column_config.CheckboxColumn(
                "Centroid Shift Drift",
                help="True if cluster centroids shifted significantly",
                disabled=True
            )
        },
        width="stretch",
        hide_index=True
    )


def _display_cluster_plot(X_before, X_after, detector):
    st.subheader("Cluster Visualization")

    labels_old = detector.cluster_labels_old
    labels_new = detector.cluster_labels_new

    if labels_old is not None and labels_new is not None:
        plot_drift_clustered(X_before, X_after, labels_old, labels_new, color_map=color_map, show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)
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
            _display_drift_details(details)
            _display_cluster_plot(X_before, X_after, detector)

        except Exception as e:
            st.error(f"An error occurred during drift detection: {e}")
            st.text(traceback.format_exc())
