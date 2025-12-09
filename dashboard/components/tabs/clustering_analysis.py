import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.clustering.clustering import ClusterBasedDriftDetector
from src.clustering.visualization import plot_drift_clustered, color_map


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
    st.header("Clustering Analysis (Cluster-Based Drift Detection)")

    st.markdown("""
    This tab uses cluster-based methods to detect and analyze concept drift.
    It compares the data distribution and cluster structures between two windows:
    **Before Drift** and **After Drift**.
    """)

    # Initialize Detector with random_state for determinism
    detector = ClusterBasedDriftDetector(X_before, y_before, X_after, y_after, random_state=42)

    with st.spinner("Running cluster-based drift detection..."):
        try:
            drift_flag, details = detector.detect()

            # 1. Information whether drift occurred
            if drift_flag:
                st.error(f"**Drift Detected:** {drift_flag}")
            else:
                st.success(f"**Drift Detected:** {drift_flag}")

            # 2. Table based on details
            st.subheader("Drift Details by Class")

            # Flatten the details dictionary for DataFrame
            # details structure: {class_label: {'nr_of_clusters': bool, 'centroid_shift': bool, ...}}
            rows = []
            for class_label, class_details in details.items():
                row = {'Class': class_label}
                row.update(class_details)
                rows.append(row)

            details_df = pd.DataFrame(rows)

            # Ensure Class is string to avoid checkbox rendering for binary labels
            details_df['Class'] = details_df['Class'].astype(str)

            # Rename columns for better readability
            details_df = details_df.rename(columns={
                'nr_of_clusters': 'Cluster Count Drift',
                'centroid_shift': 'Centroid Shift Drift'
            })

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
                width=True,
                hide_index=True
            )

            # 3. Plot created with plot_drift_clustered
            st.subheader("Cluster Visualization")

            # We need labels from the detector for the plot
            labels_old = detector.cluster_labels_old
            labels_new = detector.cluster_labels_new

            if labels_old is not None and labels_new is not None:
                plot_drift_clustered(X_before, X_after, labels_old, labels_new, color_map=color_map, show=False)
                fig = plt.gcf()
                st.pyplot(fig)
                plt.close(fig)

            else:
                st.warning("Cluster labels not available for plotting.")

        except Exception as e:
            st.error(f"An error occurred during drift detection: {e}")
            # It might be useful to see the traceback in logs,
            # but for dashboard just showing error is distinct.
            import traceback
            st.text(traceback.format_exc())
