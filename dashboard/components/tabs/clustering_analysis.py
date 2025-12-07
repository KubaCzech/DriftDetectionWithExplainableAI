import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.clustering.clustering import ClusterBasedDriftDetector
from src.clustering.visualization import plot_drift_clustered, color_map


def render_clustering_analysis_tab(X, y, window_before_start, window_after_start, window_length):
    """
    Renders the Clustering Analysis tab.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series or array-like
        Target variable.
    window_before_start : int
        Start index for the 'before' window.
    window_after_start : int
        Start index for the 'after' window.
    window_length : int
        Length of the windows.
    """
    st.header("Clustering Analysis (Cluster-Based Drift Detection)")

    st.markdown("""
    This tab uses cluster-based methods to detect and analyze concept drift.
    It compares the data distribution and cluster structures between two windows:
    **Before Drift** and **After Drift**.
    """)

    # Data preparation
    # Ensure X is numpy array for clustering
    X_np = X.values if isinstance(X, pd.DataFrame) else X
    
    # Slice the data
    start_before = window_before_start
    end_before = start_before + window_length
    
    # window_after_start is typically relative to the drift point in the dashboard logic,
    # but here we receive the absolute start index if it was passed correctly from app.py.
    # Let's verify how it's passed in feature_importance_analysis.py which this is based on.
    # In feature_importance_analysis.py:
    # compute_data_drift_analysis uses slice_data(X, y, window_before_start, window_after_start, window_length)
    # The app.py passes window_after_start directly from the sidebar config.
    # So we can use it directly.
    
    start_after = window_after_start
    end_after = start_after + window_length

    # Check bounds
    if end_before > len(X_np):
        st.error(f"Window 'Before' goes out of bounds: starts at {start_before}, ends at {end_before}, data length {len(X_np)}")
        return
    if end_after > len(X_np):
         st.error(f"Window 'After' goes out of bounds: starts at {start_after}, ends at {end_after}, data length {len(X_np)}")
         return

    X_old = X_np[start_before:end_before]
    y_old = y[start_before:end_before]
    
    X_new = X_np[start_after:end_after]
    y_new = y[start_after:end_after]

    data_old = (X_old, y_old)
    data_new = (X_new, y_new)

    # Initialize Detector with random_state for determinism
    detector = ClusterBasedDriftDetector(data_old, data_new, random_state=42)

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
                use_container_width=True,
                hide_index=True
            )

            # 3. Plot created with plot_drift_clustered
            st.subheader("Cluster Visualization")
            
            # We need labels from the detector for the plot
            labels_old = detector.cluster_labels_old
            labels_new = detector.cluster_labels_new
            
            if labels_old is not None and labels_new is not None:
                # plot_drift_clustered creates a new figure. 
                # We need to capture it to show in Streamlit.
                # The function calls plt.show() which might not work well in Streamlit directly 
                # if not handled. Ideally we modify it to return fig, but I should treat src as read-only if possible 
                # or just use the side effect of it creating a figure.
                # Since I can't easily change the src without specific permission or need, 
                # I will use the standard matplotlib approach in streamlit.
                
                # Create a new figure to ensure we capture relevant plots
                # plot_drift_clustered(X_before, X_after, labels_before, labels_after, color_map=color_map, show=True, in_subplot=False)
                # It calls plt.figure() internally if not in_subplot.
                
                # We can use st.pyplot() directly after calling the function, 
                # assuming it acts on the global pyplot state.
                # To be safe and avoid "Global figure warning", let's manage the figure.
                
                # Actually, plot_drift_clustered calls plt.show() which might close the plot.
                # However, in Streamlit, plt.show() often does nothing or warns.
                # Let's try calling it and grabbing the figure.
                
                # A better way might be to pass show=False to plot_drift_clustered if possible.
                # Looking at the code: def plot_drift_clustered(..., show=True, ...)
                # Yes, I can pass show=False.
                
                plot_drift_clustered(X_old, X_new, labels_old, labels_new, color_map=color_map, show=False)
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

