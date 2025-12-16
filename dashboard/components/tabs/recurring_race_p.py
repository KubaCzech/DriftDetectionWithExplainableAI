import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


from src.recurrence.methods import (
    cluster_windows,
    get_drift_from_clusters
)
# Note: visualize_distance_matrix and show_distance_median might be needed if used
from src.recurrence.full_window_storage import FullWindowStorage
from river import forest
from src.recurrence.protree.explainers import APete


def _process_single_window(i, x_block, y_block, model, storage, feature_names):
    # Helper to convert row to dict
    x_dicts = []
    for row in x_block:
        if hasattr(row, 'tolist'):
            row = row.tolist()
        row_dict = {feature_names[k]: v for k, v in enumerate(row)}
        x_dicts.append(row_dict)

    # Convert y to simple list
    if hasattr(y_block, 'tolist'):
        y_list = y_block.tolist()
    else:
        y_list = list(y_block)

    # Training loop
    for x_d, y_val in zip(x_dicts, y_list):
        model.learn_one(x_d, y_val)

    # Create prototypes
    current_explainer = APete(model=model, alpha=0.01)
    current_prototypes = current_explainer.select_prototypes(x_dicts, y_list)

    # Store window data
    storage.store_window(
        iteration=i,
        x=x_dicts,
        y=y_list,
        prototypes=current_prototypes,
        explainer=None,
        drift=False
    )


def _render_processing_tab(X, y, window_length):
    st.header("Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Windowing")
        total_samples = len(X)
        st.info(f"Total dataset size: {total_samples} samples")

        # Use global window length
        window_size = window_length
        st.metric("Samples per Window", window_size, help="Determined by global configuration")

        # Calculated windows
        num_windows = total_samples // window_size
        st.write(f"Resulting number of windows: **{num_windows}**")

        if num_windows < 5:
            st.warning("Warning: Number of windows is very low. Decrease window size for better resolution.")

    with col2:
        st.subheader("Clustering Parameters")
        median_filter_width = st.number_input(
            "Median Filter Width",
            min_value=1,
            max_value=15,
            value=3,
            step=2,
            help="Width of median filter for smoothing distance matrix before clustering (use odd numbers)",
            key="race_filter_width"
        )

        fix_outliers = st.checkbox(
            "Fix Outlier Labels",
            value=True,
            help="Smooth outlier detections by filling gaps between same labels",
            key="race_fix_outliers"
        )

    st.markdown("---")

    if st.button("🚀 Process Data", type="primary", key="race_process_btn"):
        _handle_processing(X, y, window_length, median_filter_width, fix_outliers)


def _handle_processing(X, y, window_length, median_filter_width, fix_outliers):
    try:
        total_samples = len(X)
        window_size = window_length
        num_windows = total_samples // window_size

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Chunk dataset
        status_text.text("Chunking dataset...")
        X_np = X.to_numpy() if hasattr(X, "to_numpy") else X
        y_np = y.to_numpy() if hasattr(y, "to_numpy") else y

        dataset_chunks = []
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            if end_idx <= total_samples:
                x_block = X_np[start_idx:end_idx]
                y_block = y_np[start_idx:end_idx]
                dataset_chunks.append((x_block, y_block))

        # Setup model and storage
        status_text.text("Creating prototypes...")
        model = forest.ARFClassifier()
        storage = FullWindowStorage()

        feature_names = list(X.columns) if hasattr(X, "columns") else [f"f{k}" for k in range(X.shape[1])]

        # Process each window
        for i, (x_block, y_block) in enumerate(dataset_chunks):
            status_text.text(f"Processing window {i+1}/{num_windows}...")

            _process_single_window(i, x_block, y_block, model, storage, feature_names)

            progress_bar.progress((i + 1) / (num_windows * 3))

        # Compute distance matrix
        status_text.text("Computing distance matrix...")
        matrix = storage.compute_distance_matrix(measure="centroid_displacement")
        progress_bar.progress(2 * num_windows / (num_windows * 3))

        # Cluster windows
        status_text.text("Clustering windows...")
        labels = cluster_windows(
            matrix,
            fix_outliers=fix_outliers,
            median_mask_width=median_filter_width
        )

        # Detect drifts
        drift_locations = get_drift_from_clusters(labels)

        # Store results
        st.session_state.race_storage = storage
        st.session_state.race_matrix = matrix
        st.session_state.race_labels = labels
        st.session_state.race_drift_locations = drift_locations
        st.session_state.race_processing_complete = True

        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")

        st.success(f"""
        Processing Complete!
        - Stored {len(storage.get_all_iterations())} windows
        - Detected {len(drift_locations)} drifts at windows: {drift_locations}
        - Found {len(set(labels[labels != -1]))} distinct concepts
        """)

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def _render_global_view_tab():
    st.header("Global Stream Analysis")

    if not st.session_state.race_processing_complete:
        st.info("ℹ️ Please process data in the 'Processing' tab first.")
    else:
        storage = st.session_state.race_storage
        labels = st.session_state.race_labels
        drift_locations = st.session_state.race_drift_locations

        # Global statistics
        st.subheader("📈 Global Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Windows", len(storage.get_all_iterations()))
        with col2:
            st.metric("Detected Drifts", len(drift_locations))
        with col3:
            num_concepts = len(set(labels[labels != -1]))
            st.metric("Distinct Concepts", num_concepts)
        with col4:
            num_outliers = sum(labels == -1)
            st.metric("Outlier Windows", num_outliers)

        # Timeline Plot
        st.subheader("🎨 Concept Timeline")
        fig, ax = plt.subplots(figsize=(14, 2))

        unique_labels = sorted(set(labels))
        # Use a colormap
        if len(unique_labels) > 0:
            cmap = plt.get_cmap('tab10')
            # Map labels to colors
            # Handle -1 (outlier) separately
            real_clusters = [lbl for lbl in unique_labels if lbl != -1]
            cluster_color_map = {lbl: cmap(i % 10) for i, lbl in enumerate(real_clusters)}
            cluster_color_map[-1] = 'black'

            for i, label in enumerate(labels):
                ax.barh(0, 1, left=i, color=cluster_color_map[label], edgecolor='none', height=0.8)

            # Mark drifts
            for drift in drift_locations:
                ax.axvline(x=drift, color='red', linestyle='-', linewidth=2)

            ax.set_yticks([])
            ax.set_xlabel("Window Index")
            ax.set_title("Concept Evolution (Colors = Concepts, Red Line = Drift, Black = Outlier)")
            st.pyplot(fig)
            plt.close()

        # Class Distribution Over Time
        st.subheader("📊 Class Distribution Over Time")

        # Gather distribution data
        dist_data = []
        all_iterations = storage.get_all_iterations()
        all_classes_seen = set()

        for i in all_iterations:
            _, win_y, _, _ = storage.get_window_data(i)
            counts = pd.Series(win_y).value_counts(normalize=True)
            row = counts.to_dict()
            row['window'] = i
            dist_data.append(row)
            all_classes_seen.update(counts.index)

        df_dist = pd.DataFrame(dist_data).fillna(0)

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        # Stackplot or line plot? Line plot is clearer for multiple classes usually
        for cls in sorted(list(all_classes_seen)):
            if cls in df_dist.columns:
                ax2.plot(df_dist['window'], df_dist[cls], label=f"Class {cls}")

        for drift in drift_locations:
            ax2.axvline(x=drift, color='red', linestyle='--', alpha=0.5)

        ax2.set_ylabel("Class Proportion")
        ax2.set_xlabel("Window")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close()


def _render_comparison_tab():
    st.header("Window Comparison")

    if not st.session_state.race_processing_complete:
        st.info("ℹ️ Please process data in the 'Processing' tab first.")
    else:
        storage = st.session_state.race_storage
        all_iterations = storage.get_all_iterations()

        st.write("Select windows to compare their prototypes.")

        selected_windows = st.multiselect(
            "Select Windows",
            options=all_iterations,
            default=all_iterations[:2] if len(all_iterations) >= 2 else all_iterations
        )

        if selected_windows:
            # Plot setup
            # We want to see prototypes for each class in each window

            # Collect all classes across selected windows
            classes_in_selection = set()
            for w in selected_windows:
                _, w_y, _, _ = storage.get_window_data(w)
                classes_in_selection.update(set(w_y))
            classes_sorted = sorted(list(classes_in_selection))

            n_rows = len(classes_sorted)
            n_cols = len(selected_windows)

            if n_rows > 0 and n_cols > 0:
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

                for r, cls in enumerate(classes_sorted):
                    for c, win_idx in enumerate(selected_windows):
                        ax = axes[r, c]
                        _, _, protos, _ = storage.get_window_data(win_idx)

                        if cls in protos:
                            # protos[cls] is a list of dicts (prototypes)
                            for p_idx, p_dict in enumerate(protos[cls]):
                                # Sort by feature name
                                items = sorted(p_dict.items())
                                feats = [str(k) for k, v in items]
                                vals = [v for k, v in items]
                                ax.plot(feats, vals, alpha=0.7, marker='o', markersize=3, label=f"P{p_idx}")

                            ax.set_title(f"Win {win_idx} - Class {cls} ({len(protos[cls])} protos)", fontsize=9)
                        else:
                            ax.text(0.5, 0.5, "No samples", ha='center', va='center', alpha=0.5)

                        # Rotate x labels if many features
                        if r == n_rows - 1:
                            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
                        else:
                            ax.set_xticklabels([])

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()


def _render_local_analysis_tab():
    st.header("Local Analysis")

    if not st.session_state.race_processing_complete:
        st.info("ℹ️ Please process data in the 'Processing' tab first.")
    else:
        storage = st.session_state.race_storage
        labels = st.session_state.race_labels
        all_iterations = storage.get_all_iterations()

        start_win = 0
        if st.session_state.race_drift_locations:
            start_win = st.session_state.race_drift_locations[0]

        selected_window = st.selectbox(
            "Select Focus Window",
            options=all_iterations,
            index=all_iterations.index(start_win) if start_win in all_iterations else 0
        )

        if selected_window is not None:
            x_data, y_data, prototypes, _ = storage.get_window_data(selected_window)
            win_label = labels[selected_window]

            st.metric("Cluster Label", str(win_label) if win_label != -1 else "Outlier")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Class Distribution**")
                cls_counts = pd.Series(y_data).value_counts()
                st.bar_chart(cls_counts)

            with col2:
                st.write("**Prototype Counts**")
                p_counts = {k: len(v) for k, v in prototypes.items()}
                st.write(p_counts)

            st.write("**Prototypes Visualization**")
            # Show prototypes for this window
            # One plot per class
            classes = sorted(prototypes.keys())
            for cls in classes:
                with st.expander(f"Class {cls} Prototypes"):
                    fig, ax = plt.subplots(figsize=(10, 3))
                    for p in prototypes[cls]:
                        items = sorted(p.items())
                        ax.plot([str(k) for k, v in items], [v for k, v in items], alpha=0.6)
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                    st.pyplot(fig)
                    plt.close()

            # Distance to other windows plot
            st.write("**Similiarity to Stream**")
            dist_fig, dist_ax = plt.subplots(figsize=(10, 3))

            # We need to compute/retrieve distance to all other windows
            # The storage doesn't pre-compute all-to-all in a queryable way easily unless we use the matrix
            # Actually we have the matrix! st.session_state.race_matrix

            matrix = st.session_state.race_matrix
            if matrix is not None:
                # Matrix is N x N. Row i is distances from i to others.
                dists = matrix.iloc[selected_window, :]
                dist_ax.plot(dists, label="Distance")
                dist_ax.axvline(x=selected_window, color='red', linestyle='--')
                dist_ax.set_xlabel("Window Index")
                dist_ax.set_ylabel("Distance")
                dist_ax.set_title(f"Distance from Window {selected_window} to others")
                st.pyplot(dist_fig)
                plt.close()


def _render_distance_matrix_tab():
    st.header("Distance Matrix")

    if not st.session_state.race_processing_complete:
        st.info("ℹ️ Please process data in the 'Processing' tab first.")
    else:
        matrix = st.session_state.race_matrix
        if matrix is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(matrix, cmap="viridis", aspect='auto')
            plt.colorbar(im, ax=ax, label="Distance")
            ax.set_title("Pairwise Window Distances")
            ax.set_xlabel("Window Index")
            ax.set_ylabel("Window Index")
            st.pyplot(fig)
            plt.close()


def render_recurring_race_p_tab(X, y, window_length):
    st.header("Recurring RACE-P Analysis")

    # Initialize session state keys specific to this tab
    if 'race_storage' not in st.session_state:
        st.session_state.race_storage = None
    if 'race_matrix' not in st.session_state:
        st.session_state.race_matrix = None
    if 'race_labels' not in st.session_state:
        st.session_state.race_labels = None
    if 'race_drift_locations' not in st.session_state:
        st.session_state.race_drift_locations = None
    if 'race_processing_complete' not in st.session_state:
        st.session_state.race_processing_complete = False

    # Tabs for the recurrence analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Processing",
        "🌍 Global View",
        "⚖️ Comparison",
        "🔬 Local Analysis",
        "🗺️ Distance Matrix"
    ])

    # ============================================================================
    # TAB 1: DATA PROCESSING
    # ============================================================================
    with tab1:
        _render_processing_tab(X, y, window_length)

    # ============================================================================
    # TAB 2: GLOBAL VIEW
    # ============================================================================
    with tab2:
        _render_global_view_tab()

    # ============================================================================
    # TAB 3: COMPARISON
    # ============================================================================
    with tab3:
        _render_comparison_tab()

    # ============================================================================
    # TAB 4: LOCAL ANALYSIS
    # ============================================================================
    with tab4:
        _render_local_analysis_tab()

    # ============================================================================
    # TAB 5: DISTANCE MATRIX
    # ============================================================================
    with tab5:
        _render_distance_matrix_tab()
