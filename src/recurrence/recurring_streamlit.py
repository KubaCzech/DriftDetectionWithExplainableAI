import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import random
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from src.recurrence.methods import (  # noqa: E402
    cluster_windows,
    get_drift_from_clusters,
    median_mask
)

from river import forest  # noqa: E402
from src.datasets.protree_data.stream_generators import Sine, Plane, RandomTree  # noqa: E402
from src.datasets.protree_data.river_generators import Sea, Rbf, Stagger, Mixed  # noqa: E402

from src.recurrence.protree.explainers import APete  # noqa: E402
from src.recurrence.full_window_storage import FullWindowStorage  # noqa: E402


def _handle_stream_processing(drift_positions_str, generator_name, num_windows, window_size, drift_duration, seed_value,
                              median_filter_width, fix_outliers):
    try:
        # Parse drift positions
        drift_positions = []
        if drift_positions_str.strip():
            try:
                drift_positions = [int(x.strip()) for x in drift_positions_str.split(',') if x.strip()]
                drift_positions = sorted(drift_positions)  # Sort them in order
            except ValueError:
                st.error("Invalid drift positions format. Please enter comma-separated numbers "
                         "(e.g., '28000, 52000, 70000')")
                return

        # Initialize generator
        generator_map = {
            "Sine": Sine,
            "Plane": Plane,
            "RandomTree": RandomTree,
            "Sea": Sea,
            "Rbf": Rbf,
            "Stagger": Stagger,
            "Mixed": Mixed
        }

        generator_class = generator_map[generator_name]

        # Create generator with drift positions
        if drift_positions:
            ds = generator_class(
                drift_position=drift_positions,
                drift_duration=drift_duration,
                seed=seed_value
            )
        else:
            ds = generator_class(seed=seed_value)

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Generate dataset
        status_text.text("Generating dataset...")
        dataset = []
        for i in range(num_windows):
            x_block, y_block = zip(*ds.take(window_size))
            dataset.append([x_block, y_block])
            progress_bar.progress((i + 1) / (num_windows * 3))

        st.session_state.dataset = dataset

        # Setup model and storage
        status_text.text("Creating prototypes...")
        model = forest.ARFClassifier()
        storage = FullWindowStorage()

        # Process each window
        for i in range(len(dataset)):
            status_text.text(f"Processing window {i+1}/{num_windows}...")
            x_block, y_block = dataset[i]

            # Learn from data
            for x, y in zip(x_block, y_block):
                model.learn_one(x, y)

            # Create prototypes
            current_explainer = APete(model=model, alpha=0.01)
            current_prototypes = current_explainer.select_prototypes(x_block, y_block)

            # Store window data
            storage.store_window(
                iteration=i,
                x=x_block,
                y=y_block,
                prototypes=current_prototypes,
                explainer=None,
                drift=False
            )

            progress_bar.progress((num_windows + i + 1) / (num_windows * 3))

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

        # Store results in session state
        st.session_state.storage = storage
        st.session_state.matrix = matrix
        st.session_state.labels = labels
        st.session_state.drift_locations = drift_locations
        st.session_state.processing_complete = True

        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")

        st.success(f"Processing Complete!\\n"
                   f"- Stored {len(storage.get_all_iterations())} windows\\n"
                   f"- Detected {len(drift_locations)} drifts at windows: {drift_locations}\\n"
                   f"- Found {len(set(labels[labels != -1]))} distinct concepts")

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


# Page configuration
st.set_page_config(page_title="Drift Detection Tool", layout="wide")
st.title("🔍 Explainable Drift Detection with Prototypes")

# Initialize session state
if 'storage' not in st.session_state:
    st.session_state.storage = None
if 'matrix' not in st.session_state:
    st.session_state.matrix = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'drift_locations' not in st.session_state:
    st.session_state.drift_locations = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Processing",
    "🌍 Global View",
    "⚖️ Comparison",
    "🔬 Local Analysis",
    "🗺️ Distance Matrix"
])

# ============================================================================
# TAB 1: DATA PROCESSING
# ============================================================================
with tab1:
    st.header("Data Processing Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stream Configuration")

        # Generator selection
        generator_name = st.selectbox(
            "Select Data Generator",
            ["Rbf", "Sine", "Plane", "RandomTree", "Sea", "Stagger", "Mixed"],
            help="Choose the type of synthetic data stream"
        )

        # Stream parameters
        num_windows = st.number_input("Number of Windows", min_value=10, max_value=500, value=100, step=10)
        window_size = st.number_input("Samples per Window", min_value=100, max_value=5000, value=1000, step=100)

        # Drift parameters
        st.subheader("Drift Configuration")
        drift_positions_str = st.text_input(
            "Drift Positions (comma-separated sample numbers)",
            value="28000, 52000, 70000",
            help="Enter sample positions where drifts occur, e.g., '28000, 52000, 70000'. Leave empty for no drifts."
        )
        drift_duration = st.number_input("Drift Duration (samples)", min_value=1, value=1, step=1000)

        # Seed
        use_random_seed = st.checkbox("Use Random Seed", value=True)
        if not use_random_seed:
            seed_value = st.number_input("Seed Value", min_value=0, value=42)
        else:
            seed_value = random.randint(1, 10000)

    with col2:
        st.subheader("Clustering Configuration")
        median_filter_width = st.number_input(
            "Median Filter Width",
            min_value=1,
            max_value=15,
            value=3,
            step=2,
            help="Width of median filter for smoothing distance matrix before clustering (use odd numbers)"
        )

        fix_outliers = st.checkbox(
            "Fix Outlier Labels",
            value=True,
            help="Smooth outlier detections by filling gaps between same labels"
        )

    # Process button
    st.markdown("---")
    if st.button("🚀 Process Data", type="primary"):
        _handle_stream_processing(drift_positions_str, generator_name, num_windows, window_size,
                                  drift_duration, seed_value, median_filter_width, fix_outliers)

    # Show current status
    if st.session_state.processing_complete:
        st.info("✓ Data processed and ready for analysis. Navigate to other tabs to explore!")


def _render_global_stats(storage, labels, drift_locations):
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

    # Drift locations
    if drift_locations:
        st.info(f"🔄 **Detected drift locations (windows):** {', '.join(map(str, drift_locations))}")
    else:
        st.success("✓ No drifts detected - stream appears stable")

    # Additional statistics
    st.subheader("📊 Stream Statistics")

    # Collect data across all windows
    total_samples = 0
    total_prototypes = 0
    all_classes = set()
    samples_per_window = []
    prototypes_per_window = []

    for i in storage.get_all_iterations():
        _, y, prototypes, _ = storage.get_window_data(i)
        total_samples += len(y)
        window_proto_count = sum(len(p) for p in prototypes.values())
        total_prototypes += window_proto_count
        all_classes.update(y)
        samples_per_window.append(len(y))
        prototypes_per_window.append(window_proto_count)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Samples", f"{total_samples:,}")
    with col2:
        st.metric("Total Prototypes", f"{total_prototypes:,}")
    with col3:
        st.metric("Avg Prototypes/Window", f"{np.mean(prototypes_per_window):.1f}")
    with col4:
        st.metric("Number of Classes", len(all_classes))

    # Prototype distribution statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        compression_ratio = (total_prototypes / total_samples) * 100
        st.metric("Compression Ratio", f"{compression_ratio:.2f}%")
    with col2:
        st.metric("Min Prototypes/Window", int(np.min(prototypes_per_window)))
    with col3:
        st.metric("Max Prototypes/Window", int(np.max(prototypes_per_window)))
    with col4:
        st.metric("Std Prototypes/Window", f"{np.std(prototypes_per_window):.2f}")

    st.markdown("---")


def _plot_class_distribution(storage, drift_locations):
    # Class distribution over time
    st.subheader("📊 Class Distribution Over Time")

    class_distributions = []
    all_classes = set()

    for i in storage.get_all_iterations():
        _, y, _, _ = storage.get_window_data(i)
        class_counts = pd.Series(y).value_counts()
        all_classes.update(class_counts.index)
        class_distributions.append(class_counts)

    # Create dataframe with percentages
    all_classes = sorted(list(all_classes))
    class_pct_data = []

    for i, dist in enumerate(class_distributions):
        total = sum(dist.values)
        row = {'window': i}
        for cls in all_classes:
            row[f'class_{cls}'] = (dist.get(cls, 0) / total) * 100
        class_pct_data.append(row)

    df_class = pd.DataFrame(class_pct_data)

    fig, ax = plt.subplots(figsize=(12, 4))
    for cls in all_classes:
        ax.plot(df_class['window'], df_class[f'class_{cls}'], label=f'Class {cls}', linewidth=2)

    # Mark drift locations
    for drift in drift_locations:
        ax.axvline(x=drift, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Window')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Class Distribution Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()


def _plot_prototype_counts(storage, drift_locations):
    # Prototype counts over time
    st.subheader("🎯 Prototype Counts Over Time")

    # Need all classes
    all_classes = set()
    for i in storage.get_all_iterations():
        _, y, _, _ = storage.get_window_data(i)
        all_classes.update(y)
    all_classes = sorted(list(all_classes))

    prototype_counts = []
    for i in storage.get_all_iterations():
        _, _, prototypes, _ = storage.get_window_data(i)
        row = {'window': i}
        for cls in all_classes:
            row[f'class_{cls}'] = len(prototypes.get(cls, []))
        prototype_counts.append(row)

    df_proto = pd.DataFrame(prototype_counts)

    fig, ax = plt.subplots(figsize=(12, 4))
    for cls in all_classes:
        ax.plot(df_proto['window'], df_proto[f'class_{cls}'], label=f'Class {cls}', linewidth=2)

    # Mark drift locations
    for drift in drift_locations:
        ax.axvline(x=drift, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Window')
    ax.set_ylabel('Number of Prototypes')
    ax.set_title('Prototype Counts Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()


def _plot_concept_timeline(labels, drift_locations):
    # Concept timeline
    st.subheader("🎨 Concept Timeline")

    fig, ax = plt.subplots(figsize=(14, 2))

    # Create color map
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] if label != -1 else 'black'
                 for i, label in enumerate(unique_labels)}

    # Plot colored bars
    for i, label in enumerate(labels):
        ax.barh(0, 1, left=i, color=color_map[label], edgecolor='none')

    # Mark drift locations
    for drift in drift_locations:
        ax.axvline(x=drift, color='red', linestyle='-', linewidth=3, alpha=0.8)
        ax.text(drift, 0.5, 'DRIFT', rotation=90,
                verticalalignment='center', color='red', fontsize=10, fontweight='bold')

    ax.set_xlim(0, len(labels))
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Window')
    ax.set_yticks([])
    ax.set_title('Concept Clustering Timeline (Black = Outliers, Red Lines = Detected Drifts)')
    st.pyplot(fig)
    plt.close()

    # Legend for clusters
    st.markdown("**Cluster Labels:**")
    cluster_info = []
    for label in unique_labels:
        if label == -1:
            cluster_info.append(f"• **Outliers** (black): {sum(labels == -1)} windows")
        else:
            cluster_info.append(f"• **Cluster {label}**: {sum(labels == label)} windows")
    st.markdown("\\n".join(cluster_info))


def _render_global_plots(storage, labels, drift_locations):
    _plot_class_distribution(storage, drift_locations)
    _plot_prototype_counts(storage, drift_locations)
    _plot_concept_timeline(labels, drift_locations)


# ============================================================================
# TAB 2: GLOBAL VIEW
# ============================================================================
with tab2:
    st.header("Global Stream Analysis")

    if not st.session_state.processing_complete:
        st.warning("⚠️ Please process data in the 'Data Processing' tab first.")
    else:
        storage = st.session_state.storage
        labels = st.session_state.labels
        drift_locations = st.session_state.drift_locations

        _render_global_stats(storage, labels, drift_locations)
        _render_global_plots(storage, labels, drift_locations)


def _get_comparison_bounds(storage, selected_windows):
    global_min = float('inf')
    global_max = -float('inf')

    for window_nr in selected_windows:
        x, y, prototypes, explainer = storage.get_window_data(window_nr)
        for class_name in set(y):
            if class_name in prototypes:
                for prototype in prototypes[class_name]:
                    values = [v for _, v in sorted(prototype.items(), key=lambda x: x[0])]
                    if values:
                        global_min = min(global_min, min(values))
                        global_max = max(global_max, max(values))

    if global_min == float('inf'):
        return 0, 1  # Default range if no data

    margin = 0.05 * (global_max - global_min) if global_max != global_min else 0.1
    return global_min - margin, global_max + margin


def _plot_window_comparison(storage, selected_windows):
    # Display comparison plot
    st.subheader("Prototype Comparison")

    fig = plt.figure(figsize=(14, 8))

    used_min, used_max = _get_comparison_bounds(storage, selected_windows)

    # Get all classes
    all_classes = set()
    for window_nr in selected_windows:
        _, y, prototypes, _ = storage.get_window_data(window_nr)
        all_classes.update(prototypes.keys())
    classes_sorted = sorted(list(all_classes))

    # Create subplots
    n_rows = len(classes_sorted)
    n_cols = len(selected_windows)

    for row, class_name in enumerate(classes_sorted):
        for col, window_nr in enumerate(selected_windows):
            ax = plt.subplot(n_rows, n_cols, row * n_cols + col + 1)

            x, y, prototypes, explainer = storage.get_window_data(window_nr)

            if class_name in prototypes:
                for prototype in prototypes[class_name]:
                    items = sorted(prototype.items(), key=lambda x: x[0])
                    feature_names = [str(k) for k, _ in items]
                    feature_values = [v for _, v in items]
                    ax.plot(feature_names, feature_values, alpha=0.7)

                num_prototypes = len(prototypes[class_name])
                ax.text(0.02, 0.95, f"n={num_prototypes}",
                        transform=ax.transAxes, fontsize=9,
                        verticalalignment='top')

            ax.set_ylim(used_min, used_max)
            ax.tick_params(axis='x', labelrotation=45)

            if row == 0:
                ax.set_title(f"Window {window_nr}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Class {class_name}", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def _render_window_stats(storage, selected_windows):
    # Statistics table
    st.subheader("Window Statistics")

    # Get all classes for consistent columns
    all_classes = set()
    for window_nr in selected_windows:
        _, y, prototypes, _ = storage.get_window_data(window_nr)
        all_classes.update(prototypes.keys())
    classes_sorted = sorted(list(all_classes))

    stats_data = []
    for window_nr in selected_windows:
        x, y, prototypes, _ = storage.get_window_data(window_nr)

        stats = {
            'Window': window_nr,
            'Samples': len(y),
            'Total Prototypes': sum(len(p) for p in prototypes.values())
        }

        # Add per-class prototype counts
        for class_name in classes_sorted:
            stats[f'Class {class_name} Prototypes'] = len(prototypes.get(class_name, []))

        # Class distribution
        class_dist = pd.Series(y).value_counts()
        for class_name in classes_sorted:
            pct = (class_dist.get(class_name, 0) / len(y)) * 100
            stats[f'Class {class_name} %'] = f"{pct:.1f}%"

        stats_data.append(stats)

    df_stats = pd.DataFrame(stats_data)
    st.dataframe(df_stats, width='stretch')


def _render_comparison_tab(storage, all_iterations, drift_locations):
    st.write("Select multiple windows to compare their prototypes:")

    # Create default selection: window 0 + drift locations
    default_windows = [0] + drift_locations if drift_locations else all_iterations[:min(4, len(all_iterations))]
    # Ensure defaults are within valid range
    default_windows = [w for w in default_windows if w in all_iterations]

    # Multi-select for windows
    selected_windows = st.multiselect(
        "Select Windows to Compare",
        options=all_iterations,
        default=default_windows,
        help="Choose 2-6 windows for comparison"
    )

    if len(selected_windows) < 2:
        st.info("Please select at least 2 windows to compare.")
    elif len(selected_windows) > 6:
        st.warning("Comparing more than 6 windows may result in cluttered visualization.")
    else:
        _plot_window_comparison(storage, selected_windows)
        _render_window_stats(storage, selected_windows)


# ============================================================================
# TAB 3: COMPARISON
# ============================================================================
with tab3:
    st.header("Window Comparison")

    if not st.session_state.processing_complete:
        st.warning("⚠️ Please process data in the 'Data Processing' tab first.")
    else:
        storage = st.session_state.storage
        all_iterations = storage.get_all_iterations()
        drift_locations = st.session_state.drift_locations

        _render_comparison_tab(storage, all_iterations, drift_locations)


def _render_local_metrics(selected_window, x, y, prototypes, labels, drift_locations, storage):
    _render_class_distribution(y, selected_window)
    _render_prototype_details(x, y, prototypes)
    _render_distance_plots(selected_window, storage)

    # Cluster label info
    window_label = labels[selected_window]
    if window_label == -1:
        cluster_info = "⚠️ This window is classified as an **outlier**"
    else:
        cluster_info = f"✓ This window belongs to **Cluster {window_label}**"

    # Check if this is a drift location
    is_drift = selected_window in drift_locations
    drift_info = " 🔄 **DRIFT DETECTED AT THIS WINDOW**" if is_drift else ""

    st.info(f"{cluster_info}{drift_info}")

    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Window ID", selected_window)
    with col2:
        st.metric("Total Samples", len(y))
    with col3:
        st.metric("Total Prototypes", sum(len(p) for p in prototypes.values()))
    with col4:
        st.metric("Number of Classes", len(prototypes))

    # Additional window statistics
    col1, col2, col3, col4 = st.columns(4)

    # Feature statistics from all samples
    x_dicts = []
    for sample in x:
        if isinstance(sample, dict):
            x_dicts.append(sample)
        else:
            x_dicts.append({i: float(v) for i, v in enumerate(sample)})

    with col1:
        # Compression ratio for this window
        compression = (sum(len(p) for p in prototypes.values()) / len(y)) * 100
        st.metric("Window Compression", f"{compression:.2f}%")
    with col2:
        # Entropy of class distribution
        class_counts = pd.Series(y).value_counts()
        probs = class_counts / len(y)
        entropy = -np.sum(probs * np.log2(probs))
        st.metric("Class Entropy", f"{entropy:.3f}")
    with col3:
        # Imbalance ratio (max class / min class)
        if len(class_counts) > 1:
            imbalance = class_counts.max() / class_counts.min()
            st.metric("Class Imbalance Ratio", f"{imbalance:.2f}")
        else:
            st.metric("Class Imbalance Ratio", "N/A")
    with col4:
        pass

    st.markdown("---")


def _render_class_distribution(y, selected_window):
    st.subheader("Class Distribution in Window")
    class_counts = pd.Series(y).value_counts().sort_index()

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        class_counts.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title(f'Class Distribution - Window {selected_window}')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.write("**Class Statistics:**")
        for cls, count in class_counts.items():
            pct = (count / len(y)) * 100
            st.write(f"• Class {cls}: {count} samples ({pct:.1f}%)")


def _render_prototype_details(x, y, prototypes):
    st.subheader("Prototype Analysis")
    for class_name in sorted(prototypes.keys()):
        _render_single_class_prototypes(class_name, prototypes[class_name], x, y)


def _render_single_class_prototypes(class_name, proto_list, x, y):
    with st.expander(f"📌 Class {class_name} - {len(proto_list)} prototypes"):
        # Plot all prototypes for this class
        fig, ax = plt.subplots(figsize=(10, 4))

        for i, prototype in enumerate(proto_list):
            items = sorted(prototype.items(), key=lambda x: x[0])
            feature_names = [str(k) for k, _ in items]
            feature_values = [v for _, v in items]
            ax.plot(feature_names, feature_values, alpha=0.7)

        ax.set_xlabel('Features')
        ax.set_ylabel('Values')
        ax.set_title(f'All Prototypes for Class {class_name}')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelrotation=45)
        st.pyplot(fig)
        plt.close()

        _calculate_coverage_for_class_prototypes(proto_list, x, y)

        # Feature statistics across prototypes
        if len(proto_list) > 0:
            _render_prototype_feature_stats(proto_list)


def _render_prototype_feature_stats(proto_list):
    # Collect all feature values
    feature_data = {}
    for prototype in proto_list:
        for feat, val in prototype.items():
            if feat not in feature_data:
                feature_data[feat] = []
            feature_data[feat].append(val)

    stats_rows = []
    for feat in sorted(feature_data.keys()):
        values = feature_data[feat]
        stats_rows.append({
            'Feature': feat,
            'Mean': np.mean(values),
            'Std': np.std(values),
            'Min': np.min(values),
            'Max': np.max(values)
        })

    st.write("**Feature Statistics Across Prototypes:**")
    st.dataframe(pd.DataFrame(stats_rows), width='stretch')


def _calculate_coverage_for_class_prototypes(proto_list, x, y):
    st.write("**Prototype Coverage:**")

    # Convert all samples to dictionaries for distance calculation
    x_dicts = []
    for sample in x:
        if isinstance(sample, dict):
            x_dicts.append(sample)
        else:
            x_dicts.append({i: float(v) for i, v in enumerate(sample)})

    for proto_idx, prototype in enumerate(proto_list):
        distances = _calculate_distances_to_samples(prototype, x_dicts)
        _display_closest_samples(proto_idx, prototype, proto_list, distances, x_dicts, y)


def _calculate_distances_to_samples(prototype, x_dicts):
    distances = []
    for sample in x_dicts:
        common_features = set(prototype.keys()) & set(sample.keys())
        if common_features:
            dist = np.sqrt(sum((prototype[f] - sample[f])**2 for f in common_features))
            distances.append(dist)
        else:
            distances.append(float('inf'))
    return distances


def _display_closest_samples(proto_idx, prototype, proto_list, distances, x_dicts, y):
    # Find samples closest to this prototype compared to other prototypes in same class
    closest_samples = []
    for i, sample in enumerate(x_dicts):
        min_dist_to_this_proto = distances[i]
        is_closest = True

        for other_proto in proto_list:
            if other_proto is prototype:
                continue
            common_features = set(other_proto.keys()) & set(sample.keys())
            if common_features:
                other_dist = np.sqrt(sum((other_proto[f] - sample[f])**2 for f in common_features))
                if other_dist < min_dist_to_this_proto:
                    is_closest = False
                    break

        if is_closest:
            closest_samples.append(i)

    # Count classes of closest samples
    class_counts = {}
    for sample_idx in closest_samples:
        sample_class = y[sample_idx]
        class_counts[sample_class] = class_counts.get(sample_class, 0) + 1

    # Display info
    class_breakdown = ", ".join([f"Class {cls}: {cnt}" for cls, cnt in sorted(class_counts.items())])
    st.write(f"• **Prototype {proto_idx + 1}**: Closest to {len(closest_samples)} samples ({class_breakdown})")


def _render_distance_plots(selected_window, storage):
    # Distance to all other windows
    st.subheader("Distance to All Windows")

    k_median = st.slider("Median Filter Width", 1, 11, 3, 2,
                         key="local_median_filter")

    fig, ax = plt.subplots(figsize=(12, 4))
    data_to_plot = storage.compare_window_to_all(selected_window,
                                                 measure='centroid_displacement')
    data_to_plot = [float(x) for x in data_to_plot]
    if k_median > 1:
        data_to_plot = median_mask(data_to_plot, k_median)

    ax.plot(data_to_plot, linewidth=2)
    ax.axvline(x=selected_window, color='red', linestyle='--',
               linewidth=2, label='Current Window')

    # Mark drift locations
    for drift in st.session_state.drift_locations:
        ax.axvline(x=drift, color='orange', linestyle=':',
                   alpha=0.5, linewidth=1)

    ax.set_xlabel('Window')
    ax.set_ylabel('Distance')
    ax.set_title(f'Distance from Window {selected_window} to All Other Windows')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()


# ============================================================================
# TAB 5: DISTANCE MATRIX
# ============================================================================
with tab5:
    st.header("Distance Matrix Visualization")

    if not st.session_state.processing_complete:
        st.warning("⚠️ Please process data in the 'Data Processing' tab first.")
    else:
        matrix = st.session_state.matrix
        drift_locations = st.session_state.drift_locations

        st.write(f"Matrix shape: {matrix.shape}")

        # Matrix visualization options
        show_drift_markers = st.checkbox("Show Drift Markers", value=True)

        fig, ax = plt.subplots(figsize=(14, 12))

        # Create heatmap
        sns.heatmap(matrix, cmap='RdYlGn_r', square=True,
                    linewidths=0, cbar_kws={'label': 'Distance'},
                    ax=ax)

        # Mark drift positions if enabled
        if show_drift_markers and drift_locations:
            for drift_iter in drift_locations:
                if drift_iter in matrix.index:
                    idx = list(matrix.index).index(drift_iter)
                    ax.axhline(y=idx, color='red', linewidth=2, alpha=0.7)
                    ax.axvline(x=idx, color='red', linewidth=2, alpha=0.7)

        ax.set_title('Window Distance Matrix (Centroid Displacement)')
        ax.set_xlabel('Window')
        ax.set_ylabel('Window')

        st.pyplot(fig)
        plt.close()

        # Matrix statistics
        st.subheader("Matrix Statistics")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Distance", f"{matrix.values.mean():.4f}")
        with col2:
            st.metric("Std Distance", f"{matrix.values.std():.4f}")
        with col3:
            st.metric("Max Distance", f"{matrix.values.max():.4f}")
