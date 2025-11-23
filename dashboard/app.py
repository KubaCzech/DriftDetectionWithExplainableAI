import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
from io import StringIO
import contextlib
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets import DATASETS, get_all_datasets
from src.feature_importance import visualize_data_stream
from dashboard.tabs import (
    render_data_visualization_tab,
    render_feature_importance_analysis_tab
)

# --- App Configuration ---
st.set_page_config(
    page_title="Concept Drift Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Main App ---
st.title("📊 Concept Drift Analysis Dashboard")

st.markdown("""
Welcome to the Concept Drift Analysis Dashboard. This tool allows you to:
1.  **Generate** synthetic datasets with known concept drift.
2.  **Visualize** the data stream and the drift itself.
3.  **Analyze** the drift using various feature importance techniques to understand its root causes.
""")

# --- Sidebar for User Input ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # 1. Select Dataset
    dataset_key = st.selectbox(
        "Choose a Dataset",
        options=list(DATASETS.keys()),
        format_func=lambda x: DATASETS[x].display_name,
        help="Select the synthetic dataset to analyze."
    )
    
    selected_dataset = DATASETS[dataset_key]

    # Conditionally display options based on dataset type
    n_features = None
    n_drift_features = None
    csv_file = None
    target_col = "target"
    
    if dataset_key == "hyperplane_drift":
        st.subheader("Hyperplane Drift Settings")
        n_features = st.number_input(
            "Number of Features (n_features)",
            min_value=2,
            value=2,
            step=1,
            help="Total number of features for the hyperplane. Must be >= 2."
        )
        n_drift_features = st.number_input(
            "Number of Drifting Features (n_drift_features)",
            min_value=2,
            max_value=n_features,
            value=min(2, n_features),
            step=1,
            help="Number of features that will drift. Must be <= n_features."
        )
    elif dataset_key == "controlled_concept_drift":
        st.subheader("Controlled Concept Drift Settings")
        n_features = st.number_input(
            "Number of Features (n_features)",
            min_value=2,
            value=11,
            step=1,
            help="Total number of features for the dataset. Must be >= 2."
        )
        n_drift_features = st.number_input(
            "Number of Drifting Features (n_drift_features)",
            min_value=1,
            max_value=n_features,
            value=min(5, n_features),
            step=1,
            help="Number of features that will drift. Must be <= n_features."
        )
    elif dataset_key == "csv_dataset":
        st.subheader("CSV Dataset Settings")
        csv_file = st.file_uploader("Upload CSV File", type=["csv"])
        target_col = st.text_input("Target Column Name", value="target")

    # 2. Toggle for Boxplots
    show_boxplot = st.checkbox(
        "Show Importance Boxplots",
        value=True,
        help="Display boxplots for feature importance distributions."
    )

    st.markdown("---")
    st.info("Adjust the settings above to configure the data and analysis.")


# --- Data Generation ---
@st.cache_data
def generate_data(dataset_name, n_features=None, n_drift_features=None, csv_file=None, target_col="target"):
    """Cached function to generate data."""
    dataset = DATASETS.get(dataset_name)
    if not dataset:
        st.error(f"Unknown dataset: {dataset_name}")
        return None, None, None, None

    gen_params = dataset.get_params()
    
    # Update params with user input
    if n_features is not None:
        gen_params['n_features'] = n_features
    if n_drift_features is not None:
        gen_params['n_drift_features'] = n_drift_features
        
    if dataset_name == "csv_dataset":
        if csv_file is not None:
            gen_params['file_path'] = csv_file
            gen_params['target_column'] = target_col
        else:
            return None, None, None, None

    try:
        return dataset.generate(**gen_params)
    except Exception as e:
        st.error(f"Error generating data: {e}")
        return None, None, None, None


X, y, drift_point, feature_names = generate_data(
    dataset_key, 
    n_features=n_features, 
    n_drift_features=n_drift_features,
    csv_file=csv_file,
    target_col=target_col
)

if X is None:
    st.warning("Please upload a CSV file or select a valid dataset to proceed.")
    st.stop()

# --- Plot Generation and Capturing (Modified Logic) ---

@st.cache_data(show_spinner="Generating data stream visualizations...")
def generate_and_capture_plots(X, y, drift_point, feature_names):
    """Generates all visualization plots and captures them."""
    # Redirect stdout to capture print statements
    stdout_capture = StringIO()
    with contextlib.redirect_stdout(stdout_capture):
        # This function call creates multiple figures and leaves them open
        visualize_data_stream(X, y, drift_point, feature_names)

    # Capture the figures and close them immediately
    all_figs = []
    for fig_id in plt.get_fignums():
        fig = plt.figure(fig_id)
        all_figs.append(fig)
        plt.close(fig)  # Close the figure to free up memory

    return all_figs, stdout_capture.getvalue()


all_figs, info_log = generate_and_capture_plots(X, y, drift_point, feature_names)

# --- Tabs ---
tab1, tab2 = st.tabs(["Dataset Selection and Visualization", "Feature Importance Analysis"])

with tab1:
    render_data_visualization_tab(X, y, drift_point, feature_names, all_figs)

with tab2:
    render_feature_importance_analysis_tab(X, y, drift_point, feature_names, show_boxplot)

st.markdown("---")
st.markdown("Developed as part of the xAI and Data Analysis Tools for Drift Detection project.")
