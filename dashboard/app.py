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
from dashboard.components.settings import render_settings_from_schema
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

    # 2. Render Dataset Settings
    if selected_dataset.name != "custom_normal" and selected_dataset.name != "custom_3d_drift" and selected_dataset.name != "sea_drift":
         st.subheader(f"{selected_dataset.display_name} Settings")
    
    dataset_params = render_settings_from_schema(selected_dataset.get_settings_schema())

    # 3. Toggle for Boxplots
    show_boxplot = st.checkbox(
        "Show Importance Boxplots",
        value=True,
        help="Display boxplots for feature importance distributions."
    )

    st.markdown("---")
    st.info("Adjust the settings above to configure the data and analysis.")


# --- Data Generation ---
@st.cache_data
def generate_data(dataset_name, **kwargs):
    """Cached function to generate data."""
    dataset = DATASETS.get(dataset_name)
    if not dataset:
        st.error(f"Unknown dataset: {dataset_name}")
        return None, None, None, None

    gen_params = dataset.get_params()
    gen_params.update(kwargs)

    try:
        return dataset.generate(**gen_params)
    except Exception as e:
        st.error(f"Error generating data: {e}")
        return None, None, None, None


X, y, drift_point, feature_names = generate_data(
    dataset_key,
    **dataset_params
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
