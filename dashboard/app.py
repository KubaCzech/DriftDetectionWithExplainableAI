import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
from io import StringIO
import contextlib

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets import DATASETS  # noqa: E402
from src.feature_importance import visualize_data_stream  # noqa: E402
from dashboard.components.tabs import (  # noqa: E402
    render_data_visualization_tab,
    render_feature_importance_analysis_tab,
    render_drift_detection_tab,
    render_decision_boundary_tab,
    render_recurring_race_p_tab
)
from dashboard.components.sidebar import render_configuration_sidebar, render_feature_selection_sidebar  # noqa: E402


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
sidebar_config = render_configuration_sidebar()

window_length = sidebar_config["window_length"]
window_before_start = sidebar_config["window_before_start"]
window_after_start = sidebar_config["window_after_start"]
dataset_key = sidebar_config["dataset_key"]
dataset_params = sidebar_config["dataset_params"]
selected_model_class = sidebar_config["selected_model_class"]
model_params = sidebar_config["model_params"]
show_boxplot = sidebar_config["show_boxplot"]


# --- Data Generation ---
@st.cache_data
def generate_data(dataset_name, window_length_val, **kwargs):
    """Cached function to generate data."""
    dataset = DATASETS.get(dataset_name)
    if not dataset:
        st.error(f"Unknown dataset: {dataset_name}")
        return None, None

    gen_params = dataset.get_params()
    gen_params.update(kwargs)

    # Convert window-based parameters to samples if present
    if "n_windows_before" in gen_params:
        gen_params["n_samples_before"] = gen_params.pop("n_windows_before") * window_length_val
    if "n_windows_after" in gen_params:
        gen_params["n_samples_after"] = gen_params.pop("n_windows_after") * window_length_val

    try:
        return dataset.generate(**gen_params)
    except Exception as e:
        st.error(f"Error generating data: {e}")
        return None, None


X, y = generate_data(
    dataset_key,
    window_length,
    **dataset_params
)

if X is not None:
    feature_names = X.columns.tolist()
else:
    feature_names = []

if X is None:
    st.warning("Please upload a CSV file or select a valid dataset to proceed.")
    st.stop()

# --- Feature Selection ---
selected_features = render_feature_selection_sidebar(X)

# Filter X and update feature_names
X = X[selected_features]
feature_names = selected_features

# --- Plot Generation and Capturing (Modified Logic) ---


@st.cache_data(show_spinner="Generating data stream visualizations...")
def generate_and_capture_plots(X, y, window_before_start, window_after_start, window_length, feature_names):
    """Generates all visualization plots and captures them."""
    # Redirect stdout to capture print statements
    stdout_capture = StringIO()
    with contextlib.redirect_stdout(stdout_capture):
        # This function call creates multiple figures and leaves them open
        visualize_data_stream(X, y, window_before_start, window_after_start, window_length, feature_names)

    # Capture the figures and close them immediately
    all_figs = []
    for fig_id in plt.get_fignums():
        fig = plt.figure(fig_id)
        all_figs.append(fig)
        plt.close(fig)  # Close the figure to free up memory

    return all_figs, stdout_capture.getvalue()


all_figs, info_log = generate_and_capture_plots(
    X, y, window_before_start, window_after_start, window_length, feature_names
)

# --- Tabs (Navigation) ---
tabs = [
    "Dataset Selection and Visualization",
    "Drift Detection",
    "Decision Boundary",
    "Feature Importance Analysis",
    "Recurring RACE-P"
]

# Initialize session state for active tab if it doesn't exist
if "active_tab" not in st.session_state:
    st.session_state.active_tab = tabs[0]


# Custom CSS to style radio buttons as tabs
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


css_file = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
load_css(css_file)

# Use radio buttons for navigation to allow state persistence
active_tab = st.radio(
    "Select Analysis Module",
    tabs,
    horizontal=True,
    key="active_tab",
    label_visibility="collapsed"
)

if active_tab == tabs[0]:
    render_data_visualization_tab(X, y, feature_names, all_figs)

elif active_tab == tabs[1]:
    render_drift_detection_tab()

elif active_tab == tabs[2]:
    render_decision_boundary_tab(X, y,
                                 window_before_start=window_before_start,
                                 window_after_start=window_after_start,
                                 window_length=window_length,
                                 model_class=selected_model_class,
                                 model_params=model_params)

elif active_tab == tabs[3]:
    render_feature_importance_analysis_tab(X, y, feature_names, show_boxplot,
                                           window_before_start, window_after_start, window_length,
                                           model_class=selected_model_class,
                                           model_params=model_params)

elif active_tab == tabs[4]:
    render_recurring_race_p_tab()

st.markdown("---")
st.markdown("Developed as part of the xAI and Data Analysis Tools for Drift Detection project.")
