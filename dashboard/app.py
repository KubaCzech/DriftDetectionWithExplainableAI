import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
from io import StringIO
import contextlib

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.datasets import (
    DatasetName,
    generate_custom_3d_drift_data,
    generate_custom_normal_data,
    generate_hyperplane_data,
    generate_sea_drift_data,
)
from src.feature_importance.feature_importance import (
    FeatureImportanceMethod,
    visualize_data_stream,
    analyze_data_drift,
    analyze_concept_drift,
    analyze_predictive_importance_shift,
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
    dataset_name = st.selectbox(
        "Choose a Dataset",
        options=DatasetName.all_available(),
        format_func=lambda x: x.replace("_", " ").title(),
        help="Select the synthetic dataset to analyze."
    )

    # 2. Toggle for Boxplots
    show_boxplot = st.checkbox(
        "Show Importance Boxplots",
        value=True,
        help="Display boxplots for feature importance distributions."
    )

    st.markdown("---")
    st.info("Adjust the settings above and click the 'Run Analysis' button in the main panel.")


# --- Data Generation ---
@st.cache_data
def generate_data(dataset):
    """Cached function to generate data."""
    gen_params = {
        "n_samples_before": 500,
        "n_samples_after": 500,
        "random_seed": 42
    }
    if dataset == DatasetName.CUSTOM_NORMAL:
        return generate_custom_normal_data(**gen_params)
    elif dataset == DatasetName.CUSTOM_3D_DRIFT:
        return generate_custom_3d_drift_data(**gen_params)
    elif dataset == DatasetName.SEA_DRIFT:
        return generate_sea_drift_data(**gen_params)
    elif dataset == DatasetName.HYPERPLANE_DRIFT:
        return generate_hyperplane_data(**gen_params)
    else:
        st.error(f"Unknown dataset: {dataset}")
        return None, None, None, None

X, y, drift_point, feature_names = generate_data(dataset_name)

# --- Plot Generation and Capturing (Modified Logic) ---

# Define plot names and their assumed index based on creation order in visualize_data_stream
# I have added the 4th plot back as 'Classification Boundary Plot' (index 3).
PLOT_OPTIONS = {
    "Feature Space Distribution": 0, # Assumed first plot
    "Feature vs Index Plots": 1,     # Assumed second plot
    "Target vs Index Plot": 2,       # Assumed third plot
    "Classification Boundary Plot": 3, # Assumed fourth plot
}
DEFAULT_PLOT = "Feature Space Distribution"

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
        plt.close(fig) # Close the figure to free up memory
        
    return all_figs, stdout_capture.getvalue()

all_figs, info_log = generate_and_capture_plots(X, y, drift_point, feature_names)

# --- Analysis Trigger ---
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

if st.button("🚀 Run Full Analysis", key="run_analysis"):
    st.session_state.analysis_done = True

# --- Tabs ---
tab1, tab2 = st.tabs(["Dataset Selection and Visualization", "Feature Importance Analysis"])

with tab1:
    st.header("1. Data Stream Visualization")
    st.markdown("This section visualizes the generated data before and after the drift point.")

    # Display captured print output
    st.text_area("Class Distribution Info", info_log, height=150)

    # New: Plot selection dropdown
    plot_choice = st.selectbox(
        "Select Plot to View",
        options=list(PLOT_OPTIONS.keys()),
        index=list(PLOT_OPTIONS.keys()).index(DEFAULT_PLOT),
        help="Choose one of the visualizations of the data stream. **Note**: The exact plots available depend on the data generation function."
    )

    # Display the selected plot
    selected_index = PLOT_OPTIONS[plot_choice]
    
    if selected_index < len(all_figs):
        st.subheader(f"Plot: {plot_choice}")
        st.pyplot(all_figs[selected_index])
    else:
        st.warning(f"The selected plot ('{plot_choice}') is not available for the currently selected dataset type. Showing the first available plot instead.")
        if all_figs:
            st.subheader(f"Plot: {list(PLOT_OPTIONS.keys())[0]}")
            st.pyplot(all_figs[0])
        else:
            st.error("No visualization plots were generated.")


with tab2:
    st.header("2. Drift Analysis Configuration & Results")

    # Move: Select Feature Importance Method into this tab
    importance_method = st.selectbox(
        "Choose a Feature Importance Method",
        options=FeatureImportanceMethod.all_available(),
        format_func=lambda x: x.upper(),
        help="Select the method to explain the drift."
    )

    if st.session_state.analysis_done:
        st.markdown(f"Running analysis with **{importance_method.upper()}** method.")

        # --- Analysis Step 1: Data Drift ---
        with st.container():
            st.subheader("Analysis 1: Data Drift - P(X) Changes")
            st.markdown("""
            This analysis trains a model to distinguish between the 'before' and 'after' periods using **only the input features (X)**.
            High accuracy indicates that the feature distribution P(X) has changed significantly.
            The feature importance scores show which features contributed most to this change.
            """)
            with st.spinner(f'Running Data Drift analysis with {importance_method.upper()}...'):
                stdout_capture = StringIO()
                with contextlib.redirect_stdout(stdout_capture):
                    analyze_data_drift(
                        X, y, drift_point, feature_names,
                        importance_method=importance_method,
                        show_importance_boxplot=show_boxplot
                    )
                st.text_area("Data Drift Analysis Log", stdout_capture.getvalue(), height=200)
                figs = [plt.figure(i) for i in plt.get_fignums()]
                for fig in figs:
                    st.pyplot(fig)
                    plt.close(fig)

        # --- Analysis Step 2: Concept Drift ---
        with st.container():
            st.subheader("Analysis 2: Concept Drift - P(Y|X) Changes")
            st.markdown("""
            This analysis trains a model to distinguish between the 'before' and 'after' periods using **both input features (X) and the target variable (Y)**.
            If the 'Y' feature has high importance, it suggests that the relationship between features and the target has changed (i.e., concept drift).
            """)
            with st.spinner(f'Running Concept Drift analysis with {importance_method.upper()}...'):
                stdout_capture = StringIO()
                with contextlib.redirect_stdout(stdout_capture):
                    analyze_concept_drift(
                        X, y, drift_point, feature_names,
                        importance_method=importance_method,
                        show_importance_boxplot=show_boxplot
                    )
                st.text_area("Concept Drift Analysis Log", stdout_capture.getvalue(), height=200)
                figs = [plt.figure(i) for i in plt.get_fignums()]
                for fig in figs:
                    st.pyplot(fig)
                    plt.close(fig)

        # --- Analysis Step 3: Predictive Power Shift ---
        with st.container():
            st.subheader("Analysis 3: Predictive Power Shift")
            st.markdown("""
            This analysis compares the importance of features for predicting the target variable (Y) in two separate models:
            1.  A model trained **only on 'before' data**.
            2.  A model trained **only on 'after' data**.
            A significant shift in feature importance between the two models indicates concept drift.
            """)
            with st.spinner(f'Running Predictive Power Shift analysis with {importance_method.upper()}...'):
                stdout_capture = StringIO()
                with contextlib.redirect_stdout(stdout_capture):
                    analyze_predictive_importance_shift(
                        X, y, drift_point, feature_names,
                        importance_method=importance_method,
                        show_importance_boxplot=show_boxplot
                    )
                st.text_area("Predictive Power Shift Analysis Log", stdout_capture.getvalue(), height=200)
                figs = [plt.figure(i) for i in plt.get_fignums()]
                for fig in figs:
                    st.pyplot(fig)
                    plt.close(fig)

        st.success("✅ Full analysis complete!")
        st.balloons()

    else:
        st.info("Click the 'Run Full Analysis' button to perform the drift explanation.")

st.markdown("---")
st.markdown("Developed as part of the xAI and Data Analysis Tools for Drift Detection project.")
