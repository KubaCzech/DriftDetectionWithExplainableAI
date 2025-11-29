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
from dashboard.components.settings import render_settings_from_schema  # noqa: E402
from dashboard.tabs import (  # noqa: E402
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

    # 1. Global Window Settings
    st.subheader("Global Settings")
    window_length = st.number_input(
        "Window Length (Samples)",
        min_value=1,
        value=1000,
        help="Length of the analysis window in samples."
    )
    
    st.markdown("---")

    st.subheader("Analysis Window Selection")
    window_before_start_windows = st.number_input(
        "Window Before Start (Windows)",
        min_value=0,
        value=0,
        help="Starting index for the first analysis window (in number of windows)."
    )
    window_after_start_windows = st.number_input(
        "Window After Start (Windows)",
        min_value=0,
        value=1,
        help="Starting index for the second analysis window (in number of windows)."
    )

    # Calculate absolute indices
    window_before_start = window_before_start_windows * window_length
    window_after_start = window_after_start_windows * window_length

    st.markdown("---")

    # 2. Select Dataset
    dataset_key = st.selectbox(
        "Choose a Dataset",
        options=list(DATASETS.keys()),
        format_func=lambda x: DATASETS[x].display_name,
        help="Select the synthetic dataset to analyze."
    )

    selected_dataset = DATASETS[dataset_key]

    # 3. Settings Preset Selection
    available_settings = selected_dataset.get_available_settings()
    
    # Initialize session state for selected setting if not exists
    if 'selected_setting' not in st.session_state:
        st.session_state.selected_setting = None
    
    # Initialize session state for parameters if not exists
    if 'dataset_params' not in st.session_state:
        st.session_state.dataset_params = {}
    
    # Track if we need to force update the widgets (when preset changes)
    if 'force_update_widgets' not in st.session_state:
        st.session_state.force_update_widgets = False
    
    # If there are available settings, show the dropdown
    if available_settings:
        setting_options = ["Not selected"] + list(available_settings.keys())
        
        # Initialize the selectbox session state if not exists
        if 'setting_selectbox' not in st.session_state:
            st.session_state.setting_selectbox = "Not selected"
        
        # Update selectbox value based on selected_setting
        if st.session_state.selected_setting in available_settings:
            st.session_state.setting_selectbox = st.session_state.selected_setting
        elif st.session_state.selected_setting is None:
            st.session_state.setting_selectbox = "Not selected"
        
        def on_setting_change():
            """Callback when settings dropdown changes"""
            selected = st.session_state.setting_selectbox
            if selected == "Not selected":
                st.session_state.selected_setting = None
                st.session_state.dataset_params = {}
            else:
                st.session_state.selected_setting = selected
                # Update the parameter values in session state
                # We need to be careful here if presets are in samples but UI is in windows
                # For now, assuming presets might need manual adjustment or we just load them as is
                # But wait, if we change the schema, we need to adapt the presets too?
                # The user requirement says: "When choosing window before drift and window after drift, you will choose window numbers instead of sample numbers."
                # And "For synthetic datasets, size of data before drift and after drift should be specified in the dashboard as number of windows instead of number of samples."
                
                # Let's assume presets in code are still in samples (as they are in the dataset classes).
                # We might need to convert them to windows for display if we want them to work nicely with the new schema.
                # However, modifying the presets logic is complex. 
                # Let's just load them. If the schema expects 'n_windows_before', and preset has 'n_samples_before', 
                # render_settings_from_schema might ignore it or we need to map it.
                
                st.session_state.dataset_params = available_settings[selected].copy()
                # Set flag to force update widgets
                st.session_state.force_update_widgets = True
        
        st.selectbox(
            "Select Preset Settings",
            options=setting_options,
            key='setting_selectbox',
            on_change=on_setting_change,
            help="Choose a preset configuration or customize parameters manually."
        )

    # 4. Render Dataset Settings
    if (selected_dataset.name != "custom_normal" and
            selected_dataset.name != "custom_3d_drift" and
            selected_dataset.name != "sea_drift"):
        st.subheader(f"{selected_dataset.display_name} Settings")
        
        # Standard rendering for non-synthetic datasets
        schema_to_render = selected_dataset.get_settings_schema()
    else:
        st.subheader(f"{selected_dataset.display_name} Settings")
        # Intercept schema for synthetic datasets
        original_schema = selected_dataset.get_settings_schema()
        schema_to_render = []
        for item in original_schema:
            new_item = item.copy()
            if item["name"] == "n_samples_before":
                new_item["name"] = "n_windows_before"
                new_item["label"] = "Number of Windows Before Drift"
                new_item["default"] = int(item["default"] / window_length) if item["default"] else 1
                new_item["min_value"] = 1
                new_item["step"] = 1
                new_item["help"] = "Number of windows generated before the concept drift occurs."
            elif item["name"] == "n_samples_after":
                new_item["name"] = "n_windows_after"
                new_item["label"] = "Number of Windows After Drift"
                new_item["default"] = int(item["default"] / window_length) if item["default"] else 1
                new_item["min_value"] = 1
                new_item["step"] = 1
                new_item["help"] = "Number of windows generated after the concept drift occurs."
            schema_to_render.append(new_item)

    def on_param_change():
        """Callback when any parameter changes - mark as custom"""
        if available_settings:
            st.session_state.selected_setting = None
            st.session_state.dataset_params = {}

    dataset_params = render_settings_from_schema(
        schema_to_render,
        on_change=on_param_change,
        initial_values=st.session_state.dataset_params if st.session_state.dataset_params else None,
        force_update=st.session_state.force_update_widgets
    )
    
    # Post-processing: Convert windows back to samples for synthetic datasets
    if "n_windows_before" in dataset_params:
        dataset_params["n_samples_before"] = dataset_params.pop("n_windows_before") * window_length
    
    if "n_windows_after" in dataset_params:
        dataset_params["n_samples_after"] = dataset_params.pop("n_windows_after") * window_length
    
    # Reset the force update flag after rendering
    st.session_state.force_update_widgets = False

    # 5. Toggle for Boxplots
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
        return None, None

    gen_params = dataset.get_params()
    gen_params.update(kwargs)

    try:
        return dataset.generate(**gen_params)
    except Exception as e:
        st.error(f"Error generating data: {e}")
        return None, None


X, y = generate_data(
    dataset_key,
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
with st.sidebar:
    st.markdown("---")
    st.subheader("Feature Selection")
    
    selected_features = []
    if feature_names:
        st.write("Select features to include in the analysis:")
        for feature in feature_names:
            if st.checkbox(feature, value=True, key=f"feature_select_{feature}"):
                selected_features.append(feature)
        
        if not selected_features:
            st.warning("Please select at least one feature.")
            st.stop()
            
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


all_figs, info_log = generate_and_capture_plots(X, y, window_before_start, window_after_start, window_length, feature_names)

# --- Tabs ---
tab1, tab2 = st.tabs(["Dataset Selection and Visualization", "Feature Importance Analysis"])

with tab1:
    render_data_visualization_tab(X, y, feature_names, all_figs)

with tab2:
    render_feature_importance_analysis_tab(X, y, feature_names, show_boxplot,
                                           window_before_start, window_after_start, window_length)

st.markdown("---")
st.markdown("Developed as part of the xAI and Data Analysis Tools for Drift Detection project.")
