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

from src.feature_importance.feature_importance import (
    FeatureImportanceMethod,
    visualize_data_stream,
    compute_data_drift_analysis,
    compute_concept_drift_analysis,
    compute_predictive_importance_shift,
    visualize_data_drift_analysis,
    visualize_concept_drift_analysis,
    visualize_predictive_importance_shift,
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

# Define plot names and their assumed index based on creation
# order in visualize_data_stream
PLOT_OPTIONS = {
    "Feature Space Distribution for particular features": 0,  # Assumed first plot
    "Feature vs Index Plots": 1,      # Assumed second plot
    "Target vs Index Plot": 2,        # Assumed third plot
    "Feature Space Distribution": 3,  # Assumed fourth plot
}
DEFAULT_PLOT = "Feature Space Distribution for particular features"

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
    st.header("1. Data Stream Visualization")
    st.markdown("This section visualizes the generated data before and after the drift point.")

    # Plot selection dropdown
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

    # Select Feature Importance Method
    importance_method = st.selectbox(
        "Choose a Feature Importance Method",
        options=FeatureImportanceMethod.all_available(),
        format_func=lambda x: x.upper(),
        help="Select the method to explain the drift."
    )

    # Analysis type selector dropdown
    ANALYSIS_OPTIONS = {
        "Concept Drift - P(Y|X) Changes": "concept_drift",
        "Data Drift - P(X) Changes": "data_drift",
        "Predictive Power Shift": "predictive_shift"
    }
    
    analysis_choice = st.selectbox(
        "Select Analysis Type",
        options=list(ANALYSIS_OPTIONS.keys()),
        index=0,  # Default to Concept Drift (P(Y|X))
        help="Choose which drift analysis to display."
    )
    
    selected_analysis = ANALYSIS_OPTIONS[analysis_choice]

    st.markdown(f"Running **{analysis_choice}** analysis with **{importance_method.upper()}** method.")

    # --- Conditional Analysis Display ---
    
    if selected_analysis == "data_drift":
        # --- Analysis: Data Drift ---
        with st.container():
            st.subheader("Analysis: Data Drift - P(X) Changes")
            st.markdown("""
            This analysis trains a model to distinguish between the 'before' and 'after' periods using **only the input features (X)**.
            High accuracy indicates that the feature distribution P(X) has changed significantly.
            The feature importance scores show which features contributed most to this change.
            """)
            with st.spinner(f'Running Data Drift analysis with {importance_method.upper()}...'):
                # Compute the analysis results
                data_drift_result = compute_data_drift_analysis(
                    X, y, drift_point, feature_names,
                    importance_method=importance_method
                )

                # Display the importance table
                st.markdown("#### Feature Importance Summary")
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Mean Importance': data_drift_result['importance_mean'],
                    'Std Deviation': data_drift_result['importance_std']
                })
                importance_df = importance_df.sort_values('Mean Importance', ascending=False)
                st.dataframe(
                    importance_df.style.format({
                        'Mean Importance': '{:.4f}',
                        'Std Deviation': '{:.4f}'
                    }),
                    use_container_width=True
                )

                # Display visualizations
                stdout_capture = StringIO()
                with contextlib.redirect_stdout(stdout_capture):
                    visualize_data_drift_analysis(
                        data_drift_result, feature_names,
                        show_boxplot=show_boxplot
                    )
                figs = [plt.figure(i) for i in plt.get_fignums()]
                for fig in figs:
                    st.pyplot(fig)
                    plt.close(fig)

    elif selected_analysis == "concept_drift":
        # --- Analysis: Concept Drift ---
        with st.container():
            st.subheader("Analysis: Concept Drift - P(Y|X) Changes")
            st.markdown("""
            This analysis trains a model to distinguish between the 'before' and 'after' periods using **both input features (X) and the target variable (Y)**.
            If the 'Y' feature has high importance, it suggests that the relationship between features and the target has changed (i.e., concept drift).
            """)
            with st.spinner(f'Running Concept Drift analysis with {importance_method.upper()}...'):
                # Compute the analysis results
                concept_drift_result = compute_concept_drift_analysis(
                    X, y, drift_point, feature_names,
                    importance_method=importance_method
                )
                
                # Display the importance table
                st.markdown("#### Feature Importance Summary")
                feature_names_with_y = concept_drift_result['feature_names_with_y']
                importance_df = pd.DataFrame({
                    'Feature': feature_names_with_y,
                    'Mean Importance': concept_drift_result['importance_mean'],
                    'Std Deviation': concept_drift_result['importance_std']
                })
                importance_df = importance_df.sort_values('Mean Importance', ascending=False)
                st.dataframe(
                    importance_df.style.format({
                        'Mean Importance': '{:.4f}',
                        'Std Deviation': '{:.4f}'
                    }),
                    use_container_width=True
                )
                
                # Display visualizations
                stdout_capture = StringIO()
                with contextlib.redirect_stdout(stdout_capture):
                    visualize_concept_drift_analysis(
                        concept_drift_result, feature_names_with_y,
                        show_boxplot=show_boxplot
                    )
                figs = [plt.figure(i) for i in plt.get_fignums()]
                for fig in figs:
                    st.pyplot(fig)
                    plt.close(fig)

    elif selected_analysis == "predictive_shift":
        # --- Analysis: Predictive Power Shift ---
        with st.container():
            st.subheader("Analysis: Predictive Power Shift")
            st.markdown("""
            This analysis compares the importance of features for predicting the target variable (Y) in two separate models:
            1.  A model trained **only on 'before' data**.
            2.  A model trained **only on 'after' data**.
            A significant shift in feature importance between the two models indicates concept drift.
            """)
            with st.spinner(f'Running Predictive Power Shift analysis with {importance_method.upper()}...'):
                # Compute the analysis results
                shift_result = compute_predictive_importance_shift(
                    X, y, drift_point, feature_names,
                    importance_method=importance_method
                )
                
                # Display the importance tables side by side
                st.markdown("#### Feature Importance Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Before Drift**")
                    importance_before_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Mean Importance': shift_result['fi_before']['importances_mean'],
                        'Std Deviation': shift_result['fi_before']['importances_std']
                    })
                    importance_before_df = importance_before_df.sort_values('Mean Importance', ascending=False)
                    st.dataframe(
                        importance_before_df.style.format({
                            'Mean Importance': '{:.4f}',
                            'Std Deviation': '{:.4f}'
                        }),
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("**After Drift**")
                    importance_after_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Mean Importance': shift_result['fi_after']['importances_mean'],
                        'Std Deviation': shift_result['fi_after']['importances_std']
                    })
                    importance_after_df = importance_after_df.sort_values('Mean Importance', ascending=False)
                    st.dataframe(
                        importance_after_df.style.format({
                            'Mean Importance': '{:.4f}',
                            'Std Deviation': '{:.4f}'
                        }),
                        use_container_width=True
                    )
                
                # Display visualizations
                stdout_capture = StringIO()
                with contextlib.redirect_stdout(stdout_capture):
                    visualize_predictive_importance_shift(
                        shift_result, feature_names,
                        show_boxplot=show_boxplot
                    )
                figs = [plt.figure(i) for i in plt.get_fignums()]
                for fig in figs:
                    st.pyplot(fig)
                    plt.close(fig)

    st.success("✅ Analysis complete!")

st.markdown("---")
st.markdown("Developed as part of the xAI and Data Analysis Tools for Drift Detection project.")
