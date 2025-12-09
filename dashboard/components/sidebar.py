import streamlit as st
import pandas as pd
from src.datasets import DATASETS, reload_datasets, DatasetRegistry
from src.models import MODELS
from dashboard.components.modals.dataset_settings import open_dataset_settings_modal
from dashboard.components.modals.model_settings import open_model_settings_modal


def render_configuration_sidebar():  # noqa: C901
    """
    Renders the main configuration sidebar (Dataset, Model, Global Settings).
    Returns a dictionary containing the configuration.
    """
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

        st.subheader("Analysis Window Selection")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            window_before_start_windows = st.number_input(
                "Before",
                min_value=0,
                value=0,
                help="Starting index for the first analysis window (in number of windows)."
            )
        with col_w2:
            window_after_start_windows = st.number_input(
                "After",
                min_value=0,
                value=1,
                help="Starting index for the second analysis window (in number of windows)."
            )

        # Calculate absolute indices
        window_before_start = window_before_start_windows * window_length
        window_after_start = window_after_start_windows * window_length

        st.subheader("Dataset Selection")

        # 2. Select Dataset
        # Define Import Option
        IMPORT_OPTION = "➕ Import dataset..."

        dataset_options = list(DATASETS.keys()) + [IMPORT_OPTION]

        # Check if we should select a specific dataset (e.g. after import)
        index = 0
        if 'selected_dataset_key' in st.session_state and st.session_state.selected_dataset_key in dataset_options:
            index = dataset_options.index(st.session_state.selected_dataset_key)

        col_ds1, col_ds2 = st.columns([0.75, 0.25], vertical_alignment="bottom")
        with col_ds1:
            dataset_key = st.selectbox(
                "Choose a Dataset",
                options=dataset_options,
                index=index,
                format_func=lambda x: x if x == IMPORT_OPTION else DATASETS[x].display_name,
                help="Select the synthetic dataset to analyze or import a new one."
            )

        if dataset_key == IMPORT_OPTION:
            @st.dialog("Import Dataset")
            def open_import_dataset_modal():
                st.write("Upload a CSV file to import a new dataset.")
                uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

                if uploaded_file is not None:
                    try:
                        # Read header only
                        uploaded_file.seek(0)
                        df_preview = pd.read_csv(uploaded_file, nrows=0)
                        columns = df_preview.columns.tolist()
                    except Exception as e:
                        st.error(f"Error reading CSV: {e}")
                        return

                    dataset_name = st.text_input("Dataset Name", value=uploaded_file.name.replace(".csv", ""))

                    # Default target is the last column
                    default_target_idx = len(columns) - 1 if columns else 0

                    target_col = st.selectbox(
                        "Target Variable",
                        options=columns,
                        index=default_target_idx,
                        help="Select the column containing the target variable."
                    )

                    # Default features are all except target
                    available_features = [c for c in columns if c != target_col]
                    features = st.multiselect(
                        "Features to Include",
                        options=available_features,
                        default=available_features,
                        help="Select the features to be used for the dataset."
                    )

                    if st.button("Import Dataset"):
                        if not dataset_name:
                            st.error("Please provide a dataset name.")
                            return
                        if not features:
                            st.error("Please select at least one feature.")
                            return

                        # Save
                        registry = DatasetRegistry()
                        registry.save_dataset(dataset_name, uploaded_file, target_col, features)

                        # Reload datasets
                        reload_datasets()

                        # Update session state to select the new dataset
                        st.session_state.selected_dataset_key = dataset_name
                        st.success(f"Dataset '{dataset_name}' imported successfully!")
                        st.rerun()

            open_import_dataset_modal()
            # Stop execution so the rest of the app doesn't try to render with "Import dataset..." key
            st.stop()
        else:
            st.session_state.selected_dataset_key = dataset_key

        selected_dataset = DATASETS[dataset_key]

        # Add delete option for imported datasets
        # Check if it's an imported dataset by checking registry
        registry = DatasetRegistry()
        if registry.get_dataset_info(dataset_key):
            if st.sidebar.button("🗑️ Delete Dataset", help="Permanently remove this imported dataset."):
                registry.delete_dataset(dataset_key)
                reload_datasets()
                st.session_state.selected_dataset_key = list(DATASETS.keys())[0]  # specific fallback or let it reset
                st.rerun()

        # Initialize session state for parameters if not exists
        if 'dataset_params' not in st.session_state:
            st.session_state.dataset_params = {}

        with col_ds2:
            if st.button("⚙️", key="dataset_settings_btn", help="Configure dataset settings", width="stretch"):
                # Clear temporary settings widgets to ensure a fresh start from committed params
                keys_to_clear = [k for k in st.session_state.keys() if k.startswith("temp_dataset_param_")]
                for k in keys_to_clear:
                    del st.session_state[k]
                open_dataset_settings_modal(selected_dataset, window_length, dataset_key)

        # 5. Model Selection
        st.subheader("Model Configuration")

        col_m1, col_m2 = st.columns([0.75, 0.25], vertical_alignment="bottom")
        with col_m1:
            model_key = st.selectbox(
                "Choose a Model",
                options=list(MODELS.keys()),
                format_func=lambda x: MODELS[x]().display_name,
                help="Select the machine learning model to use for drift detection."
            )

        selected_model_class = MODELS[model_key]

        # Initialize session state for model parameters if not exists
        if 'model_params' not in st.session_state:
            st.session_state.model_params = {}

        with col_m2:
            if st.button("⚙️", key="model_settings_btn", help="Configure model settings", width="stretch"):
                # Clear temporary settings widgets to ensure a fresh start
                keys_to_clear = [k for k in st.session_state.keys() if k.startswith("temp_model_param_")]
                for k in keys_to_clear:
                    del st.session_state[k]
                open_model_settings_modal(selected_model_class)

        # Ensure dataset_params and model_params are populated for the main app execution
        dataset_params = st.session_state.dataset_params
        model_params = st.session_state.model_params

    return {
        "window_length": window_length,
        "window_before_start": window_before_start,
        "window_after_start": window_after_start,
        "dataset_key": dataset_key,
        "dataset_params": dataset_params,
        "selected_model_class": selected_model_class,
        "model_params": model_params,
        "selected_features": st.session_state.get(f"selected_features_{dataset_key}", [])
    }
