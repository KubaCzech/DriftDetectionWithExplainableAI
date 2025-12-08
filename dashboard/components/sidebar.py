import streamlit as st
import pandas as pd
from src.datasets import DATASETS, reload_datasets, DatasetRegistry
from src.models import MODELS
from dashboard.components.settings import render_settings_from_schema
from dashboard.utils import get_dataset_settings_schema


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

        st.subheader("Dataset Selection")

        # 2. Select Dataset
        # Define Import Option
        IMPORT_OPTION = "➕ Import dataset..."

        # 2. Select Dataset
        dataset_options = list(DATASETS.keys()) + [IMPORT_OPTION]

        # Check if we should select a specific dataset (e.g. after import)
        index = 0
        if 'selected_dataset_key' in st.session_state and st.session_state.selected_dataset_key in dataset_options:
            index = dataset_options.index(st.session_state.selected_dataset_key)

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

        # 4. Dataset Settings Modal
        @st.dialog("Dataset Settings")
        def open_dataset_settings_modal():
            st.write(f"Configure advanced settings for **{selected_dataset.display_name}**.")

            # --- Dataset Settings ---
            schema_to_render = get_dataset_settings_schema(selected_dataset, window_length)

            # Render dataset settings with temporary key prefix
            initial_dataset_params = st.session_state.dataset_params.copy() if st.session_state.dataset_params else {}

            temp_dataset_params = render_settings_from_schema(
                schema_to_render,
                initial_values=initial_dataset_params,
                key_prefix="temp_dataset_",
                force_update=st.session_state.force_update_widgets
            )

            # Reset force update flag
            if st.session_state.force_update_widgets:
                st.session_state.force_update_widgets = False

            st.markdown("---")

            if st.button("Apply Dataset Changes"):
                # Update session state directly
                st.session_state.dataset_params = temp_dataset_params.copy()
                st.rerun()

        if st.button("Configure Dataset Settings"):
            open_dataset_settings_modal()

        # 5. Model Selection
        st.subheader("Model Configuration")

        model_key = st.selectbox(
            "Choose a Model",
            options=list(MODELS.keys()),
            format_func=lambda x: MODELS[x]().display_name,
            help="Select the machine learning model to use for drift detection."
        )

        selected_model_class = MODELS[model_key]
        # Instantiate temporarily to get schema/settings
        temp_model = selected_model_class()

        # 6. Model Preset Selection
        model_available_settings = temp_model.get_available_settings()

        # Initialize session state for selected model setting if not exists
        if 'selected_model_setting' not in st.session_state:
            st.session_state.selected_model_setting = None

        # Initialize session state for model parameters if not exists
        if 'model_params' not in st.session_state:
            st.session_state.model_params = {}

        # Track if we need to force update the model widgets
        if 'force_update_model_widgets' not in st.session_state:
            st.session_state.force_update_model_widgets = False

        if model_available_settings:
            model_setting_options = ["Not selected"] + list(model_available_settings.keys())

            if 'model_setting_selectbox' not in st.session_state:
                st.session_state.model_setting_selectbox = "Not selected"

            if st.session_state.selected_model_setting in model_available_settings:
                st.session_state.model_setting_selectbox = st.session_state.selected_model_setting
            elif st.session_state.selected_model_setting is None:
                st.session_state.model_setting_selectbox = "Not selected"

            def on_model_setting_change():
                selected = st.session_state.model_setting_selectbox
                if selected == "Not selected":
                    st.session_state.selected_model_setting = None
                    st.session_state.model_params = {}
                else:
                    st.session_state.selected_model_setting = selected
                    st.session_state.model_params = model_available_settings[selected].copy()
                    st.session_state.force_update_model_widgets = True

            st.selectbox(
                "Select Model Preset",
                options=model_setting_options,
                key='model_setting_selectbox',
                on_change=on_model_setting_change,
                help="Choose a preset configuration for the model."
            )

        # 7. Model Settings Modal
        @st.dialog("Model Settings")
        def open_model_settings_modal():
            st.write(f"Configure advanced settings for **{temp_model.display_name}**.")

            # --- Model Settings ---
            model_schema = temp_model.get_settings_schema()

            temp_model_params = render_settings_from_schema(
                model_schema,
                initial_values=st.session_state.model_params if st.session_state.model_params else None,
                key_prefix="temp_model_",
                force_update=st.session_state.force_update_model_widgets
            )

            if st.session_state.force_update_model_widgets:
                st.session_state.force_update_model_widgets = False

            if st.button("Apply Model Changes"):
                # Update session state
                st.session_state.model_params = temp_model_params
                st.rerun()

        if st.button("Configure Model Settings"):
            open_model_settings_modal()

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
        "model_params": model_params
    }


def render_feature_selection_sidebar(X):
    """
    Renders the feature selection sidebar.

    Args:
        X (pd.DataFrame): The dataframe containing generated data.

    Returns:
        list: The list of selected feature names.
    """
    if X is not None:
        feature_names = X.columns.tolist()
    else:
        feature_names = []

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

        return selected_features
