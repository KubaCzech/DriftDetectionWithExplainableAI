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

        # 4. Dataset Settings Modal
        @st.dialog("Dataset Settings")
        def open_dataset_settings_modal():
            st.write(f"Configure advanced settings for **{selected_dataset.display_name}**.")

            # 3. Settings Preset Selection
            available_settings = selected_dataset.get_available_settings()
            # Initialize session state for selected setting if not exists
            if 'selected_setting' not in st.session_state:
                st.session_state.selected_setting = None
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

            # --- Feature Selection ---
            st.subheader("Feature Selection")
            
            # Get available features
            preview_features = []
            try:
                # Check if imported
                registry_info = registry.get_dataset_info(selected_dataset.name)
                if registry_info:
                    preview_features = registry_info.get("selected_features", [])
                    # If empty (old format?), try getting target and generic features logic if available, 
                    # but ImportedCSVDataset usually saves 'selected_features'.
                else:
                    # Synthetic: generate small sample to get columns
                    # Prepare params based on current temp or session params
                    gen_params = st.session_state.dataset_params.copy() if st.session_state.dataset_params else selected_dataset.get_params()
                    
                    # Merge with temporary changes if any (though schema rendering updates state directly on change/rerun, 
                    # temp_dataset_params is local. But here we want preview based on currently applied or about-to-be-applied?)
                    # Usually we want based on current applied params + defaults.
                    
                    # Handle window->sample conversion locally for preview
                    # Note: We need to handle keys that might be 'n_windows_before' in params but 'n_samples_before' in generate.
                    if "n_windows_before" in gen_params:
                        gen_params["n_samples_before"] = gen_params.pop("n_windows_before") * window_length
                    if "n_windows_after" in gen_params:
                        gen_params["n_samples_after"] = gen_params.pop("n_windows_after") * window_length
                        
                    # Override for minimal generation
                    gen_params["n_samples_before"] = 5
                    gen_params["n_samples_after"] = 0
                    
                    X_preview, _ = selected_dataset.generate(**gen_params)
                    if X_preview is not None:
                        preview_features = X_preview.columns.tolist()
            except Exception as e:
                st.warning(f"Could not preview features: {e}")
                preview_features = []

            # Initialize selected features for this dataset if not in state
            feature_key = f"selected_features_{dataset_key}"
            if feature_key not in st.session_state:
                st.session_state[feature_key] = preview_features

            if preview_features:
                selected_feat = st.multiselect(
                    "Include Features",
                    options=preview_features,
                    default=st.session_state[feature_key] if st.session_state[feature_key] else preview_features,
                    key=f"multiselect_{dataset_key}" # Use a distinct key to avoid conflicts, synched manually below or just use simple key
                )
                # Note: multiselect with key auto-updates session_state[key]. 
                # But we want to persist it across reloads in a stable key.
                # Let's just use the return value and update the main state key.
                st.session_state[feature_key] = selected_feat
            else:
                st.info("No features available to select.")
                st.session_state[feature_key] = []

            st.markdown("---")

            if st.button("Apply Dataset Changes"):
                # Update session state directly
                st.session_state.dataset_params = temp_dataset_params.copy()
                st.rerun()

        with col_ds2:
            if st.button("⚙️", key="dataset_settings_btn", help="Configure dataset settings", use_container_width=True):
                open_dataset_settings_modal()

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
        # Instantiate temporarily to get schema/settings
        temp_model = selected_model_class()

        # Initialize session state for model parameters if not exists
        if 'model_params' not in st.session_state:
            st.session_state.model_params = {}

        # 7. Model Settings Modal
        @st.dialog("Model Settings")
        def open_model_settings_modal():
            st.write(f"Configure advanced settings for **{temp_model.display_name}**.")

            # 6. Model Preset Selection
            model_available_settings = temp_model.get_available_settings()

            # Initialize session state for selected model setting if not exists
            if 'selected_model_setting' not in st.session_state:
                st.session_state.selected_model_setting = None

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

        with col_m2:
            if st.button("⚙️", key="model_settings_btn", help="Configure model settings", use_container_width=True):
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
        "model_params": model_params,
        "selected_features": st.session_state.get(f"selected_features_{dataset_key}", [])
    }



