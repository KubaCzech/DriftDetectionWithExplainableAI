import streamlit as st
from src.datasets import DatasetRegistry
from dashboard.components.settings import render_settings_from_schema
from dashboard.utils import get_dataset_settings_schema

@st.dialog("Dataset Settings")
def open_dataset_settings_modal(selected_dataset, window_length, dataset_key):
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
    
    registry = DatasetRegistry()
    
    # Get available features
    preview_features = []
    try:
        # Check if imported
        registry_info = registry.get_dataset_info(selected_dataset.name)
        if registry_info:
            preview_features = registry_info.get("selected_features", [])
        else:
            # Synthetic: generate small sample to get columns
            # Prepare params based on current temp or session params
            gen_params = st.session_state.dataset_params.copy() if st.session_state.dataset_params else selected_dataset.get_params()
            
            # Feature selection preview based on params about to be applied? 
            # Or currently applied? Using current state for simplicity, or 
            # ideally the temp params if we want to reflect "what if".
            # Using temp_dataset_params because if parameters change columns (rare but possible), preview should update?
            # But schema rendering returns temp_dataset_params.
            # Let's use temp_dataset_params if available, else defaults.
            gen_params.update(temp_dataset_params)

            # Handle window->sample conversion locally for preview
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
            key=f"multiselect_{dataset_key}" 
        )
        st.session_state[feature_key] = selected_feat
    else:
        st.info("No features available to select.")
        st.session_state[feature_key] = []

    st.markdown("---")

    if st.button("Apply Dataset Changes"):
        # Update session state directly
        st.session_state.dataset_params = temp_dataset_params.copy()
        st.rerun()
