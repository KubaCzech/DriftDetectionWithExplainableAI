import streamlit as st
from src.datasets import DatasetRegistry
from dashboard.components.settings import render_settings_from_schema
from dashboard.utils import get_dataset_settings_schema


def _get_preview_features(registry, selected_dataset, temp_dataset_params, window_length):
    """
    Helper to get available features for preview.
    """
    try:
        # Check if imported
        registry_info = registry.get_dataset_info(selected_dataset.name)
        if registry_info:
            return registry_info.get("selected_features", [])

        # Synthetic: generate small sample to get columns
        if st.session_state.dataset_params:
            gen_params = st.session_state.dataset_params.copy()
        else:
            gen_params = selected_dataset.get_params()

        if temp_dataset_params:
            gen_params.update(temp_dataset_params)

        # Handle window->sample conversion locally for preview
        if "n_windows_before" in gen_params:
            gen_params["n_samples_before"] = gen_params.pop("n_windows_before") * window_length
        if "n_windows_after" in gen_params:
            gen_params["n_samples_after"] = gen_params.pop("n_windows_after") * window_length

        # Override for minimal generation
        gen_params["n_samples_before"] = 5
        gen_params["n_samples_after"] = 5

        X_preview, _ = selected_dataset.generate(**gen_params)
        if X_preview is not None:
            return X_preview.columns.tolist()
    except Exception as e:
        st.warning(f"Could not preview features: {e}")
    return []


def _render_feature_selection(selected_dataset, dataset_key, registry, temp_dataset_params, window_length):
    """
    Renders feature selection UI.
    """
    st.subheader("Feature Selection")

    preview_features = _get_preview_features(registry, selected_dataset, temp_dataset_params, window_length)

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


def _on_setting_change_handler(available_settings):
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


@st.dialog("Dataset Settings")
def open_dataset_settings_modal(selected_dataset, window_length, dataset_key):
    st.write(f"Configure advanced settings for **{selected_dataset.display_name}**.")

    # 3. Settings Preset Selection
    available_settings = selected_dataset.get_available_settings()

    # Identify default preset if exists
    default_preset_name = None
    if available_settings:
        for name, params in available_settings.items():
            if params.get('default'):
                default_preset_name = name
                break

    # Clean available_settings by removing 'default' key from all params
    cleaned_available_settings = {}
    if available_settings:
        for name, params in available_settings.items():
            cleaned_params = params.copy()
            cleaned_params.pop('default', None)
            cleaned_available_settings[name] = cleaned_params

    # Initialize session state for selected setting if not exists
    if 'selected_setting' not in st.session_state:
        st.session_state.selected_setting = None

    # Reverse Lookup: If this is a fresh open (no temp keys), sync selected_setting with current params
    # This recovers the "Preset Name" if the user had previously applied that preset, 
    # or recovers "Not selected" if they had custom params.
    has_temp_keys = any(k.startswith("temp_dataset_param_") for k in st.session_state.keys())
    
    if not has_temp_keys and st.session_state.dataset_params:
        # Assume "Not selected" unless we find a match
        st.session_state.selected_setting = None
        
        current_params = st.session_state.dataset_params
        for name, preset_params in cleaned_available_settings.items():
            # Compare current params with preset params
            # We must safeguard against type differences if possible, but exact match is baseline
            is_match = True
            for k, v in preset_params.items():
                if current_params.get(k) != v:
                    is_match = False
                    break
            
            # Also check if current_params has extra keys that preset doesn't? 
            # Usually dataset_params might have 'selected_features' or others? 
            # No, 'dataset_params' usually only contains the generator params. 
            # But let's stick to checking if all preset params match.
            
            if is_match:
                st.session_state.selected_setting = name
                break

    # Auto-select default if nothing selected yet and default exists
    # Only if we don't already have some params set (e.g. custom ones)
    if st.session_state.selected_setting is None and default_preset_name and not st.session_state.dataset_params:
        st.session_state.selected_setting = default_preset_name
        st.session_state.dataset_params = cleaned_available_settings[default_preset_name].copy()
        st.session_state.force_update_widgets = True

    # Track if we need to force update the widgets (when preset changes)
    if 'force_update_widgets' not in st.session_state:
        st.session_state.force_update_widgets = False

    # Early detection of modifications to preset
    # Check if we have an active preset and are NOT currently forcing an update
    if st.session_state.selected_setting and st.session_state.selected_setting in cleaned_available_settings and not st.session_state.force_update_widgets:
        current_preset_params = cleaned_available_settings[st.session_state.selected_setting]
        is_modified = False
        
        # We need to check the widget states. The helper 'render_settings_from_schema' uses 'temp_dataset_' prefix.
        # But we don't have the param names handy unless we iterate the preset params.
        # Assuming the preset params map 1:1 to the widget keys.
        
        for k, v in current_preset_params.items():
            # The helper 'render_settings_from_schema' uses '{key_prefix}param_{name}' format.
            widget_key = f"temp_dataset_param_{k}"
            # Only check if the widget key exists in session state (meaning it has been rendered/interacted with)
            if widget_key in st.session_state:
                # Compare values. 
                if st.session_state[widget_key] != v:
                    is_modified = True
                    break
        
        if is_modified:
            st.session_state.selected_setting = None
            st.session_state.setting_selectbox = "Not selected"
            # No need to rerun, we just updated the state before rendering the selectbox.

    # If there are available settings, show the dropdown
    if cleaned_available_settings:
        setting_options = list(cleaned_available_settings.keys())

        # Only add "Not selected" if currently in that state
        if st.session_state.selected_setting is None:
            setting_options = ["Not selected"] + setting_options

        # Initialize the selectbox session state if not exists
        if 'setting_selectbox' not in st.session_state:
            st.session_state.setting_selectbox = "Not selected"

        # Update selectbox value based on selected_setting
        if st.session_state.selected_setting in cleaned_available_settings:
            st.session_state.setting_selectbox = st.session_state.selected_setting
        elif st.session_state.selected_setting is None:
            st.session_state.setting_selectbox = "Not selected"

        st.selectbox(
            "Select Preset Settings",
            options=setting_options,
            key='setting_selectbox',
            on_change=_on_setting_change_handler,
            args=(cleaned_available_settings,),
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
    registry = DatasetRegistry()
    _render_feature_selection(selected_dataset, dataset_key, registry, temp_dataset_params, window_length)

    st.markdown("---")

    if st.button("Apply Dataset Changes"):
        # Update session state directly
        st.session_state.dataset_params = temp_dataset_params.copy()
        st.rerun()
