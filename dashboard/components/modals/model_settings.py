import streamlit as st
from dashboard.components.settings import render_settings_from_schema


def _on_model_setting_change_handler(model_available_settings):
    """Callback when model settings dropdown changes"""
    selected = st.session_state.model_setting_selectbox
    if selected == "Not selected":
        st.session_state.selected_model_setting = None
        st.session_state.model_params = {}
    else:
        st.session_state.selected_model_setting = selected
        st.session_state.model_params = model_available_settings[selected].copy()
        st.session_state.force_update_model_widgets = True


def _get_cleaned_model_settings(temp_model):
    """Retrieves and cleans available settings from the model."""
    model_available_settings = temp_model.get_available_settings()
    cleaned_available_settings = {}
    default_preset_name = None

    if model_available_settings:
        for name, params in model_available_settings.items():
            if params.get('default'):
                default_preset_name = name
            cleaned_params = params.copy()
            cleaned_params.pop('default', None)
            cleaned_available_settings[name] = cleaned_params

    return cleaned_available_settings, default_preset_name


def _sync_selected_model_setting(cleaned_available_settings):
    """Syncs selected setting with current params if fresh open."""
    has_temp_keys = any(k.startswith("temp_model_param_") for k in st.session_state.keys())

    if not has_temp_keys and st.session_state.model_params:
        st.session_state.selected_model_setting = None
        current_params = st.session_state.model_params
        for name, preset_params in cleaned_available_settings.items():
            is_match = True
            for k, v in preset_params.items():
                if current_params.get(k) != v:
                    is_match = False
                    break
            if is_match:
                st.session_state.selected_model_setting = name
                break


def _check_model_preset_modification(cleaned_available_settings):
    """Checks if the current preset has been modified."""
    if (st.session_state.selected_model_setting and
            st.session_state.selected_model_setting in cleaned_available_settings and
            not st.session_state.force_update_model_widgets):

        current_preset_params = cleaned_available_settings[st.session_state.selected_model_setting]
        is_modified = False

        for k, v in current_preset_params.items():
            widget_key = f"temp_model_param_{k}"
            if widget_key in st.session_state:
                if st.session_state[widget_key] != v:
                    is_modified = True
                    break

        if is_modified:
            st.session_state.selected_model_setting = None
            st.session_state.model_setting_selectbox = "Not selected"


def _render_model_preset_selectbox(cleaned_available_settings):
    """Renders the model preset dropdown."""
    if not cleaned_available_settings:
        return

    model_setting_options = list(cleaned_available_settings.keys())

    if st.session_state.selected_model_setting is None:
        model_setting_options = ["Not selected"] + model_setting_options

    if 'model_setting_selectbox' not in st.session_state:
        st.session_state.model_setting_selectbox = "Not selected"

    if st.session_state.selected_model_setting in cleaned_available_settings:
        st.session_state.model_setting_selectbox = st.session_state.selected_model_setting
    elif st.session_state.selected_model_setting is None:
        st.session_state.model_setting_selectbox = "Not selected"

    st.selectbox(
        "Select Model Preset",
        options=model_setting_options,
        key='model_setting_selectbox',
        on_change=_on_model_setting_change_handler,
        args=(cleaned_available_settings,),
        help="Choose a preset configuration for the model."
    )


def _render_model_preset_selection(temp_model):
    """Renders the dropdown for selecting model presets."""
    cleaned_available_settings, default_preset_name = _get_cleaned_model_settings(temp_model)

    # Initialize session state for selected model setting if not exists
    if 'selected_model_setting' not in st.session_state:
        st.session_state.selected_model_setting = None

    _sync_selected_model_setting(cleaned_available_settings)

    # Auto-select default if nothing selected yet and default exists
    if (st.session_state.selected_model_setting is None and
            default_preset_name and
            not st.session_state.model_params):
        st.session_state.selected_model_setting = default_preset_name
        st.session_state.model_params = cleaned_available_settings[default_preset_name].copy()
        st.session_state.force_update_model_widgets = True

    # Track if we need to force update the model widgets
    if 'force_update_model_widgets' not in st.session_state:
        st.session_state.force_update_model_widgets = False

    _check_model_preset_modification(cleaned_available_settings)
    _render_model_preset_selectbox(cleaned_available_settings)


@st.dialog("Model Settings")
def open_model_settings_modal(selected_model_class):
    # Instantiate temporarily to get schema/settings/display_name
    temp_model = selected_model_class()

    st.write(f"Configure advanced settings for **{temp_model.display_name}**.")

    # 6. Model Preset Selection
    _render_model_preset_selection(temp_model)

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
