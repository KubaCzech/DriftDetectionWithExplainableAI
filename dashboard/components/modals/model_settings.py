import streamlit as st
from dashboard.components.settings import render_settings_from_schema

@st.dialog("Model Settings")
def open_model_settings_modal(selected_model_class):
    # Instantiate temporarily to get schema/settings/display_name
    temp_model = selected_model_class()
    
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
