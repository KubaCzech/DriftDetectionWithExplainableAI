import streamlit as st


def render_settings_from_schema(schema: list[dict], on_change=None, initial_values=None, force_update=False) -> dict:
    """
    Render Streamlit widgets based on a settings schema.

    Parameters
    ----------
    schema : list[dict]
        A list of dictionaries describing the settings.
    on_change : callable, optional
        Callback function to call when any parameter changes.
    initial_values : dict, optional
        Dictionary of initial values to use instead of defaults.
    force_update : bool, optional
        If True, force update widget values even if they already exist in session state.

    Returns
    -------
    dict
        A dictionary of selected parameter values.
    """
    params = {}

    if not schema:
        return params

    for setting in schema:
        name = setting["name"]
        label = setting["label"]
        help_text = setting.get("help", "")
        default = setting.get("default")
        key = f"param_{name}"
        
        # Use initial value if provided, otherwise use default
        value = initial_values.get(name, default) if initial_values else default
        
        # Initialize session state if not exists, or force update if requested
        if key not in st.session_state:
            st.session_state[key] = value
        elif force_update and initial_values and name in initial_values:
            # Only update if force_update is True (when preset changes)
            st.session_state[key] = value

        if setting["type"] == "int":
            params[name] = st.number_input(
                label,
                min_value=setting.get("min_value"),
                max_value=setting.get("max_value"),
                step=setting.get("step", 1),
                help=help_text,
                key=key,
                on_change=on_change
            )
        elif setting["type"] == "float":
            params[name] = st.number_input(
                label,
                min_value=setting.get("min_value"),
                max_value=setting.get("max_value"),
                step=setting.get("step", 0.1),
                help=help_text,
                key=key,
                on_change=on_change
            )
        elif setting["type"] == "text":
            params[name] = st.text_input(
                label,
                help=help_text,
                key=key,
                on_change=on_change
            )
        elif setting["type"] == "bool":
            params[name] = st.checkbox(
                label,
                help=help_text,
                key=key,
                on_change=on_change
            )
        elif setting["type"] == "file":
            params[name] = st.file_uploader(
                label,
                type=setting.get("allowed_types"),
                help=help_text,
                key=key,
                on_change=on_change
            )

    return params
