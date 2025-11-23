import streamlit as st


def render_settings_from_schema(schema: list[dict]) -> dict:
    """
    Render Streamlit widgets based on a settings schema.

    Parameters
    ----------
    schema : list[dict]
        A list of dictionaries describing the settings.

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

        if setting["type"] == "int":
            params[name] = st.number_input(
                label,
                min_value=setting.get("min_value"),
                max_value=setting.get("max_value"),
                value=default,
                step=setting.get("step", 1),
                help=help_text
            )
        elif setting["type"] == "float":
            params[name] = st.number_input(
                label,
                min_value=setting.get("min_value"),
                max_value=setting.get("max_value"),
                value=default,
                step=setting.get("step", 0.1),
                help=help_text
            )
        elif setting["type"] == "text":
            params[name] = st.text_input(
                label,
                value=default,
                help=help_text
            )
        elif setting["type"] == "bool":
            params[name] = st.checkbox(
                label,
                value=default,
                help=help_text
            )
        elif setting["type"] == "file":
            params[name] = st.file_uploader(
                label,
                type=setting.get("allowed_types"),
                help=help_text
            )

    return params
