import streamlit as st


@st.dialog("About this Dashboard")
def show_info_modal():
    """Displays the general information modal for the dashboard."""
    st.markdown("""
    Welcome to the Concept Drift Analysis Dashboard. This tool allows you to:
    1.  **Generate** synthetic datasets with known concept drift.
    2.  **Visualize** the data stream and the drift itself.
    3.  **Analyze** the drift using various feature importance techniques to understand its root causes.
    """)
