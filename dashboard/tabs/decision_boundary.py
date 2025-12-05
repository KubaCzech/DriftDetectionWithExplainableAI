import streamlit as st
import matplotlib.pyplot as plt
from src.decision_boundary.analysis import compute_decision_boundary_analysis
from src.decision_boundary.visualization import visualize_decision_boundary

def render_decision_boundary_tab(X, y,
                                 window_before_start=0,
                                 window_after_start=0,
                                 window_length=1000,
                                 model_class=None,
                                 model_params=None):
    """
    Renders the Decision Boundary Analysis tab.

    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target variable
    window_before_start : int
        Start index for the pre-drift window (absolute)
    window_after_start : int
        Start index for the post-drift window (absolute)
    window_length : int
        Length of the analysis window
    model_class : class
        Classifier class
    model_params : dict
        Parameters for the classifier
    """
    st.header("3. Decision Boundary Analysis")
    st.markdown("""
    This tab visualizes the decision boundary of a classifier trained on the pre-drift and post-drift data.
    It uses **SSNP (Semi-Supervised Neural Projection)** to project the high-dimensional data into 2D while 
    preserving the separation between classes.
    """)

    # Configuration for SSNP
    with st.expander("SSNP Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            ssnp_epochs = st.number_input("SSNP Epochs", min_value=1, max_value=200, value=10, step=5,
                                         help="Number of epochs for training the SSNP projector.")
        with col2:
            grid_size = st.number_input("Grid Resolution", min_value=50, max_value=500, value=200, step=50,
                                       help="Resolution of the grid for visualizing probabilities.")

    # Initialize session state for results if not exists
    if 'decision_boundary_results' not in st.session_state:
        st.session_state.decision_boundary_results = None

    if st.button("Run Decision Boundary Analysis", key="run_decision_boundary_btn"):
        with st.spinner("Running Analysis (Training SSNP and Classifiers)..."):
            try:
                # Note: window_after_start from app.py is the absolute start index of the second window
                results = compute_decision_boundary_analysis(
                    X, y,
                    start_index_pre=window_before_start,
                    start_index_post=window_after_start,
                    window_length=window_length,
                    model_class=model_class,
                    model_params=model_params,
                    ssnp_epochs=ssnp_epochs,
                    grid_size=grid_size
                )
                # Store results in session state
                st.session_state.decision_boundary_results = results
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                # Optional: print stack trace for debugging
                import traceback
                st.text(traceback.format_exc())
                st.session_state.decision_boundary_results = None

    # Display results if they exist in session state
    if st.session_state.decision_boundary_results is not None:
        try:
            st.markdown("### Decision Boundaries (Pre vs Post)")
            
            # Create and display plot
            fig = visualize_decision_boundary(st.session_state.decision_boundary_results)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error displaying visualization: {e}")
            # In case of stale state or error, allow clearing
            if st.button("Clear Results"):
                st.session_state.decision_boundary_results = None
                st.rerun()
