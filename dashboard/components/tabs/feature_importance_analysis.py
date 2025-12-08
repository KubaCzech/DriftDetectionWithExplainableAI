import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import contextlib

from src.feature_importance import (
    FeatureImportanceMethod,
    compute_data_drift_analysis,
    compute_concept_drift_analysis,
    compute_predictive_importance_shift,
    visualize_data_drift_analysis,
    visualize_concept_drift_analysis,
    visualize_predictive_importance_shift,
)


def render_feature_importance_analysis_tab(X_before, y_before, X_after, y_after,
                                           feature_names, show_boxplot,
                                           model_class=None, model_params=None):
    """
    Renders the Feature Importance Analysis tab.

    Parameters
    ----------
    X_before : array-like
        Feature matrix for 'before' window
    y_before : array-like
        Target variable for 'before' window
    X_after : array-like
        Feature matrix for 'after' window
    y_after : array-like
        Target variable for 'after' window
    feature_names : list
        List of feature names
    show_boxplot : bool
        Whether to show boxplots
    """
    st.header("2. Drift Analysis Configuration & Results")

    # Select Feature Importance Method
    importance_method = st.selectbox(
        "Choose a Feature Importance Method",
        options=FeatureImportanceMethod.all_available(),
        format_func=lambda x: x.upper(),
        help="Select the method to explain the drift."
    )

    # Analysis type selector dropdown
    ANALYSIS_OPTIONS = {
        "Concept Drift - P(Y|X) Changes": "concept_drift",
        "Data Drift - P(X) Changes": "data_drift",
        "Predictive Power Shift": "predictive_shift"
    }
    analysis_choice = st.selectbox(
        "Select Analysis Type",
        options=list(ANALYSIS_OPTIONS.keys()),
        index=0,  # Default to Concept Drift (P(Y|X))
        help="Choose which drift analysis to display."
    )

    selected_analysis = ANALYSIS_OPTIONS[analysis_choice]

    st.markdown(f"Running **{analysis_choice}** analysis with **{importance_method.upper()}** method.")

    # --- Conditional Analysis Display ---

    if selected_analysis == "data_drift":
        # --- Analysis: Data Drift ---
        with st.container():
            st.subheader("Analysis: Data Drift - P(X) Changes")
            st.markdown("""
            This analysis trains a model to distinguish between the 'before' and 'after' periods
            using **only the input features (X)**.
            High accuracy indicates that the feature distribution P(X) has changed significantly.
            The feature importance scores show which features contributed most to this change.
            """)
            with st.spinner(f'Running Data Drift analysis with {importance_method.upper()}...'):
                # Compute the analysis results
                data_drift_result = compute_data_drift_analysis(
                    X_before, y_before, X_after, y_after,
                    feature_names=feature_names,
                    importance_method=importance_method,
                    model_class=model_class,
                    model_params=model_params
                )

                # Display the importance table
                st.markdown("#### Feature Importance Summary")
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Mean Importance': data_drift_result['importance_mean'],
                    'Std Deviation': data_drift_result['importance_std']
                })
                importance_df = importance_df.sort_values('Mean Importance', ascending=False)
                st.dataframe(
                    importance_df.style.format({
                        'Mean Importance': '{:.4f}',
                        'Std Deviation': '{:.4f}'
                    }),
                    use_container_width=True
                )

                # Display visualizations
                stdout_capture = StringIO()
                with contextlib.redirect_stdout(stdout_capture):
                    visualize_data_drift_analysis(
                        data_drift_result, feature_names,
                        show_boxplot=show_boxplot
                    )
                figs = [plt.figure(i) for i in plt.get_fignums()]
                for fig in figs:
                    st.pyplot(fig)
                    plt.close(fig)

    elif selected_analysis == "concept_drift":
        # --- Analysis: Concept Drift ---
        with st.container():
            st.subheader("Analysis: Concept Drift - P(Y|X) Changes")
            st.markdown("""
            This analysis trains a model to distinguish between the 'before' and 'after' periods
            using **both input features (X) and the target variable (Y)**.
            If the 'Y' feature has high importance, it suggests that the relationship between features
            and the target has changed (i.e., concept drift).
            """)
            with st.spinner(f'Running Concept Drift analysis with {importance_method.upper()}...'):
                # Compute the analysis results
                concept_drift_result = compute_concept_drift_analysis(
                    X_before, y_before, X_after, y_after,
                    feature_names=feature_names,
                    importance_method=importance_method,
                    model_class=model_class,
                    model_params=model_params
                )

                # Display the importance table
                st.markdown("#### Feature Importance Summary")
                feature_names_with_y = concept_drift_result['feature_names_with_y']
                importance_df = pd.DataFrame({
                    'Feature': feature_names_with_y,
                    'Mean Importance': concept_drift_result['importance_mean'],
                    'Std Deviation': concept_drift_result['importance_std']
                })
                importance_df = importance_df.sort_values('Mean Importance', ascending=False)
                st.dataframe(
                    importance_df.style.format({
                        'Mean Importance': '{:.4f}',
                        'Std Deviation': '{:.4f}'
                    }),
                    use_container_width=True
                )

                # Display visualizations
                stdout_capture = StringIO()
                with contextlib.redirect_stdout(stdout_capture):
                    visualize_concept_drift_analysis(
                        concept_drift_result, feature_names_with_y,
                        show_boxplot=show_boxplot
                    )
                figs = [plt.figure(i) for i in plt.get_fignums()]
                for fig in figs:
                    st.pyplot(fig)
                    plt.close(fig)

    elif selected_analysis == "predictive_shift":
        # --- Analysis: Predictive Power Shift ---
        with st.container():
            st.subheader("Analysis: Predictive Power Shift")
            st.markdown("""
            This analysis compares the importance of features for predicting the target variable (Y) in two separate models:
            1.  A model trained **only on 'before' data**.
            2.  A model trained **only on 'after' data**.
            A significant shift in feature importance between the two models indicates concept drift.
            """)
            with st.spinner(f'Running Predictive Power Shift analysis with {importance_method.upper()}...'):
                # Compute the analysis results
                shift_result = compute_predictive_importance_shift(
                    X_before, y_before, X_after, y_after,
                    feature_names=feature_names,
                    importance_method=importance_method,
                    model_class=model_class,
                    model_params=model_params
                )

                # Display the importance tables side by side
                st.markdown("#### Feature Importance Summary")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Before Drift**")
                    importance_before_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Mean Importance': shift_result['fi_before']['importances_mean'],
                        'Std Deviation': shift_result['fi_before']['importances_std']
                    })
                    importance_before_df = importance_before_df.sort_values('Mean Importance', ascending=False)
                    st.dataframe(
                        importance_before_df.style.format({
                            'Mean Importance': '{:.4f}',
                            'Std Deviation': '{:.4f}'
                        }),
                        use_container_width=True
                    )

                with col2:
                    st.markdown("**After Drift**")
                    importance_after_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Mean Importance': shift_result['fi_after']['importances_mean'],
                        'Std Deviation': shift_result['fi_after']['importances_std']
                    })
                    importance_after_df = importance_after_df.sort_values('Mean Importance', ascending=False)
                    st.dataframe(
                        importance_after_df.style.format({
                            'Mean Importance': '{:.4f}',
                            'Std Deviation': '{:.4f}'
                        }),
                        use_container_width=True
                    )

                # Display visualizations
                stdout_capture = StringIO()
                with contextlib.redirect_stdout(stdout_capture):
                    visualize_predictive_importance_shift(
                        shift_result, feature_names,
                        show_boxplot=show_boxplot
                    )
                figs = [plt.figure(i) for i in plt.get_fignums()]
                for fig in figs:
                    st.pyplot(fig)
                    plt.close(fig)

    st.success("✅ Analysis complete!")
