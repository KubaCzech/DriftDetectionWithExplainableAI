import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.feature_importance import (
    FeatureImportanceMethod,
    FeatureImportanceDriftAnalyzer,
    visualize_drift_importance,
    visualize_predictive_importance_shift,
)


def render_feature_importance_analysis_tab(X_before, y_before, X_after, y_after,
                                           feature_names,
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
    model_class : class (optional)
        The model class to use for drift detection
    model_params : dict (optional)
        Parameters for the model
    """
    st.header("Feature Importance Analysis")

    # Analysis type selector dropdown
    ANALYSIS_OPTIONS = {
        "Drift Analysis (P(X) or P(Y|X))": "drift_analysis",
        "Predictive Power Shift": "predictive_shift"
    }

    col1, col2 = st.columns(2)

    with col1:
        analysis_choice = st.selectbox(
            "Select Analysis Type",
            options=list(ANALYSIS_OPTIONS.keys()),
            index=0,
            help="Choose which drift analysis to display."
        )

    with col2:
        # Select Feature Importance Method
        importance_method = st.selectbox(
            "Choose a Feature Importance Method",
            options=FeatureImportanceMethod.all_available(),
            format_func=lambda x: x.upper(),
            help="Select the method to explain the drift."
        )

    selected_analysis = ANALYSIS_OPTIONS[analysis_choice]
    
    # Checkbox for including target (only for drift analysis)
    include_target = True
    if selected_analysis == "drift_analysis":
        include_target = st.checkbox(
            "Include Target (Y) in Analysis",
            value=True,
            help="If checked, analyzes Concept Drift (P(Y|X)). If unchecked, analyzes Data Drift (P(X))."
        )

    # Plot Type Selector
    plot_type_display = st.radio(
        "Select Plot Type",
        options=["Bar Chart", "Box Plot"],
        index=0,
        horizontal=True,
        help="Choose visualization type."
    )
    plot_type_map = {"Bar Chart": "bar", "Box Plot": "box"}
    selected_plot_type = plot_type_map[plot_type_display]

    # Initialize DriftAnalyzer
    analyzer = FeatureImportanceDriftAnalyzer(X_before, y_before, X_after, y_after, feature_names=feature_names)

    # --- Conditional Analysis Display ---

    if selected_analysis == "drift_analysis":
        # --- Analysis: Drift Analysis ---
        with st.container():
            drift_title = "Concept Drift - P(Y|X) Changes" if include_target else "Data Drift - P(X) Changes"
            st.subheader(f"Analysis: {drift_title}")
            
            if include_target:
                st.markdown("""
                This analysis trains a model to distinguish between the 'before' and 'after' periods
                using **both input features (X) and the target variable (Y)**.
                If the 'Y' feature has high importance, it suggests that the relationship between features
                and the target has changed (i.e., concept drift).
                """)
            else:
                st.markdown("""
                This analysis trains a model to distinguish between the 'before' and 'after' periods
                using **only the input features (X)**.
                High accuracy indicates that the feature distribution P(X) has changed significantly.
                The feature importance scores show which features contributed most to this change.
                """)
                
            with st.spinner(f'Running Drift analysis with {importance_method.upper()}...'):
                # Compute the analysis results
                drift_result = analyzer.compute_drift_importance(
                    importance_method=importance_method,
                    include_target=include_target,
                    model_class=model_class,
                    model_params=model_params
                )

                # Create columns for side-by-side layout
                col_viz, col_table = st.columns([3, 2])

                with col_viz:
                    # Display visualizations
                    st.markdown("#### Drift Visualization")
                    fig = visualize_drift_importance(
                        drift_result, drift_result['feature_names'],
                        plot_type=selected_plot_type,
                        include_target=include_target
                    )
                    st.pyplot(fig)
                    plt.close(fig)

                with col_table:
                    # Display the importance table
                    st.markdown("#### Feature Importance Summary")
                    feature_names_result = drift_result['feature_names']
                    importance_df = pd.DataFrame({
                        'Feature': feature_names_result,
                        'Mean Importance': drift_result['importance_mean'],
                        'Std Deviation': drift_result['importance_std']
                    })
                    importance_df = importance_df.sort_values('Mean Importance', ascending=False)
                    st.dataframe(
                        importance_df.style.format({
                            'Mean Importance': '{:.4f}',
                            'Std Deviation': '{:.4f}'
                        }),
                        width="stretch"
                    )

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
                shift_result = analyzer.compute_predictive_importance_shift(
                    importance_method=importance_method,
                    model_class=model_class,
                    model_params=model_params
                )

                # Create columns for side-by-side layout
                col_viz, col_table = st.columns([3, 2])

                with col_viz:
                    # Display visualizations
                    st.markdown("#### Drift Visualization")
                    fig = visualize_predictive_importance_shift(
                        shift_result, feature_names,
                        plot_type=selected_plot_type
                    )
                    st.pyplot(fig)
                    plt.close(fig)

                with col_table:
                    # Display the importance tables stacked
                    st.markdown("#### Feature Importance Summary")

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
                        width="stretch",
                        height=200  # Fixed height to avoid overtaking the column
                    )

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
                        width="stretch",
                        height=200  # Fixed height
                    )
