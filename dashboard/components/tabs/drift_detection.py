import streamlit as st
import numpy as np
import plotly.graph_objects as go
from river import drift as river_drift
from sklearn.neural_network import MLPClassifier
from src.DDM.BinaryErrorDriftDescriptor import BinaryErrorDriftDescriptor


def generate_error_stream(X_np, y_np, model):
    """
    Runs test-then-train once and returns the error stream and predictions.
    """
    classes = np.unique(y_np)

    error_stream = []
    predictions = []

    progress_bar = st.progress(0)

    # Initialize model
    model.partial_fit(X_np[0].reshape(1, -1), [y_np[0]], classes=classes)

    for i in range(1, len(X_np)):
        x_i = X_np[i].reshape(1, -1)
        y_true = y_np[i]

        # Predict
        y_pred = model.predict(x_i)[0]
        predictions.append(y_pred)

        # Binary error
        error_stream.append(int(y_pred != y_true))

        # Learn
        model.partial_fit(x_i, [y_true])

        if i % 100 == 0:
            progress_bar.progress(i / len(X_np))

    progress_bar.progress(1.0)

    return error_stream, predictions


def run_drift_detection(
    error_stream,
    detector_type,
    warning_grace_period,
    rate_calculation_sample_size,
    lookback_method='gradient',
    confidence_level=None
):
    # Create detector
    if detector_type == "DDM":
        detector = river_drift.binary.DDM()
    elif detector_type == "EDDM":
        detector = river_drift.binary.EDDM()
    elif detector_type == "FHDDM":
        detector = river_drift.binary.FHDDM(confidence_level=confidence_level)
    elif detector_type == "HDDM_A":
        detector = river_drift.binary.HDDM_A()
    elif detector_type == "HDDM_W":
        detector = river_drift.binary.HDDM_W()

    drift_descriptor = BinaryErrorDriftDescriptor(
        warning_grace_period=warning_grace_period,
        rate_calculation_sample_size=rate_calculation_sample_size,
        ddm=detector,
        lookback_method=lookback_method
    )

    drift_descriptions = []

    for i, error in enumerate(error_stream):
        drift_descriptor.update(error)

        if drift_descriptor.drift_detected:
            drift = drift_descriptor.last_detected_drift
            drift.detected_at = i
            drift_descriptions.append(drift)

    return drift_descriptions


def render_drift_detection_tab(X, y, window_length):  # noqa: C901
    """
    Main entry point for DDM (Drift Detection Method) Analysis tab.

    Uses online learning with River's MLPClassifier to track prediction errors
    and detect concept drift using binary drift detectors.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature data for the entire dataset
    y : pd.Series or np.ndarray
        Target labels for the entire dataset
    window_length : int
        Number of samples per window (not used directly but kept for consistency)
    """

    st.header("DDM Analysis")
    st.markdown("Binary Error Drift Detection using Online Learning")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Detector Configuration**")
        detector_type = st.selectbox(
            "Detector Type",
            ["DDM", "EDDM", "FHDDM", "HDDM_A", "HDDM_W"],
            help="Type of drift detector to use",
            key="ddm_detector_type"
        )

        # Detector-specific parameters
        if detector_type == "FHDDM":
            confidence_level = st.select_slider(
                "Confidence Level",
                options=[0.0001, 0.001, 0.01, 0.05, 0.1],
                value=0.001,
                help="Confidence level for FHDDM detector",
                key="ddm_confidence"
            )

    with col2:
        st.markdown("**Descriptor Parameters**")
        warning_grace_period = st.slider(
            "Warning Grace Period",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Number of non-warning iterations allowed in warning chain",
            key="ddm_grace"
        )
        rate_calculation_sample_size = st.slider(
            "Rate Calculation Sample Size",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="Window size for calculating error rates",
            key="ddm_rate_size"
        )

        lookback_method = st.selectbox(
            "Drift Start Detection Method",
            ["threshold", "gradient", "cusum"],
            help="Method for detecting actual drift start point:\n"
                 "- Threshold: Sustained error rate increase\n"
                 "- Gradient: Error rate slope analysis\n"
                 "- CUSUM: Cumulative sum change detection",
            key="ddm_lookback_method"
        )

    st.markdown("---")

    # Convert data to numpy once
    X_np = X.to_numpy() if hasattr(X, "to_numpy") else X
    y_np = y.to_numpy() if hasattr(y, "to_numpy") else y

    # Check if error stream needs to be generated
    need_error_stream = 'ddm_error_stream' not in st.session_state

    # Button to generate error stream (only if not already done)
    if need_error_stream:
        if st.button("Generate Error Stream", type="primary", key="ddm_generate"):
            try:
                status_text = st.empty()
                status_text.text("Training model and generating error stream...")

                # Initialize model
                model = MLPClassifier(
                    hidden_layer_sizes=(10,),
                    max_iter=1,
                    random_state=42
                )

                # Generate error stream using helper function
                error_stream, predictions = generate_error_stream(X_np, y_np, model)

                # Calculate sliding window error rate
                error_array = np.array(error_stream)
                error_rate = [
                    np.mean(error_array[i:i + rate_calculation_sample_size])
                    for i in range(len(error_stream) - rate_calculation_sample_size + 1)
                ]

                # Store in session state
                st.session_state.ddm_error_stream = error_stream
                st.session_state.ddm_predictions = predictions
                st.session_state.ddm_error_rate = error_rate
                st.session_state.ddm_sample_size_at_rate_creation = rate_calculation_sample_size

                status_text.text("✅ Error stream generated!")
                st.success("Error stream generated! Now you can run drift detection with different detectors.")
                st.rerun()

            except Exception as e:
                st.error(f"Error generating error stream: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("✓ Error stream already generated. You can now run drift detection with different parameters.")

        if st.button("Regenerate Error Stream", key="ddm_regenerate"):
            # Clear the cached error stream
            del st.session_state.ddm_error_stream
            del st.session_state.ddm_predictions
            del st.session_state.ddm_error_rate
            del st.session_state.ddm_sample_size_at_rate_creation
            if 'ddm_drift_descriptions' in st.session_state:
                del st.session_state.ddm_drift_descriptions
            if 'ddm_processing_complete' in st.session_state:
                del st.session_state.ddm_processing_complete
            st.rerun()

    # Run drift detection button (only if error stream exists)
    if not need_error_stream:
        st.markdown("---")
        if st.button("Run Drift Detection", type="primary", key="ddm_run"):
            try:
                status_text = st.empty()
                status_text.text("Running drift detection...")

                # Get error stream from session state
                error_stream = st.session_state.ddm_error_stream

                # Run drift detection with current parameters using helper function
                confidence = confidence_level if detector_type == "FHDDM" else None
                drift_descriptions = run_drift_detection(
                    error_stream,
                    detector_type,
                    warning_grace_period,
                    rate_calculation_sample_size,
                    lookback_method=lookback_method,
                    confidence_level=confidence
                )

                # Store results in session state
                st.session_state.ddm_drift_descriptions = drift_descriptions
                st.session_state.ddm_processing_complete = True

                status_text.text("✅ Drift detection complete!")
                st.success(f"Detected {len(drift_descriptions)} drift(s) using {detector_type}!")

            except Exception as e:
                st.error(f"Error during drift detection: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Display results if processing is complete
    if 'ddm_processing_complete' in st.session_state and st.session_state.ddm_processing_complete:
        st.markdown("---")

        error_rate = st.session_state.ddm_error_rate
        drift_descriptions = st.session_state.ddm_drift_descriptions
        error_stream = st.session_state.ddm_error_stream
        predictions = st.session_state.ddm_predictions

        # Create interactive plot with Plotly
        st.subheader("Error Rate Over Time")

        fig = go.Figure()

        # Add error rate line
        fig.add_trace(go.Scatter(
            x=list(range(st.session_state.ddm_sample_size_at_rate_creation,
                         st.session_state.ddm_sample_size_at_rate_creation + len(error_rate))),
            y=error_rate,
            mode='lines',
            name='Error Rate',
            line=dict(color='royalblue', width=2),
            hovertemplate='Index: %{x}<br>Error Rate: %{y:.3f}<extra></extra>'
        ))

        # Add drift annotations using ACTUAL drift start
        for idx, drift in enumerate(drift_descriptions):
            # Use the actual drift start index if available
            if hasattr(drift, 'drift_start_index') and drift.drift_start_index is not None:
                start = drift.drift_start_index
            else:
                start = max(0, drift.detected_at - drift.drift_duration)

            end = drift.detected_at

            # Make sure they're within the error_rate bounds
            start_plot = max(st.session_state.ddm_sample_size_at_rate_creation,
                             min(start, st.session_state.ddm_sample_size_at_rate_creation + len(error_rate) - 1))
            end_plot = max(st.session_state.ddm_sample_size_at_rate_creation,
                           min(end, st.session_state.ddm_sample_size_at_rate_creation + len(error_rate) - 1))

            # Calculate error_rate array indices
            start_idx = start - st.session_state.ddm_sample_size_at_rate_creation
            end_idx = end - st.session_state.ddm_sample_size_at_rate_creation
            start_idx = max(0, min(start_idx, len(error_rate) - 1))
            end_idx = max(0, min(end_idx, len(error_rate) - 1))

            # Add drift region as shaded area
            fig.add_vrect(
                x0=start_plot,
                x1=end_plot,
                fillcolor="rgba(255, 107, 53, 0.2)",
                layer="below",
                line_width=0,
            )

            # Add drift line
            fig.add_trace(go.Scatter(
                x=[start_plot, end_plot],
                y=[error_rate[start_idx], error_rate[end_idx]],
                mode='lines+markers',
                name=f'Drift {idx+1}',
                line=dict(color='#FF6B35', width=3, dash='dash'),
                marker=dict(size=10, color='#FF6B35', symbol=['circle', 'x']),
                hovertemplate=(
                    f'<b>Drift {idx+1}</b><br>'
                    f'Start Index: {start}<br>'
                    f'End Index: {end}<br>'
                    f'Duration: {drift.drift_duration}<br>'
                    f'Error Rate at Start: {error_rate[start_idx]:.3f}<br>'
                    f'Error Rate at Detection: {error_rate[end_idx]:.3f}<br>'
                    f'<extra></extra>'
                )
            ))

        # Update layout
        fig.update_layout(
            title=f'Drift Detection using {detector_type} (Drift Start Detection: {lookback_method})',
            xaxis_title='Data Point Index',
            yaxis_title='Error Rate',
            hovermode='closest',
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Display plot
        st.plotly_chart(fig, width='stretch')

        # Display statistics
        st.markdown("---")
        st.subheader("Detection Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Data Points", len(error_stream))
            st.metric("Drifts Detected", len(drift_descriptions))
            overall_accuracy = 1 - np.mean(error_stream)
            st.metric("Overall Accuracy", f"{overall_accuracy:.3f}")

        with col2:
            if drift_descriptions:
                avg_duration = np.mean([d.drift_duration for d in drift_descriptions])
                st.metric("Average Drift Duration", f"{avg_duration:.1f}")
                avg_error_change = np.mean([
                    d.error_rate_at_detection - d.error_rate_at_warning
                    for d in drift_descriptions
                ])
                st.metric("Avg Error Rate Change", f"{avg_error_change:.3f}")
            else:
                st.metric("Average Drift Duration", "N/A")
                st.metric("Avg Error Rate Change", "N/A")

        with col3:
            if drift_descriptions:
                min_duration = min([d.drift_duration for d in drift_descriptions])
                max_duration = max([d.drift_duration for d in drift_descriptions])
                st.metric("Min/Max Duration", f"{min_duration} / {max_duration}")
            else:
                st.metric("Min/Max Duration", "N/A")

            # Calculate accuracy in first and last quarters
            quarter = len(error_stream) // 4
            if quarter > 0:
                first_quarter_acc = 1 - np.mean(error_stream[:quarter])
                last_quarter_acc = 1 - np.mean(error_stream[-quarter:])
                st.metric("First/Last Quarter Acc", f"{first_quarter_acc:.3f} / {last_quarter_acc:.3f}")

        # Detailed drift information
        if drift_descriptions:
            st.markdown("---")
            st.subheader("🔍 Detailed Drift Information")

            for idx, drift in enumerate(drift_descriptions):
                # Determine actual start index
                if hasattr(drift, 'drift_start_index') and drift.drift_start_index is not None:
                    actual_start = drift.drift_start_index
                else:
                    actual_start = drift.detected_at - drift.drift_duration

                with st.expander(f"Drift {idx+1} - Detected at index {drift.detected_at}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Duration", drift.drift_duration)
                        st.metric("Start Index", actual_start)
                        st.metric("End Index", drift.detected_at)

                    with col2:
                        st.metric("Error Rate at Start", f"{error_rate[actual_start]:.3f}")
                        st.metric("Error Rate at Detection", f"{error_rate[drift.detected_at]:.3f}")

                    with col3:
                        error_change = error_rate[drift.detected_at] - error_rate[actual_start]
                        st.metric("Error Rate Change", f"{error_change:.3f}")
                        change_pct = (error_change / error_rate[actual_start] *
                                      100) if error_rate[actual_start] > 0 else 0
                        st.metric("Change Percentage", f"{change_pct:.1f}%")
        else:
            st.info("No drifts detected with current configuration. Try adjusting the parameters.")
