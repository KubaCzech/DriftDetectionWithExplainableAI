import numpy as np
from river import drift


class DriftDescription():
    def __init__(self, error_rate_at_warning=None, error_rate_at_detection=None,
                 drift_duration=None, drift_start_index=None):
        self.error_rate_at_warning = error_rate_at_warning
        self.error_rate_at_detection = error_rate_at_detection
        self.drift_duration = drift_duration
        self.drift_start_index = drift_start_index  # New: actual start index

        self.detected_at = None


class BinaryErrorDriftDescriptor():
    """
    Binary Error Drift Descriptor creates descriptions of drifts identified by detectors
    from BinaryDriftAndWarningDetector family of river.

    The moment a drift signal is detected is also considered the end of the drift.
    The amount of iterations where warning is not detected by BinaryDriftAndWarningDetector,
    but BinaryErrorDriftDescriptor assumes the chain of warnings was not interrupted in a row
    can be set using warning grace period.
    When warning grace period is equal to n, there can be n-1 non-warning iterations between
    warning iterations such that the algorithm assumes the chain of warnings was not broken.
    The method with which the start of the drift is determined depends on lookback_method.

    The detected drifts are described using  DritDescription object.
    When describing a drift 3 statistics are noted:
    - The duration of the drift
    - The error rate at the time of the first warning
    - The error rate at the time of the drift being detected
    """

    def __init__(self, warning_grace_period=3, rate_calculation_sample_size=100,
                 ddm=drift.binary.DDM(), lookback_method='cusum'):
        self.warning_grace_period = warning_grace_period
        self.rate_calculation_sample_size = rate_calculation_sample_size
        self.ddm = ddm
        self.lookback_method = lookback_method  # 'cusum', 'threshold', or 'gradient'

        self.warning_grace_period_left = warning_grace_period
        self.error_history = []
        self.complete_error_history = []
        self.previous_was_warning = False
        self.assume_warning = False
        self.last_detected_drift = None
        self.drift_detected = False
        self.current_index = 0

    def find_drift_start_cusum(self, detection_idx, lookback_window=300):
        """
        Use CUSUM to detect change point - where error rate started increasing.
        """
        if detection_idx < 50:
            return 0

        start_idx = max(0, detection_idx - lookback_window)
        history_segment = self.complete_error_history[start_idx:detection_idx]

        if len(history_segment) < 20:
            return start_idx

        # Establish baseline from early portion
        baseline_size = min(50, len(history_segment) // 4)
        baseline_mean = np.mean(history_segment[:baseline_size])

        # Apply CUSUM
        cusum = 0
        threshold = 1.5  # Sensitivity parameter

        for i in range(baseline_size, len(history_segment)):
            cusum = max(0, cusum + (history_segment[i] - baseline_mean - 0.1))

            # Detect when CUSUM exceeds threshold
            if cusum > threshold:
                return start_idx + i

        return start_idx + baseline_size

    def find_drift_start_threshold(self, detection_idx, lookback_window=300):
        """
        Find drift start by looking for sustained error rate increase.
        """
        if detection_idx < 50:
            return 0

        window_size = self.rate_calculation_sample_size
        start_idx = max(0, detection_idx - lookback_window)

        # Calculate baseline from early history
        baseline_end = min(start_idx + window_size, detection_idx - 50)
        if baseline_end <= start_idx:
            return start_idx

        baseline = self.complete_error_history[start_idx:baseline_end]
        baseline_rate = np.mean(baseline)

        # Look for point where error rate exceeds baseline significantly
        increase_threshold = 1.5  # 50% increase
        sustained_count = 0
        sustained_threshold = 20  # Need 20 consecutive high-error samples

        for i in range(baseline_end, detection_idx):
            if i + window_size > detection_idx:
                window_size_local = detection_idx - i
            else:
                window_size_local = window_size

            if window_size_local < 10:
                continue

            window = self.complete_error_history[i:i + window_size_local]
            current_rate = np.mean(window)

            if current_rate > baseline_rate * increase_threshold:
                sustained_count += 1
                if sustained_count >= sustained_threshold:
                    return max(start_idx, i - sustained_threshold)
            else:
                sustained_count = 0

        return baseline_end

    def find_drift_start_gradient(self, detection_idx, lookback_window=300):
        """
        Find drift start using error rate gradient analysis.
        """
        if detection_idx < 50:
            return 0

        window_size = 20
        start_idx = max(0, detection_idx - lookback_window)

        # Calculate moving average of error rate
        error_rates = []
        for i in range(start_idx, detection_idx - window_size):
            window = self.complete_error_history[i:i + window_size]
            error_rates.append(np.mean(window))

        if len(error_rates) < 2:
            return start_idx

        # Calculate gradient
        gradients = np.diff(error_rates)

        # Find first significant positive gradient
        gradient_threshold = 0.01  # Adjust based on your data
        for i, grad in enumerate(gradients):
            if grad > gradient_threshold:
                return start_idx + i

        return start_idx

    def update(self, x):
        self.ddm.update(x)
        self.complete_error_history.append(x)
        self.drift_detected = False

        if self.ddm.warning_detected:
            self.warning_grace_period_left = self.warning_grace_period
        elif self.previous_was_warning is True:
            self.warning_grace_period_left -= 1

        if self.previous_was_warning and self.warning_grace_period_left > 0:
            self.assume_warning = True
        elif self.ddm.warning_detected:
            self.assume_warning = True
        else:
            self.assume_warning = False

        self.error_history.append(x)

        if not self.assume_warning and not self.ddm.drift_detected:
            self.error_history = self.error_history[-self.rate_calculation_sample_size:]

        if self.ddm.drift_detected:
            detection_idx = self.current_index

            # Find actual drift start using selected method
            if self.lookback_method == 'cusum':
                drift_start_idx = self.find_drift_start_cusum(detection_idx)
            elif self.lookback_method == 'threshold':
                drift_start_idx = self.find_drift_start_threshold(detection_idx)
            elif self.lookback_method == 'gradient':
                drift_start_idx = self.find_drift_start_gradient(detection_idx)
            else:
                drift_start_idx = max(0, detection_idx - len(self.error_history))

            # Calculate error rates at drift start and detection
            window_size = self.rate_calculation_sample_size

            # Error rate at actual drift start
            start_window_begin = max(0, drift_start_idx - window_size // 2)
            start_window_end = min(drift_start_idx + window_size // 2, len(self.complete_error_history))
            start_window = self.complete_error_history[start_window_begin:start_window_end]
            error_rate_at_warning = np.mean(start_window) if len(start_window) > 0 else 0

            # Error rate at detection
            detection_window_begin = max(0, detection_idx - window_size)
            detection_window = self.complete_error_history[detection_window_begin:detection_idx]
            error_rate_at_detection = np.mean(detection_window) if len(detection_window) > 0 else 0

            # Duration is from actual start to detection
            drift_duration = detection_idx - drift_start_idx

            self.last_detected_drift = DriftDescription(
                error_rate_at_detection=error_rate_at_detection,
                error_rate_at_warning=error_rate_at_warning,
                drift_duration=drift_duration,
                drift_start_index=drift_start_idx
            )

            self.drift_detected = True

            # Reset after drift
            self.warning_grace_period_left = self.warning_grace_period
            self.error_history = []
            self.previous_was_warning = False
            self.assume_warning = False

        self.previous_was_warning = self.assume_warning
        self.current_index += 1
