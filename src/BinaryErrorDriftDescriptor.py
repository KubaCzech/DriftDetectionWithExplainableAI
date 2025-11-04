from river import drift

class DriftDescription():
    def __init__(self, error_rate_at_warning = None, error_rate_at_detection = None, drift_duration = None):
        self.error_rate_at_warning = error_rate_at_warning
        self.error_rate_at_detection = error_rate_at_detection
        self.drift_duration = drift_duration
        
        self.detected_at = None

class BinaryErrorDriftDescriptor():
    """
    Binary Error Drift Descriptor creates descriptions of drifts identified by detectors from BinaryDriftAndWarningDetector family of river.

    A drift is said to begin at the moment of the first detected warning that was followed by an uninterrupted series of warnings until the moment drift detected signal is detected from BinaryDriftAndWarningDetector.
    The moment a drift signal is detected is also considered the end of the drift.
    The amount of iterations where warning is not detected by BinaryDriftAndWarningDetector, but BinaryErrorDriftDescriptor assumes the chain of warnings was not interrupted in a row can be set using warning grace period.
    When warning grace period is equal to n, there can be n-1 non-warning iterations between warning iterations such that the algorithm assumes the chain of warnings was not broken.

    The detected drifts are described using  DritDescription object.
    When describing a drift 3 statistics are noted:
    - The duration of the drift
    - The error rate at the time of the first warning
    - The error rate at the time of the drift being detected
    """
    def __init__(self, warning_grace_period = 3, rate_calculation_sample_size = 100, ddm = drift.binary.DDM()):
        self.warning_grace_period = warning_grace_period
        self.rate_calculation_sample_size = rate_calculation_sample_size
        self.ddm = ddm

        self.warning_grace_period_left = warning_grace_period
        self.error_history = []
        self.previous_was_warning = False
        self.assume_warning = False
        self.last_detected_drift = None
        self.drift_detected = False

    def update(self, x):
        self.ddm.update(x)

        self.drift_detected = False

        if self.ddm.warning_detected:
            self.warning_grace_period_left = self.warning_grace_period
        elif self.previous_was_warning == True: 
            self.warning_grace_period_left -= 1

        if self.previous_was_warning and self.warning_grace_period_left > 0:
            self.assume_warning = True
        elif self.ddm.warning_detected: self.assume_warning = True
        else: self.assume_warning = False


        self.error_history.append(x)

        if not self.assume_warning and not self.ddm.drift_detected:
            self.error_history = self.error_history[-self.rate_calculation_sample_size:]

        
        if self.ddm.drift_detected:
            if len(self.error_history) >= (self.rate_calculation_sample_size * 2):
                drift_duration = len(self.error_history) - self.rate_calculation_sample_size

                error_rate_at_warning = sum(self.error_history[:self.rate_calculation_sample_size]) / self.rate_calculation_sample_size
                error_rate_at_detection = sum(self.error_history[-self.rate_calculation_sample_size:]) / self.rate_calculation_sample_size

            else:
                warning_sample_size = len(self.error_history)//2
                detection_sample_size = len(self.error_history) - warning_sample_size

                drift_duration = min(len(self.error_history), self.rate_calculation_sample_size)

                error_rate_at_warning = sum(self.error_history[:warning_sample_size]) / warning_sample_size
                error_rate_at_detection = sum(self.error_history[-detection_sample_size:]) / detection_sample_size  

            self.last_detected_drift = DriftDescription(error_rate_at_detection=error_rate_at_detection, error_rate_at_warning=error_rate_at_warning, drift_duration=drift_duration)

            self.drift_detected = True

            # Reset after drift
            self.warning_grace_period_left = self.warning_grace_period
            self.error_history = []
            self.previous_was_warning = False
            self.assume_warning = False


        self.previous_was_warning = self.assume_warning