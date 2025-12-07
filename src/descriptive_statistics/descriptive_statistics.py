from enum import Enum


class StatisticsType(Enum):
    Mean = 'mean'
    StandardDeviation = 'sd'
    Min = 'min'
    Max = 'max'
    Median = 'median'
    ImbalanceRatio = 'imbalance_ratio'
    All = 'all'  # TODO: implement all


class DescriptiveStatisticsDriftDetector:
    def __init__(self):
        pass

    def calculate_range(self, old_value, new_value, thr):
        if old_value == 0 and new_value == 0:
            return False
        elif old_value == 0 and new_value != 0:
            return True
        elif abs(new_value - old_value) / abs(old_value) > thr:
            return True
        return False

    def detect_mean(self, old_data, new_data, thr):
        old_mean = old_data.groupby('label').mean().T
        new_mean = new_data.groupby('label').mean().T

        assert old_mean.shape == new_mean.shape
        for label in old_mean.columns:
            for feature in old_mean.index:
                if self.calculate_range(old_mean.at[feature, label], new_mean.at[feature, label], thr):
                    return True
        return False

    def detect_median(self, old_series, new_series, thr):
        old_median = old_series.groupby('label').median().T
        new_median = new_series.groupby('label').median().T

        assert old_median.shape == new_median.shape
        for label in old_median.columns:
            for feature in old_median.index:
                if self.calculate_range(old_median.at[feature, label], new_median.at[feature, label], thr):
                    return True
        return False

    def detect_sd(self, old_data, new_data, thr):
        old_sd = old_data.groupby('label').std().T
        new_sd = new_data.groupby('label').std().T

        assert old_sd.shape == new_sd.shape
        for label in old_sd.columns:
            for feature in old_sd.index:
                if self.calculate_range(old_sd.at[feature, label], old_sd.at[feature, label], thr):
                    return True
        return False

    def detect_min(self, old_data, new_data, thr):
        old_min = old_data.groupby('label').min().T
        new_min = new_data.groupby('label').min().T

        assert old_min.shape == new_min.shape
        for label in old_min.columns:
            for feature in old_min.index:
                if self.calculate_range(old_min.at[feature, label], new_min.at[feature, label], thr):
                    return True
        return False

    def detect_max(self, old_data, new_data, thr):
        old_max = old_data.groupby('label').max().T
        new_max = new_data.groupby('label').max().T

        assert old_max.shape == new_max.shape
        for label in old_max.columns:
            for feature in old_max.index:
                if self.calculate_range(old_max.at[feature, label], new_max.at[feature, label], thr):
                    return True
        return False

    def detect_imbalance_ratio(self, old_data, new_data, thr):
        old_ir = old_data['label'].value_counts(normalize=True)
        new_ir = new_data['label'].value_counts(normalize=True)

        assert old_ir.shape == new_ir.shape
        for label in old_ir.index:
            if self.calculate_range(old_ir.at[label], new_ir.at[label], thr):
                return True
        return False

    def _detect_single_statistic(self, old_data, new_data, stat_type, thr):
        if stat_type == StatisticsType.Mean:
            return self.detect_mean(old_data, new_data, thr)
        elif stat_type == StatisticsType.StandardDeviation:
            return self.detect_sd(old_data, new_data, thr)
        elif stat_type == StatisticsType.Min:
            return self.detect_min(old_data, new_data, thr)
        elif stat_type == StatisticsType.Max:
            return self.detect_max(old_data, new_data, thr)
        elif stat_type == StatisticsType.Median:
            return self.detect_median(old_data, new_data, thr)
        elif stat_type == StatisticsType.ImbalanceRatio:
            return self.detect_imbalance_ratio(old_data, new_data, thr)
        else:
            raise ValueError("Unsupported statistics type")

    def detect(self, old_data, new_data, stat_type, features=None, thr=0.2):
        if features is not None:
            old_data = old_data[features]
            new_data = new_data[features]

        if type(stat_type) is list:
            drifts = []
            for st in stat_type:
                drifts.append(self._detect_single_statistic(old_data, new_data, st, thr))
            return sum(drifts) / len(drifts) > 0.4
        else:
            return self._detect_single_statistic(old_data, new_data, stat_type, thr)
