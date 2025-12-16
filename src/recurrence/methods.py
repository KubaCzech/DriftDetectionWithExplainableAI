import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
import pandas as pd
import numpy as np


from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

from src.recurrence.full_window_storage import FullWindowStorage


def visualize_distance_matrix(matrix: pd.DataFrame,
                              drift_positions: list[int] = None,
                              title: str = "Window Distance Matrix"):
    """Visualize the distance matrix as a heatmap.

    Args:
        matrix: Distance matrix DataFrame
        drift_positions: List of iterations where drift was detected
        title: Plot title
    """

    fig, ax = plt.subplots(figsize=(14, 12))

    # Create heatmap
    sns.heatmap(matrix, cmap='RdYlGn_r', square=True,
                linewidths=0, cbar_kws={'label': 'Distance'},
                ax=ax)

    # Mark drift positions if provided
    if drift_positions:
        for drift_iter in drift_positions:
            if drift_iter in matrix.index:
                idx = list(matrix.index).index(drift_iter)
                # Draw lines to mark drifts
                ax.axhline(y=idx, color='red', linewidth=2, alpha=0.7)
                ax.axvline(x=idx, color='red', linewidth=2, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Iteration')

    plt.tight_layout()
    plt.show()


def median_mask(arr, k=3):
    arr = np.asarray(arr)
    pad = k // 2

    # Same/edge padding
    padded = np.pad(arr, pad_width=pad, mode='edge')

    # Sliding window view
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)

    return np.median(windows, axis=1).astype(arr.dtype)


def show_distance_median(storage: FullWindowStorage, window_nr, k=3, measure='centroid_displacement'):
    data_to_plot = storage.compare_window_to_all(window_nr, measure=measure)
    data_to_plot = [float(x) for x in data_to_plot]
    data_to_plot = median_mask(data_to_plot, k)

    plt.plot(data_to_plot)
    plt.title("distance from window nr " + str(window_nr))
    plt.show()


def cluster_windows(matrix: pd.DataFrame, fix_outliers=True, median_mask_width=1):
    clusterer = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=3,
    )

    if median_mask_width != 1:
        matrix = pd.DataFrame(median_mask_2d(matrix, kx=median_mask_width, ky=median_mask_width))

    labels = clusterer.fit_predict(matrix.values)

    if fix_outliers:
        # If both left and right neighbours are the same, replace the middle label with their value
        # in order to denoise
        for i, label in enumerate(labels[1:-1]):
            if labels[i] == labels[i+2] and labels[i] != -1:
                labels[i+1] = labels[i]

    return labels


def get_drift_from_clusters(labels):
    drift_locations = []
    last_label = labels[0]
    for i, label in enumerate(labels):
        if label == -1:
            continue

        if label != last_label:
            last_label = label
            drift_locations.append(i)

    return drift_locations


def clustered_labels_accuracy(labels, true_concept, outliers_included=True):
    mask = labels != -1
    pred = labels[mask]
    true_non_out = true_concept[mask]

    # confusion matrix between predicted cluster IDs and true cluster IDs
    cm = confusion_matrix(true_non_out, pred)

    # Hungarian algorithm finds best 1-to-1 mapping
    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)

    # Build mapping: pred_label → true_label
    # The clustering algorighms labels and true labels are different, but they still can be correct.
    # All that is important is taht the same windows are grouped together in both system. The label itself is not imoprtant
    # However they need to be the same during evaluation, so we are re-maping
    mapping = dict(zip(col_ind, row_ind))

    pred_mapped = np.array([mapping[p] for p in pred])
    accuracy_no_outliers = np.mean(pred_mapped == true_non_out)

    n_outliers = sum([x == -1 for x in labels])
    # assume outliers are always incorrectly classified
    accuracy_with_outliers = accuracy_no_outliers * (1-(n_outliers/len(labels)))

    if outliers_included:
        return accuracy_with_outliers
    else:
        return accuracy_no_outliers


def median_mask_2d(arr, ky=3, kx=3):
    """
    2D median filter with SAME (edge) padding and independent kernel sizes.

    arr : 2D array-like
    ky  : kernel height (odd)
    kx  : kernel width  (odd)
    """
    arr = np.asarray(arr, dtype=np.float64)

    if ky % 2 == 0 or kx % 2 == 0:
        raise ValueError("Both ky and kx must be odd.")

    py = ky // 2
    px = kx // 2

    padded = np.pad(
        arr,
        pad_width=((py, py), (px, px)),
        mode="edge"
    )

    windows = np.lib.stride_tricks.sliding_window_view(padded, (ky, kx))
    return np.median(windows, axis=(2, 3))


def evaluate_threshold(concepts_same, distance, threshold):
    if concepts_same:
        return distance < threshold   # predict similar
    else:
        return distance > threshold   # predict different


def threshold_test(M: pd.DataFrame, true_concept: list, median_mask_dimensions=(1, 1)):
    # Test the accuracy of separation of 'similar' and 'different' concept for various threshold values
    M = M.to_numpy()
    M = median_mask_2d(M, ky=median_mask_dimensions[0], kx=median_mask_dimensions[1])

    thresholds = [x * 0.005 for x in range(200)]

    results = []

    for threshold in thresholds:
        TP = FP = TN = FN = 0

        for i in range(len(M)):
            for j in range(i-1):
                concepts_same = true_concept[i] == true_concept[j]
                distance = M[i][j]

                if distance is None:
                    continue

                prediction_correct = evaluate_threshold(concepts_same, distance, threshold)

                if concepts_same and prediction_correct:
                    TP += 1
                elif not concepts_same and prediction_correct:
                    TN += 1
                elif concepts_same and not prediction_correct:
                    FN += 1
                else:
                    FP += 1

        total = TP + FP + TN + FN

        accuracy = (TP + TN) / total
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'fnr': fnr,
            'f1': f1
        })

    return results


def plot_threshold_analysis_results(results):
    # Plot All Metrics
    thresholds = sorted([x['threshold'] for x in results])

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, [r['accuracy'] for r in results], label="Accuracy")
    plt.plot(thresholds, [r['precision'] for r in results], label="Precision")
    plt.plot(thresholds, [r['recall'] for r in results], label="Recall")
    plt.plot(thresholds, [r['f1'] for r in results], label="F1 Score")
    plt.plot(thresholds, [r['fpr'] for r in results], label="FPR")
    plt.plot(thresholds, [r['fnr'] for r in results], label="FNR")

    plt.title("Threshold Performance Metrics (Median Distance)")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Best Threshold by F1
    best = max(results, key=lambda x: x['f1'])
    print("Best Threshold by F1:")
    print(best)
