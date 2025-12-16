import matplotlib.pyplot as plt
from full_window_storage import FullWindowStorage


def plot_prototype_comparison(storage: FullWindowStorage, windows_to_compare: list[int]):
    # First compute global min/max across ALL prototypes for nice y scale
    global_min = float('inf')
    global_max = -float('inf')

    for window_nr in windows_to_compare:
        x, y, prototypes, explainer = storage.get_window_data(window_nr)
        for class_name in set(y):
            for prototype in prototypes[class_name]:
                values = [v for _, v in sorted(prototype.items(), key=lambda x: x[0])]
                local_min = min(values)
                local_max = max(values)
                global_min = min(global_min, local_min)
                global_max = max(global_max, local_max)

    margin = 0.05*(global_max - global_min)
    used_min = global_min - margin
    used_max = global_max + margin

    fig, ax = plt.subplots(len(set(prototypes.keys())), len(windows_to_compare), figsize=(12, 6), sharey=True)

    # Plot prototypes
    for col, window_nr in enumerate(windows_to_compare):
        x, y, prototypes, explainer = storage.get_window_data(window_nr)
        classes_sorted = sorted(list(set(prototypes.keys())))

        for row, class_name in enumerate(classes_sorted):
            for prototype in prototypes[class_name]:

                items = sorted(prototype.items(), key=lambda x: x[0])
                feature_names = [str(k) for k, _ in items]
                feature_values = [v for _, v in items]

                ax[row, col].plot(feature_names, feature_values)
                ax[row, col].tick_params(axis='x', labelrotation=45)
                num_prototypes = len(prototypes[class_name])
                ax[row, col].text(
                    0.02, 0.95,
                    f"n={num_prototypes}",
                    transform=ax[row, col].transAxes,
                    fontsize=9,
                    verticalalignment='top'
                )

            ax[row, col].set_ylim(used_min, used_max)

    # Column titles (windows)
    for col, window_nr in enumerate(windows_to_compare):
        ax[0, col].set_title(f"Window {window_nr}")

    # Row titles (classes)
    for row, class_name in enumerate(classes_sorted):
        ax[row, 0].set_ylabel(f"Class {class_name}", rotation=90, labelpad=10)

    plt.tight_layout()
    plt.show()
