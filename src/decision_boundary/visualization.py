import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def visualize_decision_boundary(results, title="Decision Boundary Analysis"):
    """
    Visualize the decision boundary analysis results using HSV style.
    Hue = Class, Value = Confidence.

    Parameters
    ----------
    results : dict
        Output from compute_decision_boundary_analysis
    title : str
        Main title for the figure

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    res_pre = results['pre']
    res_post = results['post']

    # Identify all unique classes to ensure consistent coloring
    y_all = np.concatenate([res_pre['y_train'], res_post['y_train']])
    classes = np.unique(y_all)
    n_classes = len(classes)
    
    # Create color map
    # Using tab20 or tab10 depending on n_classes
    if n_classes <= 10:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = plt.get_cmap('tab20')
        
    class_to_idx = {c: i for i, c in enumerate(classes)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)

    def plot_window(ax, res, window_name):
        # 1. Prepare Grid Image from HSV Logic
        grid_labels = res['grid_labels']
        grid_probs = res['grid_probs']
        
        # Determine the RGB baselines for each pixel based on class
        idx_grid = np.zeros_like(grid_labels, dtype=int)
        for c, idx in class_to_idx.items():
            idx_grid[grid_labels == c] = idx
            
        # Get RGB colors (ignore alpha from cmap)
        # normalize index for cmap
        norm_indices = idx_grid / max(1, n_classes - 1) if n_classes > 1 else np.zeros_like(idx_grid, dtype=float)
        rgba_grid = cmap(norm_indices) # Shape: (H, W, 4)
        rgb_grid = rgba_grid[..., :3]  # Shape: (H, W, 3)
        
        # Convert to HSV
        hsv_grid = mcolors.rgb_to_hsv(rgb_grid)
        
        # Modulate Value (brightness) by probability
        # prob is 0..1. 
        # In original code: data_hsv[:, :, 2] = prob_matrix
        # This means high confidence = bright, low confidence = dark.
        hsv_grid[..., 2] = grid_probs 
        
        # Convert back to RGB
        final_rgb = mcolors.hsv_to_rgb(hsv_grid)
        
        # 2. Plot Image
        extent = res['grid_bounds'] # (xmin, xmax, ymin, ymax)
        
        ax.imshow(final_rgb, extent=extent, origin='lower', aspect='auto')
        
        # 3. Plot Data Points (Optional but requested to show them on plot)
        X_2d = res['X_2d']
        y_train = res['y_train']
        
        for c in classes:
            mask = (y_train == c)
            if not np.any(mask):
                continue
            
            idx = class_to_idx[c]
            # Solid color for points
            c_color = cmap(idx / max(1, n_classes - 1) if n_classes > 1 else 0)
            
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[c_color], label=f"Class {c}",
                       edgecolor='white', s=30, alpha=0.9, linewidth=0.5)

        ax.set_title(f"{window_name} Drift")
        ax.set_xlabel("SSNP Component 1")
        ax.set_ylabel("SSNP Component 2")
        # ax.grid(True, linestyle='--', alpha=0.3) # Grid might interfere with background visibility

    plot_window(axes[0], res_pre, "Pre")
    plot_window(axes[1], res_post, "Post")

    # Add legend (using handles from one of the plots)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(n_classes, 5), bbox_to_anchor=(0.5, 0.0))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust for legend
    return fig
