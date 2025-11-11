import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from time import time
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.extmath import cartesian
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from skimage.color import rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ssnp 

def results_to_png(np_matrix, prob_matrix, grid_size, n_classes,
                   dataset_name, classifier_name, real_points=None,
                   max_value_hsv=1.0, suffix=None):
    """
    Render decision boundary images with:
      - hue = class label (consistent mapping across runs)
      - saturation = classifier confidence (probability)
      - value  = max_value_hsv (brightness; default 1.0)
    This makes low-confidence areas desaturated/pale (visible as boundaries).
    """
    import os
    import numpy as np
    from PIL import Image
    import matplotlib.cm as cm
    from skimage.color import hsv2rgb, rgb2hsv

    output_dir = f"./results_{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    suffix = f"_{suffix}" if suffix else ""

    # --- Ensure integer labels and consistent mapping across runs ---
    np_matrix = np_matrix.astype(int)
    unique_labels = np.unique(np_matrix)
    # Build a full mapping for labels 0..(n_classes-1). If some labels are missing,
    # we still want a stable mapping for the full class set.
    sorted_labels = sorted(unique_labels.tolist())
    label_map = {lab: i for i, lab in enumerate(sorted_labels)}

    mapped = np.vectorize(lambda v: label_map.get(int(v), 0))(np_matrix)
    mapped = mapped.astype(int)

    # --- Build hue image: map class index -> categorical colormap (tab20 or other) ---
    # Use tab20 to obtain RGB colors per class, then extract hue.
    # We sample evenly across the colormap using the full n_classes range for stability.
    palette = cm.get_cmap("tab20", max(1, n_classes))
    # Create an RGB image (grid_size x grid_size x 3) from mapped class indices:
    normalized_indices = mapped / max(1, (n_classes - 1))  # in [0,1]
    rgb_image = palette(normalized_indices)[:, :, :3]  # returns floats in [0,1]

    # Convert RGB -> HSV to control S (saturation) and V (value)
    hsv_image = rgb2hsv(rgb_image)

    # --- Apply probability as saturation (S) ---
    prob_matrix = np.clip(prob_matrix.astype(float), 0.0, 1.0)
    # If prob_matrix is not the same shape as grid, try to broadcast or raise
    if prob_matrix.shape != (grid_size, grid_size):
        # attempt safe broadcast/reshape: if flattened, reshape
        try:
            prob_matrix = prob_matrix.reshape((grid_size, grid_size))
        except Exception:
            raise ValueError("prob_matrix has incompatible shape with grid_size")

    # Set saturation to probability (low prob -> desaturated -> pale/white)
    hsv_image[:, :, 1] = prob_matrix

    # Set value (brightness) to provided max value (defaults to 1.0)
    v_val = float(max_value_hsv) if max_value_hsv is not None else 1.0
    v_val = np.clip(v_val, 0.0, 1.0)
    hsv_image[:, :, 2] = v_val

    # Optionally highlight real_points by forcing value=1 and saturation=1 there
    if real_points is not None:
        rp = np.asarray(real_points, dtype=int)
        # Clip indices
        rp[:, 0] = np.clip(rp[:, 0], 0, grid_size - 1)
        rp[:, 1] = np.clip(rp[:, 1], 0, grid_size - 1)
        hsv_image[rp[:, 0], rp[:, 1], 1] = 1.0
        hsv_image[rp[:, 0], rp[:, 1], 2] = 1.0

    # Convert back to RGB for saving
    rgb_from_hsv = hsv2rgb(hsv_image)
    rgb_from_hsv = np.clip(rgb_from_hsv, 0.0, 1.0)

    # --- Save vanilla (pure class color, fixed brightness) ---
    vanilla_rgb = palette(normalized_indices)[:, :, :3].copy()
    # If user asked for a max_value_hsv override for vanilla, apply (optional):
    if max_value_hsv is not None:
        vanilla_hsv = rgb2hsv(vanilla_rgb)
        vanilla_hsv[:, :, 2] = v_val
        vanilla_rgb = hsv2rgb(vanilla_hsv)
    vanilla_img = (vanilla_rgb * 255).astype(np.uint8)
    Image.fromarray(vanilla_img).save(
        os.path.join(output_dir, f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_vanilla{suffix}.png")
    )

    # --- Save alpha (RGBA) version where alpha = probability ---
    rgba = palette(normalized_indices).copy()  # shape (..., 4)
    if rgba.shape[2] == 4:
        rgba = rgba.copy()
        rgba[:, :, 3] = prob_matrix
    else:
        # palette returned RGB; append alpha channel
        alpha_chan = prob_matrix[:, :, np.newaxis]
        rgba = np.concatenate([rgba, alpha_chan], axis=2)
    rgba_img = (rgba * 255).astype(np.uint8)
    Image.fromarray(rgba_img, mode="RGBA").save(
        os.path.join(output_dir, f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_alpha{suffix}.png")
    )

    # --- Save HSV-based brightness/confidence image (the one papers often show) ---
    hsv_based_img = (rgb_from_hsv * 255).astype(np.uint8)
    Image.fromarray(hsv_based_img).save(
        os.path.join(output_dir, f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_hsv{suffix}.png")
    )



# ===========================================================
# Main Experiment
# ===========================================================
if __name__ == "__main__":
   

    dataset_name="breast"

    print(f"[INFO] Running experiment with {dataset_name} data.")

    patience = 5
    epochs = 10
    verbose = False
    grid_size = 300
    output_dir = f"./results_{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    # ===========================================================
    # Load Data
    # ===========================================================
    print(f"[INFO] Loading {dataset_name} data...")
    X_file = f"X_{dataset_name}.npy"
    y_file = f"y_{dataset_name}.npy"

    if not os.path.exists(X_file) or not os.path.exists(y_file):
        raise FileNotFoundError(f"Missing required .npy files: {X_file}, {y_file}")

    X = np.load(X_file)
    y = np.load(y_file)
    print(f"[INFO] Loaded X shape: {X.shape}, y shape: {y.shape}")

    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    train_size = min(int(n_samples * 0.9), 5000)
    test_size = 1000

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42, stratify=y
    )

    # ===========================================================
    # Define Classifiers
    # ===========================================================
    classifier_names = ["lr", "svm", "rf", "mlp"]
    classifiers = [
        linear_model.LogisticRegression(max_iter=500),
        SVC(kernel="rbf", probability=True),
        RandomForestClassifier(n_estimators=200),
        MLPClassifier(hidden_layer_sizes=(200,) * 3, max_iter=300),
    ]

    # ===========================================================
    # Run Each Classifier
    # ===========================================================
    for clf_name, clf in zip(classifier_names, classifiers):
        print(f"\n==================== {clf_name.upper()} ====================")
        out_name = f"{clf_name}_{grid_size}x{grid_size}_{dataset_name}"
        out_file = os.path.join(output_dir, out_name + ".npy")

        if os.path.exists(out_file):
            print(f"[INFO] Found existing results for {clf_name}. Skipping computation.")
            img_grid_ssnp = np.load(out_file)
            prob_grid_ssnp = np.load(os.path.join(output_dir, out_name + "_prob.npy"))
            prob_grid_ssnp = prob_grid_ssnp.clip(max=0.8)

            X_ssnpgt = np.load(os.path.join(output_dir, f"X_SSNP_{dataset_name}.npy"))

            scaler = MinMaxScaler()
            scaler.fit(X_ssnpgt)
            normalized = scaler.transform(X_ssnpgt)
            normalized *= (grid_size - 1)
            normalized = normalized.astype(int)

            normalized[:, 0] = (grid_size - 1) - normalized[:, 0]

            img_grid_ssnp[normalized[:, 0], normalized[:, 1]] = y_train
            prob_grid_ssnp[normalized[:, 0], normalized[:, 1]] = 1.0

            results_to_png(
                np_matrix=img_grid_ssnp,
                prob_matrix=prob_grid_ssnp,
                grid_size=grid_size,
                n_classes=n_classes,
                real_points=normalized,
                max_value_hsv=0.8,
                dataset_name=dataset_name,
                classifier_name=clf_name,
                suffix="ssnp_w_real",
            )
            continue

        # ===========================================================
        # SSNP Projection
        # ===========================================================
        X_ssnpgt_proj_file = f"X_SSNP_{dataset_name}.npy"
        n_features = X_train.shape[1]
        pre_drift_dataset_name = f"hyperplane{n_features}d_pre_drift"
        projector_name = f"{pre_drift_dataset_name}_ssnp"
        projector_path = os.path.join(output_dir, projector_name)

        ssnpgt = ssnp.SSNP(
            epochs=epochs,
            verbose=verbose,
            patience=patience,
            opt="adam",
            bottleneck_activation="linear",
        )

       
        if os.path.exists(projector_path):
            print(f"[INFO] Loading existing pre-drift SSNP model: {projector_name}")
            ssnpgt.load_model(projector_path)
        else:
            print(f"[INFO] Training new pre-drift SSNP model: {projector_name}...")
            ssnpgt.fit(X_train, y_train)
            ssnpgt.save_model(projector_path)
            print(f"[INFO] Saved pre-drift projector.")
        
      
        

        # ===========================================================
        # Train Classifier
        # ===========================================================
        name = f"{dataset_name}_{clf_name}.pkl"
        if os.path.exists(os.path.join(output_dir, name)):
            clf = pickle.load(open(os.path.join(output_dir, name), "rb"))
        else:
            print("[INFO] Training classifier...")
            start = time()
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            print(f"[INFO] Test Accuracy: {acc:.4f}")
            endtime = time() - start
            print(f"[INFO] Training completed in {endtime:.2f}s")
            # pickle.dump(clf, open(os.path.join(output_dir, name), "wb"))
            # with open(os.path.join(output_dir, f"{dataset_name}_{clf_name}.txt"), "w") as f:
            #     f.write(f"Accuracy: {acc}\nTraining time: {endtime:.2f}s\n")

        # ===========================================================
        # Project Training Data
        # ===========================================================
        if os.path.exists(os.path.join(output_dir, X_ssnpgt_proj_file)):
            print("[INFO] Found projected points.")
            X_ssnpgt = np.load(os.path.join(output_dir, X_ssnpgt_proj_file))
        else:
            print("[INFO] Projecting data with SSNP...")
            X_ssnpgt = ssnpgt.transform(X_train)
            np.save(os.path.join(output_dir, X_ssnpgt_proj_file), X_ssnpgt)
            print("[INFO] Saved projected points.")

       # ===========================================================
        # Save 2D Projection Scatter Plot (with Legend)
        # ===========================================================
        projection_png_file = os.path.join(output_dir, f"projection_scatter_{dataset_name}.png")
        
        if not os.path.exists(projection_png_file):
            print(f"[INFO] Saving 2D projection scatter plot to {projection_png_file}...")
            plt.figure(figsize=(10, 8))
            
            plt.title(f"SSNP 2D Projection - {dataset_name}")
            
            # --- AXIS LABELS ARE SWAPPED ---
            plt.xlabel("SSNP Component 2") 
            plt.ylabel("SSNP Component 1")

            # Get unique classes and map them to colors from tab20
            unique_classes = np.unique(y_train)
            n_classes = len(unique_classes)
            
            # Plot each class separately to build the legend
            for i, class_label in enumerate(unique_classes):
                # Get all points for the current class
                class_points = X_ssnpgt[y_train == class_label]
                
                # Get the corresponding color from the colormap
                # We divide by n_classes to map the index to the [0, 1] range
                color = cm.tab20(i / n_classes) 

                # --- COMPONENTS ARE SWAPPED ---
                plt.scatter(
                    class_points[:, 1],  # Component 2 on X-axis
                    class_points[:, 0],  # Component 1 on Y-axis
                    c=[color],  # Pass color as a list for a single-color scatter
                    s=3,        # Use small points
                    alpha=0.6,  # Add transparency
                    label=f"Class {int(class_label)}" # Label for the legend
                )
            
            # Add the legend
            # 'markerscale=3' makes the legend markers larger and easier to see
            plt.legend(title="Classes", markerscale=3, loc="best") 
            
            # plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.savefig(projection_png_file, dpi=200, bbox_inches='tight')
            plt.close() # Close the figure to free up memory
            print(f"[INFO] Saved 2D projection plot: {projection_png_file}")
        # ===========================================================
        # Create Decision Boundary Map
        # ===========================================================
        print("[INFO] Creating DBM grid...")
        scaler = MinMaxScaler()
        scaler.fit(X_ssnpgt)
        xmin, xmax = np.min(X_ssnpgt[:, 0]), np.max(X_ssnpgt[:, 0])
        ymin, ymax = np.min(X_ssnpgt[:, 1]), np.max(X_ssnpgt[:, 1])

        img_grid = np.zeros((grid_size, grid_size))
        prob_grid = np.zeros((grid_size, grid_size))

        x_intrvls = np.linspace(xmin, xmax, num=grid_size)
        y_intrvls = np.linspace(ymin, ymax, num=grid_size)
        x_grid = np.linspace(0, grid_size - 1, num=grid_size)
        y_grid = np.linspace(0, grid_size - 1, num=grid_size)

        pts = cartesian((x_intrvls, y_intrvls))
        pts_grid = cartesian((x_grid, y_grid)).astype(int)

        batch_size = 100000
        pbar = tqdm(total=len(pts))
        position = 0

        while position < len(pts):
            pts_batch = pts[position:position + batch_size]
            image_batch = ssnpgt.inverse_transform(pts_batch)

            probs = clf.predict_proba(image_batch)
            alpha = np.amax(probs, axis=1)
            label_indices = probs.argmax(axis=1) 
            labels = clf.classes_[label_indices]
          
            pts_grid_batch = pts_grid[position:position + batch_size]
            img_grid[pts_grid_batch[:, 0], pts_grid_batch[:, 1]] = labels
            prob_grid[pts_grid_batch[:, 0], pts_grid_batch[:, 1]] = alpha
            position += batch_size
            pbar.update(batch_size)

        pbar.close()
        img_grid = np.flipud(img_grid)
        prob_grid = np.flipud(prob_grid)

        np.save(out_file, img_grid)
        np.save(os.path.join(output_dir, f"{out_name}_prob.npy"), prob_grid)

        results_to_png(
            np_matrix=img_grid,
            prob_matrix=prob_grid,
            grid_size=grid_size,
            dataset_name=dataset_name,
            classifier_name=clf_name,
            n_classes=n_classes,
        )

    print("\n Experiment finished for all classifiers.")
