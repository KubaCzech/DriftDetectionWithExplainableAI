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

# ===========================================================
# Image Saving Utility
# ===========================================================
def results_to_png(np_matrix, prob_matrix, grid_size, n_classes,
                   dataset_name, classifier_name, real_points=None,
                   max_value_hsv=None, suffix=None):
    output_dir = "./results_hyperplane"
    os.makedirs(output_dir, exist_ok=True)

    suffix = f"_{suffix}" if suffix else ""
    data = cm.tab20(np_matrix / n_classes)
    data_vanilla = data[:, :, :3].copy()

    if max_value_hsv is not None:
        data_vanilla = rgb2hsv(data_vanilla)
        data_vanilla[:, :, 2] = max_value_hsv
        data_vanilla = hsv2rgb(data_vanilla)

    if real_points is not None:
        data_vanilla = rgb2hsv(data_vanilla)
        data_vanilla[real_points[:, 0], real_points[:, 1], 2] = 1
        data_vanilla = hsv2rgb(data_vanilla)

    data_alpha = data.copy()
    data_hsv = data[:, :, :3].copy()
    data_alpha[:, :, 3] = prob_matrix

    data_hsv = rgb2hsv(data_hsv)
    data_hsv[:, :, 2] = prob_matrix
    data_hsv = hsv2rgb(data_hsv)

    rescaled_vanilla = (data_vanilla * 255.0).astype(np.uint8)
    im = Image.fromarray(rescaled_vanilla)
    im.save(os.path.join(output_dir, f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_vanilla{suffix}.png"))

    rescaled_alpha = (255.0 * data_alpha).astype(np.uint8)
    im = Image.fromarray(rescaled_alpha)
    im.save(os.path.join(output_dir, f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_alpha{suffix}.png"))

    rescaled_hsv = (255.0 * data_hsv).astype(np.uint8)
    im = Image.fromarray(rescaled_hsv)
    im.save(os.path.join(output_dir, f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_hsv{suffix}.png"))


# ===========================================================
# Main Experiment
# ===========================================================
if __name__ == "__main__":
    # ===========================================================
    # CLI Switch
    # ===========================================================
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "pre"
    if mode not in ["pre", "post"]:
        raise ValueError("Invalid mode. Use 'pre' or 'post'.")

    drift_label = "pre_drift" if mode == "pre" else "post_drift"

    print(f"[INFO] Running experiment with {drift_label} data.")

    patience = 5
    epochs = 10
    verbose = False
    grid_size = 300
    dataset_name = f"hyperplane4d_{drift_label}"
    output_dir = "./results_hyperplane"
    os.makedirs(output_dir, exist_ok=True)

    # ===========================================================
    # Load Data
    # ===========================================================
    print(f"[INFO] Loading 4D hyperplane {drift_label} data...")
    X_file = f"X_{drift_label}.npy"
    y_file = f"y_{drift_label}.npy"

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

            img_grid_ssnp[normalized[:, 0], normalized[:, 1]] = y_train
            prob_grid_ssnp[normalized[:, 0], normalized[:, 1]] = 1.0

            normalized[:, 0] = (grid_size - 1) - normalized[:, 0]
            

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

        if mode == "pre":
            # If we're in "pre" mode, we train or load the projector as usual
            if os.path.exists(projector_path):
                print(f"[INFO] Loading existing pre-drift SSNP model: {projector_name}")
                ssnpgt.load_model(projector_path)
            else:
                print(f"[INFO] Training new pre-drift SSNP model: {projector_name}...")
                ssnpgt.fit(X_train, y_train)
                ssnpgt.save_model(projector_path)
                print(f"[INFO] Saved pre-drift projector.")
        
        elif mode == "post":
            # If we're in "post" mode, we MUST load the "pre-drift" projector
            if not os.path.exists(projector_path):
                raise FileNotFoundError(
                    f"[ERROR] Pre-drift projector not found at: {projector_path}\n"
                    "Please run the 'pre' mode first to train the SSNP model."
                )
            
            print(f"[INFO] Loading PRE-DRIFT SSNP model for post-drift data: {projector_name}")
            ssnpgt.load_model(projector_path)
        

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
