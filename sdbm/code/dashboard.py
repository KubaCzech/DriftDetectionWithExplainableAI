import streamlit as st
import os
from PIL import Image

# --- Configuration ---
RESULTS_DIR = "./results_hyperplane"
CLASSIFIERS = ["lr", "svm", "rf", "mlp"]
GRID_SIZE = 300 # Must match the grid_size in your experiment script
# ---------------------

st.set_page_config(layout="wide")
st.title("Drift Analysis Dashboard: Pre-Drift vs. Post-Drift")

# Helper function to find and display an image
def display_image(image_path, caption):
    """Checks if an image exists and displays it or a 'not found' message."""
    if os.path.exists(image_path):
        try:
            image = Image.open(image_path)
            # --- THIS LINE IS THE FIX ---
            st.image(image, caption=caption, width="stretch")
        except Exception as e:
            st.error(f"Error loading image {image_path}: {e}")
    else:
        st.warning(f"Image not found:\n{image_path}\n\n(Have you run the experiment script for both 'pre' and 'post' modes?)")
# --- Section 1: 2D Projections (Classifier-Independent) ---
st.header("SSNP 2D Projection Scatter Plots")
st.markdown("This shows the 2D projection of the *training data* for each drift stage.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pre-Drift Projection")
    pre_proj_name = f"projection_scatter_hyperplane4d_pre_drift.png"
    pre_proj_path = os.path.join(RESULTS_DIR, pre_proj_name)
    display_image(pre_proj_path, "Pre-Drift SSNP Projection")

with col2:
    st.subheader("Post-Drift Projection")
    post_proj_name = f"projection_scatter_hyperplane4d_post_drift.png"
    post_proj_path = os.path.join(RESULTS_DIR, post_proj_name)
    display_image(post_proj_path, "Post-Drift SSNP Projection")


# --- Section 2: Decision Boundary Maps (Classifier-Dependent) ---
st.divider()
st.header("Classifier Decision Boundary Maps (DBM)")
st.markdown("This shows the decision boundaries learned by each classifier, visualized on the 2D projection.")

# Create a dropdown menu to select the classifier
selected_clf = st.selectbox("Select Classifier to Compare:", CLASSIFIERS)

if selected_clf:
    col3, col4 = st.columns(2)

    # --- Pre-Drift DBM ---
    with col3:
        st.subheader(f"Pre-Drift DBM ({selected_clf.upper()})")
        dataset_name_pre = "hyperplane4d_pre_drift"
        
        # Your script saves two types of images, let's prioritize the one with real points
        img_name_real_pre = f"{selected_clf}_{GRID_SIZE}x{GRID_SIZE}_{dataset_name_pre}_hsv_ssnp_w_real.png"
        img_name_plain_pre = f"{selected_clf}_{GRID_SIZE}x{GRID_SIZE}_{dataset_name_pre}_hsv.png"
        
        # Check which file exists
        pre_dbm_path = os.path.join(RESULTS_DIR, img_name_real_pre)
        if not os.path.exists(pre_dbm_path):
             pre_dbm_path = os.path.join(RESULTS_DIR, img_name_plain_pre)
             
        display_image(pre_dbm_path, f"Pre-Drift DBM for {selected_clf.upper()}")

    # --- Post-Drift DBM ---
    with col4:
        st.subheader(f"Post-Drift DBM ({selected_clf.upper()})")
        dataset_name_post = "hyperplane4d_post_drift"
        
        img_name_real_post = f"{selected_clf}_{GRID_SIZE}x{GRID_SIZE}_{dataset_name_post}_hsv_ssnp_w_real.png"
        img_name_plain_post = f"{selected_clf}_{GRID_SIZE}x{GRID_SIZE}_{dataset_name_post}_hsv.png"

        # Check which file exists
        post_dbm_path = os.path.join(RESULTS_DIR, img_name_real_post)
        if not os.path.exists(post_dbm_path):
             post_dbm_path = os.path.join(RESULTS_DIR, img_name_plain_post)

        display_image(post_dbm_path, f"Post-Drift DBM for {selected_clf.upper()}")