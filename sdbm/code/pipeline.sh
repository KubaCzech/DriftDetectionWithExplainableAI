#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- (1/5) Activating 'river_env' and generating hyperplane data... ---"
conda run -n river_env python generate_hyperplane_data.py

echo "--- (2/5) Activating 'bachelors' and running 'pre' experiment... ---"
conda run -n bachelors python experiments_4d_data_all_classifiers.py pre

echo "--- (3/5) Staying in 'bachelors' and running 'post' experiment... ---"
conda run -n bachelors python experiments_4d_data_all_classifiers.py post

echo "--- (4/5) Activating 'streamlit_env' and launching dashboard... ---"
echo "Warnings will be ignored."

# We use 'bash -c' to ensure the environment variable is set 
# for the streamlit command within the conda-run session.
conda run -n streamlit_env bash -c "PYTHONWARNINGS=ignore streamlit run dashboard.py"

echo "--- (5/5) Pipeline finished. Streamlit is running. ---"
