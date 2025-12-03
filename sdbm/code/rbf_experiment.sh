#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- (1/5) Activating 'river_env' and generating hyperplane data... ---"
conda run -n river_env python generate_rbf_data.py

echo "--- (2/5) Activating 'bachelors' and running 'pre' experiment (FIRST RUN - Computes DBMs)... ---"
conda run -n bachelors python experiments_4d_data_all_classifiers.py pre

echo "--- (3/5) Activating 'bachelors' and running 'post' experiment (FIRST RUN - Computes DBMs)... ---"
conda run -n bachelors python experiments_4d_data_all_classifiers.py post

echo "--- (4/5) Re-running 'pre' (SECOND RUN - Generates DBMs w/ points)... ---"
conda run -n bachelors python experiments_4d_data_all_classifiers.py pre

echo "--- (5/5) Re-running 'post' (SECOND RUN - Generates DBMs w/ points)... ---"
conda run -n bachelors python experiments_4d_data_all_classifiers.py post

conda run -n bachelors python disagreement.py

echo "--- Activating 'streamlit_env' and launching dashboard... ---"
echo "Warnings will be ignored."

conda run -n streamlit_env bash -c "PYTHONWARNINGS=ignore streamlit run dashboard.py"

echo "--- (5/5) Pipeline finished. Streamlit is running. ---"
