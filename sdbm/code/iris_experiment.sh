#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -euo pipefail

echo "--- (1/4) Activating 'river_env' and generating cleveland data... ---"
# Note: Your echo said "iris data", I corrected it to "cleveland" to match the script
conda run -n river_env python generate_breast_data.py

echo "--- (2/4) Activating 'bachelors' and running experiment (FIRST RUN - Computes DBMs)... ---"
conda run -n bachelors python experiments_dataset.py

echo "--- (3/4) Re-running 'bachelors' experiment (SECOND RUN - Generates DBMs w/ points)... ---"
conda run -n bachelors python experiments_dataset.py

echo "--- (4/4) All steps completed successfully. ---"