# xAI and Data Analysis Tools for Drift Detection, Characterization, and Explanation

This repository contains the source code and notebooks for the Bachelor Thesis by Deniz Aksoy, Kuba Czech, Wojciech Nagórka, and Michał Redmer.

## About The Project

Modern machine learning systems often face performance degradation due to changes in the underlying data distribution over time, a phenomenon known as **concept drift**. While numerous methods exist to detect such drift, few provide meaningful explanations for *why* it occurs.

This project aims to develop a novel framework that not only detects but also explains concept drift using explainable artificial intelligence (xAI) techniques. The proposed approach integrates statistical drift detection methods with model-agnostic explainability tools and prototype-based analysis.

The expected outcome is a Python-based tool that generates interpretable explanations for drift phenomena, providing data scientists and machine learning practitioners with actionable insights and improved trust in model monitoring processes.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Make sure you have Python 3.8+ and pip installed.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/your_username/DriftDetectionWithExplainableAI.git
   ```
2. Install Python packages
   ```sh
   pip install -r requirements.txt
   ```

## Usage

The primary explorations and experiments are conducted in Jupyter Notebooks.

-   `notebooks/data_drift.ipynb`: Contains initial data generation, drift visualization, and implementation of various drift detection methods.
-   `src/`: Contains source code for specific functionalities, such as feature importance analysis.

## Dashboard

We provide an interactive dashboard to visualize data streams, generate synthetic drift scenarios, and analyze drift using various drift explanation methods.

To run the dashboard:

```sh
streamlit run dashboard/app.py
```

The dashboard allows you to:
1.  **Generate Data**: Create synthetic datasets with known concept drift (e.g., Hyperplane, Sea, Controlled).
2.  **Visualize**: View the data stream and drift points.
3.  **Analyze**: Apply different drift explanation methods to detect and explain data drift, concept drift, and predictive power shift.


## License

Distributed under the MIT License. See `LICENSE` for more information.