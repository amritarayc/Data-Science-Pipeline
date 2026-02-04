# Data Science Pipeline with DVC

This repository implements a modular, reproducible data preprocessing pipeline using DVC and YAML-based configuration. It is currently designed to split a raw dataset into train/test sets for further modeling.


## Tools & Technologies

DVC (Data Version Control): Tracks data, models, and pipeline stages.

Python: Core language used for scripting.

Git: For version control of code and DVC files.

## Project Structure
    .
    ├── configs/
    │   └── data/
    │       └── preprocess.yaml      # Configuration file
    ├── data/
    │   ├── raw/                     # Raw input data (not modified)
    │   │   └── Data1.xlsx
    │   │   └── Variable_Mapping_Data1.xlsx  
    │   └── processed/               # Outputs: train/test splits
    ├── src/data/preprocess.py       # Preprocessing script
    ├── src/model/model.py           # Training script
    ├── dvc.yaml                     # DVC pipeline definition
    ├── dvc.lock                     # Auto-generated pipeline state
    ├── env.yaml                     # To recreate python env
    ├── .gitignore
    └── README.md

## How to use

### Add your dataset

Place your raw CSV or XLSX file inside data/raw/ and update the path in 
    configs/data/preprocess.yaml

### Update configuration

Edit the following file to customize the pipeline for your dataset: configs/data/preprocess.yaml

    raw_data_path: data/raw/your_data.xlsx          # Path to your input file
    processed_dir: data/processed                  # Directory where outputs will be saved
    target_column: "your_target_column"            # Target column to predict

    test_size: 0.2                                  # Test split ratio
    random_state: 42                                # Random seed for reproducibility

    output_files:                                   # Filenames for outputs
    X_train: X_train.csv
    X_test: X_test.csv
    y_train: y_train.csv
    y_test: y_test.csv

### Model Training & Experiment Tracking

This project uses a Random Forest Regressor for regression modeling, with metrics and visualizations tracked via Weights & Biases (wandb).

The model configuration lives in: configs/model/model.yaml

### Viewing Metrics in W&B

After running, visit wandb.ai and navigate to your project to view:

    Logged evaluation metrics

    Actual vs Predicted scatter plot

    Experiment metadata (config, runtime, etc.)

### Set up Environment

First, make sure you have Anaconda or Miniconda installed.

Then run:

    conda env create -f env.yaml

This will create a new environment with all required packages.
To activate the environment:
s
    conda activate DSpipeline

### Set up Weights and Biases

Make sure you are logged into W&B:

    wandb login

Enter API Key


### Run the pipeline

If you're cloning this project for the first time:

    dvc init

Once the environment is set and activated:

    dvc repro