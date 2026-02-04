import os
import pandas as pd
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig
import numpy as np

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def preprocess(cfg: DictConfig):
    data_cfg = cfg.data
    metadata_path = data_cfg.metadata_path
    raw_data_path = data_cfg.raw_data_path

    # Load data
    if raw_data_path.endswith(".csv"):
        df = pd.read_csv(raw_data_path, decimal=",", thousands=".")
    elif raw_data_path.endswith(".xlsx"):
        df = pd.read_excel(raw_data_path)
    else:
        raise ValueError(f"Unsupported file type: {raw_data_path}")

    # Drop columns with >90% missing values
    df.dropna(axis=1, thresh=int((1 - 0.9) * len(df)), inplace=True)
    df = df.dropna()

    # Handle outputs from metadata
    metadata = pd.read_excel(metadata_path)
    output_vars = metadata.loc[metadata["VarType"] == "Output", "VarID"].tolist()
    print(f"Identified output variables: {', '.join(output_vars)}")

    #Create mapping of VarName to VarID
    varname_to_varid = dict(zip(metadata["VarName"], metadata["VarID"]))
    df.rename(columns=varname_to_varid, inplace=True)

    # Remove empty outputs
    existing_outputs = [v for v in output_vars if v in df.columns]
    missing_outputs = [v for v in output_vars if v not in df.columns]
    if missing_outputs:
        print(f"Dropped missing outputs: {', '.join(missing_outputs)}")

    # Drop rows with >90% missing
    threshold_row = 0.1 * df.shape[1]
    df = df.dropna(thresh=threshold_row, axis=0)
    df.reset_index(drop=True, inplace=True)

    # Replace - with 0s
    df.replace('-', np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Categorical encoding for all columns for which metadata ValueType is Categorical
    categorical_vars = metadata.loc[metadata["ValueType"] == "Categorical", "VarID"].tolist()
    for col in categorical_vars:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    # Remove variables that have metadata ValueType ID
    id_vars = metadata.loc[metadata["ValueType"] == "ID", "VarID"].tolist()
    df.drop(columns=[col for col in id_vars if col in df.columns], inplace=True)

    #Remove variables with zero variance
    nunique = df.nunique()
    zero_variance_cols = nunique[nunique <= 1].index.tolist()
    if zero_variance_cols:
        df.drop(columns=zero_variance_cols, inplace=True)
        print(f"Dropped zero variance columns: {', '.join(zero_variance_cols)}")
    
    # Find outliers in Vars where ValueType is Continuous and replace with mean of neighbors
    continuous_vars = metadata.loc[metadata["ValueType"] == "Continuous", "VarID"].tolist()
    for col in continuous_vars:
        if col in df.columns:
            series = df[col]
            q1 = series.quantile(0.10)
            q3 = series.quantile(0.90)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = (series < lower_bound) | (series > upper_bound)
            outlier_indices = series[outliers].index

            for idx in outlier_indices:
                if 0 < idx < len(series) - 1:
                    mean_neighbors = (series[idx - 1] + series[idx + 1]) / 2
                elif idx == 0:
                    mean_neighbors = series[idx + 1]
                else:  # idx == len(series) - 1
                    mean_neighbors = series[idx - 1]
                df.at[idx, col] = mean_neighbors
            if not outlier_indices.empty:
                print(f"Replaced outliers in column {col} at indices: {', '.join(map(str, outlier_indices.tolist()))}")
    
    # Define X, y (multi-output)
    X = df.drop(columns=existing_outputs)
    y = df[existing_outputs]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_cfg.test_size,
        random_state=data_cfg.random_state,
    )

    # Save
    os.makedirs(data_cfg.processed_dir, exist_ok=True)
    out_files = data_cfg.output_files
    X_train.to_csv(os.path.join(data_cfg.processed_dir, out_files["X_train"]), index=False)
    X_test.to_csv(os.path.join(data_cfg.processed_dir, out_files["X_test"]), index=False)
    y_train.to_csv(os.path.join(data_cfg.processed_dir, out_files["y_train"]), index=False)
    y_test.to_csv(os.path.join(data_cfg.processed_dir, out_files["y_test"]), index=False)

    print(f"Preprocessing complete. Files saved to {data_cfg.processed_dir}")

if __name__ == "__main__":
    preprocess()
