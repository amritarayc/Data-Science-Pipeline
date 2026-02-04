import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import wandb

import hydra
from omegaconf import DictConfig, OmegaConf


def compute_metrics(y_true, y_pred):
    metrics = {}
    for i in range(y_true.shape[1]):
        metrics[f"r2_output_{i}"] = r2_score(y_true[:, i], y_pred[:, i])
        metrics[f"mae_output_{i}"] = mean_absolute_error(y_true[:, i], y_pred[:, i])
        metrics[f"rmse_output_{i}"] = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
    metrics["r2_mean"] = r2_score(y_true, y_pred, multioutput="uniform_average")
    metrics["mae_mean"] = mean_absolute_error(y_true, y_pred)
    metrics["rmse_mean"] = np.sqrt(mean_squared_error(y_true, y_pred))
    return metrics


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Full config:\n", OmegaConf.to_yaml(cfg))

    data_cfg = cfg.data
    model_cfg = cfg.model
    log_cfg = cfg.logging

    # Load metadata
    metadata = pd.read_excel(data_cfg.metadata_path)

    input_vars = metadata.loc[metadata["VarType"] == "Input", "VarID"].tolist()
    output_vars = metadata.loc[metadata["VarType"] == "Output", "VarID"].tolist()

    # Load processed train/test data
    X_train = pd.read_csv(os.path.join(data_cfg.processed_dir, data_cfg.output_files.X_train))
    X_test  = pd.read_csv(os.path.join(data_cfg.processed_dir, data_cfg.output_files.X_test))
    y_train_df = pd.read_csv(os.path.join(data_cfg.processed_dir, data_cfg.output_files.y_train))
    y_test_df  = pd.read_csv(os.path.join(data_cfg.processed_dir, data_cfg.output_files.y_test))

    # Drop missing inputs/outputs
    input_vars = [v for v in input_vars if v in X_train.columns]
    output_vars = [v for v in output_vars if v in y_train_df.columns]

    if not output_vars:
        raise ValueError("No valid output variables found for training.")

    X_train = X_train[input_vars]
    X_test  = X_test[input_vars]
    y_train = y_train_df[output_vars].values
    y_test  = y_test_df[output_vars].values

    # Init W&B with descriptive name
    run_name = f"{model_cfg.model_name}_alpha{model_cfg.alpha}_outputs{len(output_vars)}"
    wandb.init(
        project=log_cfg.project,
        entity=log_cfg.entity,
        name=run_name,
        tags=log_cfg.tags,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Train model
    base_lasso = Lasso(alpha=model_cfg.alpha, max_iter=model_cfg.max_iter, random_state=data_cfg.random_state)
    model = MultiOutputRegressor(base_lasso)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    metrics = compute_metrics(y_test, preds)

    cv = KFold(n_splits=model_cfg.cv_folds, shuffle=True, random_state=data_cfg.random_state)
    metrics["cv_r2_mean"] = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2").mean()

    # Save coefficients
    coef_data = []
    for output_name, est in zip(output_vars, model.estimators_):
        for feature, coef in zip(input_vars, est.coef_):
            coef_data.append({
                "output_var": output_name,
                "output_name_real": metadata.loc[metadata["VarID"] == output_name, "VarName"].values[0],
                "feature_var": feature,
                "feature_name_real": metadata.loc[metadata["VarID"] == feature, "VarName"].values[0],
                "coefficient": coef
            })
    coef_df = pd.DataFrame(coef_data)
    os.makedirs(os.path.join(data_cfg.processed_dir, "DS2Model"), exist_ok=True)
    coef_df.to_csv(os.path.join(data_cfg.processed_dir, "DS2Model", "model_coefficients.csv"), index=False)

    # Save model
    model_path = os.path.join(data_cfg.processed_dir, "DS2Model", "lasso.pkl")
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")
    print("Metrics:", metrics)

    # Log to W&B
    wandb.log(metrics)
    wandb.log({"coefficients": wandb.Table(dataframe=coef_df)})
    wandb.finish()


if __name__ == "__main__":
    main()
