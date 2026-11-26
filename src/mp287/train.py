"""Train scikit-learn regressors, track runs in MLflow (SQLite), and register champion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Silence MLflow agent hint during automated runs
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

from src.mp287.cleaning import clean
from src.mp287.data_checks import compute_sha256


def load_params(params_path: str | Path = "params.yaml") -> dict[str, Any]:
    """Load project configuration from params.yaml using yaml.safe_load."""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pipeline(
    regressor: Any,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    """Build a preprocessing + regression Pipeline accepting the 8 API features."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def train_and_register(params_path: str | Path = "params.yaml") -> dict[str, Any]:
    """Run training experiments, log to SQLite MLflow, register best model as @champion."""
    params = load_params(params_path)

    raw_train_path = Path(params["data"]["raw_train"])
    data_sha256 = compute_sha256(raw_train_path)

    raw_df = pd.read_csv(raw_train_path)
    df = clean(raw_df)

    numeric_cols: list[str] = params["features"]["numeric"]
    categorical_cols: list[str] = params["features"]["categorical"]
    feature_cols = numeric_cols + categorical_cols
    target_col: str = params["features"]["target"]

    X = df[feature_cols]
    y = df[target_col]

    test_size = float(params["training"]["test_size"])
    random_state = int(params["training"]["random_state"])
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    tracking_uri: str = params["mlflow"]["tracking_uri"]
    experiment_name: str = params["mlflow"]["experiment_name"]
    model_name: str = params["mlflow"]["registered_model_name"]
    model_output_path = Path(params["training"]["model_output_path"])

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    candidates: list[tuple[str, dict[str, Any], Any]] = [
        ("ridge_alpha_1.0", {"model_type": "Ridge", "alpha": 1.0}, Ridge(alpha=1.0)),
        (
            "rf_n200_d10",
            {"model_type": "RandomForestRegressor", "n_estimators": 200, "max_depth": 10},
            RandomForestRegressor(n_estimators=200, max_depth=10, random_state=random_state),
        ),
        (
            "gbr_n200_lr0.05_d3",
            {
                "model_type": "GradientBoostingRegressor",
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 3,
            },
            GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=3, random_state=random_state
            ),
        ),
        (
            "gbr_n300_lr0.03_d3",
            {
                "model_type": "GradientBoostingRegressor",
                "n_estimators": 300,
                "learning_rate": 0.03,
                "max_depth": 3,
            },
            GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=3, random_state=random_state
            ),
        ),
    ]

    best_run_id: str | None = None
    best_val_mae = float("inf")
    best_pipeline: Pipeline | None = None
    best_run_name: str | None = None
    run_summaries: list[dict[str, Any]] = []

    for run_name, model_params, estimator in candidates:
        pipeline = build_pipeline(estimator, numeric_cols, categorical_cols)
        pipeline.fit(X_train, y_train)

        train_preds = pipeline.predict(X_train)
        val_preds = pipeline.predict(X_val)

        train_mae = float(mean_absolute_error(y_train, train_preds))
        val_mae = float(mean_absolute_error(y_val, val_preds))
        val_r2 = float(r2_score(y_val, val_preds))

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("train_rows", int(len(X_train)))
            mlflow.log_param("val_rows", int(len(X_val)))
            mlflow.log_params(model_params)
            mlflow.log_metric("train_mae", round(train_mae, 4))
            mlflow.log_metric("val_mae", round(val_mae, 4))
            mlflow.log_metric("val_r2", round(val_r2, 4))
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                serialization_format="cloudpickle",
            )

            run_summaries.append(
                {
                    "run_name": run_name,
                    "run_id": run.info.run_id,
                    "train_mae": round(train_mae, 4),
                    "val_mae": round(val_mae, 4),
                    "val_r2": round(val_r2, 4),
                }
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_run_id = run.info.run_id
                best_pipeline = pipeline
                best_run_name = run_name

    assert best_run_id is not None and best_pipeline is not None

    # Register best model and assign @champion alias
    model_uri = f"runs:/{best_run_id}/model"
    reg = mlflow.register_model(model_uri=model_uri, name=model_name)
    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=str(reg.version),
    )

    # Save champion pipeline to models/model.joblib
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_output_path)

    return {
        "experiment_name": experiment_name,
        "best_run_id": best_run_id,
        "best_run_name": best_run_name,
        "best_val_mae": round(best_val_mae, 4),
        "model_name": model_name,
        "champion_version": int(reg.version),
        "model_path": str(model_output_path),
        "runs": run_summaries,
    }


if __name__ == "__main__":
    summary = train_and_register()
    print(yaml.safe_dump(summary, sort_keys=False))
