"""Drift monitoring, per-feeder MAE, v2 retraining and the promotion rule (PROBLEM.md section 7)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from src.mp287.cleaning import clean
from src.mp287.data_checks import compute_sha256
from src.mp287.train import load_params


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index of `actual` against the reference `expected`.

    Bin edges are the deciles of the reference (cleaned train), with open
    outer edges so every later value falls in some bin. Empty bins get a
    tiny share `eps` so the log never divides by zero.
    PSI = sum((actual% - expected%) * ln(actual% / expected%)).
    """
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    exp_share = np.histogram(expected, bins=edges)[0] / len(expected)
    act_share = np.histogram(actual, bins=edges)[0] / len(actual)
    exp_share = np.clip(exp_share, eps, None)
    act_share = np.clip(act_share, eps, None)
    return float(np.sum((act_share - exp_share) * np.log(act_share / exp_share)))


def _register_v2(
    v2_model: Any,
    params: dict[str, Any],
    data_sha256: str,
    metrics: dict[str, float],
    train_rows: int,
) -> int:
    """Log v2 as an MLflow run and register it. Reuses version 2 if it already exists."""
    tracking_uri = params["mlflow"]["tracking_uri"]
    model_name = params["mlflow"]["registered_model_name"]
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.tags.get("role") == "v2":
            return int(mv.version)

    # Own experiment with a local artifact folder (v1's experiment points at another PC's disk).
    exp_name = params["monitoring"]["retrain_experiment_name"]
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name, artifact_location=Path("mlartifacts").resolve().as_uri())
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name="v2_train_plus_later70") as run:
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("train_rows", train_rows)
        mlflow.log_param("later_train_fraction", params["monitoring"]["later_train_fraction"])
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(v2_model, name="model", serialization_format="cloudpickle")
        run_id = run.info.run_id

    mv = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client.set_model_version_tag(model_name, mv.version, "role", "v2")
    return int(mv.version)


def run_monitoring(params_path: str | Path = "params.yaml", update_findings: bool = True) -> dict[str, Any]:
    params = load_params(params_path)
    feats = params["features"]
    numeric_cols: list[str] = feats["numeric"]
    feature_cols = numeric_cols + feats["categorical"]
    target = feats["target"]
    mon = params["monitoring"]

    raw_train_path = params["data"]["raw_train"]
    train = clean(pd.read_csv(raw_train_path))
    later = pd.read_csv(params["data"]["raw_later"])  # PROBLEM.md: later.csv arrives clean

    # v1 = the model Member 1 trained on cleaned train.csv only. It has its own file,
    # because models/model.joblib always holds the current champion (v2 after promotion).
    v1 = joblib.load(mon["v1_model_path"])

    # 1. Drift: PSI for every numeric column, cleaned train vs later.
    psi_scores = {c: round(psi(train[c], later[c], bins=int(mon["psi_bins"])), 4) for c in numeric_cols}
    most_drifted = max(psi_scores, key=psi_scores.get)

    # 2. v1's MAE on later.csv for each feeder.
    later_pred = v1.predict(later[feature_cols])
    mae_by_group = {
        str(feeder): round(float(mean_absolute_error(g[target], later_pred[g.index])), 4)
        for feeder, g in later.groupby("feeder")
    }
    worst_group = max(mae_by_group, key=mae_by_group.get)

    # 3. Sort by date, retrain v2 on cleaned train + first 70% of later, compare on last 30%.
    later_sorted = later.sort_values(["date", "record_id"], kind="stable").reset_index(drop=True)
    cut = int(round(len(later_sorted) * float(mon["later_train_fraction"])))
    later_fit, check = later_sorted.iloc[:cut], later_sorted.iloc[cut:]

    v2_train = pd.concat([train, later_fit], ignore_index=True)
    v2 = clone(v1).fit(v2_train[feature_cols], v2_train[target])  # same settings as v1, new data

    mae_v1 = round(float(mean_absolute_error(check[target], v1.predict(check[feature_cols]))), 4)
    mae_v2 = round(float(mean_absolute_error(check[target], v2.predict(check[feature_cols]))), 4)

    # 4. Promotion rule: v2 only if its MAE is at least 8% lower than v1's.
    min_gain = float(mon["promotion_min_mae_improvement"])
    improvement = (mae_v1 - mae_v2) / mae_v1
    decision = "promote" if improvement >= min_gain else "keep"

    v2_version = _register_v2(
        v2,
        params,
        compute_sha256(raw_train_path),
        {"mae_v1_on_check": mae_v1, "mae_v2_on_check": mae_v2},
        train_rows=len(v2_train),
    )
    model_name = params["mlflow"]["registered_model_name"]
    client = MlflowClient(tracking_uri=params["mlflow"]["tracking_uri"])
    if decision == "promote":
        client.set_registered_model_alias(model_name, "champion", str(v2_version))
        joblib.dump(v2, params["training"]["model_output_path"])  # the API serves the champion
    champion_version = int(client.get_model_version_by_alias(model_name, "champion").version)

    results = {
        "psi": psi_scores,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_group,
        "mae_v1_on_check": mae_v1,
        "mae_v2_on_check": mae_v2,
        "decision": decision,
    }
    if update_findings:
        findings_path = Path(mon["findings_path"])
        findings = yaml.safe_load(findings_path.read_text(encoding="utf-8"))
        findings["monitoring"] = results
        findings_path.write_text(yaml.safe_dump(findings, sort_keys=False), encoding="utf-8")

    return {
        **results,
        "mae_improvement": round(improvement, 4),
        "check_rows": len(check),
        "v2_version": v2_version,
        "champion_version": champion_version,
    }


if __name__ == "__main__":
    print(yaml.safe_dump(run_monitoring(), sort_keys=False))
