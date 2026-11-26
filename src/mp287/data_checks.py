"""Detect the seven potential data problems in MP287 training data."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

CANONICAL_CONNECTIONS = {"domestic", "domestic_plus", "shop_cum_home"}
CANONICAL_FEEDERS = {"Feeder-1", "Feeder-2", "Feeder-3"}
NUMERIC_FEATURE_COLS = [
    "rooms",
    "members",
    "ac_hours_per_day",
    "avg_temp_c",
    "house_area_sqft",
    "solar_kw",
]
KNOWN_SENTINELS = {-999, -99, -1, 999, 9999, -999.0, -99.0, -1.0, 999.0, 9999.0}


def compute_sha256(path: str | Path) -> str:
    """Return lowercase hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_leaking_column(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame | None = None,
    target_col: str = "units_kwh",
    suspect_col: str = "billed_units_after_reading",
) -> dict[str, Any]:
    """Check if suspect_col leaks the target and is absent from holdout."""
    in_train = suspect_col in train_df.columns
    in_holdout = suspect_col in holdout_df.columns if holdout_df is not None else None
    corr = (
        round(float(train_df[suspect_col].corr(train_df[target_col])), 6)
        if in_train and target_col in train_df.columns
        else None
    )
    max_abs_diff = (
        int((train_df[suspect_col] - train_df[target_col]).abs().max())
        if in_train and target_col in train_df.columns
        else None
    )
    detected = bool(in_train and (in_holdout is False or (corr is not None and abs(corr) > 0.95)))
    return {
        "present": detected,
        "column": suspect_col if detected else None,
        "in_train": in_train,
        "in_holdout": in_holdout,
        "correlation_with_target": corr,
        "max_abs_diff": max_abs_diff,
    }


def check_unit_mixup(train_df: pd.DataFrame) -> dict[str, Any]:
    """Check whether any feeder has a systematic unit scale shift (>2x median ratio)."""
    if "feeder" not in train_df.columns:
        return {"present": False, "details": None}

    anomalies: list[dict[str, Any]] = []
    for col in NUMERIC_FEATURE_COLS + ["units_kwh"]:
        if col not in train_df.columns:
            continue
        global_med = float(train_df[col].median())
        if global_med == 0:
            continue
        feeder_meds = train_df.groupby("feeder")[col].median()
        for feeder, med in feeder_meds.items():
            ratio = float(med) / global_med
            if ratio > 2.0 or ratio < 0.5:
                anomalies.append({"column": col, "feeder": feeder, "ratio": round(ratio, 3)})

    return {
        "present": len(anomalies) > 0,
        "anomalies": anomalies if anomalies else None,
    }


def check_sentinel_values(train_df: pd.DataFrame) -> dict[str, Any]:
    """Check for negative values or known numeric sentinel codes."""
    found: dict[str, list[float]] = {}
    for col in NUMERIC_FEATURE_COLS + ["units_kwh"]:
        if col not in train_df.columns:
            continue
        s = train_df[col]
        bad = sorted(set(s[(s < 0) | s.isin(KNOWN_SENTINELS)].tolist()))
        if bad:
            found[col] = bad

    return {
        "present": len(found) > 0,
        "columns": found if found else None,
    }


def check_duplicates(train_df: pd.DataFrame) -> dict[str, Any]:
    """Check for exact duplicate rows."""
    dup_count = int(train_df.duplicated().sum())
    return {
        "present": dup_count > 0,
        "total_rows": int(len(train_df)),
        "duplicate_rows": dup_count,
        "unique_rows": int(len(train_df) - dup_count),
    }


def check_stuck_sensor(train_df: pd.DataFrame, min_run_length: int = 5) -> dict[str, Any]:
    """Detect consecutive runs of identical values in continuous sensor columns."""
    sensor_cols = ["ac_hours_per_day", "avg_temp_c", "house_area_sqft", "solar_kw"]
    findings: list[dict[str, Any]] = []

    for col in sensor_cols:
        if col not in train_df.columns:
            continue
        s = train_df[col]
        run_id = (s != s.shift()).cumsum()
        runs = train_df.groupby(run_id).agg(
            value=(col, "first"),
            length=(col, "size"),
            start_record=("record_id", "first") if "record_id" in train_df.columns else (col, "size"),
            end_record=("record_id", "last") if "record_id" in train_df.columns else (col, "size"),
            start_date=("date", "first") if "date" in train_df.columns else (col, "size"),
            end_date=("date", "last") if "date" in train_df.columns else (col, "size"),
        )
        stuck = runs[runs["length"] >= min_run_length]
        for _, row in stuck.iterrows():
            findings.append(
                {
                    "column": col,
                    "stuck_value": float(row["value"]),
                    "consecutive_rows": int(row["length"]),
                    "total_occurrences": int((s == row["value"]).sum()),
                    "start_record": str(row["start_record"]),
                    "end_record": str(row["end_record"]),
                    "start_date": str(row["start_date"]),
                    "end_date": str(row["end_date"]),
                }
            )

    return {
        "present": len(findings) > 0,
        "details": findings[0] if len(findings) == 1 else (findings if findings else None),
    }


def check_target_typos(train_df: pd.DataFrame, target_col: str = "units_kwh") -> dict[str, Any]:
    """Check for extreme target typos via range bounds and linear-model residuals."""
    if target_col not in train_df.columns:
        return {"present": False, "outliers": None}

    df = train_df.drop_duplicates().copy()
    y = df[target_col]
    neg_or_extreme = df[(y <= 0) | (y > 5000)]

    df["connection_norm"] = (
        df["connection"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    feature_cols = [c for c in NUMERIC_FEATURE_COLS if c in df.columns] + ["connection_norm", "feeder"]
    X = pd.get_dummies(df[feature_cols], drop_first=True)
    lr = LinearRegression().fit(X, y)
    residuals = np.abs(y - lr.predict(X))
    max_res = float(residuals.max())
    extreme_res = int((residuals > 300).sum())

    present = len(neg_or_extreme) > 0 or extreme_res > 0
    return {
        "present": present,
        "min_target": float(y.min()),
        "max_target": float(y.max()),
        "r2_linear": round(float(lr.score(X, y)), 4),
        "max_abs_residual": round(max_res, 2),
        "outlier_count": int(len(neg_or_extreme) + extreme_res),
    }


def check_spelling_variants(train_df: pd.DataFrame) -> dict[str, Any]:
    """Check categorical columns for non-canonical spelling/case/whitespace variants."""
    variants: dict[str, list[str]] = {}
    if "connection" in train_df.columns:
        raw_vals = sorted(train_df["connection"].dropna().unique().tolist())
        if set(raw_vals) != CANONICAL_CONNECTIONS:
            variants["connection"] = raw_vals
    if "feeder" in train_df.columns:
        raw_feeders = sorted(train_df["feeder"].dropna().unique().tolist())
        if not set(raw_feeders).issubset(CANONICAL_FEEDERS):
            variants["feeder"] = raw_feeders

    return {
        "present": len(variants) > 0,
        "connection_variant_count": len(variants.get("connection", [])),
        "connection_variants": variants.get("connection") or None,
        "feeder_variants": variants.get("feeder") or None,
    }


def run_all_checks(
    train_path: str | Path = "data/raw/train.csv",
    holdout_path: str | Path | None = "data/raw/holdout_features.csv",
) -> dict[str, Any]:
    """Run all seven data quality checks and return structured findings."""
    train_df = pd.read_csv(train_path)
    holdout_df = pd.read_csv(holdout_path) if holdout_path and Path(holdout_path).exists() else None

    return {
        "data_sha256": compute_sha256(train_path),
        "leaking_column": check_leaking_column(train_df, holdout_df),
        "unit_mixup": check_unit_mixup(train_df),
        "sentinel_values": check_sentinel_values(train_df),
        "duplicate_rows": check_duplicates(train_df),
        "stuck_sensor": check_stuck_sensor(train_df),
        "target_typos": check_target_typos(train_df),
        "spelling_variants": check_spelling_variants(train_df),
    }


if __name__ == "__main__":
    import yaml

    results = run_all_checks()
    print(yaml.safe_dump(results, sort_keys=False))
