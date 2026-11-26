"""Reusable data cleaning functions for MP287."""

from __future__ import annotations

import numpy as np
import pandas as pd

LEAK_COLUMNS = ["billed_units_after_reading"]
SENTINEL_VALUES = {-999, -99, -1, 999, 9999}

CONNECTION_ALIASES = {
    "dom.": "domestic",
    "dom": "domestic",
    "domestic": "domestic",
    "domestic_plus": "domestic_plus",
    "domestic plus": "domestic_plus",
    "sho.": "shop_cum_home",
    "sho": "shop_cum_home",
    "shop_cum_home": "shop_cum_home",
    "shop cum home": "shop_cum_home",
}


def _normalize_connection(series: pd.Series) -> pd.Series:
    """Normalize whitespace, casing, and abbreviations in connection type."""
    cleaned = series.astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    return cleaned.map(lambda val: CONNECTION_ALIASES.get(val, val))


def _fix_stuck_sensor(
    df: pd.DataFrame,
    col: str = "solar_kw",
    min_run_length: int = 5,
) -> pd.DataFrame:
    """Replace consecutive stuck-sensor runs with the non-stuck median."""
    if col not in df.columns or len(df) < min_run_length:
        return df

    s = df[col]
    run_id = (s != s.shift()).cumsum()
    run_sizes = run_id.map(run_id.value_counts())
    stuck_mask = run_sizes >= min_run_length
    if stuck_mask.any():
        non_stuck_median = float(s.loc[~stuck_mask].median()) if (~stuck_mask).any() else float(s.median())
        df = df.copy()
        df.loc[stuck_mask, col] = non_stuck_median
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw MP287 dataframe by fixing all detected data quality issues.

    Steps applied:
    1. Drop leaking columns (e.g. ``billed_units_after_reading``).
    2. Remove exact duplicate rows.
    3. Normalize ``connection`` spelling/whitespace/abbreviation variants.
    4. Replace sentinel values (if any) with NaN and impute numeric medians.
    5. Detect and replace stuck-sensor runs in ``solar_kw``.
    """
    out = df.copy()

    # 1. Drop leaking column(s)
    cols_to_drop = [c for c in LEAK_COLUMNS if c in out.columns]
    if cols_to_drop:
        out = out.drop(columns=cols_to_drop)

    # 2. Remove exact duplicates
    out = out.drop_duplicates().reset_index(drop=True)

    # 3. Normalize connection spelling variants
    if "connection" in out.columns:
        out["connection"] = _normalize_connection(out["connection"])

    # 4. Handle sentinel values if present
    numeric_cols = [
        c
        for c in [
            "rooms",
            "members",
            "ac_hours_per_day",
            "avg_temp_c",
            "house_area_sqft",
            "solar_kw",
        ]
        if c in out.columns
    ]
    for col in numeric_cols:
        mask = out[col].isin(SENTINEL_VALUES) | (out[col] < 0)
        if mask.any():
            out.loc[mask, col] = np.nan
            out[col] = out[col].fillna(out[col].median())

    # 5. Handle stuck sensor in solar_kw
    out = _fix_stuck_sensor(out, col="solar_kw", min_run_length=5)

    return out
