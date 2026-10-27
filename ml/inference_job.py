"""Generate next-hour flood predictions and save them to Gold."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

# Make the project root importable when this script runs directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from feature_columns import (
    FEATURE_COLUMNS,
    FORECAST_HORIZON_HOURS,
    NUMERIC_FEATURE_COLUMNS,
    RISK_LEVELS,
)
from warehouse.db_connection import get_engine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("inference_job")

MODEL_DIR = Path(__file__).resolve().parent / "models"

# Latest live input per city. SQL Server DATETIME2 values are treated as UTC.
FEATURE_SELECT = ",\n        ".join(
    f"[{column}]" for column in NUMERIC_FEATURE_COLUMNS
)

LATEST_LIVE_SQL = f"""
WITH ranked_live AS (
    SELECT
        city,
        event_time,
        source,
        {FEATURE_SELECT},
        ROW_NUMBER() OVER (
            PARTITION BY city
            ORDER BY event_time DESC
        ) AS row_num
    FROM silver.fact_weather_features
    WHERE LOWER(LTRIM(RTRIM(source))) = N'live'
      AND event_time <= SYSUTCDATETIME()
)
SELECT
    city,
    event_time,
    source,
    {FEATURE_SELECT}
FROM ranked_live
WHERE row_num = 1
ORDER BY city;
"""

INSERT_PREDICTION_SQL = text("""
INSERT INTO gold.predictions (
    city,
    predicted_for,
    predicted_rainfall,
    predicted_risk_score,
    predicted_risk_level,
    model_version
)
SELECT
    :city,
    :predicted_for,
    :predicted_rainfall,
    :predicted_risk_score,
    :predicted_risk_level,
    :model_version
WHERE NOT EXISTS (
    SELECT 1
    FROM gold.predictions WITH (UPDLOCK, HOLDLOCK)
    WHERE city = :city
      AND predicted_for = :predicted_for
      AND model_version = :model_version
);
""")


def load_production_model(
    name: str,
    *,
    required: bool = True,
) -> dict[str, Any] | None:
    """Load a promoted model and validate its feature contract."""
    path = MODEL_DIR / f"{name}_production.joblib"

    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"Required production model is missing: {path}"
            )

        logger.warning(
            "No production rainfall model found. "
            "The persistence baseline will be used instead."
        )
        return None

    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise ValueError(f"Invalid model artifact: {path.name}")

    if artifact.get("model") is None:
        raise ValueError(f"Model object is missing from {path.name}")

    if artifact.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError(
            f"Feature schema mismatch in {path.name}. "
            "Retrain or validate the artifact before inference."
        )

    if (
        artifact.get("forecast_horizon_hours")
        != FORECAST_HORIZON_HOURS
    ):
        raise ValueError(
            f"Forecast horizon mismatch in {path.name}"
        )

    if not artifact.get("model_version"):
        raise ValueError(
            f"Model version is missing from {path.name}"
        )

    if name == "risk_level":
        encoder = artifact.get("label_encoder")
        if encoder is None or not hasattr(encoder, "inverse_transform"):
            raise ValueError(
                "Risk-level model is missing its label encoder."
            )

    logger.info(
        "Loaded %s production model: version=%s",
        name,
        artifact["model_version"],
    )
    return artifact


def load_latest_live_features(engine) -> pd.DataFrame:
    """Read the latest live feature row for each city."""
    with engine.connect() as connection:
        frame = pd.read_sql_query(
            text(LATEST_LIVE_SQL),
            connection,
        )

    if frame.empty:
        raise RuntimeError(
            "No live features found in silver.fact_weather_features. "
            "Trigger the hourly_ingestion DAG and verify that "
            "live_incremental_job completed successfully."
        )

    frame["city"] = (
        frame["city"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    frame["event_time"] = pd.to_datetime(
        frame["event_time"],
        errors="coerce",
    )

    # The fitted preprocessing pipeline can impute missing numeric
    # features, but infinity values must be converted to NaN first.
    for column in NUMERIC_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )
        frame[column] = frame[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    valid_keys = (
        frame["city"].ne("")
        & frame["event_time"].notna()
        & frame["precipitation"].notna()
    )
    frame = frame.loc[valid_keys].copy()

    if frame.empty:
        raise RuntimeError(
            "No usable live rows remain after validating city, "
            "event_time, and precipitation."
        )

    # Do not save a purported forecast if its target time has already
    # passed. Source timestamps in this warehouse are UTC-naive.
    frame["predicted_for"] = (
        frame["event_time"]
        + pd.to_timedelta(FORECAST_HORIZON_HOURS, unit="h")
    )

    now_utc = pd.Timestamp(
        datetime.now(timezone.utc).replace(tzinfo=None)
    )
    future_target = frame["predicted_for"] > now_utc

    skipped = int((~future_target).sum())
    if skipped:
        logger.warning(
            "Skipping %s live rows because their forecast target "
            "is not in the future.",
            skipped,
        )

    frame = frame.loc[future_target].copy()

    if frame.empty:
        raise RuntimeError(
            "All live rows are stale for next-hour forecasting. "
            "Run hourly_ingestion and verify the latest live event_time "
            "in Silver before retrying inference."
        )

    if frame["city"].duplicated().any():
        raise ValueError(
            "Duplicate city rows found after selecting the latest "
            "live features."
        )

    logger.info(
        "Prepared %s current live city rows for inference.",
        len(frame),
    )
    return frame.reset_index(drop=True)


def _check_predictions(values: np.ndarray, name: str) -> np.ndarray:
    """Ensure predictions contain one finite value per input row."""
    values = np.asarray(values).reshape(-1)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name} model generated non-finite predictions."
        )

    return values


def build_pipeline_version(
    risk_score_version: str,
    risk_level_version: str,
    rainfall_version: str,
) -> str:
    """Create a deterministic pipeline ID that fits NVARCHAR(20)."""
    signature = "|".join(
        [
            risk_score_version,
            risk_level_version,
            rainfall_version,
            str(FORECAST_HORIZON_HOURS),
        ]
    )

    digest = hashlib.sha1(
        signature.encode("utf-8")
    ).hexdigest()[:12]

    # 16 characters; identifies the complete model stack.
    return f"mix-{digest}"


def create_predictions(
    frame: pd.DataFrame,
    risk_score_artifact: dict[str, Any],
    risk_level_artifact: dict[str, Any],
    rainfall_artifact: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Run all predictions and create validated SQL insert records."""
    model_input = frame.loc[:, list(FEATURE_COLUMNS)].copy()

    risk_score_values = _check_predictions(
        risk_score_artifact["model"].predict(model_input),
        "Risk-score",
    )

    risk_score_values = np.rint(
        np.clip(risk_score_values, 0, 100)
    ).astype(int)

    level_codes = _check_predictions(
        risk_level_artifact["model"].predict(model_input),
        "Risk-level",
    )
    level_codes = np.rint(level_codes).astype(int)

    encoder = risk_level_artifact["label_encoder"]
    predicted_levels = encoder.inverse_transform(level_codes)

    valid_levels = np.isin(
        predicted_levels,
        np.asarray(RISK_LEVELS),
    )
    if not valid_levels.all():
        unexpected = sorted(
            set(predicted_levels[~valid_levels].tolist())
        )
        raise ValueError(
            f"Unexpected predicted risk levels: {unexpected}"
        )

    if rainfall_artifact is not None:
        rainfall_values = _check_predictions(
            rainfall_artifact["model"].predict(model_input),
            "Rainfall",
        )
        rainfall_method = rainfall_artifact["model_version"]
    else:
        # This is the persistence baseline used in train_model.py:
        # the next-hour rainfall prediction equals the latest observed
        # precipitation. It is not an approved ML rainfall model.
        rainfall_values = _check_predictions(
            frame["precipitation"].to_numpy(dtype=float),
            "Rainfall persistence baseline",
        )
        rainfall_method = "persistence-baseline-v1"

    rainfall_values = np.clip(rainfall_values, 0, None)

    pipeline_version = build_pipeline_version(
        risk_score_artifact["model_version"],
        risk_level_artifact["model_version"],
        rainfall_method,
    )

    records: list[dict[str, Any]] = []

    for index, row in frame.iterrows():
        records.append(
            {
                "city": str(row["city"]),
                "predicted_for": row["predicted_for"].to_pydatetime(),
                "predicted_rainfall": float(rainfall_values[index]),
                "predicted_risk_score": int(risk_score_values[index]),
                "predicted_risk_level": str(predicted_levels[index]),
                "model_version": pipeline_version,
            }
        )

    logger.info(
        "Prediction stack: rainfall=%s, risk_score=%s, risk_level=%s, "
        "pipeline_version=%s",
        rainfall_method,
        risk_score_artifact["model_version"],
        risk_level_artifact["model_version"],
        pipeline_version,
    )

    return records


def save_predictions(
    engine,
    records: list[dict[str, Any]],
) -> int:
    """Insert predictions while avoiding duplicate city/time/version keys."""
    inserted = 0

    with engine.begin() as connection:
        for record in records:
            result = connection.execute(
                INSERT_PREDICTION_SQL,
                record,
            )
            if result.rowcount == 1:
                inserted += 1

    logger.info(
        "Prediction write completed: inserted=%s, existing_or_skipped=%s",
        inserted,
        len(records) - inserted,
    )
    return inserted


def main() -> None:
    """Execute the hourly ML inference workflow."""
    risk_score_artifact = load_production_model("risk_score")
    risk_level_artifact = load_production_model("risk_level")
    rainfall_artifact = load_production_model(
        "rainfall",
        required=False,
    )

    engine = get_engine()
    try:
        latest_features = load_latest_live_features(engine)

        records = create_predictions(
            latest_features,
            risk_score_artifact,
            risk_level_artifact,
            rainfall_artifact,
        )

        inserted = save_predictions(engine, records)

        logger.info(
            "Inference completed successfully. "
            "Prepared=%s, inserted=%s.",
            len(records),
            inserted,
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()