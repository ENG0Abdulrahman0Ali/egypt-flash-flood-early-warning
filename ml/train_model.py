
"""Train candidate XGBoost models for hourly flash-flood risk forecasting."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier, XGBRegressor


# Make the project root importable when this script runs from ml/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = Path(__file__).resolve().parent
MODEL_DIR = ML_DIR / "models"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_columns import (  # noqa: E402
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURES_TABLE,
    FEATURE_COLUMNS,
    FORECAST_HORIZON_HOURS,
    NUMERIC_FEATURE_COLUMNS,
    RAINFALL_TARGET_COLUMN,
    RISK_LEVELS,
    RISK_LEVEL_TARGET_COLUMN,
    RISK_SCORE_TARGET_COLUMN,
    RISK_TABLE,
)
from warehouse.db_connection import get_engine  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("train_model")

TARGET_RAINFALL = "_target_rainfall"
TARGET_RISK_SCORE = "_target_risk_score"
TARGET_RISK_LEVEL = "_target_risk_level"

TRAIN_FRACTION = 0.80
RANDOM_STATE = 42


def load_training_data() -> pd.DataFrame:
    """Read historical Silver features and align targets one hour ahead."""

    numeric_sql_columns = ", ".join(
        f"[{column}]" for column in NUMERIC_FEATURE_COLUMNS
    )

    features_sql = f"""
        SELECT city, event_time, source, {numeric_sql_columns}
        FROM {FEATURES_TABLE}
        WHERE LOWER(LTRIM(RTRIM(source))) = 'historical'
        ORDER BY event_time, city
    """

    risk_sql = f"""
        SELECT city, event_time,
               [{RISK_SCORE_TARGET_COLUMN}],
               [{RISK_LEVEL_TARGET_COLUMN}]
        FROM {RISK_TABLE}
    """

    logger.info("Reading historical features from %s", FEATURES_TABLE)

    engine = get_engine()

    try:
        with engine.connect() as connection:
            weather = pd.read_sql_query(
                features_sql,
                connection,
                parse_dates=["event_time"],
            )
            risk = pd.read_sql_query(
                risk_sql,
                connection,
                parse_dates=["event_time"],
            )
    finally:
        engine.dispose()

    if weather.empty:
        raise ValueError(
            f"No historical rows found in {FEATURES_TABLE}. "
            "Check the historical ingestion pipeline."
        )

    if risk.empty:
        raise ValueError(
            f"No risk labels found in {RISK_TABLE}. "
            "Run the Silver-to-Gold load first."
        )

    for frame in (weather, risk):
        frame["city"] = frame["city"].astype("string").str.strip()
        frame["event_time"] = pd.to_datetime(
            frame["event_time"], errors="coerce"
        )

    keys = ["city", "event_time"]

    # Ambiguous city-hour records must not enter the training set.
    if weather.duplicated(keys).any():
        raise ValueError(
            "Duplicate city/event_time keys found in historical Silver."
        )

    if risk.duplicated(keys).any():
        raise ValueError(
            "Duplicate city/event_time keys found in the Gold risk table."
        )

    # Build labels from the actual historical weather record at t + 1 hour.
    # Shifting the label timestamp back aligns it with features at time t.
    next_hour_labels = weather[
        ["city", "event_time", RAINFALL_TARGET_COLUMN]
    ].merge(
        risk[
            [
                "city",
                "event_time",
                RISK_SCORE_TARGET_COLUMN,
                RISK_LEVEL_TARGET_COLUMN,
            ]
        ],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    next_hour_labels["event_time"] -= pd.Timedelta(
        hours=FORECAST_HORIZON_HOURS
    )

    next_hour_labels = next_hour_labels.rename(
        columns={
            RAINFALL_TARGET_COLUMN: TARGET_RAINFALL,
            RISK_SCORE_TARGET_COLUMN: TARGET_RISK_SCORE,
            RISK_LEVEL_TARGET_COLUMN: TARGET_RISK_LEVEL,
        }
    )

    data = weather.merge(
        next_hour_labels,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    if data.empty:
        raise ValueError(
            "No training rows have matching next-hour targets. "
            "Check the historical timestamps in Silver and Gold."
        )

    # Convert measurement and target columns to numeric values.
    numeric_columns = [
        *NUMERIC_FEATURE_COLUMNS,
        TARGET_RAINFALL,
        TARGET_RISK_SCORE,
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    required_columns = [
        *FEATURE_COLUMNS,
        "event_time",
        TARGET_RAINFALL,
        TARGET_RISK_SCORE,
        TARGET_RISK_LEVEL,
    ]

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required_columns).copy()

    data[TARGET_RISK_LEVEL] = (
        data[TARGET_RISK_LEVEL].astype(str).str.strip()
    )

    unknown_levels = sorted(
        set(data[TARGET_RISK_LEVEL]) - set(RISK_LEVELS)
    )

    if unknown_levels:
        raise ValueError(
            f"Unexpected risk-level labels in Gold: {unknown_levels}. "
            f"Expected one of: {list(RISK_LEVELS)}"
        )

    data = data.sort_values(
        ["event_time", "city"]
    ).reset_index(drop=True)

    if data.empty:
        raise ValueError(
            "No valid training rows remain after data validation."
        )

    logger.info(
        "Prepared %s rows with complete next-hour targets.",
        f"{len(data):,}",
    )

    return data


def make_model_pipeline(model):
    """Apply numeric imputation and categorical city encoding."""

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                list(NUMERIC_FEATURE_COLUMNS),
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURE_COLUMNS),
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )


def regression_metrics(actual, predicted) -> dict[str, float]:
    """Calculate holdout metrics for a regression model."""

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
    }


def save_candidate(
    filename: str,
    model,
    metrics: dict[str, float],
    model_version: str,
    cutoff: pd.Timestamp,
    test_start: pd.Timestamp,
    row_counts: dict[str, int],
    label_encoder: LabelEncoder | None = None,
) -> None:
    """Save a candidate artifact for the evaluation/promotion stage."""

    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "model_version": model_version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "feature_columns": list(FEATURE_COLUMNS),
        "metrics": metrics,
        "training_cutoff": cutoff.isoformat(),
        "test_start": test_start.isoformat(),
        "row_counts": row_counts,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    destination = MODEL_DIR / filename
    joblib.dump(artifact, destination, compress=3)

    logger.info("Saved candidate artifact: %s", destination)


def main() -> None:
    data = load_training_data()

    unique_times = np.sort(data["event_time"].unique())

    if len(unique_times) < 10:
        raise ValueError(
            "Insufficient distinct timestamps for a chronological "
            "train/test split."
        )

    cutoff_index = int(len(unique_times) * TRAIN_FRACTION)
    cutoff = pd.Timestamp(unique_times[cutoff_index])

    train = data[data["event_time"] < cutoff].copy()
    test = data[data["event_time"] >= cutoff].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Chronological split produced an empty training or test set."
        )

    X_train = train[list(FEATURE_COLUMNS)]
    X_test = test[list(FEATURE_COLUMNS)]

    logger.info(
        "Chronological split: train=%s rows, test=%s rows, cutoff=%s",
        f"{len(train):,}",
        f"{len(test):,}",
        cutoff,
    )

    common_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    # Model 1: forecast rainfall in the next hour.
    rainfall_model = make_model_pipeline(
        XGBRegressor(
            objective="reg:squarederror",
            **common_params,
        )
    )

    rainfall_model.fit(
        X_train,
        train[TARGET_RAINFALL],
    )

    rainfall_prediction = np.clip(
        rainfall_model.predict(X_test),
        0.0,
        None,
    )

    rainfall_metrics = regression_metrics(
        test[TARGET_RAINFALL],
        rainfall_prediction,
    )

    # Persistence baseline: next-hour rainfall equals current rainfall.
    rainfall_baseline = np.clip(
        X_test[RAINFALL_TARGET_COLUMN].to_numpy(dtype=float),
        0.0,
        None,
    )

    rainfall_metrics["persistence_baseline_mae"] = float(
        mean_absolute_error(
            test[TARGET_RAINFALL],
            rainfall_baseline,
        )
    )

    logger.info("Rainfall holdout metrics: %s", rainfall_metrics)

    # Model 2: forecast the next-hour rule-based risk score.
    risk_score_model = make_model_pipeline(
        XGBRegressor(
            objective="reg:squarederror",
            **common_params,
        )
    )

    risk_score_model.fit(
        X_train,
        train[TARGET_RISK_SCORE],
    )

    risk_score_prediction = np.clip(
        risk_score_model.predict(X_test),
        0.0,
        100.0,
    )

    risk_score_metrics = regression_metrics(
        test[TARGET_RISK_SCORE],
        risk_score_prediction,
    )

    logger.info("Risk-score holdout metrics: %s", risk_score_metrics)

    # Model 3: classify the next-hour Gold risk level.
    label_encoder = LabelEncoder()

    y_train_level = label_encoder.fit_transform(
        train[TARGET_RISK_LEVEL]
    )

    if len(label_encoder.classes_) < 2:
        raise ValueError(
            "Training data contains only one risk-level class. "
            "More historical class variation is required."
        )

    y_test_level = test[TARGET_RISK_LEVEL].to_numpy()

    risk_level_model = make_model_pipeline(
        XGBClassifier(
            objective="binary:logistic"
            if len(label_encoder.classes_) == 2
            else "multi:softprob",
            eval_metric="mlogloss",
            **common_params,
        )
    )

    risk_level_model.fit(X_train, y_train_level)

    predicted_level_ids = risk_level_model.predict(X_test).astype(int)
    predicted_level_ids = np.clip(
        predicted_level_ids,
        0,
        len(label_encoder.classes_) - 1,
    )

    predicted_levels = label_encoder.inverse_transform(
        predicted_level_ids
    )

    risk_level_metrics = {
        "accuracy": float(
            accuracy_score(y_test_level, predicted_levels)
        ),
        "macro_f1": float(
            f1_score(
                y_test_level,
                predicted_levels,
                labels=list(RISK_LEVELS),
                average="macro",
                zero_division=0,
            )
        ),
    }

    logger.info("Risk-level holdout metrics: %s", risk_level_metrics)

    # These are candidates, not automatically promoted production models.
    model_version = (
        "xgb-" + datetime.now(timezone.utc).strftime("%y%m%d%H%M%S")
    )

    row_counts = {
        "total": int(len(data)),
        "train": int(len(train)),
        "test": int(len(test)),
    }

    test_start = pd.Timestamp(test["event_time"].min())

    save_candidate(
        "rainfall_candidate.joblib",
        rainfall_model,
        rainfall_metrics,
        model_version,
        cutoff,
        test_start,
        row_counts,
    )

    save_candidate(
        "risk_score_candidate.joblib",
        risk_score_model,
        risk_score_metrics,
        model_version,
        cutoff,
        test_start,
        row_counts,
    )

    save_candidate(
        "risk_level_candidate.joblib",
        risk_level_model,
        risk_level_metrics,
        model_version,
        cutoff,
        test_start,
        row_counts,
        label_encoder=label_encoder,
    )

    logger.info(
        "Candidate training completed. Models still require evaluation "
        "and promotion before inference can use them."
    )


if __name__ == "__main__":
    main()
