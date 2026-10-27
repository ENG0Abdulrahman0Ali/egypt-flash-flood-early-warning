
"""Shared ML feature and target definitions.

These constants are used by training, inference, and retraining jobs.
Keep feature names aligned with the current Silver and Gold SQL schemas.
"""

from __future__ import annotations


# Forecast the next hourly observation.
FORECAST_HORIZON_HOURS = 1

CITY_COLUMN = "city"
TIME_COLUMN = "event_time"


# Numeric features available in silver.fact_weather_features.
NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "precipitation",
    "relative_humidity_2m",
    "temperature_2m",
    "surface_pressure",
    "windspeed_10m",
    "cloudcover",
    "soil_moisture_0_to_7cm",
    "elevation",
    "slope_degree",
    "rain_sum_3h",
    "rain_sum_6h",
    "rain_sum_24h",
    "rain_sum_72h",
    "rain_lag_1h",
    "rain_lag_24h",
    "hour_of_day",
    "month",
)

# The city is categorical, not a numeric measurement.
CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    CITY_COLUMN,
)

FEATURE_COLUMNS: tuple[str, ...] = (
    *NUMERIC_FEATURE_COLUMNS,
    *CATEGORICAL_FEATURE_COLUMNS,
)


# The rainfall target is shifted to the next hour during training.
RAINFALL_TARGET_COLUMN = "precipitation"

# Risk targets are taken from the next hourly observation in Gold.
RISK_SCORE_TARGET_COLUMN = "risk_score"
RISK_LEVEL_TARGET_COLUMN = "risk_level"


# These are the English labels currently emitted by the Spark Gold loader.
RISK_LEVELS: tuple[str, ...] = (
    "Safe",
    "Watch",
    "Danger",
    "Severe Danger",
)


# SQL tables owned by the existing data pipeline.
FEATURES_TABLE = "silver.fact_weather_features"
RISK_TABLE = "gold.fact_hourly_risk"
PREDICTIONS_TABLE = "gold.predictions"
