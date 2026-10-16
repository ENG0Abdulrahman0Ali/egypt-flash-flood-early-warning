"""Load hourly flood-risk records from Silver into Gold."""

import logging
import math
import sys
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spark.jdbc_io import ( 
    create_spark_session,
    read_sql_table,
    write_sql_table,
)

logger = logging.getLogger("silver_to_gold_load")

SOURCE_TABLE = "silver.fact_weather_features"
TARGET_TABLE = "gold.fact_hourly_risk"

RISK_INPUT_COLUMNS = [
    "city",
    "event_time",
    "source",
    "precipitation",
    "rain_sum_1h",
    "rain_sum_3h",
    "rain_sum_6h",
    "rain_sum_24h",
    "rain_sum_72h",
    "soil_moisture_0_to_7cm",
    "elevation",
    "slope_degree",
]

NORMALIZATION_REFERENCES = {
    "rain_1h": 15.0,
    "rain_3h": 30.0,
    "rain_6h": 45.0,
    "rain_24h": 60.0,
    "rain_72h": 80.0,
    "soil_moisture": 0.5,
    "slope": 30.0,
}

ELEVATION_REFERENCE_METERS = 1600.0
RAIN_GATE_MM = 1.0

# Soil moisture is assumed to be a fraction in [0, 1].
# Convert percentage values [0, 100] before using this pipeline.
SOIL_MOISTURE_MIN = 0.0
SOIL_MOISTURE_MAX = 1.0

RISK_LEVEL_SAFE = "Safe"
RISK_LEVEL_WATCH = "Watch"
RISK_LEVEL_DANGER = "Danger"
RISK_LEVEL_SEVERE = "Severe Danger"

GOLD_OUTPUT_COLUMNS = [
    "city",
    "event_time",
    "avg_rainfall",
    "rain_sum_24h",
    "rain_sum_72h",
    "risk_score",
    "risk_level",
]

NON_NEGATIVE_COLUMNS = [
    "precipitation",
    "rain_sum_1h",
    "rain_sum_3h",
    "rain_sum_6h",
    "rain_sum_24h",
    "rain_sum_72h",
    "slope_degree",
]

# Elevation can legitimately be <= 0 (coastal / below-sea-level points), so it
# is only required to be finite, not non-negative.
FINITE_ONLY_COLUMNS = ["elevation"]


def _is_finite_double(column_name: str):
    """Return a condition indicating a finite, non-null numeric value."""
    value = F.col(column_name).cast("double")

    return (
        value.isNotNull()
        & ~F.isnan(value)
        & (value != F.lit(float("inf")))
        & (value != F.lit(float("-inf")))
    )


def _read_silver_features(spark: object) -> DataFrame:
    """Read required Silver columns and reject duplicate city-hour keys."""
    silver = read_sql_table(spark, SOURCE_TABLE)

    missing_columns = sorted(
        set(RISK_INPUT_COLUMNS) - set(silver.columns)
    )

    if missing_columns:
        raise ValueError(
            "Silver table is missing required columns: "
            + ", ".join(missing_columns)
        )

    weather = silver.select(*RISK_INPUT_COLUMNS).withColumn(
        "event_time",
        F.to_timestamp("event_time"),
    )

    duplicate_exists = (
        weather.groupBy("city", "event_time")
        .count()
        .filter(F.col("count") > 1)
        .limit(1)
        .count()
        > 0
    )

    if duplicate_exists:
        raise ValueError(
            "Duplicate (city, event_time) keys found in Silver. "
            "Deduplicate upstream using a deterministic ordering column."
        )

    # ERA5 soil moisture occasionally dips slightly below 0 (e.g. -0.015) as a
    # model artifact. A volumetric fraction cannot be negative, so clip to 0.
    return weather.withColumn(
        "soil_moisture_0_to_7cm",
        F.greatest(F.col("soil_moisture_0_to_7cm").cast("double"), F.lit(0.0)),
    ).withColumn(
        "avg_rainfall",
        F.col("precipitation"),
    )


def _validate_risk_inputs(weather: DataFrame) -> DataFrame:
    """Reject missing, non-finite, or invalid measurements."""
    invalid = (
        F.col("city").isNull()
        | (F.length(F.trim(F.col("city"))) == 0)
        | F.col("event_time").isNull()
        | F.col("source").isNull()
        | (F.length(F.trim(F.col("source"))) == 0)
    )

    for column_name in NON_NEGATIVE_COLUMNS:
        value = F.col(column_name).cast("double")

        invalid = invalid | (
            ~_is_finite_double(column_name)
            | (value < 0)
        )

    for column_name in FINITE_ONLY_COLUMNS:
        invalid = invalid | ~_is_finite_double(column_name)

    soil = F.col("soil_moisture_0_to_7cm").cast("double")

    invalid = invalid | (
        ~_is_finite_double("soil_moisture_0_to_7cm")
        | (soil < SOIL_MOISTURE_MIN)
        | (soil > SOIL_MOISTURE_MAX)
    )

    if weather.filter(invalid).limit(1).count() > 0:
        raise ValueError(
            "Invalid or missing risk inputs detected in Silver. "
            "Check nulls, non-finite values, negative measurements, "
            "and measurement units."
        )

    return weather


def _smooth_normalize(column_name: str, reference: float):
    """Normalize a non-negative input to a smooth value between 0 and 1."""
    if reference <= 0:
        raise ValueError(
            f"Normalization reference must be positive: {column_name}"
        )

    return F.lit(1.0) - F.exp(
        -F.col(column_name).cast("double") / F.lit(reference)
    )


def _calculate_risk_score(weather: DataFrame) -> DataFrame:
    """Calculate a rule-based risk score between 0 and 100."""
    normalized = (
        weather
        .withColumn(
            "_r1",
            _smooth_normalize(
                "rain_sum_1h",
                NORMALIZATION_REFERENCES["rain_1h"],
            ),
        )
        .withColumn(
            "_r3",
            _smooth_normalize(
                "rain_sum_3h",
                NORMALIZATION_REFERENCES["rain_3h"],
            ),
        )
        .withColumn(
            "_r6",
            _smooth_normalize(
                "rain_sum_6h",
                NORMALIZATION_REFERENCES["rain_6h"],
            ),
        )
        .withColumn(
            "_r24",
            _smooth_normalize(
                "rain_sum_24h",
                NORMALIZATION_REFERENCES["rain_24h"],
            ),
        )
        .withColumn(
            "_r72",
            _smooth_normalize(
                "rain_sum_72h",
                NORMALIZATION_REFERENCES["rain_72h"],
            ),
        )
        .withColumn(
            "_soil",
            _smooth_normalize(
                "soil_moisture_0_to_7cm",
                NORMALIZATION_REFERENCES["soil_moisture"],
            ),
        )
        .withColumn(
            "_slope",
            _smooth_normalize(
                "slope_degree",
                NORMALIZATION_REFERENCES["slope"],
            ),
        )
        .withColumn(
            "_elevation_factor",
            F.exp(
                -F.col("elevation").cast("double")
                / F.lit(ELEVATION_REFERENCE_METERS)
            ),
        )
    )

    factors = (
        normalized
        .withColumn(
            "_rainfall_hazard",
            0.35 * F.col("_r1")
            + 0.25 * F.col("_r3")
            + 0.15 * F.col("_r6")
            + 0.25 * F.col("_r24"),
        )
        .withColumn(
            "_antecedent_condition",
            0.60 * F.col("_r72")
            + 0.40 * F.col("_soil"),
        )
        .withColumn(
            "_terrain",
            0.70 * F.col("_slope")
            + 0.30 * F.col("_elevation_factor"),
        )
    )

    raw_score = F.round(
        100
        * (
            0.60 * F.col("_rainfall_hazard")
            + 0.25 * F.col("_antecedent_condition")
            + 0.15 * F.col("_terrain")
        )
    )

    return factors.withColumn(
        "risk_score",
        F.least(
            F.lit(100.0),
            F.greatest(F.lit(0.0), raw_score),
        ).cast("int"),
    )


def _derive_historical_thresholds(
    scored: DataFrame,
) -> dict[str, float]:
    """Calculate P80/P95/P99 using historical rainy hours only."""
    historical = scored.filter(
        (F.lower(F.trim(F.col("source"))) == "historical")
        & (F.col("rain_sum_24h") >= RAIN_GATE_MM)
        & F.col("risk_score").isNotNull()
    )

    if not historical.limit(1).count():
        raise ValueError(
            "No historical rainy hours found. Run historical_bootstrap "
            "and verify the source and rainfall columns."
        )

    values = historical.approxQuantile(
        "risk_score",
        [0.80, 0.95, 0.99],
        0.001,
    )

    if len(values) != 3 or any(
        not math.isfinite(value) for value in values
    ):
        raise ValueError(
            "Could not calculate valid risk thresholds from historical data."
        )

    if any(value < 0 or value > 100 for value in values):
        raise ValueError(
            f"Risk thresholds are outside [0, 100]: {values}"
        )

    if not (values[0] <= values[1] <= values[2]):
        raise ValueError(
            f"Risk thresholds are not ordered correctly: {values}"
        )

    if values[0] == values[1] or values[1] == values[2]:
        logger.warning(
            "Some percentile thresholds are equal: %s. "
            "One or more risk levels may contain few or no records.",
            values,
        )

    return {
        "p80": values[0],
        "p95": values[1],
        "p99": values[2],
    }


def _add_risk_level(
    scored: DataFrame,
    p80: float,
    p95: float,
    p99: float,
) -> DataFrame:
    """Assign risk labels using the rainfall gate and historical percentiles."""
    if not (0 <= p80 <= p95 <= p99 <= 100):
        raise ValueError(
            f"Invalid risk thresholds: {(p80, p95, p99)}"
        )

    return scored.withColumn(
        "risk_level",
        F.when(
            F.col("rain_sum_24h") < RAIN_GATE_MM,
            F.lit(RISK_LEVEL_SAFE),
        )
        .when(
            F.col("risk_score") < p80,
            F.lit(RISK_LEVEL_SAFE),
        )
        .when(
            F.col("risk_score") < p95,
            F.lit(RISK_LEVEL_WATCH),
        )
        .when(
            F.col("risk_score") < p99,
            F.lit(RISK_LEVEL_DANGER),
        )
        .otherwise(F.lit(RISK_LEVEL_SEVERE)),
    )


def _read_existing_gold_keys(spark: object) -> DataFrame:
    """Read existing keys to avoid appending duplicate city-hour records."""
    gold = read_sql_table(spark, TARGET_TABLE)

    missing_columns = sorted(
        {"city", "event_time"} - set(gold.columns)
    )

    if missing_columns:
        raise ValueError(
            "Gold table is missing key columns: "
            + ", ".join(missing_columns)
        )

    return (
        gold.select(
            "city",
            F.to_timestamp("event_time").alias("event_time"),
        )
        .dropDuplicates(["city", "event_time"])
    )


def _validate_gold_rows(gold_rows: DataFrame) -> DataFrame:
    """Validate final Gold columns before writing."""
    invalid = (
        F.col("city").isNull()
        | F.col("event_time").isNull()
        | F.col("avg_rainfall").isNull()
        | F.col("rain_sum_24h").isNull()
        | F.col("rain_sum_72h").isNull()
        | F.col("risk_score").isNull()
        | F.col("risk_level").isNull()
        | (F.col("risk_score") < 0)
        | (F.col("risk_score") > 100)
        | (
            ~F.col("risk_level").isin(
                RISK_LEVEL_SAFE,
                RISK_LEVEL_WATCH,
                RISK_LEVEL_DANGER,
                RISK_LEVEL_SEVERE,
            )
        )
    )

    if gold_rows.filter(invalid).limit(1).count() > 0:
        raise ValueError(
            "Gold output validation failed. No rows were written."
        )

    return gold_rows


def run() -> None:
    """Calculate hourly risk and append previously unseen records to Gold."""
    spark = create_spark_session("silver-to-gold-risk-load")

    try:
        logger.info("Reading Silver weather features")
        silver_weather = _read_silver_features(spark)
        # Cache: Silver is otherwise re-read from SQL Server for every action below.
        silver_weather = _validate_risk_inputs(silver_weather).cache()

        logger.info("Calculating risk scores")
        scored_weather = _calculate_risk_score(silver_weather)

        logger.info("Deriving thresholds from historical rainy hours")
        thresholds = _derive_historical_thresholds(scored_weather)

        logger.info(
            "Historical thresholds: P80=%s, P95=%s, P99=%s",
            thresholds["p80"],
            thresholds["p95"],
            thresholds["p99"],
        )

        classified_weather = _add_risk_level(
            scored_weather,
            thresholds["p80"],
            thresholds["p95"],
            thresholds["p99"],
        )

        gold_rows = (
            classified_weather
            .select(*GOLD_OUTPUT_COLUMNS)
            .withColumn("event_time", F.to_timestamp("event_time"))
        )

        gold_rows = _validate_gold_rows(gold_rows)

        logger.info("Checking which city-hour rows already exist in Gold")
        existing_keys = _read_existing_gold_keys(spark)

        new_gold_rows = gold_rows.join(
            existing_keys,
            on=["city", "event_time"],
            how="left_anti",
        )

        new_gold_rows = new_gold_rows.cache()
        new_count = new_gold_rows.count()

        if new_count == 0:
            logger.info("No new rows to write to Gold")
            return

        logger.info(
            "Appending %s new hourly risk rows to %s",
            new_count,
            TARGET_TABLE,
        )

        write_sql_table(new_gold_rows, TARGET_TABLE)

        logger.info("Silver-to-Gold risk load finished successfully")

    finally:
        spark.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run()