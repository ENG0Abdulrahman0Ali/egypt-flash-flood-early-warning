"""Incremental job that transforms new live Bronze rows into Silver."""

import logging
import sys
from datetime import timedelta
from pathlib import Path
from pyspark.sql import functions as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from spark.jdbc_io import (  # noqa: E402
    create_spark_session,
    read_sql_query,
    read_sql_table,
    write_sql_table,
)
from spark.shared_feature_logic import (  # noqa: E402
    SILVER_BASE_COLUMNS,
    SILVER_FEATURE_COLUMNS,
    add_time_features,
    assert_all_locations_mapped,
    clean_bronze_weather,
    enrich_with_location,
)


logger = logging.getLogger("live_incremental_job")

# rain_sum_72h يحتاج 72 قراءة (الساعة الحالية + 71 قبلها)، و rain_lag_24h يحتاج -24h.
CONTEXT_HOURS = 71
SILVER_FEATURE_COLUMNS_FOR_CONTEXT = [*SILVER_BASE_COLUMNS, "elevation", "slope_degree"]

def _read_new_live_rows(spark):
    """Read live Bronze rows whose city and event time are not in Silver yet."""
    live_rows = (
        read_sql_table(spark, "bronze.live_weather")
        .filter("source = 'live'")
        .withColumn("event_time", F.to_timestamp("event_time"))
        .dropDuplicates(["city", "event_time"])
    )

    silver_keys = (
        read_sql_table(spark, "silver.fact_weather_features")
        .select(
            "city",
            F.to_timestamp("event_time").alias("event_time"),
        )
        .dropDuplicates(["city", "event_time"])
    )

    return live_rows.join(
        silver_keys,
        on=["city", "event_time"],
        how="left_anti",
    )


def _read_silver_context(spark, earliest_new_event_time):
    """Read earlier Silver rows needed as window context for the new rows.

    Window functions only see the rows inside the DataFrame they run on. If we
    called ``add_time_features`` on the new rows alone, rain_sum_*h would collapse
    to the current hour and rain_lag_* would be NULL. So we also load the
    previous ``CONTEXT_HOURS`` hours from Silver, run the shared logic over
    context + new rows, and keep only the new rows afterwards.
    """
    cutoff = earliest_new_event_time - timedelta(hours=CONTEXT_HOURS)
    cutoff_text = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    columns = ", ".join(SILVER_FEATURE_COLUMNS_FOR_CONTEXT)

    # الفلتر يتنفذ جوه SQL Server بدل ما نسحب كل Silver.
    context = read_sql_query(
        spark,
        f"SELECT {columns} FROM silver.fact_weather_features "
        f"WHERE event_time >= '{cutoff_text}'",
    )
    return (
        context.withColumn("city", F.trim(F.col("city")))
        .withColumn("event_time", F.col("event_time").cast("timestamp"))
        .withColumn("precipitation", F.col("precipitation").cast("double"))
        .withColumn("relative_humidity_2m", F.col("relative_humidity_2m").cast("double"))
        .withColumn("temperature_2m", F.col("temperature_2m").cast("double"))
        .withColumn("surface_pressure", F.col("surface_pressure").cast("double"))
        .withColumn("windspeed_10m", F.col("windspeed_10m").cast("double"))
        .withColumn("cloudcover", F.col("cloudcover").cast("double"))
        .withColumn("soil_moisture_0_to_7cm", F.col("soil_moisture_0_to_7cm").cast("double"))
        .withColumn("elevation", F.col("elevation").cast("double"))
        .withColumn("slope_degree", F.col("slope_degree").cast("double"))
    )


def compute_new_silver_rows(enriched_new, silver_context):
    """Run the shared feature logic over context + new rows, return only new rows."""
    new_keys = enriched_new.select("city", "event_time").dropDuplicates()
    combined = silver_context.unionByName(enriched_new.select(*silver_context.columns))
    combined = combined.dropDuplicates(["city", "event_time"])

    return (
        add_time_features(combined)
        .join(new_keys, on=["city", "event_time"], how="left_semi")
        .select(*SILVER_FEATURE_COLUMNS)
    )


def run() -> None:
    """Execute live Bronze -> Silver processing."""
    spark = create_spark_session("live-incremental")

    try:
        logger.info("Reading new live Bronze rows")
        new_live_rows = _read_new_live_rows(spark)

        if not new_live_rows.take(1):
            logger.info("No new live rows to process")
            return

        logger.info("Reading the location dimension")
        locations = read_sql_table(spark, "gold.dim_location")

        logger.info("Cleaning and enriching new live rows")
        clean_weather = clean_bronze_weather(new_live_rows)
        enriched_weather = enrich_with_location(clean_weather, locations)
        assert_all_locations_mapped(enriched_weather)

        # Cache: الصفوف الجديدة بتتستخدم مرتين (earliest time + union).
        enriched_weather = enriched_weather.cache()
        earliest = enriched_weather.agg(F.min("event_time")).first()[0]

        logger.info("Reading Silver context (last %sh before %s)", CONTEXT_HOURS, earliest)
        silver_context = _read_silver_context(spark, earliest)

        silver_features = compute_new_silver_rows(enriched_weather, silver_context)

        logger.info("Appending new live features to Silver")
        write_sql_table(silver_features, "silver.fact_weather_features")
        logger.info("Live incremental job finished successfully")

    finally:
        spark.stop()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run()