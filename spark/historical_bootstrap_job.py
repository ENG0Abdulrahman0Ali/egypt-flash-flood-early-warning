"""Batch job that transforms all historical Bronze weather rows into Silver."""

import logging
import sys
from pathlib import Path

from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spark.jdbc_io import (  # noqa: E402
    create_spark_session,
    overwrite_staging_table,
    read_sql_table,
)
from spark.shared_feature_logic import (  # noqa: E402
    SILVER_FEATURE_COLUMNS,
    add_time_features,
    assert_all_locations_mapped,
    clean_bronze_weather,
    enrich_with_location,
)
from warehouse.db_connection import get_engine  # noqa: E402


logger = logging.getLogger("historical_bootstrap_job")
STAGING_TABLE = "silver.fact_weather_features_stage"
TARGET_TABLE = "silver.fact_weather_features"



def _replace_historical_silver_rows() -> None:
    """Replace historical Silver rows without overwriting live records."""
    column_list = ", ".join(SILVER_FEATURE_COLUMNS)
    engine = get_engine()

    try:
        with engine.begin() as connection:
            staged_count = connection.execute(
                text(
                    f"SELECT COUNT_BIG(*) FROM {STAGING_TABLE} "
                    "WHERE source = :source"
                ),
                {"source": "historical"},
            ).scalar_one()

            if staged_count == 0:
                raise ValueError(
                    "Historical staging table is empty; Silver rows were not changed"
                )

            # Remove the previous historical dataset.
            connection.execute(
                text(f"DELETE FROM {TARGET_TABLE} WHERE source = :source"),
                {"source": "historical"},
            )

            # Insert historical rows only when their city/time key
            # does not already exist in Silver (for example, as live data).
            insert_result = connection.execute(
                text(
                    f"INSERT INTO {TARGET_TABLE} ({column_list}) "
                    f"SELECT {', '.join(f's.{col}' for col in SILVER_FEATURE_COLUMNS)} "
                    f"FROM {STAGING_TABLE} AS s "
                    "WHERE s.source = :source "
                    f"AND NOT EXISTS ("
                    f"SELECT 1 FROM {TARGET_TABLE} AS t "
                    "WHERE t.city = s.city "
                    "AND t.event_time = s.event_time"
                    ")"
                ),
                {"source": "historical"},
            )

            logger.info(
                "Historical staging rows: %s; inserted rows: %s",
                staged_count,
                insert_result.rowcount,
            )

            connection.execute(text(f"DROP TABLE {STAGING_TABLE}"))

    finally:
        engine.dispose()



def run() -> None:
    """Execute historical Bronze -> Silver processing."""
    spark = create_spark_session("historical-bootstrap")
    try:
        logger.info("Reading historical Bronze weather and the location dimension")
        bronze = read_sql_table(spark, "bronze.historical_weather").filter(
            "source = 'historical'"
        )
        locations = read_sql_table(spark, "gold.dim_location")

        logger.info("Cleaning and enriching the historical weather rows")
        clean_weather = clean_bronze_weather(bronze)
        enriched_weather = enrich_with_location(clean_weather, locations)
        assert_all_locations_mapped(enriched_weather)
        silver_features = add_time_features(enriched_weather).select(
            *SILVER_FEATURE_COLUMNS
        )

        logger.info("Writing the calculated rows to the replaceable staging table")
        overwrite_staging_table(silver_features, STAGING_TABLE)

        logger.info("Replacing only historical rows in Silver")
        _replace_historical_silver_rows()
        logger.info("Historical bootstrap finished successfully")
    finally:
        spark.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run()
