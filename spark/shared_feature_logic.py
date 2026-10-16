"""Shared cleaning logic for the Egypt flash-flood Spark jobs.

The functions here convert and clean Bronze rows, enrich them with the small
location dimension, and calculate Silver time features. Historical and live
jobs can therefore share the same rules.
"""

from pyspark.sql import DataFrame, Window, functions as F

BRONZE_REQUIRED_COLUMNS = {
    "id",
    "city",
    "event_time",
    "precipitation",
    "relative_humidity_2m",
    "temperature_2m",
    "surface_pressure",
    "windspeed_10m",
    "cloudcover",
    "soil_moisture_0_to_7cm",
    "ingested_at",
    "source",
}

OPTIONAL_WEATHER_COLUMNS = [
    "temperature_2m",
    "surface_pressure",
    "windspeed_10m",
    "cloudcover",
    "soil_moisture_0_to_7cm",
]

SILVER_BASE_COLUMNS = [
    "city",
    "event_time",
    "source",
    "precipitation",
    "relative_humidity_2m",
    "temperature_2m",
    "surface_pressure",
    "windspeed_10m",
    "cloudcover",
    "soil_moisture_0_to_7cm",
]

LOCATION_REQUIRED_COLUMNS = {"city", "elevation", "slope_degree"}

TIME_FEATURE_REQUIRED_COLUMNS = {
    "city",
    "event_time",
    "source",
    "precipitation",
    "relative_humidity_2m",
    "temperature_2m",
    "surface_pressure",
    "windspeed_10m",
    "cloudcover",
    "soil_moisture_0_to_7cm",
    "elevation",
    "slope_degree",
}

SILVER_FEATURE_COLUMNS = [
    *SILVER_BASE_COLUMNS,
    "elevation",
    "slope_degree",
    "rain_sum_1h",
    "rain_sum_3h",
    "rain_sum_6h",
    "rain_sum_24h",
    "rain_sum_72h",
    "rain_lag_1h",
    "rain_lag_24h",
    "hour_of_day",
    "month",
]


def clean_bronze_weather(df: DataFrame) -> DataFrame:
    """Return validated, deduplicated weather rows ready for enrichment.

    Rules in this first version:
    - Parse Open-Meteo's ``yyyy-MM-ddTHH:mm`` event time into a timestamp.
    - Keep only rows with a city, source, non-negative rain, and humidity in
      the inclusive range 0..100. Missing rain/humidity cannot be loaded into
      the Silver schema, so those rows are dropped.
    - Fill the other weather measurements with zero, following the Spark task
      notes.
    - For a repeated city/hour, keep the most recently ingested Bronze row;
      ``id`` breaks ties within the source Bronze table.

    Spark transformations are lazy: this function describes the work but does
    not read or calculate all rows until a later action such as ``show`` or
    ``write`` is called.
    """
    missing_columns = BRONZE_REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Bronze DataFrame is missing required columns: {missing_list}"
        )

    typed = (
        df.withColumn("city", F.trim(F.col("city")))
        .withColumn(
            "event_time",
            F.to_timestamp(F.col("event_time"), "yyyy-MM-dd'T'HH:mm"),
        )
        .withColumn("precipitation", F.col("precipitation").cast("double"))
        .withColumn(
            "relative_humidity_2m",
            F.col("relative_humidity_2m").cast("double"),
        )
        .withColumn("ingested_at", F.col("ingested_at").cast("timestamp"))
    )

    for column_name in OPTIONAL_WEATHER_COLUMNS:
        typed = typed.withColumn(column_name, F.col(column_name).cast("double"))

    valid = typed.filter(
        F.col("city").isNotNull()
        & (F.length(F.col("city")) > 0)
        & F.col("source").isNotNull()
        & F.col("event_time").isNotNull()
        & F.col("precipitation").isNotNull()
        & (F.col("precipitation") >= 0)
        & F.col("relative_humidity_2m").isNotNull()
        & F.col("relative_humidity_2m").between(0, 100)
    )

    filled = valid.fillna(0.0, subset=OPTIONAL_WEATHER_COLUMNS).withColumn(
        # ERA5 soil moisture can be slightly negative (e.g. -0.015): clip to 0.
        "soil_moisture_0_to_7cm",
        F.greatest(F.col("soil_moisture_0_to_7cm"), F.lit(0.0)),
    )

    latest_for_city_hour = Window.partitionBy("city", "event_time").orderBy(
        F.col("ingested_at").desc_nulls_last(), F.col("id").desc()
    )

    return (
        filled.withColumn("_row_number", F.row_number().over(latest_for_city_hour))
        .filter(F.col("_row_number") == 1)
        .select(*SILVER_BASE_COLUMNS)
    )


def enrich_with_location(weather_df: DataFrame, location_df: DataFrame) -> DataFrame:
    """Add elevation and slope from the small ``gold.dim_location`` table.

    The SQL schema makes ``city`` the primary key, so each weather row should
    match at most one location. A left join keeps every weather row visible;
    callers must run ``assert_all_locations_mapped`` before writing Silver so
    an unknown city cannot silently disappear or be stored without terrain.
    """
    missing_columns = LOCATION_REQUIRED_COLUMNS.difference(location_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Location DataFrame is missing required columns: {missing_list}"
        )

    location_lookup = (
        location_df.select("city", "elevation", "slope_degree")
        .withColumn("city", F.trim(F.col("city")))
        .withColumn("elevation", F.col("elevation").cast("double"))
        .withColumn("slope_degree", F.col("slope_degree").cast("double"))
    )

    return weather_df.join(F.broadcast(location_lookup), on="city", how="left")


def assert_all_locations_mapped(enriched_df: DataFrame) -> None:
    """Fail before a Silver write if any weather city lacks terrain data.

    ``collect`` is bounded to five city names; the large weather rows remain
    distributed in Spark and are not copied to the Python process.
    """
    unmapped = (
        enriched_df.filter(F.col("elevation").isNull() | F.col("slope_degree").isNull())
        .select("city")
        .distinct()
        .limit(5)
        .collect()
    )

    if unmapped:
        city_names = ", ".join(row["city"] for row in unmapped)
        raise ValueError(
            f"No gold.dim_location row found for city/cities: {city_names}"
        )


def add_time_features(enriched_df: DataFrame) -> DataFrame:
    """Calculate Silver rolling sums, exact-time rain lags, and calendar fields.

    The rolling sums contain the current hourly reading plus previous hourly
    readings: 3 rows span 2 hours between timestamps, 6 span 5 hours, and so
    on. The range windows use elapsed seconds, so a missing hour does not make
    an older reading slide into the time window. Missing observations remain
    missing data; this function does not invent a zero-rain reading for them.

    Exact lag columns use timestamp-keyed self-joins. ``lag`` by row number
    would incorrectly treat the previous available reading as exactly one or
    24 hours earlier when timestamps are missing. The Spark session timezone
    should be UTC before parsing timestamps and deriving ``hour_of_day``.
    """
    missing_columns = TIME_FEATURE_REQUIRED_COLUMNS.difference(enriched_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Enriched DataFrame is missing required columns: {missing_list}"
        )

    timed = enriched_df.withColumn(
        "_event_epoch_seconds",
        F.col("event_time").cast("long"),
    )

    def trailing_window(hours: int):
        return (
            Window.partitionBy("city")
            .orderBy(F.col("_event_epoch_seconds"))
            .rangeBetween(-((hours - 1) * 60 * 60), 0)
        )

    with_sums = (
        timed.withColumn("rain_sum_1h", F.col("precipitation"))
        .withColumn(
            "rain_sum_3h",
            F.sum("precipitation").over(trailing_window(3)),
        )
        .withColumn(
            "rain_sum_6h",
            F.sum("precipitation").over(trailing_window(6)),
        )
        .withColumn(
            "rain_sum_24h",
            F.sum("precipitation").over(trailing_window(24)),
        )
        .withColumn(
            "rain_sum_72h",
            F.sum("precipitation").over(trailing_window(72)),
        )
        .withColumn("hour_of_day", F.hour("event_time"))
        .withColumn("month", F.month("event_time"))
    )

    lag_1h = timed.select(
        "city",
        (F.col("_event_epoch_seconds") + 60 * 60).alias("_event_epoch_seconds"),
        F.col("precipitation").alias("rain_lag_1h"),
    )
    lag_24h = timed.select(
        "city",
        (F.col("_event_epoch_seconds") + 24 * 60 * 60).alias("_event_epoch_seconds"),
        F.col("precipitation").alias("rain_lag_24h"),
    )

    return (
        with_sums.join(lag_1h, on=["city", "_event_epoch_seconds"], how="left")
        .join(lag_24h, on=["city", "_event_epoch_seconds"], how="left")
        .select(*SILVER_FEATURE_COLUMNS)
    )