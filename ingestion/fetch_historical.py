"""
ingestion/fetch_historical.py
Owner: Mohamed Tarek

ONE-TIME script. Fetches hourly historical weather for every at-risk area
from Open-Meteo and writes raw rows into bronze.historical_weather.

Range: the last HISTORY_YEARS (5) years up to the CURRENT UTC hour, computed
when the script runs. The Archive API (ERA5) lags real time by a few days, so:
  - Archive API  : start -> (today - ARCHIVE_LAG_DAYS)
  - Forecast API : the remaining recent days via past_days, cut at the current hour
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
from sqlalchemy import text
from config import (
    AT_RISK_LOCATIONS,
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_HISTORICAL_URL,
    HOURLY_VARIABLES,
)
from warehouse.db_connection import get_engine
from logger import get_logger

logger = get_logger("fetch_historical")

HISTORY_YEARS = 5
ARCHIVE_LAG_DAYS = 7   # Archive API data for the last few days may be incomplete/NULL


def get_date_range(now: datetime | None = None) -> tuple[str, str, datetime]:
    """Return (archive_start, archive_end, current_hour), all UTC."""
    now = now or datetime.now(timezone.utc)
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    today = current_hour.date()
    try:
        start = today.replace(year=today.year - HISTORY_YEARS)
    except ValueError:  # 29 Feb
        start = today.replace(year=today.year - HISTORY_YEARS, day=28)
    archive_end = today - timedelta(days=ARCHIVE_LAG_DAYS)
    return start.isoformat(), archive_end.isoformat(), current_hour


def _get_json(url: str, params: dict, label: str) -> dict:
    for attempt in range(1, 6):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            logger.warning(f"{label}: attempt {attempt}/5 failed", exc_info=True)
            if attempt == 5:
                raise
            time.sleep(20 * attempt)


def fetch_historical_for_city(city: str, lat: float, lon: float) -> pd.DataFrame:
    start_date, archive_end, current_hour = get_date_range()
    hourly_vars = ",".join(HOURLY_VARIABLES)

    # 1) Long history from the Archive API
    archive = _get_json(
        OPEN_METEO_HISTORICAL_URL,
        {
            "latitude": lat, "longitude": lon,
            "start_date": start_date, "end_date": archive_end,
            "hourly": hourly_vars, "timezone": "GMT",
        },
        f"{city} archive",
    )
    archive_df = pd.DataFrame(archive["hourly"]).rename(columns={"time": "event_time"})

    # 2) Recent days (archive_end -> now) from the Forecast API
    recent = _get_json(
        OPEN_METEO_FORECAST_URL,
        {
            "latitude": lat, "longitude": lon,
            "hourly": hourly_vars, "timezone": "GMT",
            "past_days": ARCHIVE_LAG_DAYS + 1, "forecast_days": 1,
        },
        f"{city} recent",
    )
    recent_df = pd.DataFrame(recent["hourly"]).rename(columns={"time": "event_time"})

    # Keep only hours after the archive part and up to the current hour
    # (the forecast response also contains the rest of today's future hours).
    last_archive_time = archive_df["event_time"].max()
    current_hour_str = current_hour.strftime("%Y-%m-%dT%H:%M")
    recent_df = recent_df[
        (recent_df["event_time"] > last_archive_time)
        & (recent_df["event_time"] <= current_hour_str)
    ]

    city_df = (
        pd.concat([archive_df, recent_df], ignore_index=True)
        .drop_duplicates(subset="event_time", keep="first")
        .sort_values("event_time")
    )
    city_df["city"] = city
    city_df["source"] = "historical"
    return city_df


if __name__ == "__main__":
    logger.info("fetch_historical.py started")
    engine = get_engine()
    total_rows = 0
    failed_cities = []

    for city, (lat, lon) in AT_RISK_LOCATIONS.items():
        logger.info(f"Fetching historical data for {city} (last {HISTORY_YEARS} years -> now)...")
        try:
            city_df = fetch_historical_for_city(city, lat, lon)
            columns = ["city", "event_time"] + HOURLY_VARIABLES + ["source"]
            city_df = city_df[columns]

            with engine.begin() as conn:
                conn.execute(
                    text("DELETE FROM bronze.historical_weather WHERE city = :c"),
                    {"c": city},
                )

            city_df.to_sql(
                "historical_weather", con=engine, schema="bronze",
                if_exists="append", index=False, chunksize=1000,
            )
            total_rows += len(city_df)
            logger.info(f"{city}: wrote {len(city_df)} rows")
            time.sleep(5)
        except Exception:
            logger.error(f"Failed to fetch/write historical data for {city}", exc_info=True)
            failed_cities.append(city)

    if failed_cities:
        logger.warning(f"Finished with errors. Failed cities: {failed_cities}")
        sys.exit(1)  
    logger.info(f"fetch_historical.py finished. {total_rows} total rows written.")