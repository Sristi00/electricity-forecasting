"""
Production-grade data ingestion module for electricity price forecasting.

Fetches spot prices from Energy-Charts API and weather from Open-Meteo archive API.
Supports both full-history and incremental modes.
"""
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import requests
from gnn_database import init_database, get_connection, run_query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EUR_TO_DKK = 7.46

ZONE_MAPPING = {
    "DK1": "DK1",
    "DK2": "DK2",
    "DE": "DE-LU",
    "HYDRO": "SE3",  # Sweden Zone 3 (SE4 feeds DK, but SE3 is a better hydro proxy)
}

ENERGY_CHARTS_URL = "https://api.energy-charts.info/price"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_LAT = 56.16
WEATHER_LON = 10.20
WEATHER_PARAMS = [
    "temperature_2m",
    "windspeed_10m",
    "cloudcover",
    "relativehumidity_2m",
]

WEATHER_START_DEFAULT = "2020-01-01"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF = 5.0  # seconds (doubles each retry)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _request_with_retry(url: str, params: dict) -> dict:
    """GET request with exponential back-off on 429/5xx errors."""
    delay = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code in (429, 500, 502, 503, 504):
                logger.warning(
                    "HTTP %s from %s (attempt %d/%d) — retrying in %.0fs",
                    resp.status_code, url, attempt, MAX_RETRIES, delay,
                )
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            logger.warning("Request error on attempt %d/%d: %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(delay)
            delay *= 2
    return {}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def get_last_ingested_timestamp(db_conn) -> "str | None":
    """Query spot_prices for MAX(hour_utc) — used for incremental updates."""
    try:
        cursor = db_conn.cursor()
        cursor.execute("SELECT MAX(hour_utc) FROM spot_prices")
        row = cursor.fetchone()
        return row[0] if row and row[0] else None
    except Exception as exc:
        logger.warning("Could not read last ingested timestamp: %s", exc)
        return None


def ingest_spot_prices(full_history: bool = False) -> dict:
    """
    Fetch spot prices from Energy-Charts API.

    - If full_history=True OR DB is empty: fetch max available history
      (days=2000, covers ~5+ years).
    - Otherwise: incremental — fetch only since last timestamp + 7-day buffer.
    - Zones: DK1, DK2, DE-LU (mapped to DE), SE4 (mapped to HYDRO).
    - Converts EUR -> DKK (multiply by 7.46).
    - Uses INSERT OR REPLACE into spot_prices(hour_utc, price_zone, price_dkk).

    Returns: {"rows_inserted": int, "date_range": [start, end], "zones": [...]}
    """
    conn = get_connection()
    try:
        last_ts = get_last_ingested_timestamp(conn)
    finally:
        conn.close()

    end_date = datetime.now(timezone.utc)
    end_str = end_date.strftime("%Y-%m-%d")

    if full_history or last_ts is None:
        start_date = end_date - timedelta(days=2000)
        start_str = start_date.strftime("%Y-%m-%d")
        logger.info("Full-history mode: fetching %s -> %s", start_str, end_str)
    else:
        # Incremental: go back 7 days as a buffer for late-arriving data
        last_dt = datetime.fromisoformat(last_ts.replace(" ", "T")).replace(tzinfo=timezone.utc)
        start_date = last_dt - timedelta(days=7)
        start_str = start_date.strftime("%Y-%m-%d")
        logger.info("Incremental mode: fetching %s -> %s", start_str, end_str)

    all_records = []

    for model_label, api_bzn in ZONE_MAPPING.items():
        params = {"bzn": api_bzn, "start": start_str, "end": end_str}
        logger.info("Fetching zone %s (bzn=%s) ...", model_label, api_bzn)
        try:
            data = _request_with_retry(ENERGY_CHARTS_URL, params)
        except Exception as exc:
            logger.error("Failed to fetch zone %s: %s — skipping", model_label, exc)
            continue

        timestamps = data.get("unix_seconds", [])
        prices_eur = data.get("price", [])

        if not timestamps:
            logger.warning("No data returned for zone %s", model_label)
            continue

        zone_count = 0
        for ts, p_eur in zip(timestamps, prices_eur):
            if p_eur is None:
                continue
            hour_str = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:00:00")
            p_dkk = float(p_eur) * EUR_TO_DKK
            all_records.append((hour_str, model_label, p_dkk))
            zone_count += 1

        logger.info("  %s: %d records", model_label, zone_count)

    if not all_records:
        logger.warning("No spot price records fetched — skipping DB insert")
        return {"rows_inserted": 0, "date_range": [start_str, end_str], "zones": []}

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT OR REPLACE INTO spot_prices (hour_utc, price_zone, price_dkk) VALUES (?, ?, ?)",
            all_records,
        )
        conn.commit()
        rows_inserted = len(all_records)
        logger.info("Inserted/replaced %d spot price rows", rows_inserted)
    except Exception as exc:
        logger.error("DB insert failed for spot prices: %s", exc)
        rows_inserted = 0
    finally:
        conn.close()

    zones_fetched = sorted({r[1] for r in all_records})
    return {
        "rows_inserted": rows_inserted,
        "date_range": [start_str, end_str],
        "zones": zones_fetched,
    }


def ingest_weather(full_history: bool = False) -> dict:
    """
    Fetch hourly weather for Denmark (lat=56.16, lon=10.20) from Open-Meteo archive API.

    - If full_history=True OR weather_data is empty: fetch from 2020-01-01 to today.
    - Otherwise: fetch since last weather timestamp.
    - INSERT OR REPLACE into weather_data.

    Returns: {"rows_inserted": int, "date_range": [start, end]}
    """
    # Open-Meteo archive typically lags ~5 days; cap end at 5 days ago
    end_date = datetime.now(timezone.utc) - timedelta(days=5)
    end_str = end_date.strftime("%Y-%m-%d")

    # Determine start date
    df_last = run_query("SELECT MAX(hour_utc) as last_ts FROM weather_data")
    last_weather_ts = df_last["last_ts"].iloc[0] if not df_last.empty else None

    if full_history or last_weather_ts is None or str(last_weather_ts) == "None":
        start_str = WEATHER_START_DEFAULT
        logger.info("Full-history weather mode: %s -> %s", start_str, end_str)
    else:
        last_dt = datetime.fromisoformat(
            str(last_weather_ts).replace(" ", "T")
        ).replace(tzinfo=timezone.utc)
        start_str = last_dt.strftime("%Y-%m-%d")
        logger.info("Incremental weather mode: %s -> %s", start_str, end_str)

    params = {
        "latitude": WEATHER_LAT,
        "longitude": WEATHER_LON,
        "hourly": ",".join(WEATHER_PARAMS),
        "start_date": start_str,
        "end_date": end_str,
    }

    try:
        data = _request_with_retry(OPEN_METEO_URL, params)
    except Exception as exc:
        logger.error("Failed to fetch weather data: %s", exc)
        return {"rows_inserted": 0, "date_range": [start_str, end_str]}

    hourly = data.get("hourly", {})
    time_vals = hourly.get("time", [])

    if not time_vals:
        logger.warning("No weather data returned for %s -> %s", start_str, end_str)
        return {"rows_inserted": 0, "date_range": [start_str, end_str]}

    temp_vals = hourly.get("temperature_2m", [None] * len(time_vals))
    wind_vals = hourly.get("windspeed_10m", [None] * len(time_vals))
    cloud_vals = hourly.get("cloudcover", [None] * len(time_vals))
    humid_vals = hourly.get("relativehumidity_2m", [None] * len(time_vals))

    records = []
    for t, temp, wind, cloud, humid in zip(time_vals, temp_vals, wind_vals, cloud_vals, humid_vals):
        # Open-Meteo returns ISO strings like "2020-01-01T00:00"
        try:
            dt = datetime.fromisoformat(t)
            hour_str = dt.strftime("%Y-%m-%d %H:00:00")
        except ValueError:
            hour_str = t
        records.append((hour_str, temp, wind, cloud, humid))

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.executemany(
            """INSERT OR REPLACE INTO weather_data
               (hour_utc, temperature_c, wind_speed_ms, cloud_cover_pct, humidity_pct)
               VALUES (?, ?, ?, ?, ?)""",
            records,
        )
        conn.commit()
        rows_inserted = len(records)
        logger.info("Inserted/replaced %d weather rows", rows_inserted)
    except Exception as exc:
        logger.error("DB insert failed for weather data: %s", exc)
        rows_inserted = 0
    finally:
        conn.close()

    return {"rows_inserted": rows_inserted, "date_range": [start_str, end_str]}


def validate_ingestion() -> dict:
    """
    Data quality checks:
    - Row counts per zone in spot_prices
    - Date range coverage
    - % of nulls
    - Gap detection (missing hours > 24h)
    - Weather coverage vs price coverage

    Returns: {"status": "ok"|"warning"|"error", "checks": {...}}
    """
    checks = {}
    warnings = []
    errors = []

    # Row counts per zone
    try:
        df_zone = run_query(
            "SELECT price_zone, COUNT(*) as row_count FROM spot_prices GROUP BY price_zone"
        )
        if df_zone.empty:
            errors.append("spot_prices table is empty")
        else:
            checks["zone_row_counts"] = dict(
                zip(df_zone["price_zone"].tolist(), df_zone["row_count"].tolist())
            )
    except Exception as exc:
        errors.append(f"zone row count query failed: {exc}")

    # Date range coverage
    try:
        df_range = run_query(
            "SELECT MIN(hour_utc) as min_ts, MAX(hour_utc) as max_ts FROM spot_prices"
        )
        if not df_range.empty and df_range["min_ts"].iloc[0]:
            checks["spot_price_date_range"] = {
                "min": str(df_range["min_ts"].iloc[0]),
                "max": str(df_range["max_ts"].iloc[0]),
            }
        else:
            warnings.append("No date range info in spot_prices")
    except Exception as exc:
        errors.append(f"date range query failed: {exc}")

    # Null percentage
    try:
        df_nulls = run_query(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN price_dkk IS NULL THEN 1 ELSE 0 END) as nulls "
            "FROM spot_prices"
        )
        if not df_nulls.empty:
            total = df_nulls["total"].iloc[0]
            nulls = df_nulls["nulls"].iloc[0]
            null_pct = (nulls / total * 100) if total > 0 else 0.0
            checks["spot_price_null_pct"] = round(float(null_pct), 2)
            if null_pct > 5:
                warnings.append(f"High null rate in spot_prices: {null_pct:.1f}%")
    except Exception as exc:
        warnings.append(f"null check failed: {exc}")

    # Gap detection in DK1 (representative zone)
    try:
        df_dk1 = run_query(
            "SELECT hour_utc FROM spot_prices WHERE price_zone='DK1' ORDER BY hour_utc"
        )
        if len(df_dk1) > 1:
            times = [
                datetime.fromisoformat(str(t).replace(" ", "T"))
                for t in df_dk1["hour_utc"].tolist()
            ]
            gaps = []
            for i in range(1, len(times)):
                diff_hours = (times[i] - times[i - 1]).total_seconds() / 3600
                if diff_hours > 24:
                    gaps.append({
                        "from": str(times[i - 1]),
                        "to": str(times[i]),
                        "gap_hours": round(diff_hours, 1),
                    })
            checks["dk1_gaps_over_24h"] = len(gaps)
            if gaps:
                checks["dk1_gap_examples"] = gaps[:3]
                warnings.append(f"Found {len(gaps)} gap(s) exceeding 24h in DK1 data")
        else:
            warnings.append("Insufficient DK1 data for gap detection")
    except Exception as exc:
        warnings.append(f"gap detection failed: {exc}")

    # Weather coverage vs price coverage
    try:
        df_w = run_query(
            "SELECT COUNT(*) as cnt, MIN(hour_utc) as min_ts, MAX(hour_utc) as max_ts FROM weather_data"
        )
        if not df_w.empty and df_w["cnt"].iloc[0]:
            checks["weather_row_count"] = int(df_w["cnt"].iloc[0])
            checks["weather_date_range"] = {
                "min": str(df_w["min_ts"].iloc[0]),
                "max": str(df_w["max_ts"].iloc[0]),
            }
        else:
            warnings.append("weather_data table is empty")
    except Exception as exc:
        warnings.append(f"weather coverage check failed: {exc}")

    # Determine overall status
    if errors:
        status = "error"
    elif warnings:
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
    }


def run_ingestion(full_history: bool = False) -> dict:
    """
    Orchestrate: init_database -> ingest_spot_prices -> ingest_weather -> validate.
    Returns a combined report dictionary.
    """
    logger.info("Starting ingestion pipeline (full_history=%s)", full_history)

    # 1. Ensure DB schema is initialized
    try:
        init_database()
    except Exception as exc:
        logger.error("init_database failed: %s", exc)
        return {"status": "error", "error": str(exc)}

    report: dict = {"full_history": full_history}

    # 2. Ingest spot prices
    try:
        spot_result = ingest_spot_prices(full_history=full_history)
        report["spot_prices"] = spot_result
        logger.info("Spot prices: %d rows inserted", spot_result.get("rows_inserted", 0))
    except Exception as exc:
        logger.error("Spot price ingestion failed: %s", exc)
        report["spot_prices"] = {"error": str(exc), "rows_inserted": 0}

    # 3. Ingest weather
    try:
        weather_result = ingest_weather(full_history=full_history)
        report["weather"] = weather_result
        logger.info("Weather: %d rows inserted", weather_result.get("rows_inserted", 0))
    except Exception as exc:
        logger.error("Weather ingestion failed: %s", exc)
        report["weather"] = {"error": str(exc), "rows_inserted": 0}

    # 4. Validate
    try:
        validation = validate_ingestion()
        report["validation"] = validation
        logger.info("Validation status: %s", validation["status"])
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        report["validation"] = {"status": "error", "error": str(exc)}

    report["status"] = "ok" if not report.get("validation", {}).get("errors") else "warning"
    return report


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_ingestion(full_history=False)
    print(json.dumps(result, indent=2, default=str))
