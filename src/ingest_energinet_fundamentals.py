"""
Ingest DK1/DK2 fundamentals (demand + wind/solar generation) from Energinet.

These are the strongest exogenous price drivers for the Danish zones and are
NOT in the price feed — they live in Energinet's settlement dataset:

    dataset: ProductionConsumptionSettlement   (hourly, per PriceArea)
      load_mwh      = GrossConsumptionMWh
      renewable_mwh = (all *Wind*_MWh columns summed) + SolarPower*_MWh

Writes into zone_fundamentals(hour_utc, price_zone, load_mwh, renewable_mwh)
for price_zone ∈ {DK1, DK2}. UTC timestamps (Energinet HourUTC) → aligns with
the price/weather data already in the DB.

⚠️  Like ingest_energinet.py, api.energidataservice.dk is firewall-blocked in
the Claude-on-the-web sandbox — RUN THIS ON YOUR MACHINE:

    python3 ingest_energinet_fundamentals.py
    python3 ingest_energinet_fundamentals.py --check   # connectivity test

Single big request (rate-limit-friendly) + idempotent INSERT OR REPLACE.
"""
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import requests
from gnn_database import get_connection, init_database, run_query
# Reuse the hardened request helper (browser UA, 429 backoff) from the price script.
from ingest_energinet import _request_with_retry, _to_hour_str, HEADERS  # noqa: F401

logger = logging.getLogger(__name__)

DATASET_URL = "https://api.energidataservice.dk/dataset/ProductionConsumptionSettlement"
ZONES = ["DK1", "DK2"]
START = "2020-01-01"
PAGE = 2_000_000


def _request(params):
    """Thin wrapper so we hit ProductionConsumptionSettlement, not Elspotprices."""
    import ingest_energinet as ie
    _orig = ie.ELSPOT_URL
    ie.ELSPOT_URL = DATASET_URL
    try:
        return _request_with_retry(params)
    finally:
        ie.ELSPOT_URL = _orig


def _pick_columns(sample: dict):
    """Identify load / wind / solar columns from a sample record (schema-robust)."""
    load = next((k for k in sample if k.lower() == "grossconsumptionmwh"), None)
    wind = [k for k in sample if "wind" in k.lower() and k.lower().endswith("mwh")]
    solar = [k for k in sample if "solar" in k.lower() and k.lower().endswith("mwh")]
    return load, wind, solar


def fetch():
    end_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    params = {
        "start": datetime.fromisoformat(START).strftime("%Y-%m-%dT%H:%M"),
        "end": end_iso,
        "filter": json.dumps({"PriceArea": ZONES}),
        "sort": "HourUTC ASC",
        "limit": PAGE,
        "offset": 0,
    }
    data = _request(params)
    rows = data.get("records", [])
    if not rows:
        return []
    load_c, wind_c, solar_c = _pick_columns(rows[0])
    logger.info("Detected columns — load=%s | wind=%s | solar=%s", load_c, wind_c, solar_c)
    out = []
    for r in rows:
        zone = r.get("PriceArea")
        if zone not in ZONES:
            continue
        load = r.get(load_c) if load_c else None
        wind = sum(float(r[c]) for c in wind_c if r.get(c) is not None)
        solar = sum(float(r[c]) for c in solar_c if r.get(c) is not None)
        renew = wind + solar
        out.append((_to_hour_str(r["HourUTC"]), zone,
                    float(load) if load is not None else None, float(renew)))
    return out


def ingest():
    init_database()
    logger.info("Fetching Energinet ProductionConsumptionSettlement %s → now …", START)
    records = fetch()
    if not records:
        logger.error("No records fetched — nothing inserted.")
        return
    conn = get_connection()
    cur = conn.cursor()
    cur.executemany(
        "INSERT OR REPLACE INTO zone_fundamentals(hour_utc, price_zone, load_mwh, renewable_mwh) "
        "VALUES(?,?,?,?)",
        records,
    )
    conn.commit()
    conn.close()
    summary = run_query(
        "SELECT price_zone, COUNT(*) n, MIN(hour_utc) lo, MAX(hour_utc) hi, "
        "AVG(load_mwh) load, AVG(renewable_mwh) ren "
        "FROM zone_fundamentals GROUP BY price_zone ORDER BY price_zone"
    )
    print("\n📊 zone_fundamentals coverage:")
    print(summary.to_string(index=False))


def check():
    try:
        d = _request({"limit": 1})
        rec = d.get("records", [{}])
        print("✅ Reachable. Sample keys:", sorted(rec[0].keys()) if rec else "none")
        return bool(rec)
    except Exception as e:
        print(f"❌ Not reachable: {e}")
        return False


def ingest_from_csv(csv_path: str):
    """Ingest DK1/DK2 fundamentals from a local ProductionConsumptionSettlement CSV.

    The CSV is semicolon-delimited with European decimal format (comma as decimal
    separator), as downloaded directly from Energinet's open-data portal.
    Use this when the API endpoint is firewalled (e.g. sandbox / university network):
        python3 ingest_energinet_fundamentals.py --csv ProductionConsumptionSettlement.csv
    """
    import pandas as pd
    wind_cols = [
        'OffshoreWindLt100MW_MWh', 'OffshoreWindGe100MW_MWh',
        'OnshoreWindLt50kW_MWh',   'OnshoreWindGe50kW_MWh',
    ]
    solar_cols = [
        'SolarPowerLt10kW_MWh', 'SolarPowerGe10Lt40kW_MWh',
        'SolarPowerGe40kW_MWh', 'SolarPowerSelfConMWh',
    ]
    needed = ['HourUTC', 'PriceArea', 'GrossConsumptionMWh', 'HydroPowerMWh'] + wind_cols + solar_cols

    df = pd.read_csv(csv_path, sep=';', decimal=',', usecols=needed)
    df = df[df['PriceArea'].isin(['DK1', 'DK2'])]
    df['hour_utc'] = pd.to_datetime(df['HourUTC']).dt.strftime('%Y-%m-%d %H:00:00')
    df['load_mwh'] = df['GrossConsumptionMWh'].fillna(0)
    df['renewable_mwh'] = (
        df[wind_cols].fillna(0).sum(axis=1)
        + df[solar_cols].fillna(0).sum(axis=1)
        + df['HydroPowerMWh'].fillna(0)
    )
    records = list(zip(df['hour_utc'], df['PriceArea'], df['load_mwh'], df['renewable_mwh']))
    logger.info("Parsed %d rows from %s", len(records), csv_path)

    init_database()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM zone_fundamentals WHERE price_zone IN ('DK1','DK2')")
    cur.executemany(
        "INSERT OR REPLACE INTO zone_fundamentals(hour_utc, price_zone, load_mwh, renewable_mwh) "
        "VALUES(?,?,?,?)",
        records,
    )
    conn.commit()
    summary = run_query(
        "SELECT price_zone, COUNT(*) n, MIN(hour_utc) lo, MAX(hour_utc) hi, "
        "AVG(load_mwh) load, AVG(renewable_mwh) ren "
        "FROM zone_fundamentals GROUP BY price_zone ORDER BY price_zone"
    )
    print("\n📊 zone_fundamentals coverage after CSV ingest:")
    print(summary.to_string(index=False))
    conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--csv",   metavar="FILE",
                    help="Ingest from local ProductionConsumptionSettlement CSV "
                         "(semicolon-delimited, European decimal) instead of API")
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if check() else 1)
    elif args.csv:
        ingest_from_csv(args.csv)
    else:
        ingest()
