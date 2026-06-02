"""
Ingest fundamentals that aren't in Energinet, from the SaiShadow GitHub dataset:

  • Market-wide factors → market_factors table (broadcast to all zones):
        gas  : NCG/THE day-ahead gas (EUR/MWh, DAILY → forward-filled to hourly) → DKK
        co2  : EU ETS CO2 auction (EUR/t, DAILY → forward-filled) → DKK
  • DE zone fundamentals → zone_fundamentals (price_zone='DE'):
        load_mwh      : German Load
        renewable_mwh : German Renewable generation (wind+solar+hydro+bio)

Source repo (GitHub, reachable even when data APIs are firewalled):
  SaiShadow/Cross-Border-Electricity-Price-Forecasting-with-Deep-Learning
  data/Germany_time_zone/load/data/all_<YEAR>.csv  (Load, Renewable, CO2, ts has +01:00 offset)
  data/Germany_time_zone/gas/data/gas_<YEAR>.csv   (daily gas, DD.MM.YYYY)

Timestamps: the load files carry an explicit UTC offset (…+01:00), so we parse
to UTC exactly. Gas/CO2 are daily and forward-filled across each day's 24 hours.

Run from src/:  python3 ingest_fundamentals_github.py
Idempotent (replaces DE fundamentals + market_factors for the covered range).
"""
import sys
import io
import requests
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gnn_database import get_connection, init_database

EUR_TO_DKK = 7.46
REPO = "SaiShadow/Cross-Border-Electricity-Price-Forecasting-with-Deep-Learning"
BASE = f"https://raw.githubusercontent.com/{REPO}/main/data/Germany_time_zone"
YEARS = range(2018, 2025)  # 2018..2024 inclusive


def _get_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def main():
    init_database()

    # ── DE load/renewable + CO2 (from the 'all_<year>' files) ──────────────────
    load_frames = []
    for y in YEARS:
        try:
            load_frames.append(_get_csv(f"{BASE}/load/data/all_{y}.csv"))
        except Exception as e:
            print(f"   ⚠️  all_{y}.csv: {e}")
    de = pd.concat(load_frames, ignore_index=True)
    # Timestamp carries an explicit offset (e.g. 2021-01-01T00:00+01:00) → UTC
    ts = pd.to_datetime(de["Date (GMT+1)"], utc=True)
    de["hour_utc"] = ts.dt.strftime("%Y-%m-%d %H:00:00")
    de = de.drop_duplicates("hour_utc").sort_values("hour_utc")

    de_load = de["Load"].astype(float)
    de_ren  = de["Renewable"].astype(float)
    co2_col = "CO2 Emission Allowances, Auction DE"
    co2 = de[co2_col].astype(float) * EUR_TO_DKK if co2_col in de.columns else None

    # ── Gas (daily) → hourly via ffill over the DE timeline ────────────────────
    gas_frames = []
    for y in YEARS:
        try:
            g = _get_csv(f"{BASE}/gas/data/gas_{y}.csv")
            gas_frames.append(g)
        except Exception as e:
            print(f"   ⚠️  gas_{y}.csv: {e}")
    gas = pd.concat(gas_frames, ignore_index=True)
    gas_col = [c for c in gas.columns if "Gas" in c][0]
    gas["day"] = pd.to_datetime(gas["Day"], format="%d.%m.%Y").dt.strftime("%Y-%m-%d")
    gas = gas[["day", gas_col]].dropna().drop_duplicates("day").sort_values("day")
    gas_map = dict(zip(gas["day"], gas[gas_col].astype(float) * EUR_TO_DKK))

    conn = get_connection()
    cur = conn.cursor()

    # zone_fundamentals: DE
    cur.execute("DELETE FROM zone_fundamentals WHERE price_zone='DE'")
    cur.executemany(
        "INSERT OR REPLACE INTO zone_fundamentals(hour_utc, price_zone, load_mwh, renewable_mwh) "
        "VALUES(?,?,?,?)",
        list(zip(de["hour_utc"], ["DE"] * len(de), de_load, de_ren)),
    )

    # market_factors: gas (ffill from daily) + co2
    gas_series = pd.Series(de["hour_utc"].str[:10].map(gas_map).values, index=de["hour_utc"].values)
    gas_series = gas_series.ffill().bfill()
    rows = []
    for hu, g, c in zip(de["hour_utc"].values, gas_series.values,
                        (co2.values if co2 is not None else [None] * len(de))):
        rows.append((hu, float(g) if pd.notna(g) else None,
                     float(c) if (c is not None and pd.notna(c)) else None))
    cur.execute("DELETE FROM market_factors")
    cur.executemany(
        "INSERT OR REPLACE INTO market_factors(hour_utc, gas_dkk, co2_dkk) VALUES(?,?,?)",
        rows,
    )
    conn.commit()

    # Report
    for tbl, q in [
        ("zone_fundamentals (DE)",
         "SELECT COUNT(*), MIN(hour_utc), MAX(hour_utc), AVG(load_mwh), AVG(renewable_mwh) "
         "FROM zone_fundamentals WHERE price_zone='DE'"),
        ("market_factors",
         "SELECT COUNT(*), MIN(hour_utc), MAX(hour_utc), AVG(gas_dkk), AVG(co2_dkk) FROM market_factors"),
    ]:
        print(f"  {tbl}: {cur.execute(q).fetchone()}")
    conn.close()
    print("✅ GitHub fundamentals ingested (DE load/renewable + global gas/CO2).")


if __name__ == "__main__":
    main()
