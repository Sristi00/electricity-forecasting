"""
Fill the DE (Germany/Luxembourg) zone from a GitHub-hosted source.

Energinet's Elspotprices dataset is Nordic-only (PriceArea ∈ {DK1, DK2, SE3,
SE4, NO2}) — it has NO German prices. So DE cannot come from ingest_energinet.py.
This script sources real DE-LU day-ahead prices from a committed research
dataset on GitHub (reachable even when data APIs are firewalled):

    SaiShadow/Cross-Border-Electricity-Price-Forecasting-with-Deep-Learning
    data/Germany_time_zone/all/{train,test}_with_all.csv  (column `y` = DE-LU EUR/MWh)
    Coverage: 2018-10 → 2024-12, hourly, continuous.

Timezone: the source `ds` is German LOCAL time (Europe/Berlin, DST-aware). We
convert it to UTC so it lines up with the Energinet zones (which are stored in
UTC). Validation: after conversion, corr(DK1, DE) ≈ 0.97 at zero offset.

Units: EUR/MWh → DKK at the fixed EUR/DKK peg (7.46), matching the rest of the DB.

Idempotent: replaces the entire DE zone with this single consistent source.
Run from src/:  python3 ingest_de_github.py
"""
import sys
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gnn_database import get_connection, init_database

EUR_TO_DKK = 7.46
REPO = "SaiShadow/Cross-Border-Electricity-Price-Forecasting-with-Deep-Learning"
BASE = f"https://raw.githubusercontent.com/{REPO}/main/data/Germany_time_zone/all"
FILES = [f"{BASE}/train_with_all.csv", f"{BASE}/test_with_all.csv"]


def main():
    init_database()
    print("📥 Downloading DE-LU series from GitHub …")
    frames = [pd.read_csv(u, usecols=["ds", "y"]) for u in FILES]
    de = pd.concat(frames).drop_duplicates("ds")
    de["ds"] = pd.to_datetime(de["ds"])

    # German local time → UTC (DST-aware). Drop the ~1 ambiguous/nonexistent hr/yr.
    loc = de["ds"].dt.tz_localize(ZoneInfo("Europe/Berlin"),
                                  ambiguous="NaT", nonexistent="NaT")
    de = de.assign(utc=loc.dt.tz_convert("UTC")).dropna(subset=["utc"])
    de["hour_utc"] = de["utc"].dt.strftime("%Y-%m-%d %H:00:00")
    de["price_dkk"] = de["y"].astype(float) * EUR_TO_DKK
    de = de.drop_duplicates("hour_utc")
    print(f"   {len(de)} rows | {de['hour_utc'].iloc[0]} .. {de['hour_utc'].iloc[-1]}")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM spot_prices WHERE price_zone='DE'")
    cur.executemany(
        "INSERT OR REPLACE INTO spot_prices(hour_utc, price_zone, price_dkk) VALUES(?,?,?)",
        list(zip(de["hour_utc"], ["DE"] * len(de), de["price_dkk"])),
    )
    conn.commit()

    # Alignment sanity check
    dk1 = pd.read_sql("SELECT hour_utc, price_dkk dk1 FROM spot_prices WHERE price_zone='DK1'", conn)
    ded = pd.read_sql("SELECT hour_utc, price_dkk de FROM spot_prices WHERE price_zone='DE'", conn)
    m = dk1.merge(ded, on="hour_utc")
    if len(m):
        print(f"   corr(DK1, DE) = {np.corrcoef(m['dk1'], m['de'])[0,1]:.4f} (UTC-aligned)")

    print("\n📊 spot_prices coverage:")
    for r in cur.execute("SELECT price_zone, COUNT(*), MIN(hour_utc), MAX(hour_utc) "
                         "FROM spot_prices GROUP BY price_zone ORDER BY price_zone"):
        print(f"  {r[0]:6} {r[1]:6d}  {r[2]} .. {r[3]}")
    conn.close()


if __name__ == "__main__":
    main()
