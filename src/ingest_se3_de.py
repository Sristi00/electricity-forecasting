"""
Ingest SE3 (→ HYDRO zone) and DE-LU (→ DE zone) hourly spot prices
from the committed CSV snapshot into electricity.db.

Source: /tmp/eu_energy/data/europeanenergy.csv
  Coverage: 2023-01-05 to 2024-08-20
  Zones used: BZN SE3 (Sweden), BZN DE-LU (Germany only)
  Price unit: EUR/MWh → converted to DKK at 7.46

Timestamps in the CSV and the existing DK1/DK2 rows both use CET,
so no timezone conversion is applied — timestamps are stored as-is
to stay consistent with the existing data.

Run once; re-running is safe (INSERT OR IGNORE).
"""
import sys
import sqlite3
import pandas as pd
from pathlib import Path

CSV_PATH = Path("/tmp/eu_energy/data/europeanenergy.csv")
DB_PATH  = Path(__file__).parent / "electricity.db"
EUR_TO_DKK = 7.46


def parse_date(d: str) -> str:
    """Handle mixed date formats: MM/DD/YYYY and DD.MM.YYYY → YYYY-MM-DD."""
    if "/" in d:
        # MM/DD/YYYY
        m, day, y = d.split("/")
        return f"{y}-{m.zfill(2)}-{day.zfill(2)}"
    else:
        # DD.MM.YYYY
        day, m, y = d.split(".")
        return f"{y}-{m.zfill(2)}-{day.zfill(2)}"


def parse_hour(t: str) -> int:
    """Extract start hour from '00:00 - 01:00' format."""
    return int(t.split(":")[0])


def load_zone(df_raw: pd.DataFrame, area: str, country_filter: str | None) -> pd.DataFrame:
    mask = df_raw["area"] == area
    if country_filter:
        mask &= df_raw["country"] == country_filter
    sub = df_raw[mask].copy()

    sub["date_str"] = sub["date"].apply(parse_date)
    sub["hour"]     = sub["time"].apply(parse_hour)
    sub["hour_utc"] = sub["date_str"] + " " + sub["hour"].astype(str).str.zfill(2) + ":00:00"
    sub["price_dkk"] = pd.to_numeric(sub["price(eur/mwh)"], errors="coerce") * EUR_TO_DKK

    # Drop rows with missing/invalid prices
    sub = sub.dropna(subset=["price_dkk"])

    return sub[["hour_utc", "price_dkk"]].drop_duplicates("hour_utc").sort_values("hour_utc").reset_index(drop=True)


def insert_zone(conn: sqlite3.Connection, df: pd.DataFrame, zone: str) -> int:
    cur = conn.cursor()
    inserted = 0
    for _, row in df.iterrows():
        try:
            cur.execute(
                "INSERT OR IGNORE INTO spot_prices (hour_utc, price_zone, price_dkk) VALUES (?, ?, ?)",
                (row["hour_utc"], zone, float(row["price_dkk"])),
            )
            if cur.rowcount > 0:
                inserted += 1
        except Exception as e:
            print(f"  ⚠️  Skipped {row['hour_utc']}: {e}")
    conn.commit()
    return inserted


def main():
    if not CSV_PATH.exists():
        print(f"❌ CSV not found at {CSV_PATH}")
        sys.exit(1)

    print(f"📂 Reading {CSV_PATH} …")
    df_raw = pd.read_csv(CSV_PATH)
    print(f"   Total rows: {len(df_raw):,}")

    df_se3  = load_zone(df_raw, "BZN SE3",   country_filter=None)
    df_delu = load_zone(df_raw, "BZN DE-LU", country_filter="Germany")

    print(f"   SE3  rows after parsing: {len(df_se3):,}  ({df_se3['hour_utc'].min()} → {df_se3['hour_utc'].max()})")
    print(f"   DE-LU rows after parsing: {len(df_delu):,}  ({df_delu['hour_utc'].min()} → {df_delu['hour_utc'].max()})")

    conn = sqlite3.connect(DB_PATH)

    # Pre-check existing counts
    cur = conn.cursor()
    for zone in ("HYDRO", "DE"):
        cur.execute("SELECT COUNT(*) FROM spot_prices WHERE price_zone=?", (zone,))
        n = cur.fetchone()[0]
        print(f"   Existing {zone} rows in DB: {n}")

    print("\n💾 Inserting into spot_prices …")
    n_hydro = insert_zone(conn, df_se3,  "HYDRO")
    n_de    = insert_zone(conn, df_delu, "DE")

    print(f"   HYDRO (SE3)  inserted: {n_hydro:,} new rows")
    print(f"   DE    (DE-LU) inserted: {n_de:,} new rows")

    # Verify
    for zone in ("HYDRO", "DE"):
        cur.execute("SELECT COUNT(*), MIN(hour_utc), MAX(hour_utc) FROM spot_prices WHERE price_zone=?", (zone,))
        row = cur.fetchone()
        print(f"   {zone}: {row[0]:,} total rows  ({row[1]} → {row[2]})")

    conn.close()
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
