# OVERWRITE EXACTLY: src/gnn_database.py
import sqlite3
import os
import pandas as pd

# Canonical project database — lives next to this module (src/electricity.db).
# This is the single source of truth read by the graph builders and trainers.
DB_PATH = os.path.join(os.path.dirname(__file__), 'electricity.db')

def get_connection():
    """Connect to the canonical project database (src/electricity.db)."""
    return sqlite3.connect(DB_PATH)

def init_database():
    """Initialize localized database tables."""
    print(f"📁 Initializing database at: {DB_PATH}")
    conn = get_connection()
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spot_prices (
            hour_utc TEXT,
            price_zone TEXT,
            price_dkk REAL,
            PRIMARY KEY (hour_utc, price_zone)
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            hour_utc TEXT PRIMARY KEY,
            temperature_c REAL,
            wind_speed_ms REAL,
            cloud_cover_pct REAL,
            humidity_pct REAL
        )
    """)

    # Per-zone fundamentals (demand + renewable generation).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS zone_fundamentals (
            hour_utc TEXT,
            price_zone TEXT,
            load_mwh REAL,
            renewable_mwh REAL,
            PRIMARY KEY (hour_utc, price_zone)
        )
    """)

    # Market-wide factors (fuel + carbon), broadcast to all zones.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_factors (
            hour_utc TEXT PRIMARY KEY,
            gas_dkk REAL,
            co2_dkk REAL
        )
    """)

    conn.commit()
    conn.close()
    print(f"✅ Database ready: {DB_PATH}")

def run_query(query: str, params=None):
    """Run SQL query and return securely as a Pandas DataFrame."""
    conn = get_connection()
    try:
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

if __name__ == "__main__":
    init_database()