# OVERWRITE EXACTLY: src/hetero_pipeline.py
import pandas as pd
import numpy as np
from gnn_database import run_query

def prepare_multi_area_data():
    """Extracts, processes, and aligns multi-zone spot prices and weather inputs cleanly."""
    print("\n🔄 Extracting Production 4-Area System Data from SQLite...")
    
    # 1. Pull market spot prices
    df_prices = run_query("SELECT hour_utc as timestamp, price_zone, price_dkk FROM spot_prices")
    if df_prices.empty:
        raise ValueError("❌ spot_prices table is completely empty! Please run your scraper data ingestion scripts first.")
        
    df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp']).dt.strftime('%Y-%m-%d %H:00:00')
    
    # 2. Pull local atmospheric weather measurements
    df_weather = run_query("SELECT * FROM weather_data")
    if df_weather.empty:
        print("⚠️ Warning: weather_data table is empty. Generating synthetic baseline parameters.")
        unique_times = df_prices['timestamp'].unique()
        df_weather = pd.DataFrame({
            'hour_utc': unique_times,
            'temperature_c': np.random.uniform(5.0, 18.0, len(unique_times)),
            'wind_speed_ms': np.random.uniform(2.0, 12.0, len(unique_times)),
            'cloud_cover_pct': np.random.uniform(10.0, 90.0, len(unique_times)),
            'humidity_pct': np.random.uniform(60.0, 95.0, len(unique_times))
        })
    
    df_weather['timestamp'] = pd.to_datetime(df_weather['hour_utc']).dt.strftime('%Y-%m-%d %H:00:00')
    df_weather = df_weather.drop(columns=['hour_utc'])
    
    # 3. Extract individual zone data frames
    df_dk1 = df_prices[df_prices['price_zone'] == 'DK1'].copy().reset_index(drop=True)
    df_dk2 = df_prices[df_prices['price_zone'] == 'DK2'].copy().reset_index(drop=True)
    df_de = df_prices[df_prices['price_zone'] == 'DE'].copy().reset_index(drop=True)
    df_hydro = df_prices[df_prices['price_zone'] == 'HYDRO'].copy().reset_index(drop=True)
    
    # 4. Timeline Synchronization - Strict ffill only (historical lookup), NO global bfill
    print("   Synchronizing timelines via outer-join...")
    df_master = df_dk1[['timestamp', 'price_dkk']].rename(columns={'price_dkk': 'dk1'})
    df_master = df_master.merge(df_dk2[['timestamp', 'price_dkk']].rename(columns={'price_dkk': 'dk2'}), on='timestamp', how='outer')
    df_master = df_master.merge(df_de[['timestamp', 'price_dkk']].rename(columns={'price_dkk': 'de'}), on='timestamp', how='outer')
    df_master = df_master.merge(df_hydro[['timestamp', 'price_dkk']].rename(columns={'price_dkk': 'hydro'}), on='timestamp', how='outer')
    
    # Leakage fixed: Forward-fill values up to the gap, fill remaining historic edges with neutral 0.0
    df_master = df_master.sort_values('timestamp').ffill().fillna(0.0)
    
    # Re-split back into standardized individual zone views
    df_dk1 = df_master[['timestamp', 'dk1']].rename(columns={'dk1': 'price_dkk'})
    df_dk2 = df_master[['timestamp', 'dk2']].rename(columns={'dk2': 'price_dkk'})
    df_de = df_master[['timestamp', 'de']].rename(columns={'de': 'price_dkk'})
    df_hydro = df_master[['timestamp', 'hydro']].rename(columns={'hydro': 'price_dkk'})
    
    # 5. Connect weather factors
    df_dk1 = df_dk1.merge(df_weather, on='timestamp', how='left')
    df_dk2 = df_dk2.merge(df_weather, on='timestamp', how='left')
    
    # Fill remaining gaps using safe forward-fills and neutral medians to prevent look-ahead leaks
    for df in [df_dk1, df_dk2]:
        df['temperature_c'] = df['temperature_c'].ffill().fillna(12.0)
        df['wind_speed_ms'] = df['wind_speed_ms'].ffill().fillna(5.0)
        df['cloud_cover_pct'] = df['cloud_cover_pct'].ffill().fillna(50.0)
        df['humidity_pct'] = df['humidity_pct'].ffill().fillna(75.0)
        
    # 6. Apply Feature Engineering
    print("   Engineering historical lag attributes on synchronized frames...")
    df_dk1 = _add_autoregressive_features(df_dk1)
    df_dk2 = _add_autoregressive_features(df_dk2)
    df_de = _add_autoregressive_features(df_de)
    df_hydro = _add_autoregressive_features(df_hydro)

    # 7. Attach fundamentals (demand + renewable generation) and market factors.
    #    Per-zone load/renewable from zone_fundamentals; global gas/CO2 from
    #    market_factors broadcast to every zone. Missing (zone,feature) → 0.0.
    df_dk1 = _attach_fundamentals(df_dk1, 'DK1')
    df_dk2 = _attach_fundamentals(df_dk2, 'DK2')
    df_de = _attach_fundamentals(df_de, 'DE')
    df_hydro = _attach_fundamentals(df_hydro, 'HYDRO')

    print(f"   Pipeline successfully returning data frames. Total size: {len(df_dk1)} records.")
    return df_dk1, df_dk2, df_hydro, df_de, df_weather


def _attach_fundamentals(df, zone):
    """Merge per-zone load/renewable and global gas/CO2 onto a zone frame.

    Day-ahead-safe: load/renewable are 'perfect forecast' proxies (like weather);
    gas/CO2 are slow daily series. All columns default to 0.0 when absent so the
    feature stays well-defined for zones without a data source (e.g. SE3/HYDRO).
    """
    try:
        zf = run_query(
            "SELECT hour_utc as timestamp, load_mwh, renewable_mwh "
            "FROM zone_fundamentals WHERE price_zone=?", params=(zone,)
        )
    except Exception:
        zf = pd.DataFrame(columns=['timestamp', 'load_mwh', 'renewable_mwh'])
    try:
        mf = run_query("SELECT hour_utc as timestamp, gas_dkk, co2_dkk FROM market_factors")
    except Exception:
        mf = pd.DataFrame(columns=['timestamp', 'gas_dkk', 'co2_dkk'])

    for frame in (zf, mf):
        if not frame.empty:
            frame['timestamp'] = pd.to_datetime(frame['timestamp']).dt.strftime('%Y-%m-%d %H:00:00')

    df = df.merge(zf, on='timestamp', how='left') if not zf.empty else df.assign(load_mwh=0.0, renewable_mwh=0.0)
    df = df.merge(mf, on='timestamp', how='left') if not mf.empty else df.assign(gas_dkk=0.0, co2_dkk=0.0)

    for c in ['load_mwh', 'renewable_mwh', 'gas_dkk', 'co2_dkk']:
        if c not in df.columns:
            df[c] = 0.0
        # ffill slow/again-daily series, then neutral 0.0 for any remaining edges
        df[c] = df[c].ffill().fillna(0.0)

    # Per-zone z-score for demand/generation so each zone's WITHIN-zone variation
    # survives the downstream global scaler. Without this, German load (~55 GWh)
    # would dominate the shared scaler and crush DK's (~2 GWh) variation to noise.
    # Stats from the first 80% (chronological train split) → day-ahead-safe.
    n_train = int(len(df) * 0.8)
    for c in ['load_mwh', 'renewable_mwh']:
        tr = df[c].iloc[:n_train]
        mu, sd = tr.mean(), tr.std()
        if sd and sd > 1e-6:  # skip all-zero zones (no data source)
            df[c] = (df[c] - mu) / sd
    return df

def _add_autoregressive_features(df):
    """
    Day-ahead-safe autoregressive features.

    All lags are anchored at >= 24h, so every feature is known at gate closure
    (12:00 CET on day D-1) when forecasting the 24 delivery hours of day D.
    Nord Pool publishes the full day-D-1 price curve at noon on day D-2, so the
    entire previous day (lag_24h) is available for all target hours — no leakage.

    Replaces the previous next-hour design (shift(1)/shift(2)/shift(6)), which
    leaked the most recent actual price and made the model a one-step-ahead
    predictor rather than a genuine day-ahead forecaster.

    Weather columns (merged upstream at the target timestamp) are kept as a
    "perfect weather forecast" proxy — the standard EPF convention, since 12–36h
    weather forecasts are highly accurate.
    """
    df['price_lag_24h']  = df['price_dkk'].shift(24)    # same hour, previous day
    df['price_lag_48h']  = df['price_dkk'].shift(48)    # same hour, two days ago
    df['price_lag_168h'] = df['price_dkk'].shift(168)   # same hour, previous week

    # Rolling stats over the last fully-known day (24–47h ago) — leakage-free
    df['price_rolling_24h_mean'] = df['price_dkk'].shift(24).rolling(24).mean()
    df['price_rolling_24h_std']  = df['price_dkk'].shift(24).rolling(24).std()

    df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
    df['minute'] = pd.to_datetime(df['timestamp']).dt.minute

    # Cold-start NaNs (first 168h) filled with 0.0 — never backfilled from future
    return df.fillna(0.0).reset_index(drop=True)