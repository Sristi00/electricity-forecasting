# OVERWRITE EXACTLY: src/gnn_data_ingestion.py
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
from gnn_database import get_connection, init_database

def fetch_prices_from_api(days=120):
    """
    Fetches historical electricity prices zone-by-zone from Energy-Charts API.
    Maps real bidding zones to your model's 4-Area topology tags.
    """
    print("📥 Commencing Day-Ahead market extraction from Energy-Charts...")
    
    # Calculate dates matching your data strategy windows
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Dictionary mapping: GNN_Model_Zone_Label -> Official_API_Bidding_Zone_String
    # We MUST pull DE and HYDRO (SE4) because your baseline and graph networks require them as neighbor features!
    zone_mapping = {
        'DK1': 'DK1',
        'DK2': 'DK2',
        'DE': 'DE-LU',
        'HYDRO': 'SE3'  # Sweden Zone 3 (better hydro proxy than SE4)
    }
    
    all_records = []
    
    for model_label, api_bzn in zone_mapping.items():
        url = "https://api.energy-charts.info/price"
        params = {
            'bzn': api_bzn,
            'start': start_str,
            'end': end_str
        }
        
        print(f"   📡 Fetching boundary data for area node: {model_label} (bzn={api_bzn})...")
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # De-serialize parallel lists from the API structural layout
            timestamps = data.get('unix_seconds', [])
            prices_eur = data.get('price', [])
            
            zone_count = 0
            for ts, p_eur in zip(timestamps, prices_eur):
                if p_eur is None:
                    continue
                
                # Format timestamps as standard strings matching your database schema
                hour_str = datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d %H:00:00')
                p_dkk = float(p_eur) * 7.46 # FX peg optimization conversion
                
                all_records.append({
                    'hour_utc': hour_str,
                    'price_zone': model_label,
                    'price_dkk': p_dkk
                })
                zone_count += 1
                
            print(f"   ✅ Processed {zone_count} hourly nodes for {model_label}")
            
        except Exception as e:
            print(f"   ❌ Network/Parsing skip on zone {model_label}: {e}")
            continue
            
    return pd.DataFrame(all_records)

def run_ingestion():
    print("="*60)
    print("🚀 PRODUCTION ENERGY DATA INGESTION ENGINE")
    print("="*60)

    # 1. Ensure schema parameters are initialized in /tmp/energy.db
    init_database()

    # 2. Extract price frames across the 120-day lookback window
    df_prices = fetch_prices_from_api(days=120)

    if df_prices.empty:
        print("❌ Ingestion failure: Combined DataFrame matrix is entirely empty.")
        return False

    # 3. Stream frames straight into SQLite engine
    conn = get_connection()
    try:
        # Use 'append' instead of 'replace' to safely handle incremental runs 
        # while leaning on your database PRIMARY KEY constraints to eliminate duplication
        cursor = conn.cursor()
        
        # We process record inserts manually via an upsert pattern to protect key boundaries
        records_to_insert = df_prices.values.tolist()
        cursor.executemany("""
            INSERT OR REPLACE INTO spot_prices (hour_utc, price_zone, price_dkk)
            VALUES (?, ?, ?)
        """, records_to_insert)
        
        conn.commit()
        print(f"\n📦 Data synchronization finalized!")
        print(f"   Successfully verified and pushed {len(df_prices)} metrics straight to /tmp/energy.db")
        print(f"   Active Operational Hub Nodes: {df_prices['price_zone'].unique().tolist()}")
        return True
    except Exception as e:
        print(f"❌ Database transaction aborted: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    run_ingestion()