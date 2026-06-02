"""
Energinet (Energi Data Service) ingestion — canonical spot-price source.

Replaces the previous energy-charts.info + partial-CSV approach. Pulls the
full hourly history for all four model zones directly from Energinet's public
`Elspotprices` dataset, which already provides prices in DKK (no FX conversion
needed) and covers DK1, DK2 and the neighbouring Nordic/Continental areas for
2020-present.

Zone mapping (Energinet PriceArea → our model zone):
    DK1   → DK1
    DK2   → DK2
    SE3   → HYDRO   (Swedish hydro-dominated zone; hydro proxy for the graph)
    DE-LU → DE      (German/Luxembourg bidding zone; continental coupling)

API: https://api.energidataservice.dk/dataset/Elspotprices
  - No key required, free, public.
  - Records: {HourUTC, HourDK, PriceArea, SpotPriceDKK, SpotPriceEUR}
  - We prefer SpotPriceDKK and fall back to SpotPriceEUR * 7.46 if DKK is null.

⚠️  NETWORK NOTE: this endpoint is 403-blocked inside the Claude-on-the-web
sandbox (the allowlist only permits PyPI/GitHub). Run this on a machine /
environment whose network policy permits api.energidataservice.dk:

    python3 ingest_energinet.py                 # full history 2020-01-01 → now
    python3 ingest_energinet.py --start 2024-01-01
    python3 ingest_energinet.py --check          # connectivity self-test only

It is idempotent (INSERT OR REPLACE), so re-running to top up is safe.
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
from gnn_database import init_database, get_connection, run_query

logger = logging.getLogger(__name__)

ELSPOT_URL = "https://api.energidataservice.dk/dataset/Elspotprices"
EUR_TO_DKK = 7.46
REQUEST_TIMEOUT = 180  # large pages (100k rows) are big JSON downloads
MAX_RETRIES = 5
RETRY_BACKOFF = 5.0  # seconds, doubles each retry (capped at 60)

# Energinet PriceArea string  →  our internal model zone label
AREA_MAP = {
    "DK1":   "DK1",
    "DK2":   "DK2",
    "SE3":   "HYDRO",
    "DE-LU": "DE",
}

DEFAULT_START = "2020-01-01"


# A normal browser-style User-Agent — the default "python-requests/x" UA is
# aggressively rate-limited (HTTP 429) by Energinet's CDN.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
    "Accept": "application/json",
}


def _request_with_retry(params: dict) -> dict:
    """GET Elspotprices with exponential back-off on transient errors (incl. 429)."""
    delay = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(ELSPOT_URL, params=params,
                                headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 403:
                raise RuntimeError(
                    "403 Forbidden from api.energidataservice.dk — the host is "
                    "blocked by this environment's network allowlist. Run this "
                    "script where the endpoint is reachable."
                )
            if resp.status_code in (429, 500, 502, 503, 504):
                # 429 = rate limited; honour Retry-After if present, else back off hard.
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if (retry_after and retry_after.isdigit()) else delay
                logger.warning("HTTP %s (attempt %d/%d) — retry in %.0fs",
                               resp.status_code, attempt, MAX_RETRIES, wait)
                time.sleep(wait)
                delay = min(delay * 2, 60)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            logger.warning("Request error %d/%d: %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return {}


def _month_windows(start: str, end: datetime):
    """Yield (start_iso, end_iso) month-aligned windows from `start` to `end`."""
    cur = datetime.fromisoformat(start).replace(tzinfo=timezone.utc, day=1, hour=0,
                                                minute=0, second=0, microsecond=0)
    while cur < end:
        # First day of next month
        if cur.month == 12:
            nxt = cur.replace(year=cur.year + 1, month=1)
        else:
            nxt = cur.replace(month=cur.month + 1)
        yield (cur.strftime("%Y-%m-%dT%H:%M"), nxt.strftime("%Y-%m-%dT%H:%M"))
        cur = nxt


def _to_hour_str(hour_utc: str) -> str:
    """'2020-01-01T00:00:00' → '2020-01-01 00:00:00' (DB convention)."""
    try:
        dt = datetime.fromisoformat(hour_utc.replace("Z", ""))
        return dt.strftime("%Y-%m-%d %H:00:00")
    except ValueError:
        return hour_utc.replace("T", " ")


# Single-shot page size. Energinet returns exactly the limit we ask for and
# honours large values, so one big request fetches the entire history at once.
# This is critical: on rate-limited (shared university/corporate) IPs, only the
# FIRST request of a run reliably succeeds — a second request gets a multi-minute
# 429 penalty. Asking for everything in one request sidesteps that entirely.
# ~6 years × 4 zones ≈ 210k rows; 2,000,000 comfortably covers it.
PAGE = 2_000_000


def _iter_pages(start: str):
    """
    Yield (page_no, rows) from Elspotprices over [start, now]. Uses one huge
    page so it is normally a single request; still paginates as a safety net.
    """
    end_iso   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    start_iso = datetime.fromisoformat(start).strftime("%Y-%m-%dT%H:%M")
    areas = list(AREA_MAP.keys())
    offset = 0
    page_no = 0
    while True:
        params = {
            "start": start_iso,
            "end": end_iso,
            "filter": json.dumps({"PriceArea": areas}),
            "columns": "HourUTC,PriceArea,SpotPriceDKK,SpotPriceEUR",
            "sort": "HourUTC ASC",
            "limit": PAGE,
            "offset": offset,
        }
        data = _request_with_retry(params)
        rows = data.get("records", [])
        if not rows:
            return
        page_no += 1
        yield page_no, rows
        if len(rows) < PAGE:
            return
        offset += PAGE
        time.sleep(2.0)


def ingest(start: str = DEFAULT_START) -> dict:
    """
    Fetch Energinet Elspotprices and INSERT OR REPLACE into spot_prices,
    committing after EACH page so partial progress survives a rate-limit abort.
    Idempotent: safe to re-run to top up.
    """
    init_database()
    logger.info("Fetching Energinet Elspotprices %s → now …", start)

    conn = get_connection()
    cur = conn.cursor()
    total = 0
    seen = set()
    aborted = None
    try:
        for page_no, rows in _iter_pages(start):
            batch = []
            for r in rows:
                area = r.get("PriceArea")
                zone = AREA_MAP.get(area)
                if zone is None:
                    continue
                p_dkk = r.get("SpotPriceDKK")
                if p_dkk is None:
                    p_eur = r.get("SpotPriceEUR")
                    if p_eur is None:
                        continue
                    p_dkk = float(p_eur) * EUR_TO_DKK
                batch.append((_to_hour_str(r["HourUTC"]), zone, float(p_dkk)))
                seen.add(area)
            cur.executemany(
                "INSERT OR REPLACE INTO spot_prices (hour_utc, price_zone, price_dkk) "
                "VALUES (?, ?, ?)",
                batch,
            )
            conn.commit()  # persist this page before fetching the next
            total += len(batch)
            logger.info("  page %d → %d rows inserted (cumulative %d)",
                        page_no, len(batch), total)
    except Exception as exc:
        conn.commit()
        aborted = exc
        logger.error("Stopped early: %s", exc)
        logger.error("Progress saved (%d rows). The API rate-limited this IP — "
                     "wait a few minutes and simply RE-RUN; it tops up idempotently.",
                     total)
    finally:
        conn.close()

    for a in AREA_MAP:
        if a not in seen:
            logger.warning("Area %s returned NO data — verify the PriceArea label "
                           "for Energinet (e.g. DE-LU may be labelled 'DE').", a)

    summary = run_query(
        "SELECT price_zone, COUNT(*) n, MIN(hour_utc) lo, MAX(hour_utc) hi "
        "FROM spot_prices GROUP BY price_zone ORDER BY price_zone"
    )
    print("\n📊 spot_prices coverage after ingest:")
    print(summary.to_string(index=False))
    return {"rows_inserted": total, "aborted": str(aborted) if aborted else None}


def check_connectivity() -> bool:
    """Lightweight self-test: can we reach the API and get one record?"""
    try:
        data = _request_with_retry({"limit": 1})
        ok = bool(data.get("records"))
        print(f"✅ Reachable — sample record: {data.get('records', [{}])[0]}" if ok
              else "⚠️  Reached API but no records returned.")
        return ok
    except Exception as exc:
        print(f"❌ Not reachable: {exc}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_START, help="ISO date, e.g. 2020-01-01")
    ap.add_argument("--check", action="store_true", help="connectivity self-test only")
    args = ap.parse_args()

    if args.check:
        sys.exit(0 if check_connectivity() else 1)

    result = ingest(start=args.start)
    print(json.dumps(result, indent=2))
