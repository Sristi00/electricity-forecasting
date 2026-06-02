"""
Model and data monitoring for production use.

Tracks: prediction logs, rolling MAE, feature drift vs training baseline.
Stores to monitoring_log table in SQLite (same DB as the rest of the project).
"""
import sys
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from gnn_database import get_connection

logger = logging.getLogger(__name__)

_SRC = Path(__file__).parent
_ARTIFACTS_HETERO = _SRC / "artifacts_hetero"
_DEFAULT_BASELINE_PATH = str(_ARTIFACTS_HETERO / "feature_baseline_stats.json")


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def init_monitoring_db():
    """Create monitoring_log table if it does not exist."""
    conn = get_connection()
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS monitoring_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT NOT NULL,
                zone          TEXT,
                predicted_dkk REAL,
                actual_dkk    REAL,
                mae           REAL,
                features_json TEXT
            )"""
        )
        conn.commit()
        logger.info("monitoring_log table ready")
    except Exception as exc:
        logger.error("init_monitoring_db failed: %s", exc)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Logging predictions
# ---------------------------------------------------------------------------

def log_prediction(
    zone: str,
    predicted_dkk: Optional[float],
    features: dict,
    actual_dkk: Optional[float] = None,
):
    """Insert one prediction record into monitoring_log."""
    init_monitoring_db()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    mae = None
    if predicted_dkk is not None and actual_dkk is not None:
        mae = abs(predicted_dkk - actual_dkk)

    try:
        conn = get_connection()
        conn.execute(
            """INSERT INTO monitoring_log
               (timestamp, zone, predicted_dkk, actual_dkk, mae, features_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (now, zone, predicted_dkk, actual_dkk, mae, json.dumps(features)),
        )
        conn.commit()
    except Exception as exc:
        logger.error("log_prediction failed: %s", exc)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Rolling MAE
# ---------------------------------------------------------------------------

def compute_rolling_mae(zone: str, window_hours: int = 168) -> dict:
    """
    Compute MAE over the last window_hours predictions where actual_dkk IS NOT NULL.

    Returns: {"zone": zone, "window_hours": window_hours, "mae": float|None, "n_samples": int}
    """
    init_monitoring_db()
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=window_hours)
    ).strftime("%Y-%m-%d %H:%M:%S")

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT predicted_dkk, actual_dkk
               FROM monitoring_log
               WHERE zone = ?
                 AND actual_dkk IS NOT NULL
                 AND predicted_dkk IS NOT NULL
                 AND timestamp >= ?
               ORDER BY timestamp DESC""",
            (zone, cutoff),
        )
        rows = cursor.fetchall()
    except Exception as exc:
        logger.error("compute_rolling_mae query failed: %s", exc)
        rows = []
    finally:
        conn.close()

    n = len(rows)
    if n == 0:
        mae = None
    else:
        mae = float(np.mean([abs(r[0] - r[1]) for r in rows]))

    return {"zone": zone, "window_hours": window_hours, "mae": mae, "n_samples": n}


# ---------------------------------------------------------------------------
# Feature drift
# ---------------------------------------------------------------------------

def detect_feature_drift(
    current_features: dict,
    baseline_stats_path: str = _DEFAULT_BASELINE_PATH,
) -> dict:
    """
    Compare current feature values vs training-time baseline stats (mean, std).

    Loads baseline from artifacts_hetero/feature_baseline_stats.json if it exists.
    A feature is flagged as drifted if |z-score| > 3.

    Returns: {"drifted_features": [...], "z_scores": {...}, "status": "ok"|"drift_detected"}
    """
    baseline_path = Path(baseline_stats_path)
    if not baseline_path.exists():
        logger.warning("Baseline stats file not found at %s — skipping drift check", baseline_path)
        return {
            "drifted_features": [],
            "z_scores": {},
            "status": "ok",
            "note": "baseline stats not available",
        }

    try:
        with open(baseline_path) as f:
            baseline = json.load(f)
    except Exception as exc:
        logger.error("Could not load baseline stats: %s", exc)
        return {"drifted_features": [], "z_scores": {}, "status": "ok", "error": str(exc)}

    z_scores = {}
    drifted = []
    means = baseline.get("means", {})
    stds = baseline.get("stds", {})

    for feature, value in current_features.items():
        mu = means.get(feature)
        sigma = stds.get(feature)
        if mu is None or sigma is None:
            continue
        if sigma == 0:
            z = 0.0
        else:
            z = (value - mu) / sigma
        z_scores[feature] = round(float(z), 3)
        if abs(z) > 3:
            drifted.append(feature)

    status = "drift_detected" if drifted else "ok"
    return {
        "drifted_features": drifted,
        "z_scores": z_scores,
        "status": status,
    }


def save_baseline_stats(x_base_numpy: np.ndarray, feature_names: list, output_path: str):
    """
    Save mean and std of training features to JSON for drift detection.

    Args:
        x_base_numpy: 2D numpy array of shape (n_samples, n_features)
        feature_names: list of feature name strings
        output_path: path to write the JSON file
    """
    means = x_base_numpy.mean(axis=0).tolist()
    stds = x_base_numpy.std(axis=0).tolist()

    stats = {
        "feature_names": feature_names,
        "means": dict(zip(feature_names, means)),
        "stds": dict(zip(feature_names, stds)),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved baseline stats to %s", out)


# ---------------------------------------------------------------------------
# Monitoring report
# ---------------------------------------------------------------------------

def get_monitoring_report() -> dict:
    """Return rolling MAE for DK1 and DK2, last 24h prediction count, drift status."""
    init_monitoring_db()

    # Rolling MAE per zone (7-day window)
    dk1_mae = compute_rolling_mae("DK1", window_hours=168)
    dk2_mae = compute_rolling_mae("DK2", window_hours=168)

    # Last 24h prediction count
    cutoff_24h = (
        datetime.now(timezone.utc) - timedelta(hours=24)
    ).strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM monitoring_log WHERE timestamp >= ?",
            (cutoff_24h,),
        )
        row = cursor.fetchone()
        predictions_last_24h = int(row[0]) if row else 0
    except Exception as exc:
        logger.error("Prediction count query failed: %s", exc)
        predictions_last_24h = 0
    finally:
        conn.close()

    # Drift detection: load the most recent prediction features if any
    drift_status = {"status": "ok", "note": "no recent predictions to check"}
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT features_json FROM monitoring_log WHERE features_json IS NOT NULL ORDER BY timestamp DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row and row[0]:
            latest_features = json.loads(row[0])
            if latest_features:
                drift_status = detect_feature_drift(latest_features)
    except Exception as exc:
        logger.warning("Drift check in report failed: %s", exc)
    finally:
        conn.close()

    return {
        "rolling_mae_dk1": dk1_mae,
        "rolling_mae_dk2": dk2_mae,
        "predictions_last_24h": predictions_last_24h,
        "drift_status": drift_status,
    }


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = get_monitoring_report()
    print(json.dumps(report, indent=2, default=str))
