import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import json
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hetero_pipeline import prepare_multi_area_data
from mlflow_config import setup_mlflow


def train_xgboost_baseline():
    print("\n Training Production-Aligned XGBoost Baseline (DK1 + DK2)...")

    # ── MLflow setup ────────────────────────────────────────────────────────────
    try:
        setup_mlflow()
        import mlflow
        _mlflow_run = mlflow.start_run(run_name="XGBoost")
        _mlflow_ok = True
    except Exception as _e:
        print(f"   [MLflow] Disabled — {_e}")
        _mlflow_ok = False

    try:
        # 1. Pull perfectly synchronized frames
        df_dk1, df_dk2, df_hydro, df_de, _ = prepare_multi_area_data()

        # 2. Assign discrete structural labels
        df_dk1 = df_dk1.copy()
        df_dk2 = df_dk2.copy()
        df_dk1['zone_id'] = 0
        df_dk2['zone_id'] = 1

        # 3. Inject neighborhood inputs to maintain feature parity with GNN designs.
        #    Use the neighbor zones' 24h-lagged price (same hour, previous day) —
        #    NOT the target-hour price. Tomorrow's DE/HYDRO prices clear in the same
        #    day-ahead auction and are unknown at gate closure, so using them would
        #    leak. The 24h lag is published the day before and is leakage-free.
        df_dk1['neighbor_price_de'] = df_de['price_lag_24h']
        df_dk1['neighbor_price_hydro'] = df_hydro['price_lag_24h']

        df_dk2['neighbor_price_de'] = df_de['price_lag_24h']
        df_dk2['neighbor_price_hydro'] = df_hydro['price_lag_24h']

        # Stack features vertically
        df = pd.concat([df_dk1, df_dk2], ignore_index=True)

        # 4. Execute a strict chronological split (80% Train, 20% Test)
        unique_timestamps = sorted(df['timestamp'].unique())
        split_idx = int(0.8 * len(unique_timestamps))
        split_timestamp = unique_timestamps[split_idx]

        train_mask = df['timestamp'] < split_timestamp
        test_mask = df['timestamp'] >= split_timestamp

        feature_cols = [
            'hour_of_day', 'minute', 'zone_id',
            'price_lag_24h', 'price_lag_48h', 'price_lag_168h',
            'price_rolling_24h_mean', 'price_rolling_24h_std',
            'neighbor_price_de', 'neighbor_price_hydro',
            'temperature_c', 'wind_speed_ms', 'cloud_cover_pct'
        ]

        X_train = df.loc[train_mask, feature_cols].fillna(0)
        y_train = df.loc[train_mask, 'price_dkk']

        X_test = df.loc[test_mask, feature_cols].fillna(0)
        y_test = df.loc[test_mask, 'price_dkk']

        print(f"   Total Timeline Intervals: {len(unique_timestamps)}")
        print(f"   Training Vectors: {len(X_train)} rows")
        print(f"   Testing Vectors : {len(X_test)} rows")

        # 5. Train Tabular Regressor Model
        n_estimators = 250
        max_depth = 6
        learning_rate = 0.05
        subsample = 0.8
        colsample_bytree = 0.8

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1,
        )

        # Log hyperparameters
        if _mlflow_ok:
            try:
                mlflow.log_params({
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "random_state": 42,
                    "n_features": len(feature_cols),
                })
            except Exception:
                pass

        model.fit(X_train, y_train)

        # 6. Evaluation metrics calculation (Leakage-Proof & Zero-Price Stable)
        y_pred = model.predict(X_test)

        # Robust SMAPE calculation to handle zero/negative pricing safely
        y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test
        smape = np.mean(2.0 * np.abs(y_pred - y_test_np) / (np.abs(y_test_np) + np.abs(y_pred) + 1e-8)) * 100

        metrics = {
            "mae":   float(mean_absolute_error(y_test, y_pred)),
            "rmse":  float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2":    float(r2_score(y_test, y_pred)),
            "smape": float(smape),
        }

        print("\n XGBoost Baseline Evaluation Metrics:")
        print(f"   MAE   : {metrics['mae']:.4f} DKK")
        print(f"   RMSE  : {metrics['rmse']:.4f} DKK")
        print(f"   R2    : {metrics['r2']:.4f}")
        print(f"   SMAPE : {metrics['smape']:.2f}%")

        # Log final metrics
        if _mlflow_ok:
            try:
                mlflow.log_metrics({
                    "test_mae":   metrics["mae"],
                    "test_rmse":  metrics["rmse"],
                    "test_r2":    metrics["r2"],
                    "test_smape": metrics["smape"],
                })
            except Exception:
                pass

        # Write model outputs to an artifacts folder
        ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
        ARTIFACTS_DIR.mkdir(exist_ok=True)

        pkl_path = ARTIFACTS_DIR / "xgboost_baseline.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)

        with open(ARTIFACTS_DIR / "xgboost_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Log checkpoint artifact
        if _mlflow_ok:
            try:
                mlflow.log_artifact(str(pkl_path))
            except Exception:
                pass

        print(f"   Saved baseline metrics package to: {ARTIFACTS_DIR}")
        return model, metrics

    finally:
        if _mlflow_ok:
            try:
                mlflow.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    train_xgboost_baseline()
