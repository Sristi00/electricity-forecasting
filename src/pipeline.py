"""
End-to-end MLOps pipeline orchestrator for electricity price forecasting.

Stages:
  1. DATA    — ingest new spot prices + weather (incremental)
  2. VALIDATE — data quality checks
  3. BUILD   — rebuild heterogeneous graph if new data or force flag
  4. TRAIN   — retrain HeteroSAGE (primary model)
  5. EVAL    — evaluate and compare vs previous best
  6. REGISTER — promote to MLflow Model Registry if MAE improved
  7. REPORT  — build and return PipelineResult

Can be triggered:
  - via CLI:   python src/pipeline.py [--force]
  - via API:   POST /pipeline/run
  - via CI/CD: GitHub Actions on push to main or workflow_dispatch
"""
import sys
import json
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)

_SRC = Path(__file__).parent
_ARTIFACTS_HETERO = _SRC / "artifacts_hetero"
_METRICS_FILE = _ARTIFACTS_HETERO / "hetero_metrics_clean.json"
_PREV_METRICS_FILE = _ARTIFACTS_HETERO / "previous_best_metrics.json"
_LAST_RUN_FILE = _ARTIFACTS_HETERO / "last_pipeline_run.json"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    status: str                             # "success" | "failed" | "skipped"
    stages_completed: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    previous_metrics: dict = field(default_factory=dict)
    improved: bool = False
    duration_seconds: float = 0.0
    errors: list = field(default_factory=list)
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json_safe(path: Path) -> dict:
    """Load a JSON file; return {} on any error."""
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception as exc:
        logger.warning("Could not load %s: %s", path, exc)
    return {}


def _save_json(path: Path, data: dict):
    """Persist a dict to JSON, creating parent dirs as needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as exc:
        logger.warning("Could not save %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    skip_ingestion: bool = False,
    skip_training: bool = False,
    force_rebuild_graph: bool = False,
) -> PipelineResult:
    """
    Execute the full MLOps pipeline.

    Each stage catches its own exceptions so later stages can still run.
    Returns a PipelineResult with all stage outcomes and metrics.
    """
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    stages_completed = []
    errors = []
    new_rows_ingested = 0
    new_metrics: dict = {}
    prev_metrics: dict = _load_json_safe(_METRICS_FILE)

    logger.info("Pipeline started at %s", timestamp)

    # ------------------------------------------------------------------
    # Stage 1 — Data ingestion
    # ------------------------------------------------------------------
    if not skip_ingestion:
        try:
            from data_ingestion import run_ingestion
            ingest_report = run_ingestion(full_history=False)
            spot_rows = ingest_report.get("spot_prices", {}).get("rows_inserted", 0)
            weather_rows = ingest_report.get("weather", {}).get("rows_inserted", 0)
            new_rows_ingested = spot_rows + weather_rows
            logger.info(
                "Stage 1 [DATA] complete: %d spot + %d weather rows",
                spot_rows,
                weather_rows,
            )
            stages_completed.append("DATA")
        except Exception as exc:
            msg = f"Stage 1 [DATA] failed: {exc}"
            logger.error(msg)
            errors.append(msg)
    else:
        logger.info("Stage 1 [DATA] skipped")
        stages_completed.append("DATA:skipped")

    # ------------------------------------------------------------------
    # Stage 2 — Validation
    # ------------------------------------------------------------------
    try:
        from data_ingestion import validate_ingestion
        validation = validate_ingestion()
        val_status = validation.get("status", "unknown")
        logger.info("Stage 2 [VALIDATE] status: %s", val_status)
        if val_status == "error":
            errors.append(f"Stage 2 [VALIDATE] reported errors: {validation.get('errors', [])}")
        stages_completed.append("VALIDATE")
    except Exception as exc:
        msg = f"Stage 2 [VALIDATE] failed: {exc}"
        logger.error(msg)
        errors.append(msg)

    # ------------------------------------------------------------------
    # Stage 3 — Build graph
    #
    # Incremental run (new data, no --force): freeze the scaler so the feature
    #   distribution stays fixed → the model can be warm-started safely.
    # Full run (--force): refit the scaler and retrain from scratch.
    # ------------------------------------------------------------------
    incremental = not force_rebuild_graph
    should_rebuild = force_rebuild_graph or new_rows_ingested > 0
    if not skip_training and should_rebuild:
        try:
            from hetero_graph_builder import build_heterogeneous_spatiotemporal_graph
            build_heterogeneous_spatiotemporal_graph(freeze_scaler=incremental)
            logger.info(
                "Stage 3 [BUILD] graph rebuilt (%s scaler)",
                "frozen" if incremental else "fresh",
            )
            stages_completed.append("BUILD")
        except Exception as exc:
            msg = f"Stage 3 [BUILD] failed: {exc}"
            logger.error(msg)
            errors.append(msg)
            # BUILD failure is fatal for TRAIN — mark training as skipped
            skip_training = True
    else:
        reason = "skip_training=True" if skip_training else "no new rows"
        logger.info("Stage 3 [BUILD] skipped (%s)", reason)
        stages_completed.append("BUILD:skipped")

    # ------------------------------------------------------------------
    # Stage 4 — Train
    # ------------------------------------------------------------------
    if not skip_training:
        try:
            from quick_retrain import quick_retrain
            # Warm-start fine-tune for incremental runs; full fresh train on --force.
            new_metrics = quick_retrain(warm_start=incremental)
            logger.info(
                "Stage 4 [TRAIN] complete (%s): MAE=%.2f",
                "warm-start" if incremental else "fresh",
                new_metrics.get("mae", float("inf")),
            )
            stages_completed.append("TRAIN")
        except Exception as exc:
            msg = f"Stage 4 [TRAIN] failed: {exc}"
            logger.error(msg)
            errors.append(msg)
    else:
        logger.info("Stage 4 [TRAIN] skipped")
        stages_completed.append("TRAIN:skipped")
        # Load existing metrics for EVAL if training was skipped
        new_metrics = _load_json_safe(_METRICS_FILE)

    # ------------------------------------------------------------------
    # Stage 5 — Evaluate / compare vs previous best
    # ------------------------------------------------------------------
    improved = False
    try:
        # Reload from file in case quick_retrain wrote it directly
        if not new_metrics:
            new_metrics = _load_json_safe(_METRICS_FILE)

        new_mae = new_metrics.get("mae", float("inf"))
        prev_mae = prev_metrics.get("mae", float("inf"))

        if new_mae < prev_mae:
            improved = True
            logger.info(
                "Stage 5 [EVAL] improvement: %.2f -> %.2f DKK MAE", prev_mae, new_mae
            )
            # Save current as new best reference
            _save_json(_PREV_METRICS_FILE, new_metrics)
        else:
            logger.info(
                "Stage 5 [EVAL] no improvement: new=%.2f prev=%.2f DKK MAE",
                new_mae,
                prev_mae,
            )
        stages_completed.append("EVAL")
    except Exception as exc:
        msg = f"Stage 5 [EVAL] failed: {exc}"
        logger.error(msg)
        errors.append(msg)

    # ------------------------------------------------------------------
    # Stage 6 — Register model in MLflow if improved (or no prior best)
    # ------------------------------------------------------------------
    try:
        from mlflow_config import setup_mlflow
        setup_mlflow()
        import mlflow

        if improved or not prev_metrics:
            ckpt_path = _ARTIFACTS_HETERO / "best_hetero_model.pt"
            if ckpt_path.exists():
                with mlflow.start_run(run_name="HeteroSAGE-pipeline-register"):
                    mlflow.log_metrics(new_metrics)
                    mlflow.log_artifact(str(ckpt_path))
                    mlflow.register_model(
                        f"runs:/{mlflow.active_run().info.run_id}/best_hetero_model.pt",
                        "HeteroSAGE-production",
                    )
                logger.info("Stage 6 [REGISTER] model registered in MLflow")
            else:
                logger.warning("Stage 6 [REGISTER] checkpoint not found, skipping registration")
        else:
            logger.info("Stage 6 [REGISTER] skipped — no improvement")
        stages_completed.append("REGISTER")
    except Exception as exc:
        # MLflow is optional — log but don't fail pipeline
        msg = f"Stage 6 [REGISTER] MLflow unavailable or failed: {exc}"
        logger.warning(msg)
        stages_completed.append("REGISTER:skipped")

    # ------------------------------------------------------------------
    # Stage 7 — Report / leaderboard
    # ------------------------------------------------------------------
    leaderboard = {}
    try:
        from compare_models import load_metrics as _compare
        # compare_models.load_metrics() prints; capture silently
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _compare()
        leaderboard = {"compare_output_length": len(buf.getvalue())}
        stages_completed.append("REPORT")
    except Exception as exc:
        msg = f"Stage 7 [REPORT] failed: {exc}"
        logger.warning(msg)
        stages_completed.append("REPORT:skipped")

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    elapsed = round(time.time() - t0, 2)
    overall_status = "failed" if errors and "TRAIN" not in "".join(stages_completed) else "success"
    if not new_metrics and skip_training:
        overall_status = "skipped"

    result = PipelineResult(
        status=overall_status,
        stages_completed=stages_completed,
        metrics=new_metrics,
        previous_metrics=prev_metrics,
        improved=improved,
        duration_seconds=elapsed,
        errors=errors,
        timestamp=timestamp,
    )

    # Persist run info for /pipeline/status endpoint
    _save_json(_LAST_RUN_FILE, result.__dict__)
    logger.info(
        "Pipeline finished in %.1fs — status=%s improved=%s",
        elapsed,
        overall_status,
        improved,
    )
    return result


# ---------------------------------------------------------------------------
# Status query
# ---------------------------------------------------------------------------

def get_pipeline_status() -> dict:
    """Return last pipeline run info from artifacts_hetero/last_pipeline_run.json if it exists."""
    data = _load_json_safe(_LAST_RUN_FILE)
    if not data:
        return {"status": "no_run", "message": "Pipeline has not been run yet"}
    return data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    force = "--force" in sys.argv
    result = run_pipeline(force_rebuild_graph=force)
    print(json.dumps(result.__dict__, indent=2, default=str))
