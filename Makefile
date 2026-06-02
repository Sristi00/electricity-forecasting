.PHONY: help train-all train-hetero train-hetero-warm train-homo train-xgboost compare serve dashboard mlflow-ui docker-up lint ingest ingest-full pipeline pipeline-force monitor

help:
	@echo "Electricity Price Forecasting — Developer Workflow"
	@echo ""
	@echo "Training:"
	@echo "  make train-all        Train all three models"
	@echo "  make train-hetero     Train HeteroSAGE from scratch (quick_retrain.py)"
	@echo "  make train-hetero-warm Fine-tune HeteroSAGE from existing weights (fast)"
	@echo "  make train-homo       Train HomoGNN GraphSAGE (homo_retrain.py)"
	@echo "  make train-xgboost    Train XGBoost baseline (xgboost_baseline.py)"
	@echo ""
	@echo "Data Ingestion:"
	@echo "  make ingest           Incremental ingest (spot prices + weather)"
	@echo "  make ingest-full      Full-history ingest (all available data)"
	@echo ""
	@echo "Pipeline:"
	@echo "  make pipeline         Incremental run (frozen scaler + warm-start fine-tune)"
	@echo "  make pipeline-force   Full rebuild + from-scratch retrain (fresh scaler)"
	@echo ""
	@echo "Monitoring:"
	@echo "  make monitor          Print rolling MAE and drift report"
	@echo ""
	@echo "Evaluation:"
	@echo "  make compare          Print model comparison report"
	@echo ""
	@echo "Serving:"
	@echo "  make serve            Start FastAPI server on :8000"
	@echo "  make dashboard        Start Streamlit dashboard on :8501"
	@echo "  make mlflow-ui        Start MLflow UI on :5000"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up        Start MLflow + API containers"
	@echo ""
	@echo "Quality:"
	@echo "  make lint             Syntax-check all src/*.py files"

train-all: train-xgboost train-homo train-hetero compare

train-hetero:
	PYTHONPATH=src python src/quick_retrain.py

train-hetero-warm:
	PYTHONPATH=src python src/quick_retrain.py --warm-start

train-homo:
	PYTHONPATH=src python src/homo_retrain.py

train-xgboost:
	PYTHONPATH=src python src/xgboost_baseline.py

compare:
	PYTHONPATH=src python src/compare_models.py

serve:
	python -m uvicorn src.model_api:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run src/dashboard.py --server.port=8501

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns.db --host 0.0.0.0 --port 5000

docker-up:
	docker compose up mlflow api

ingest:
	python -c "import sys; sys.path.insert(0,'src'); from data_ingestion import run_ingestion; import json; print(json.dumps(run_ingestion(), indent=2, default=str))"

ingest-full:
	python -c "import sys; sys.path.insert(0,'src'); from data_ingestion import run_ingestion; import json; print(json.dumps(run_ingestion(full_history=True), indent=2, default=str))"

pipeline:
	python src/pipeline.py

pipeline-force:
	python src/pipeline.py --force

monitor:
	python -c "import sys; sys.path.insert(0,'src'); from monitoring import get_monitoring_report; import json; print(json.dumps(get_monitoring_report(), indent=2))"

lint:
	python -m py_compile src/*.py && echo All OK
