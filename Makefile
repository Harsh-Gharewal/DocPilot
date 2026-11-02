.PHONY: install test seed run-api run-dashboard eval compare docker-build docker-up

install:
	pip install -r requirements.txt matplotlib

test:
	pytest -v

seed:
	python scripts/seed.py

run-api:
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	streamlit run dashboard/app.py --server.port 8501

eval:
	python -m eval.run_eval --strategy structure_aware --mode hybrid_rerank --label baseline

compare:
	python -m eval.compare_strategies

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

