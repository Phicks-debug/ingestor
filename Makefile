run:
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

qdrant-up:
	docker compose up -d

qdrant-down:
	docker compose down -v