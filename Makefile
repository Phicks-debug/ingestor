run:
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Docker commands
docker-build:
	docker build -t ingestor:latest .

docker-run:
	docker run -d \
		--name ingestor \
		-p 8000:8000 \
		--env-file .env \
		ingestor:latest

docker-stop:
	docker stop ingestor && docker rm ingestor

docker-logs:
	docker logs -f ingestor

docker-shell:
	docker exec -it ingestor /bin/bash

# Qdrant commands
qdrant-up:
	docker compose up -d

qdrant-down:
	docker compose down -v