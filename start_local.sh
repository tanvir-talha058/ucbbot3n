#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
QDRANT_CONTAINER="ucb-qdrant"
QDRANT_IMAGE="qdrant/qdrant:latest"
HOST="127.0.0.1"
PORT="${PORT:-8000}"
QDRANT_ONLY="false"
RELOAD_FLAG="--reload"

for arg in "$@"; do
  case "$arg" in
    --qdrant-only)
      QDRANT_ONLY="true"
      ;;
    --no-reload)
      RELOAD_FLAG=""
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: ./start_local.sh [--qdrant-only] [--no-reload]"
      exit 1
      ;;
  esac
done

echo "==> UCB Chatbot local startup"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed or not in PATH."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker daemon is not reachable. Start Docker first."
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python virtual environment not found at $PYTHON_BIN"
  echo "Create it first: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$QDRANT_CONTAINER"; then
  if [[ -z "$(docker ps --format '{{.Names}}' | grep -x "$QDRANT_CONTAINER" || true)" ]]; then
    echo "==> Starting existing Qdrant container: $QDRANT_CONTAINER"
    docker start "$QDRANT_CONTAINER" >/dev/null
  else
    echo "==> Qdrant container already running: $QDRANT_CONTAINER"
  fi
else
  echo "==> Creating and starting Qdrant container: $QDRANT_CONTAINER"
  docker run -d \
    --name "$QDRANT_CONTAINER" \
    -p 6333:6333 \
    -p 6334:6334 \
    "$QDRANT_IMAGE" >/dev/null
fi

echo "==> Waiting for Qdrant on http://127.0.0.1:6333"
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:6333/collections >/dev/null 2>&1; then
    echo "==> Qdrant is ready"
    break
  fi
  sleep 1
done

if ! curl -fsS http://127.0.0.1:6333/collections >/dev/null 2>&1; then
  echo "ERROR: Qdrant did not become ready in time."
  exit 1
fi

if [[ "$QDRANT_ONLY" == "true" ]]; then
  echo "==> Done. Qdrant is up."
  exit 0
fi

echo "==> Starting API at http://${HOST}:${PORT}"
if [[ -n "$RELOAD_FLAG" ]]; then
  exec "$PYTHON_BIN" -m uvicorn api.main:app --host "$HOST" --port "$PORT" --reload
else
  exec "$PYTHON_BIN" -m uvicorn api.main:app --host "$HOST" --port "$PORT"
fi
