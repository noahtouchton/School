#!/usr/bin/env bash
# Starts the GridironIQ backend (FastAPI) and frontend (Next.js) together.
# Ctrl+C stops both.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

if [ ! -d "$BACKEND_DIR/.venv" ]; then
  echo "Backend venv not found. Run: cd backend && python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt"
  exit 1
fi

if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "Frontend dependencies not found. Run: cd frontend && npm install"
  exit 1
fi

cleanup() {
  echo ""
  echo "Stopping..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting backend (FastAPI) on http://localhost:8000 ..."
(cd "$BACKEND_DIR" && ./.venv/bin/uvicorn app.main:app --port 8000 --reload) &
BACKEND_PID=$!

echo "Starting frontend (Next.js) on http://localhost:3000 ..."
(cd "$FRONTEND_DIR" && npm run dev -- --port 3000) &
FRONTEND_PID=$!

echo ""
echo "GridironIQ is starting. Frontend: http://localhost:3000  |  Backend: http://localhost:8000"
echo "Press Ctrl+C to stop both."

wait "$BACKEND_PID" "$FRONTEND_PID"
