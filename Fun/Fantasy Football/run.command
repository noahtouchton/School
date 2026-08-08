#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

(
    until curl -s http://localhost:8501 > /dev/null; do
        sleep 1
    done
    open "http://localhost:8501"
) &

python3 run.py
