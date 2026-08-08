#!/bin/bash
# Double-click this file in Finder to launch the Fantasy Football AI app!

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "======================================================================"
echo "🏈 Starting Antigravity Fantasy Football AI Platform..."
echo "======================================================================"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Poll until server is active, then open browser automatically
(
    until curl -s http://localhost:8501 > /dev/null; do
        sleep 1
    done
    open "http://localhost:8501"
) &

# Run the platform application (with auto-install for any missing dependencies)
python3 run.py
