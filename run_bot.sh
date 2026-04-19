#!/usr/bin/env bash
# run_bot.sh — Linux/Mac startup script for Zerodha Trading Bot
#
# Usage:
#   ./run_bot.sh                        (paper trading, NIFTY)
#   ./run_bot.sh --symbol BANKNIFTY
#   ./run_bot.sh --paper --no-dashboard
#   ./run_bot.sh --help
#
# First run: set your credentials in .env before starting.

set -euo pipefail

# ── locate project root ───────────────────────────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── locate Python / venv ──────────────────────────────────────────────────
if [ -f "$ROOT/venv/bin/python" ]; then
    PYTHON="$ROOT/venv/bin/python"
elif [ -f "$ROOT/.venv/bin/python" ]; then
    PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "[ERROR] Python not found. Install Python 3.11+ or activate your venv first."
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# ── check .env exists ─────────────────────────────────────────────────────
if [ ! -f "$ROOT/.env" ]; then
    echo "[WARNING] .env file not found."
    if [ -f "$ROOT/.env.example" ]; then
        echo "  Copy .env.example to .env and fill in your credentials:"
        echo "    cp .env.example .env"
    fi
    echo "  Continuing with default dummy values..."
    echo
fi

# ── pre-flight dependency check ───────────────────────────────────────────
echo "Running dependency check..."
if ! "$PYTHON" scripts/check_requirements.py; then
    echo "[ERROR] Dependency check failed. Run: pip install -r requirements.txt"
    exit 1
fi

# ── start bot ─────────────────────────────────────────────────────────────
echo
echo "Starting trading bot (args: $*)..."
echo

exec "$PYTHON" main.py "$@"
