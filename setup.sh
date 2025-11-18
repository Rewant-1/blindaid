#!/usr/bin/env bash
set -euo pipefail

CYAN="\033[0;36m"
YELLOW="\033[0;33m"
GREEN="\033[0;32m"
RED="\033[0;31m"
RESET="\033[0m"

info() { printf "%b%s%b\n" "$CYAN" "$1" "$RESET"; }
warn() { printf "%b%s%b\n" "$YELLOW" "$1" "$RESET"; }
ok() { printf "%b%s%b\n" "$GREEN" "$1" "$RESET"; }
err() { printf "%b%s%b\n" "$RED" "$1" "$RESET"; }

info "BlindAid setup"
printf "====================\n\n"

PYTHON_BIN=${PYTHON:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  err "Python 3.9+ not found. Install Python before running this script."
  exit 1
fi

if [ ! -d ".venv" ]; then
  warn "Creating .venv"
  "$PYTHON_BIN" -m venv .venv
else
  ok "Virtual environment already present (.venv)"
fi

warn "Activating virtual environment"
# shellcheck disable=SC1091
source .venv/bin/activate

warn "Upgrading pip"
python -m pip install --upgrade pip wheel

warn "Installing runtime dependencies"
pip install -r requirements.txt

if [ "${1:-}" == "--dev" ]; then
  warn "Installing developer tooling"
  pip install -r requirements-dev.txt
fi

ok "Dependencies installed."
info "Populate resources/models and resources/known_faces before running: python -m blindaid"
