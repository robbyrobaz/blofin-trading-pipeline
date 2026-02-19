#!/bin/bash
# cron_phase2_check.sh
# Daily Phase 2 trigger check — runs at 06:00 MST (13:00 UTC)
# Checks gates: ≥2 weeks paper trading + ≥75 closed trades + regime diversity
# Runs retrain if all gates pass.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/logs/phase2_retrain.log"
VENV="${SCRIPT_DIR}/.venv/bin/python3"

# Use venv python if available, otherwise system python3
PYTHON="${VENV}"
if [ ! -f "${VENV}" ]; then
    PYTHON="python3"
fi

mkdir -p "${SCRIPT_DIR}/logs"

echo "" >> "${LOG_FILE}"
echo "========================================" >> "${LOG_FILE}"
echo "[cron] Phase 2 trigger check — $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "${LOG_FILE}"

cd "${SCRIPT_DIR}"

# Run the retrain (gates are checked inside; only trains if gates pass)
${PYTHON} ml_retrain_phase2.py >> "${LOG_FILE}" 2>&1
EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[cron] Phase 2 check completed successfully" >> "${LOG_FILE}"
else
    echo "[cron] Phase 2 check exited with code ${EXIT_CODE}" >> "${LOG_FILE}"
fi

echo "========================================" >> "${LOG_FILE}"
