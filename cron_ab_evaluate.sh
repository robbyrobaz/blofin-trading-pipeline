#!/bin/bash
# cron_ab_evaluate.sh
# Daily A/B evaluation — runs at 18:00 MST (01:00 UTC next day)
# Evaluates model arms after Phase 2 retrain, decides on model swap.
# Requires ≥100 closed trades per arm before any swap.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/logs/phase2_retrain.log"
VENV="${SCRIPT_DIR}/.venv/bin/python3"

PYTHON="${VENV}"
if [ ! -f "${VENV}" ]; then
    PYTHON="python3"
fi

mkdir -p "${SCRIPT_DIR}/logs"

echo "" >> "${LOG_FILE}"
echo "========================================" >> "${LOG_FILE}"
echo "[cron] A/B Evaluation — $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "${LOG_FILE}"

cd "${SCRIPT_DIR}"

${PYTHON} ab_test_models.py >> "${LOG_FILE}" 2>&1
EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[cron] A/B evaluation completed" >> "${LOG_FILE}"
else
    echo "[cron] A/B evaluation exited with code ${EXIT_CODE}" >> "${LOG_FILE}"
fi

echo "========================================" >> "${LOG_FILE}"
