#!/bin/bash
# Hourly pipeline runner with full debugging and monitoring

set -e

WORK_DIR="/home/rob/.openclaw/workspace/blofin-stack"
VENV="$WORK_DIR/.venv"
LOG_DIR="$WORK_DIR/data"
HOUR=$(date +%H)
RUN_FILE="$LOG_DIR/hourly_run_$HOUR.log"

# Activate venv
source "$VENV/bin/activate"

# Run pipeline with full output
echo "========================================" | tee -a "$RUN_FILE"
echo "BLOFIN PIPELINE RUN - $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$RUN_FILE"
echo "========================================" | tee -a "$RUN_FILE"

# Run pipeline
cd "$WORK_DIR"
python3 orchestration/daily_runner.py >> "$RUN_FILE" 2>&1

EXIT_CODE=$?

# Extract key metrics
echo "" | tee -a "$RUN_FILE"
echo "========================================" | tee -a "$RUN_FILE"
echo "RESULTS" | tee -a "$RUN_FILE"
echo "========================================" | tee -a "$RUN_FILE"

MODELS=$(grep -o '"models_trained": [0-9]*' "$RUN_FILE" | tail -1 | grep -o '[0-9]*' || echo "?")
STRATEGIES=$(grep -o '"strategies_designed": [0-9]*' "$RUN_FILE" | tail -1 | grep -o '[0-9]*' || echo "?")
DURATION=$(grep -o '"duration_seconds": [0-9.]*' "$RUN_FILE" | tail -1 | grep -o '[0-9.]*' || echo "?")

echo "Models trained: $MODELS ✓" | tee -a "$RUN_FILE"
echo "Strategies designed: $STRATEGIES ✓" | tee -a "$RUN_FILE"
echo "Duration: ${DURATION}s" | tee -a "$RUN_FILE"

# Check if it worked
if [ "$MODELS" != "0" ] && [ "$MODELS" != "?" ]; then
    echo "" | tee -a "$RUN_FILE"
    echo "✅ ML PIPELINE WORKING! ($MODELS models trained)" | tee -a "$RUN_FILE"
else
    echo "" | tee -a "$RUN_FILE"
    echo "⚠️  Models still at 0, continuing debug..." | tee -a "$RUN_FILE"
fi

echo "Log: $RUN_FILE" | tee -a "$RUN_FILE"
echo "Exit code: $EXIT_CODE" | tee -a "$RUN_FILE"

exit $EXIT_CODE
