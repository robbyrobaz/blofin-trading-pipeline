#!/bin/bash
# Run Blofin pipeline every hour with monitoring
# Usage: ./run_pipeline_hourly.sh

set -e

WORK_DIR="/home/rob/.openclaw/workspace/blofin-stack"
VENV="$WORK_DIR/.venv"
LOG_DIR="$WORK_DIR/data"
PIPELINE_LOG="$LOG_DIR/pipeline_hourly.log"
SUCCESS_LOG="$LOG_DIR/pipeline_hourly_success.log"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

cd "$WORK_DIR"

echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting hourly pipeline run" | tee -a "$PIPELINE_LOG"

# Run pipeline
{
    source "$VENV/bin/activate"
    python3 orchestration/daily_runner.py
} >> "$PIPELINE_LOG" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Pipeline completed successfully at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - SUCCESS" >> "$SUCCESS_LOG"
    
    # Check if ML models were trained
    MODELS_TRAINED=$(tail -50 "$PIPELINE_LOG" | grep -o '"models_trained": [0-9]*' | tail -1 | grep -o '[0-9]*' || echo "0")
    STRATEGIES_DESIGNED=$(tail -50 "$PIPELINE_LOG" | grep -o '"strategies_designed": [0-9]*' | tail -1 | grep -o '[0-9]*' || echo "0")
    
    echo -e "${GREEN}  Models trained: $MODELS_TRAINED${NC}"
    echo -e "${GREEN}  Strategies designed: $STRATEGIES_DESIGNED${NC}"
    
    if [ "$MODELS_TRAINED" -gt 0 ]; then
        echo -e "${GREEN}✓ ML TRAINING WORKING!${NC}"
    else
        echo -e "${YELLOW}⚠ ML training still returning 0 models${NC}"
    fi
else
    echo -e "${RED}✗ Pipeline failed with exit code $EXIT_CODE${NC}"
    echo "Check logs: $PIPELINE_LOG"
fi

exit $EXIT_CODE
