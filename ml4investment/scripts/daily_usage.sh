#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

current_date=$(date +%Y-%m-%d)

echo "Starting daily usage run at $(date)"

export WANDB_RUN_GROUP="daily-usage"
echo "WANDB_RUN_GROUP set to: ${WANDB_RUN_GROUP}"

echo "Step 1: Fetching Data from last trading day..."
export WANDB_MODE=disabled
# python fetch_data.py

echo "Step 2: Backtesting..."
export WANDB_MODE=online
export WANDB_RUN_NAME="date-${current_date}-backtest"
export WANDB_JOB_TYPE="backtest"
python backtest.py -v

echo "Step 3: Predicting..."
export WANDB_MODE=online
export WANDB_RUN_NAME="date-${current_date}-predict"
export WANDB_JOB_TYPE="predict"
python predict.py -pt -v

echo "" # Add a blank line for readability

echo "================================================="
echo "All runs completed successfully at $(date)"
echo "================================================="
