#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting full experiment run at $(date)"

# Define the sample sizes to iterate over
SAMPLE_SIZES="1000 2000 4000 8000 16000 32000 64000 128000 256000 512000"

for sample_size in $SAMPLE_SIZES; do
    echo "================================================="
    echo ">> Running with target sample size: $sample_size"
    echo ">> Start time: $(date)"
    echo "================================================="

    RESULT_DIR="retrain/${sample_size}"
    DSPP="result/${sample_size}/prod_data_sampling_proportion.json"
    FP="result/${sample_size}/prod_model_features.json"
    MHPP="result/${sample_size}/prod_model_hyperparams.json"
    LOG_DIR="logs"

    # Ensure the result directory for this run exists
    echo "Creating result directory: $RESULT_DIR"
    mkdir -p "$RESULT_DIR"

    echo "Step 1: Optimizing Model Hyperparameters..."
    python train.py -dspp "$DSPP" -fp "$FP" -mhpp "$MHPP" -tss "$sample_size" -v

    echo "Step 2: Running first backtest..."
    python backtest.py -fp "$FP"

    echo "Step 3: Optimizing Prediction Stocks..."
    python train.py -dspp "$DSPP" -fp "$FP" -mhpp "$MHPP" -ops -tss "$sample_size" -v

    echo "Step 4: Running final backtest..."
    python backtest.py -fp "$FP"

    echo "Step 5: Archiving configuration files..."
    mv "${LOG_DIR}" "${RESULT_DIR}"

    echo ">> Finished experiment for sample_size: $sample_size at $(date)"
    echo "" # Add a blank line for readability
done

echo "================================================="
echo "All experiments completed successfully at $(date)"
echo "================================================="