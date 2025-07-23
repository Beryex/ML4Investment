#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting full experiment run at $(date)"

# Define the sample sizes to iterate over
SAMPLE_SIZES="500 1000 2000 4000 8000 16000 32000 64000 128000 256000 512000"

for sample_size in $SAMPLE_SIZES; do
    echo "================================================="
    echo ">> Running with target sample size: $sample_size"
    echo ">> Start time: $(date)"
    echo "================================================="

    RESULT_DIR="result/${sample_size}"
    DATA_DIR="data"
    LOG_DIR="logs"

    # Ensure the result directory for this run exists
    echo "Creating result directory: $RESULT_DIR"
    mkdir -p "$RESULT_DIR"

    echo "Step 1: Optimizing Data Sampling Proportion..."
    python train.py -odsp -tss "$sample_size" -v

    echo "Step 2: Optimizing Model Features..."
    python train.py -omf -tss "$sample_size" -v

    echo "Step 3: Optimizing Model Hyperparameters..."
    python train.py -omhp -tss "$sample_size" -v

    echo "Step 4: Running first backtest..."
    python backtest.py

    echo "Step 5: Optimizing Prediction Stocks..."
    python train.py -ops -tss "$sample_size" -v

    echo "Step 6: Running final backtest..."
    python backtest.py

    echo "Step 7: Archiving configuration files..."
    cp "${DATA_DIR}/prod_data_sampling_proportion.json" "${RESULT_DIR}/prod_data_sampling_proportion.json"
    cp "${DATA_DIR}/prod_model_features.json" "${RESULT_DIR}/prod_model_features.json"
    cp "${DATA_DIR}/prod_model_hyperparams.json" "${RESULT_DIR}/prod_model_hyperparams.json"
    mv "${LOG_DIR}" "${RESULT_DIR}"

    echo ">> Finished experiment for sample_size: $sample_size at $(date)"
    echo "" # Add a blank line for readability
done

echo "================================================="
echo "All experiments completed successfully at $(date)"
echo "================================================="
