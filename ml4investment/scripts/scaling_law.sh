#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting scaling law experiment run at $(date)"

export WANDB_RUN_GROUP="scaling-law-$(date +%Y%m%d-%H%M%S)"
echo "WANDB_RUN_GROUP set to: ${WANDB_RUN_GROUP}"

echo "Reading the number of iterations from Python settings..."
ITERATIVE_OPTIMIZATION_STEPS=$(python -c "from config import settings; print(settings.ITERATIVE_OPTIMIZATION_STEPS)")
echo "Retrieved number of iterations: ${ITERATIVE_OPTIMIZATION_STEPS}."

YEAR_MONTHS="2023-11 2023-5 2022-11 2022-5 2021-11 2021-5"

for ym_pair in $YEAR_MONTHS; do

    # Split the pair into year and month
    start_year=$(echo "$ym_pair" | cut -d'-' -f1)
    start_month=$(echo "$ym_pair" | cut -d'-' -f2)

    echo "================================================="
    echo ">> Running with training data start year: $start_year, month: $start_month"
    echo ">> Start time: $(date)"
    echo "================================================="

    export TRAIN_START_DATE="${start_year}-${start_month}-30"
    echo "Dynamically set TRAINING_START_DATE to: ${TRAIN_START_DATE}"

    RESULT_DIR="results/TRAIN_START_DATE=${start_year}-${start_month}-30"
    CONFIG_DIR="config"
    DATA_DIR="data"
    LOG_DIR="logs"

    # Ensure the result directory for this run exists
    echo "Creating result directory: $RESULT_DIR"
    mkdir -p "$RESULT_DIR"
    
    for i in $(seq 1 $ITERATIVE_OPTIMIZATION_STEPS); do
        echo "Running optimization iteration ${i}..."

        echo "Step 1.${i}: Optimizing Data Sampling Proportion..."
        export WANDB_MODE=disabled
        python train.py -odsp -v

        echo "Step 2.${i}: Optimizing Model Features..."
        export WANDB_MODE=disabled
        python train.py -of -v

        echo "Step 3.${i}: Optimizing Model Hyperparameters..."
        export WANDB_MODE=disabled
        python train.py -omhp -v
    done

    echo "Step 4: Running backtest for optimized model with unoptimized prediction stocks..."
    export WANDB_MODE=online
    export WANDB_RUN_NAME="start-ym-${start_year}-${start_month}-model-optimized-predict-stocks-unoptimized-backtest"
    export WANDB_JOB_TYPE="backtest"
    python backtest.py -v

    echo "Step 5: Optimizing Prediction Stocks..."
    export WANDB_MODE=online
    export WANDB_RUN_NAME="start-ym-${start_year}-${start_month}-model-optimized-predict-stocks-optimized-validation"
    export WANDB_JOB_TYPE="validation"
    python train.py -ops -v

    echo "Step 6: Running backtest for optimized model with optimized prediction stocks..."
    export WANDB_MODE=online
    export WANDB_RUN_NAME="start-ym-${start_year}-${start_month}-model-optimized-predict-stocks-optimized-backtest"
    export WANDB_JOB_TYPE="backtest"
    python backtest.py -v

    echo "Step 7: Archiving configuration files..."
    mv "${DATA_DIR}/prod_data_sampling_proportion.json" "${RESULT_DIR}"
    mv "${DATA_DIR}/prod_features.json" "${RESULT_DIR}"
    mv "${DATA_DIR}/prod_model_hyperparams.json" "${RESULT_DIR}"
    mv "${DATA_DIR}/prod_model.model" "${RESULT_DIR}"
    mv "${DATA_DIR}/prod_process_feature_config.pkl" "${RESULT_DIR}"
    mv "${CONFIG_DIR}/predict_stocks.json" "${RESULT_DIR}"
    mv "${LOG_DIR}" "${RESULT_DIR}"

    echo ">> Finished experiment for start date: ${start_year}-${start_month} at $(date)"
    echo "" # Add a blank line for readability
done

echo "================================================="
echo "All experiments completed successfully at $(date)"
echo "================================================="
