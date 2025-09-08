#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting hyperparameter search experiment run at $(date)"

# Define the metrices to optimize
METRIC="mae mse"

for metric in $METRIC; do
    export WANDB_RUN_GROUP="metric-${metric}-$(date +%Y%m%d-%H%M%S)"
    echo "WANDB_RUN_GROUP set to: ${WANDB_RUN_GROUP}"

    export OPTIMIZE_METRIC=$metric
    echo "OPTIMIZE_METRIC set to: ${OPTIMIZE_METRIC}"

    # Define the sample sizes to iterate over
    START_YEARS="2023 2022 2021"

    for start_year in $START_YEARS; do
    
        START_MONTHS="5 11"

        for start_month in $START_MONTHS; do

            echo "================================================="
            echo ">> Running with training data start year: $start_year"
            echo ">> Start time: $(date)"
            echo "================================================="

            export TRAIN_START_DATE="${start_year}-${start_month}-30"
            echo "Dynamically set TRAINING_DATA_START_DATE to: ${TRAIN_START_DATE}"

            RESULT_DIR="metric-${metric}/${start_year}-${start_month}"
            CONFIG_DIR="config"
            DATA_DIR="data"
            LOG_DIR="logs"

            # Ensure the result directory for this run exists
            echo "Creating result directory: $RESULT_DIR"
            mkdir -p "$RESULT_DIR"

            echo "Step 1: Optimizing Data Sampling Proportion..."
            export WANDB_MODE=disabled
            python train.py -odsp -v

            echo "Step 2: Optimizing Model Features..."
            export WANDB_MODE=disabled
            python train.py -of -v

            echo "Step 3: Optimizing Model Hyperparameters..."
            export WANDB_MODE=disabled
            python train.py -omhp -v

            echo "Step 4: Running backtest for optimized model with unoptimized prediction stocks..."
            export WANDB_MODE=online
            export WANDB_RUN_NAME="start-YM-${start_year}-${start_month}-model-optimized-predict-stocks-unoptimized-backtest"
            export WANDB_JOB_TYPE="backtest"
            python backtest.py -v

            echo "Step 5: Optimizing Prediction Stocks..."
            export WANDB_MODE=online
            export WANDB_RUN_NAME="start-YM-${start_year}-${start_month}-model-optimized-predict-stocks-optimized-validation"
            export WANDB_JOB_TYPE="validation"
            python train.py -ops -v

            echo "Step 6: Running backtest for optimized model with optimized prediction stocks..."
            export WANDB_MODE=online
            export WANDB_RUN_NAME="start-YM-${start_year}-${start_month}-model-optimized-predict-stocks-optimized-backtest"
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
    done
done

echo "================================================="
echo "All experiments completed successfully at $(date)"
echo "================================================="
