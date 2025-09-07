#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting hyperparameter search experiment run at $(date)"

# Define the metrices to optimize
METHOD="skip stock global"

for method in $METHOD; do
    export WANDB_RUN_GROUP="method-${method}-$(date +%Y%m%d-%H%M%S)"
    echo "WANDB_RUN_GROUP set to: ${WANDB_RUN_GROUP}"

    export APPLY_CLIP=$method
    export APPLY_SCALE=$method
    echo "APPLY_CLIP set to: ${APPLY_CLIP}"
    echo "APPLY_SCALE set to: ${APPLY_SCALE}"

    # Define the sample sizes to iterate over
    START_YEARS="2023 2022 2021 2020 2019"

    for start_year in $START_YEARS; do
        echo "================================================="
        echo ">> Running with training data start year: $start_year"
        echo ">> Start time: $(date)"
        echo "================================================="

        export TRAIN_START_DATE="${start_year}-11-30"
        echo ">> Dynamically set TRAINING_DATA_START_DATE to: ${TRAIN_START_DATE}"

        RESULT_DIR="result/${start_year}"
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
        export WANDB_RUN_NAME="start-year-${start_year}-model-optimized-predict-stocks-unoptimized-backtest"
        export WANDB_JOB_TYPE="backtest"
        python backtest.py -v

        echo "Step 5: Optimizing Prediction Stocks..."
        export WANDB_MODE=online
        export WANDB_RUN_NAME="start-year-${start_year}-model-optimized-predict-stocks-optimized-validation"
        export WANDB_JOB_TYPE="validation"
        python train.py -ops -v

        echo "Step 6: Running backtest for optimized model with optimized prediction stocks..."
        export WANDB_MODE=online
        export WANDB_RUN_NAME="start-year-${start_year}-model-optimized-predict-stocks-optimized-backtest"
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

        echo ">> Finished experiment for sample_size: $sample_size at $(date)"
        echo "" # Add a blank line for readability
    done
done

echo "================================================="
echo "All experiments completed successfully at $(date)"
echo "================================================="
