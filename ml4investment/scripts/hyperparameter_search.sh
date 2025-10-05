#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

if [[ -z "${BASH_VERSINFO}" || "${BASH_VERSINFO[0]}" -lt 4 ]]; then
  echo "Error: This script requires Bash version 4.0 or higher." >&2
  echo "You are using version: $(bash --version | head -n 1)" >&2
  echo "On macOS, you can upgrade by running: brew install bash" >&2
  exit 1
fi

echo "Starting generic hyperparameter search run at $(date)"

export LOG_DIR="hyperparameter_search_logs"

# ===================================================================================
## 1. DEFINE HYPERPARAMETERS (MODIFY THIS SECTION)
#    - Key: The name of the environment variable (e.g., TRAIN_OBJECTIVE)
#    - Value: A space-separated string containing all possible values to test.
# ===================================================================================
declare -A hparams=(
    ["TRAIN_START_DATE"]="2024-6-30 2024-5-31 2024-4-30 2024-3-31 2024-2-29 2024-1-31 2023-12-31 2023-11-30 2023-10-31 2023-9-30 2023-8-31 2023-7-31 2023-6-30 2023-5-31 2023-4-30 2023-3-31 2023-2-28 2023-1-31 2022-12-31 2022-11-30 2022-10-31 2022-9-30 2022-8-31 2022-7-31"
)

# ===================================================================================
## 2. CORE EXPERIMENT LOGIC
#    This function is called once for each hyperparameter combination.
# ===================================================================================
run_experiment() {
    # Use a nameref to easily access the current combination associative array.
    declare -n current_combination=$1

    # --- Build a unique result directory name based on the current combination ---
    local dir_parts=()
    for key in "${!current_combination[@]}"; do
        dir_parts+=("${key}=${current_combination[$key]}")
    done
    # This creates a string like "key1=value1,key2=value2" to use in the directory name.
    local combination_str
    combination_str=$(IFS=,; echo "${dir_parts[*]}")
    local RESULT_DIR="results/${combination_str}"

    echo "================================================="
    echo ">> [RUNNING] Combination: ${combination_str}"
    echo ">> Result DIR: ${RESULT_DIR}"
    echo ">> Start time: $(date)"
    echo "================================================="

    # --- Export the key-value pairs of the current combination as environment variables ---
    for key in "${!current_combination[@]}"; do
        export "$key"="${current_combination[$key]}"
        echo "Exporting ${key}=${!key}"
    done

    echo "Reading the number of iterations from Python settings..."
    ITERATIVE_OPTIMIZATION_STEPS=$(python -c "from ml4investment.config.global_settings import settings; print(settings.ITERATIVE_OPTIMIZATION_STEPS)")
    echo "Retrieved number of iterations: ${ITERATIVE_OPTIMIZATION_STEPS}."

    # --- Set up and create directories ---
    local CONFIG_DIR="config"
    local DATA_DIR="data"
    mkdir -p "$RESULT_DIR"

    # --- Set a unique WANDB_RUN_GROUP for this combination ---
    export WANDB_RUN_GROUP="${combination_str}"

    for i in $(seq 1 $ITERATIVE_OPTIMIZATION_STEPS); do
        echo "Running optimization iteration ${i}..."

        echo "Step 1.${i}.1: Optimizing Data Sampling Proportion..."
        export WANDB_MODE=online
        export WANDB_RUN_NAME="optimized-sampling-validation"
        export WANDB_JOB_TYPE="validation"
        python train.py -odsp -v

        echo "Step 1.${i}.2: Running backtest for optimized data sampling proportion..."
        export WANDB_MODE=online
        export WANDB_RUN_NAME="optimized-sampling-backtest"
        export WANDB_JOB_TYPE="backtest"
        python backtest.py -v

        echo "Step 2.${i}.1: Optimizing Model Features..."
        export WANDB_MODE=online
        export WANDB_RUN_NAME="optimized-features-validation"
        export WANDB_JOB_TYPE="validation"
        python train.py -of -v

        echo "Step 2.${i}.2: Running backtest for optimized model features..."
        export WANDB_MODE=online
        export WANDB_RUN_NAME="optimized-features-backtest"
        export WANDB_JOB_TYPE="backtest"
        python backtest.py -v

        echo "Step 3.${i}.1: Optimizing Model Hyperparameters..."
        export WANDB_MODE=online
        export WANDB_RUN_NAME="optimized-hyperparams-validation"
        export WANDB_JOB_TYPE="validation"
        python train.py -omhp -v

        echo "Step 3.${i}.2: Running backtest for optimized model hyperparameters..."
        export WANDB_MODE=online
        export WANDB_RUN_NAME="optimized-hyperparams-backtest"
        export WANDB_JOB_TYPE="backtest"
        python backtest.py -v

        echo "Step 4.${i}.1: Optimizing Prediction Stocks..."
        export WANDB_MODE=online
        export WANDB_RUN_NAME="optimized-stocks-validation"
        export WANDB_JOB_TYPE="validation"
        python train.py -ops -v

        echo "Step 4.${i}.2: Running backtest for optimized model with optimized prediction stocks..."
        export WANDB_MODE=online
        export WANDB_RUN_NAME="optimized-stocks-backtest"
        export WANDB_JOB_TYPE="backtest"
        python backtest.py -v

    done

    echo "Step 5: Archiving configuration files..."
    mv "${DATA_DIR}/prod_data_sampling_proportion.json" "${RESULT_DIR}"
    mv "${DATA_DIR}/prod_features.json" "${RESULT_DIR}"
    mv "${DATA_DIR}/prod_model_hyperparams.json" "${RESULT_DIR}"
    mv "${DATA_DIR}/prod_model.model" "${RESULT_DIR}"
    mv "${DATA_DIR}/prod_process_feature_config.pkl" "${RESULT_DIR}"
    mv "${DATA_DIR}/shap_summary_correct_Validation.png" "${RESULT_DIR}"
    mv "${DATA_DIR}/shap_summary_errors_Validation.png" "${RESULT_DIR}"
    mv "${DATA_DIR}/shap_summary_global_Validation.png" "${RESULT_DIR}"
    mv "${DATA_DIR}/shap_summary_correct_Backtest.png" "${RESULT_DIR}"
    mv "${DATA_DIR}/shap_summary_errors_Backtest.png" "${RESULT_DIR}"
    mv "${DATA_DIR}/shap_summary_global_Backtest.png" "${RESULT_DIR}"
    mv "${CONFIG_DIR}/predict_stocks.json" "${RESULT_DIR}"
    mv "${LOG_DIR}" "${RESULT_DIR}"

    echo ">> [SUCCESS] Finished experiment for combination: ${combination_str} at $(date)"
    echo ""
}

# ===================================================================================
## 3. RECURSIVE FUNCTION TO GENERATE ALL HYPERPARAMETER COMBINATIONS (CARTESIAN PRODUCT)
# ===================================================================================
generate_combinations() {
    local combo_name=$1
    declare -n combo_so_far=$1
    local keys_array=("${@:2}")

    # Base case: If there are no more keys to process, a full combination has been generated.
    if [[ ${#keys_array[@]} -eq 0 ]]; then
        run_experiment combo_so_far
        return
    fi

    # Recursive step:
    local current_key=${keys_array[0]}
    local remaining_keys=("${keys_array[@]:1}")
    local values_str=${hparams[$current_key]}

    # Split the string of values into an array.
    local -a values
    read -r -a values <<< "$values_str"

    for value in "${values[@]}"; do
        combo_so_far["$current_key"]="$value"
        generate_combinations "$combo_name" "${remaining_keys[@]}"
    done
}

# ===================================================================================
## 4. SCRIPT ENTRY POINT
# ===================================================================================
main() {
    # Initialize an empty combination
    declare -A initial_combo
    # Get all the hyperparameter keys
    local param_keys=("${!hparams[@]}")
    
    # Start generating combinations and running experiments
    generate_combinations initial_combo "${param_keys[@]}"

    echo "================================================="
    echo "All hyperparameter experiments completed successfully at $(date)"
    echo "================================================="
}

main
