import json
import logging
import math
from collections import defaultdict
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm import register_logger
from optuna.importance import get_param_importances

from ml4investment.config.global_settings import settings
from ml4investment.utils.data_loader import sample_training_data
from ml4investment.utils.model_predicting import get_detailed_static_result
from ml4investment.utils.utils import OptimalIterationLogger, id_to_stock_code

logger = logging.getLogger(__name__)
register_logger(logger)


def model_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    X_validate_dict: dict,
    y_validate_dict: dict,
    categorical_features: list,
    model_hyperparams: dict,
    target_stock_list: list,
    optimize_predict_stocks: bool = True,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[lgb.Booster, list, float, float, float, float, float, float, float]:
    """Train the final model with optimized parameters"""
    logger.info("Begin model training with optimized parameters")
    logger.info(model_hyperparams)
    metric_logger_cb = OptimalIterationLogger()
    final_model = lgb.train(
        model_hyperparams,
        train_set=lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features),
        valid_sets=[
            lgb.Dataset(X_validate, label=y_validate, categorical_feature=categorical_features)
        ],
        num_boost_round=int(model_hyperparams["num_rounds"]),
        callbacks=[lgb.log_evaluation(period=100), metric_logger_cb],
    )
    final_model.best_iteration = metric_logger_cb.optimal_iteration
    logger.info(
        f"Final model training completed. Optimal iteration on validation set: "
        f"{final_model.best_iteration} with lowest MAE: {metric_logger_cb.optimal_score}"
    )

    (
        valid_mae,
        valid_mse,
        valid_sign_acc,
        valid_precision,
        valid_recall,
        valid_f1,
        valid_gain,
        predict_stock_list,
    ) = validate_model(
        final_model,
        X_validate,
        y_validate,
        X_validate_dict,
        y_validate_dict,
        target_stock_list=target_stock_list,
        optimize_predict_stocks=optimize_predict_stocks,
        verbose=verbose,
    )

    logger.info("Model training completed")

    return (
        final_model,
        predict_stock_list,
        valid_mae,
        valid_mse,
        valid_sign_acc,
        valid_precision,
        valid_recall,
        valid_f1,
        valid_gain,
    )


def validate_model(
    model: lgb.Booster,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    X_validate_dict: dict,
    y_validate_dict: dict,
    target_stock_list: list[str],
    optimize_predict_stocks: bool,
    verbose: bool = False,
) -> tuple[float, float, float, float, float, float, float, list]:
    """Validate the model on the validation dataset"""
    logger.info("Starting model validation on the provided validation set.")

    stock_actual_gains_collect = defaultdict(lambda: {"total_gain": 1.0, "sample_count": 0})

    preds = model.predict(X_validate, num_iteration=model.best_iteration)
    assert isinstance(preds, np.ndarray)

    unique_stock_ids_in_val = X_validate["stock_id"].unique()
    for stock_id_val in unique_stock_ids_in_val:
        stock_code_val = id_to_stock_code(stock_id_val)

        stock_mask = X_validate["stock_id"] == stock_id_val
        stock_preds = preds[stock_mask]
        stock_y_val_numpy = y_validate[stock_mask].to_numpy()

        if stock_code_val in target_stock_list:
            # Only optimize predict stock list from target stock list
            positive_pred_mask = stock_preds > 0
            if np.any(positive_pred_mask):
                factors_to_multiply = 1 + stock_y_val_numpy[positive_pred_mask]
                current_stock_gain_prod = float(np.prod(factors_to_multiply))

                stock_actual_gains_collect[stock_code_val]["total_gain"] *= current_stock_gain_prod
                stock_actual_gains_collect[stock_code_val]["sample_count"] += len(stock_preds)
            else:
                stock_actual_gains_collect[stock_code_val]["sample_count"] += len(stock_preds)

    stock_avg_actual_gain_dict = {
        stock_code: data["total_gain"] ** (1 / data["sample_count"])
        for stock_code, data in stock_actual_gains_collect.items()
    }

    if optimize_predict_stocks:
        logger.info("Begin predict stocks optimization")
        logger.info(
            f"Using average actual gain as the predict stocks optimization metric "
            f"with target number {settings.PREDICT_STOCK_NUMBER}"
        )
        sorted_stock_avg_actual_gain_list = sorted(
            stock_avg_actual_gain_dict.items(), key=lambda item: item[1], reverse=True
        )
        predict_stock_list = [
            stock
            for stock, _ in sorted_stock_avg_actual_gain_list[: settings.PREDICT_STOCK_NUMBER]
        ]
        logger.info(
            f"Selected {len(predict_stock_list)} stocks for prediction: "
            f"{', '.join(predict_stock_list)}"
        )
    else:
        logger.info("No predict stocks optimization. Using all target stocks as predict stocks")
        predict_stock_list = target_stock_list

    (
        valid_mae,
        valid_mse,
        valid_sign_acc,
        valid_precision,
        valid_recall,
        valid_f1,
        valid_gain,
    ) = get_detailed_static_result(
        model=model,
        X_dict=X_validate_dict,
        y_dict=y_validate_dict,
        predict_stock_list=predict_stock_list,
        start_date=settings.VALIDATION_DATA_START_DATE,
        end_date=settings.VALIDATION_DATA_END_DATE,
        name="Validation",
        verbose=verbose,
    )

    logger.info("Model validation completed.")

    return (
        valid_mae,
        valid_mse,
        valid_sign_acc,
        valid_precision,
        valid_recall,
        valid_f1,
        valid_gain,
        predict_stock_list,
    )


def optimize_data_sampling_proportion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    target_stock_list: list[str],
    given_data_sampling_proportion_pth: str,
    categorical_features: list,
    model_hyperparams: dict,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Optimize data sampling proportion for training"""

    def objective(trial: optuna.Trial) -> float:
        stock_proportion_dict = {}
        for stock in target_stock_list:
            stock_proportion_dict[stock] = trial.suggest_float(f"{stock}", 0.0, 1.0)

        cur_X_train, cur_y_train = sample_training_data(
            X_train,
            y_train,
            sampling_proportion=stock_proportion_dict,
            seed=seed,
        )

        cur_metric_logger_cb = OptimalIterationLogger()
        cur_model = lgb.train(
            model_hyperparams,
            train_set=lgb.Dataset(
                cur_X_train, label=cur_y_train, categorical_feature=categorical_features
            ),
            valid_sets=[
                lgb.Dataset(
                    X_validate,
                    label=y_validate,
                    categorical_feature=categorical_features,
                )
            ],
            num_boost_round=int(model_hyperparams["num_rounds"]),
            callbacks=[lgb.log_evaluation(False), cur_metric_logger_cb],
        )
        cur_model.best_iteration = cur_metric_logger_cb.optimal_iteration
        cur_valid_mae = cur_metric_logger_cb.optimal_score
        logger.info(
            f"Current trial model training completed. "
            f"Optimal iteration on validation set: {cur_model.best_iteration} "
            f"with lowest MAE: {cur_valid_mae}"
        )

        return cur_valid_mae

    logger.info("Optimizing data sampling proportion...")
    study = optuna.create_study(
        study_name="Data Sampling Proportion Optimization",
        directions=["minimize"],
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
    )

    logger.info("Enqueuing trial with all data sampled as a baseline.")
    uniform_params = {stock: 1.0 for stock in target_stock_list}
    study.enqueue_trial(uniform_params)

    if given_data_sampling_proportion_pth and Path(given_data_sampling_proportion_pth).exists():
        logger.info("Enqueuing trial with given sampling proportion as another baseline.")
        given_data_sampling_proportion = json.load(open(given_data_sampling_proportion_pth, "r"))
        study.enqueue_trial(given_data_sampling_proportion)

    study.optimize(
        objective,
        n_trials=settings.DATA_SAMPLING_PROPORTION_SEARCH_LIMIT,
        timeout=604800,
    )

    optimal_trial = study.best_trial
    optimal_data_sampling_proportion = optimal_trial.params.copy()
    optimal_valid_mae = optimal_trial.value

    logger.info(f"Selected Optimal Trial Number: {optimal_trial.number}")
    logger.info(f"  Optimal Trial Value (Valid MAE): {optimal_valid_mae:.4f}")
    if verbose:
        logger.info(
            f"  Optimal Trial Data Sampling Proportion: {optimal_data_sampling_proportion}"
        )

    if verbose:
        logger.info("Optimal data sampling proportion:")
        for stock, proportion in optimal_data_sampling_proportion.items():
            logger.info(f"  {stock}: {proportion * 100:.2f}%")

    logger.info("Data sampling proportion optimization completed")

    return optimal_data_sampling_proportion


def optimize_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    all_features: list[str],
    categorical_features: list[str],
    given_features_pth: str,
    model_hyperparams: dict,
    seed: int,
    verbose: bool = False,
) -> list[str]:
    """Optimize model features using Optuna to select a subset."""
    numerical_features = [f for f in all_features if f not in categorical_features]

    def objective(trial: optuna.Trial) -> float:
        candidate_features = categorical_features.copy()
        for feature in numerical_features:
            if trial.suggest_float(feature, 0.0, 1.0) >= 0.5:
                candidate_features.append(feature)

        if not candidate_features:
            logger.warning("Trial selected zero features. Pruning this trial.")
            return float("inf")

        cur_X_train = X_train[candidate_features]
        cur_X_validate = X_validate[candidate_features]

        cur_metric_logger_cb = OptimalIterationLogger()
        cur_model = lgb.train(
            model_hyperparams,
            train_set=lgb.Dataset(
                cur_X_train, label=y_train, categorical_feature=categorical_features
            ),
            valid_sets=[
                lgb.Dataset(
                    cur_X_validate,
                    label=y_validate,
                    categorical_feature=categorical_features,
                )
            ],
            num_boost_round=model_hyperparams["num_rounds"],
            callbacks=[lgb.log_evaluation(False), cur_metric_logger_cb],
        )
        cur_model.best_iteration = cur_metric_logger_cb.optimal_iteration
        cur_valid_mae = cur_metric_logger_cb.optimal_score

        if verbose:
            logger.info(
                f"Trial {trial.number} with {len(candidate_features)} features "
                f"resulted in MAE: {cur_valid_mae:.6f}"
            )

        return cur_valid_mae

    study = optuna.create_study(
        study_name="Model Feature Selection Optimization",
        directions=["minimize"],
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
    )

    logger.info("Enqueuing a baseline trial with all features included.")
    baseline_features = {f"{f}": 0.5 for f in numerical_features}
    study.enqueue_trial(baseline_features)

    if given_features_pth and Path(given_features_pth).exists():
        logger.info("Enqueuing trial with given features as another baseline.")
        given_model_features = json.load(open(given_features_pth, "r"))["features"]
        baseline_features = {}
        for feature in numerical_features:
            if feature in given_model_features:
                baseline_features[feature] = 1.0
            else:
                baseline_features[feature] = 0.0
        study.enqueue_trial(baseline_features)

    # Run the optimization
    study.optimize(
        objective,
        n_trials=settings.FEATURE_SEARCH_LIMIT,
        timeout=604800,
    )

    best_params = study.best_trial.params
    optimal_features = categorical_features.copy()
    for feature in numerical_features:
        if best_params.get(feature, 0.0) >= 0.5:
            optimal_features.append(feature)

    optimal_valid_mae = study.best_trial.value
    original_feature_number = len(all_features)

    logger.info(
        f"Final selected {len(optimal_features)} features after Optuna search, "
        f"select ratio: {len(optimal_features) / original_feature_number:.2f}"
    )
    logger.info(f"Final Valid MAE after feature selection: {optimal_valid_mae:.6f}")
    if verbose:
        logger.info(f"Optimal features: {optimal_features}")

    logger.info("Feature optimization completed")

    return optimal_features


def optimize_model_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    categorical_features: list,
    given_model_hyperparams_pth: str,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]:
    """Optimize model hyperparameters using Optuna"""
    cur_train_fixed_config = settings.FIXED_TRAINING_CONFIG.copy()
    cur_train_fixed_config.update({"seed": seed})
    cur_train_fixed_config.update(
        {"num_threads": min(max(1, cpu_count()), settings.MAX_NUM_PROCESSES)}
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "drop_rate": trial.suggest_float("drop_rate", 0.05, 0.2),
            "skip_drop": trial.suggest_float("skip_drop", 0.3, 0.7),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 1.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 1.0, log=True),
        }
        params.update(cur_train_fixed_config)

        cur_metric_logger_cb = OptimalIterationLogger()
        cur_model = lgb.train(
            params,
            train_set=lgb.Dataset(
                X_train, label=y_train, categorical_feature=categorical_features
            ),
            valid_sets=[
                lgb.Dataset(
                    X_validate,
                    label=y_validate,
                    categorical_feature=categorical_features,
                )
            ],
            num_boost_round=params["num_rounds"],
            callbacks=[lgb.log_evaluation(False), cur_metric_logger_cb],
        )
        cur_model.best_iteration = cur_metric_logger_cb.optimal_iteration
        cur_valid_mae = cur_metric_logger_cb.optimal_score
        logger.info(
            f"Current trial model training completed. "
            f"Optimal iteration on validation set: {cur_model.best_iteration} "
            f"with lowest MAE: {cur_valid_mae}"
        )

        return cur_valid_mae

    logger.info("Begin hyperparameter optimization")
    study = optuna.create_study(
        study_name="Model Hyperparameter Optimization",
        directions=["minimize"],
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
    )

    logger.info("Enqueuing trial with default hyperparameter as a baseline.")
    default_params = {
        "drop_rate": 0.1,
        "skip_drop": 0.5,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "min_data_in_leaf": 20,
        "lambda_l1": 1e-8,
        "lambda_l2": 1e-8,
    }
    default_params.update(cur_train_fixed_config)
    study.enqueue_trial(default_params)

    if given_model_hyperparams_pth and Path(given_model_hyperparams_pth).exists():
        logger.info("Enqueuing trial with given hyperparameter as another baseline.")
        given_model_hyperparams = json.load(open(given_model_hyperparams_pth, "r"))
        study.enqueue_trial(given_model_hyperparams)

    study.optimize(objective, n_trials=settings.HYPERPARAMETER_SEARCH_LIMIT, timeout=604800)

    if verbose:
        param_importance = get_param_importances(study)
        logger.info("Parameter Importances:")
        for param, importance in param_importance.items():
            logger.info(f"  - {param}: {importance:.6f}")

        completed_trials = sorted(
            study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]),
            key=lambda t: t.value if t.value is not None else math.inf,
        )

        logger.info("Top 5 Best Trials Summary:")
        for i, trial in enumerate(completed_trials[:5]):
            logger.info(f"  Trial {trial.number} (Rank {i + 1}):")
            logger.info(f"    - Value (MAE): {trial.value:.6f}")
            logger.info("    - Params:")
            for param_name, param_value in trial.params.items():
                if isinstance(param_value, float):
                    logger.info(f"      - {param_name}: {param_value:.6f}")
                else:
                    logger.info(f"      - {param_name}: {param_value}")

    optimal_trial = study.best_trial
    optimal_params = optimal_trial.params.copy()
    optimal_valid_mae = optimal_trial.value
    optimal_params.update(cur_train_fixed_config)

    logger.info(f"Selected Optimal Trial Number: {optimal_trial.number}")
    logger.info(f"  Optimal Trial Value (Valid MAE): {optimal_valid_mae:.4f}")
    if verbose:
        logger.info(f"  Optimal Trial Parameters: {optimal_params}")

    logger.info("Hyperparameter optimization completed")

    return optimal_params
