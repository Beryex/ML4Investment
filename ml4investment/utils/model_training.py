import json
import logging
import math
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any
from collections.abc import Callable

import lightgbm as lgb
import optuna
import pandas as pd
from lightgbm import register_logger
from optuna.importance import get_param_importances
from optuna.integration import LightGBMPruningCallback  # type: ignore

from ml4investment.config.global_settings import settings
from ml4investment.utils.data_loader import sample_training_data
from ml4investment.utils.model_backtesting import get_detailed_static_result
from ml4investment.utils.utils import OptimalIterationCallback, OptimalIterationLogger

logger = logging.getLogger(__name__)
register_logger(logger)


def _build_training_callbacks(
    show_training_log: bool, callbacks: list
) -> tuple[list, OptimalIterationCallback]:
    """Build LightGBM training callbacks and metric logger.

    Args:
        show_training_log: Whether to log training evaluation.
        callbacks: Additional callbacks.

    Returns:
        Tuple of (callbacks list, metric logger callback).
    """
    metric_logger_cb = OptimalIterationLogger()
    training_callbacks = [
        lgb.log_evaluation(period=100 if show_training_log else -1),
        metric_logger_cb,
    ]
    if callbacks:
        training_callbacks.extend(callbacks)
    return training_callbacks, metric_logger_cb


def _train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    categorical_features: list,
    model_hyperparams: dict,
    training_callbacks: list,
) -> lgb.Booster:
    """Train a LightGBM model.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_validate: Validation features.
        y_validate: Validation targets.
        categorical_features: Categorical feature names.
        model_hyperparams: Hyperparameter dictionary.
        training_callbacks: LightGBM callbacks.

    Returns:
        Trained LightGBM booster.
    """
    return lgb.train(
        model_hyperparams,
        train_set=lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features),
        valid_sets=[
            lgb.Dataset(X_validate, label=y_validate, categorical_feature=categorical_features)
        ],
        num_boost_round=int(model_hyperparams["num_rounds"]),
        callbacks=training_callbacks,
    )


def _log_optimal_iteration(
    model: lgb.Booster,
    metric_logger_cb: OptimalIterationCallback,
) -> float:
    """Log optimal iteration results.

    Args:
        model: Trained model.
        metric_logger_cb: Callback storing optimal iteration.

    Returns:
        Optimal score.
    """
    model.best_iteration = metric_logger_cb.optimal_iteration
    optimal_score = metric_logger_cb.optimal_score
    logger.info(
        "Optimal iteration on validation set: %s with %s: %s",
        model.best_iteration,
        settings.OPTIMIZE_METRIC,
        optimal_score,
    )
    return optimal_score


def model_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    categorical_features: list,
    model_hyperparams: dict,
    show_training_log: bool = False,
    callbacks: list = [],
) -> tuple[lgb.Booster, float]:
    """Train the model and get optimal score on validation set."""
    logger.info("Begin model training...")

    training_callbacks, metric_logger_cb = _build_training_callbacks(show_training_log, callbacks)

    model = _train_model(
        X_train,
        y_train,
        X_validate,
        y_validate,
        categorical_features,
        model_hyperparams,
        training_callbacks,
    )

    optimal_score = _log_optimal_iteration(model, metric_logger_cb)

    logger.info("Model training completed")

    return model, optimal_score


def validate_model(
    model: lgb.Booster,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    target_stock_list: list[str],
    verbose: bool = False,
) -> tuple[int, float, float, float, float, float, float, float, float, float, float, list]:
    """Validate the model on the validation dataset."""
    logger.info("Starting model validation on the provided validation set.")

    metrics = get_detailed_static_result(
        model=model,
        X=X_validate,
        y=y_validate,
        predict_stock_list=target_stock_list,
        name="Validation",
        verbose=verbose,
    )

    logger.info("Model validation completed.")

    return metrics


def _build_data_sampling_study(seed: int) -> optuna.Study:
    """Build Optuna study for data sampling proportion.

    Args:
        seed: Random seed.

    Returns:
        Optuna study instance.
    """
    return optuna.create_study(
        study_name="Data Sampling Proportion Optimization",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=seed, multivariate=settings.DATA_OPTIMIZATION_SAMPLING_MULTIVARIATE
        ),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=settings.PRUNING_WARMUP_STEPS),
    )


def _enqueue_sampling_baseline(
    study: optuna.Study,
    train_stock_list: list[str],
    given_data_sampling_proportion_pth: str,
) -> dict[str, float]:
    """Enqueue baseline data sampling proportion.

    Args:
        study: Optuna study.
        train_stock_list: Stock codes used for training.
        given_data_sampling_proportion_pth: Path to baseline proportions.

    Returns:
        Baseline sampling proportion.
    """
    if given_data_sampling_proportion_pth and Path(given_data_sampling_proportion_pth).exists():
        logger.info("Enqueuing trial with given sampling proportion as baseline.")
        baseline_dsp = json.load(open(given_data_sampling_proportion_pth, "r"))
        study.enqueue_trial(baseline_dsp)
        return baseline_dsp

    logger.info("Enqueuing trial with all data sampled as baseline.")
    uniform_params = {stock: 1.0 for stock in train_stock_list}
    study.enqueue_trial(uniform_params)
    return {}


def _build_sampling_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    train_stock_list: list[str],
    categorical_features: list,
    model_hyperparams: dict,
    seed: int,
) -> Callable:
    """Build Optuna objective for sampling proportion.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_validate: Validation features.
        y_validate: Validation targets.
        train_stock_list: Stock codes used for training.
        categorical_features: Categorical feature names.
        model_hyperparams: Hyperparameter dictionary.
        seed: Random seed.

    Returns:
        Objective callable for Optuna.
    """

    def objective(trial: optuna.Trial) -> float:
        stock_proportion_dict = {}
        for stock in train_stock_list:
            if stock in settings.SELECTIVE_ETF:
                continue
            stock_proportion_dict[stock] = trial.suggest_float(f"{stock}", 0.0, 1.0)

        cur_X_train, cur_y_train = sample_training_data(
            X_train,
            y_train,
            sampling_proportion=stock_proportion_dict,
            seed=seed,
        )

        _, cur_valid_score = model_training(
            cur_X_train,
            cur_y_train,
            X_validate,
            y_validate,
            categorical_features,
            model_hyperparams,
            callbacks=[LightGBMPruningCallback(trial, metric=settings.OPTIMIZE_METRIC)],
        )

        return cur_valid_score

    return objective


def optimize_data_sampling_proportion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    train_stock_list: list[str],
    given_data_sampling_proportion_pth: str,
    categorical_features: list,
    model_hyperparams: dict,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Optimize data sampling proportion for training."""
    logger.info("Optimizing data sampling proportion...")
    study = _build_data_sampling_study(seed)

    _enqueue_sampling_baseline(study, train_stock_list, given_data_sampling_proportion_pth)

    objective = _build_sampling_objective(
        X_train,
        y_train,
        X_validate,
        y_validate,
        train_stock_list,
        categorical_features,
        model_hyperparams,
        seed,
    )

    study.optimize(
        objective,
        n_trials=settings.DATA_SAMPLING_PROPORTION_SEARCH_LIMIT,
        timeout=604800,
    )

    optimal_trial = study.best_trial
    optimal_data_sampling_proportion = optimal_trial.params.copy()
    optimal_value = optimal_trial.value

    logger.info("Selected Optimal Trial Number: %s", optimal_trial.number)
    logger.info("  Optimal Trial Value (Valid %s): %.4f", settings.OPTIMIZE_METRIC, optimal_value)
    if verbose:
        logger.info(
            "  Optimal Trial Data Sampling Proportion: %s", optimal_data_sampling_proportion
        )

    if verbose:
        logger.info("Optimal data sampling proportion:")
        for stock, proportion in optimal_data_sampling_proportion.items():
            logger.info("  %s: %.2f%%", stock, proportion * 100)

    logger.info("Data sampling proportion optimization completed")

    return optimal_data_sampling_proportion


def _build_feature_study(seed: int) -> optuna.Study:
    """Build Optuna study for feature optimization.

    Args:
        seed: Random seed.

    Returns:
        Optuna study instance.
    """
    return optuna.create_study(
        study_name="Model Feature Selection Optimization",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=seed, multivariate=settings.FEATURE_OPTIMIZATION_SAMPLING_MULTIVARIATE
        ),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=settings.PRUNING_WARMUP_STEPS),
    )


def _enqueue_feature_baseline(
    study: optuna.Study, numerical_features: list[str], given_features_pth: str
) -> None:
    """Enqueue baseline feature selection parameters.

    Args:
        study: Optuna study.
        numerical_features: Numerical feature names.
        given_features_pth: Path to baseline features.
    """
    if given_features_pth and Path(given_features_pth).exists():
        logger.info("Enqueuing trial with given features as baseline.")
        given_model_features = json.load(open(given_features_pth, "r"))["features"]
        baseline_features = {}
        for feature in numerical_features:
            baseline_features[feature] = 0.5 if feature in given_model_features else 0.0
        study.enqueue_trial(baseline_features)
        return

    logger.info("Enqueuing trial with all features included as baseline.")
    baseline_features = {f"{f}": 0.5 for f in numerical_features}
    study.enqueue_trial(baseline_features)


def _build_feature_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    categorical_features: list[str],
    numerical_features: list[str],
    model_hyperparams: dict,
) -> Callable:
    """Build Optuna objective for feature selection.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_validate: Validation features.
        y_validate: Validation targets.
        categorical_features: Categorical feature names.
        numerical_features: Numerical feature names.
        model_hyperparams: Hyperparameter dictionary.

    Returns:
        Objective callable for Optuna.
    """

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

        _, cur_valid_score = model_training(
            cur_X_train,
            y_train,
            cur_X_validate,
            y_validate,
            categorical_features,
            model_hyperparams,
            callbacks=[LightGBMPruningCallback(trial, metric=settings.OPTIMIZE_METRIC)],
        )

        return cur_valid_score

    return objective


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
    logger.info("Optimizing features...")
    study = _build_feature_study(seed)

    numerical_features = [f for f in all_features if f not in categorical_features]
    _enqueue_feature_baseline(study, numerical_features, given_features_pth)

    objective = _build_feature_objective(
        X_train,
        y_train,
        X_validate,
        y_validate,
        categorical_features,
        numerical_features,
        model_hyperparams,
    )

    study.optimize(
        objective,
        n_trials=settings.FEATURE_SEARCH_LIMIT,
        timeout=604800,
    )

    optimal_trial = study.best_trial
    best_params = optimal_trial.params.copy()
    optimal_features = categorical_features.copy()
    for feature in numerical_features:
        if best_params.get(feature, 0.0) >= 0.5:
            optimal_features.append(feature)
    optimal_value = optimal_trial.value
    original_feature_number = len(all_features)

    logger.info("Selected Optimal Trial Number: %s", optimal_trial.number)
    logger.info("  Optimal Trial Value (Valid %s): %.4f", settings.OPTIMIZE_METRIC, optimal_value)
    if verbose:
        logger.info("  Optimal Trial Features: %s", optimal_features)
    logger.info(
        "Final selected %d features after Optuna search, select ratio: %.2f",
        len(optimal_features),
        len(optimal_features) / original_feature_number,
    )

    logger.info("Feature optimization completed")

    return optimal_features


def _build_hyperparameter_study(seed: int) -> optuna.Study:
    """Build Optuna study for hyperparameter optimization.

    Args:
        seed: Random seed.

    Returns:
        Optuna study instance.
    """
    return optuna.create_study(
        study_name="Model Hyperparameter Optimization",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=seed, multivariate=settings.MODEL_OPTIMIZATION_SAMPLING_MULTIVARIATE
        ),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=settings.PRUNING_WARMUP_STEPS),
    )


def _build_fixed_train_config(seed: int) -> dict[str, Any]:
    """Build fixed training configuration.

    Args:
        seed: Random seed.

    Returns:
        Fixed training configuration dictionary.
    """
    cur_train_fixed_config = settings.FIXED_TRAINING_CONFIG.copy()
    cur_train_fixed_config.update({"seed": seed})
    cur_train_fixed_config.update(
        {"num_threads": min(max(1, cpu_count()), settings.MAX_NUM_PROCESSES)}
    )
    return cur_train_fixed_config


def _enqueue_hyperparameter_baseline(
    study: optuna.Study, given_model_hyperparams_pth: str, fixed_config: dict[str, Any]
) -> None:
    """Enqueue baseline hyperparameters.

    Args:
        study: Optuna study.
        given_model_hyperparams_pth: Path to baseline hyperparameters.
        fixed_config: Fixed training configuration.
    """
    if given_model_hyperparams_pth and Path(given_model_hyperparams_pth).exists():
        logger.info("Enqueuing trial with given hyperparameter as baseline.")
        given_model_hyperparams = json.load(open(given_model_hyperparams_pth, "r"))
        mhp_baseline = {
            "drop_rate": given_model_hyperparams.get("drop_rate", 0.1),
            "skip_drop": given_model_hyperparams.get("skip_drop", 0.5),
            "num_leaves": given_model_hyperparams.get("num_leaves", 31),
            "learning_rate": given_model_hyperparams.get("learning_rate", 0.1),
            "min_data_in_leaf": given_model_hyperparams.get("min_data_in_leaf", 20),
            "lambda_l1": given_model_hyperparams.get("lambda_l1", 1e-8),
            "lambda_l2": given_model_hyperparams.get("lambda_l2", 1e-8),
        }
        baseline = mhp_baseline.copy()
        baseline.update(fixed_config)
        study.enqueue_trial(baseline)
        return

    logger.info("Enqueuing trial with default hyperparameter as a baseline.")
    mhp_baseline = {
        "drop_rate": 0.1,
        "skip_drop": 0.5,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "min_data_in_leaf": 20,
        "lambda_l1": 1e-8,
        "lambda_l2": 1e-8,
    }
    baseline = mhp_baseline.copy()
    baseline.update(fixed_config)
    study.enqueue_trial(baseline)


def _build_hyperparameter_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    categorical_features: list,
    fixed_config: dict[str, Any],
) -> Callable:
    """Build Optuna objective for hyperparameter search.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_validate: Validation features.
        y_validate: Validation targets.
        categorical_features: Categorical feature names.
        fixed_config: Fixed training configuration.

    Returns:
        Objective callable for Optuna.
    """

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
        params.update(fixed_config)

        _, cur_valid_score = model_training(
            X_train,
            y_train,
            X_validate,
            y_validate,
            categorical_features,
            params,
            callbacks=[LightGBMPruningCallback(trial, metric=settings.OPTIMIZE_METRIC)],
        )

        return cur_valid_score

    return objective


def _log_param_importance(study: optuna.Study) -> None:
    """Log parameter importance from Optuna study.

    Args:
        study: Optuna study.
    """
    param_importance = get_param_importances(study)
    logger.info("Parameter Importances:")
    for param, importance in param_importance.items():
        logger.info("  - %s: %.6f", param, importance)


def _log_top_trials(study: optuna.Study) -> None:
    """Log top 5 Optuna trials.

    Args:
        study: Optuna study.
    """
    completed_trials = sorted(
        study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]),
        key=lambda trial: trial.value if trial.value is not None else math.inf,
    )

    logger.info("Top 5 Best Trials Summary:")
    for i, trial in enumerate(completed_trials[:5]):
        logger.info("  Trial %s (Rank %d):", trial.number, i + 1)
        logger.info("    - Value (%s): %.6f", settings.OPTIMIZE_METRIC, trial.value)
        logger.info("    - Params:")
        for param_name, param_value in trial.params.items():
            if isinstance(param_value, float):
                logger.info("      - %s: %.6f", param_name, param_value)
            else:
                logger.info("      - %s: %s", param_name, param_value)


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
    """Optimize model hyperparameters using Optuna."""
    logger.info("Optimize model hyperparameters...")
    study = _build_hyperparameter_study(seed)

    fixed_config = _build_fixed_train_config(seed)
    _enqueue_hyperparameter_baseline(study, given_model_hyperparams_pth, fixed_config)

    objective = _build_hyperparameter_objective(
        X_train,
        y_train,
        X_validate,
        y_validate,
        categorical_features,
        fixed_config,
    )

    study.optimize(objective, n_trials=settings.HYPERPARAMETER_SEARCH_LIMIT, timeout=604800)

    if verbose:
        _log_param_importance(study)
        _log_top_trials(study)

    optimal_trial = study.best_trial
    optimal_params = optimal_trial.params.copy()
    optimal_value = optimal_trial.value
    optimal_params.update(fixed_config)

    logger.info("Selected Optimal Trial Number: %s", optimal_trial.number)
    logger.info("  Optimal Trial Value (%s): %.4f", settings.OPTIMIZE_METRIC, optimal_value)
    if verbose:
        logger.info("  Optimal Trial Parameters: %s", optimal_params)

    logger.info("Hyperparameter optimization completed")

    return optimal_params
