import json
import logging
import math
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import lightgbm as lgb
import optuna
import pandas as pd
from lightgbm import register_logger
from optuna.importance import get_param_importances

from ml4investment.config.global_settings import settings
from ml4investment.utils.data_loader import sample_training_data
from ml4investment.utils.utils import OptimalIterationLogger, get_detailed_static_result

logger = logging.getLogger(__name__)
register_logger(logger)


def model_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    categorical_features: list,
    model_hyperparams: dict,
    show_training_log: bool = False
) -> tuple[lgb.Booster, float]:
    """Train the model with optimized parameters and get optimal score"""
    logger.info("Begin model training with optimized parameters")
    metric_logger_cb = OptimalIterationLogger()
    model = lgb.train(
        model_hyperparams,
        train_set=lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features),
        valid_sets=[
            lgb.Dataset(X_validate, label=y_validate, categorical_feature=categorical_features)
        ],
        num_boost_round=int(model_hyperparams["num_rounds"]),
        callbacks=[lgb.log_evaluation(period=100 if show_training_log else -1), metric_logger_cb],
    )
    model.best_iteration = metric_logger_cb.optimal_iteration
    optimal_score = metric_logger_cb.optimal_score
    logger.info(
        f"Model training completed. Optimal iteration on validation set: "
        f"{model.best_iteration} with {settings.OPTIMIZE_METRIC}: "
        f"{optimal_score}"
    )

    logger.info("Model training completed")

    return model, optimal_score


def validate_model(
    model: lgb.Booster,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    target_stock_list: list[str],
    verbose: bool = False,
) -> tuple[float, float, float, float, float, float, float, float, list]:
    """Validate the model on the validation dataset"""
    logger.info("Starting model validation on the provided validation set.")

    (
        valid_mae,
        valid_mse,
        valid_sign_acc,
        valid_precision,
        valid_recall,
        valid_f1,
        vaild_average_daily_gain,
        vaild_overall_gain,
        sorted_stocks,
    ) = get_detailed_static_result(
        model=model,
        X=X_validate,
        y=y_validate,
        predict_stock_list=target_stock_list,
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
        vaild_average_daily_gain,
        vaild_overall_gain,
        sorted_stocks,
    )


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
    """Optimize data sampling proportion for training"""
    logger.info("Optimizing data sampling proportion...")
    study = optuna.create_study(
        study_name="Data Sampling Proportion Optimization",
        directions=["minimize"],
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
    )

    if given_data_sampling_proportion_pth and Path(given_data_sampling_proportion_pth).exists():
        logger.info("Enqueuing trial with given sampling proportion as baseline.")
        baseline_dsp = json.load(open(given_data_sampling_proportion_pth, "r"))
        study.enqueue_trial(baseline_dsp)
    else:
        baseline_dsp = {}
        logger.info("Enqueuing trial with all data sampled as baseline.")
        uniform_params = {stock: 1.0 for stock in train_stock_list}
        study.enqueue_trial(uniform_params)
    
    dsp_search_range = {}
    train_stock_list = train_stock_list.copy()
    train_stock_list = [s for s in train_stock_list if s not in settings.SELECTIVE_ETF]
    dsp_search_amplitude = settings.DATA_SAMPLING_PROPORTION_SEARCH_AMPLITUDE
    logger.info(
        f"Updating each stock search range based on given data sampling proportion "
        f"with amplitude: {dsp_search_amplitude}"
    )
    for stock in train_stock_list:
        dsp_search_range[stock] = {
            "min": max(
                0.0,
                baseline_dsp.get(stock, 0.5) - dsp_search_amplitude
            ),
            "max": min(
                1.0,
                baseline_dsp.get(stock, 0.5) + dsp_search_amplitude
            ),
        }
        logger.info(
            f"  - {stock}: "
            f"[{dsp_search_range[stock]['min']:.2f}, {dsp_search_range[stock]['max']:.2f}]"
        )

    def objective(trial: optuna.Trial) -> float:
        stock_proportion_dict = {}
        for stock in train_stock_list:
            stock_proportion_dict[stock] = trial.suggest_float(
                f"{stock}", dsp_search_range[stock]['min'], dsp_search_range[stock]['max']
            )

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
        )

        return cur_valid_score

    study.optimize(
        objective,
        n_trials=settings.DATA_SAMPLING_PROPORTION_SEARCH_LIMIT,
        timeout=604800,
    )

    optimal_trial = study.best_trial
    optimal_data_sampling_proportion = optimal_trial.params.copy()
    optimal_value = optimal_trial.value

    logger.info(f"Selected Optimal Trial Number: {optimal_trial.number}")
    logger.info(f"  Optimal Trial Value (Valid {settings.OPTIMIZE_METRIC}): {optimal_value:.4f}")
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
    logger.info("Optimizing features...")
    study = optuna.create_study(
        study_name="Model Feature Selection Optimization",
        directions=["minimize"],
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
    )

    numerical_features = [f for f in all_features if f not in categorical_features]
    if given_features_pth and Path(given_features_pth).exists():
        logger.info("Enqueuing trial with given features as baseline.")
        given_model_features = json.load(open(given_features_pth, "r"))["features"]
        baseline_features = {}
        for feature in numerical_features:
            if feature in given_model_features:
                baseline_features[feature] = 0.75
            else:
                baseline_features[feature] = 0.25
        study.enqueue_trial(baseline_features)
    else:
        logger.info("Enqueuing trial with all features included as baseline.")
        baseline_features = {f"{f}": 0.5 for f in numerical_features}
        study.enqueue_trial(baseline_features)

    feature_search_range = {}
    feature_search_amplitude = settings.FEATURE_SEARCH_AMPLITUDE
    logger.info(
        f"Updating each feature search range based on given features "
        f"with amplitude: {feature_search_amplitude}"
    )
    for feature in numerical_features:
        feature_search_range[feature] = {
            "min": max(
                0.0, 
                baseline_features.get(feature, 0.5) - feature_search_amplitude
            ),
            "max": min(
                1.0, 
                baseline_features.get(feature, 0.5) + feature_search_amplitude
            ),
        }
        logger.info(
            f"  - {feature}: "
            f"[{feature_search_range[feature]['min']:.2f}, "
            f"{feature_search_range[feature]['max']:.2f}]"
        )

    def objective(trial: optuna.Trial) -> float:
        candidate_features = categorical_features.copy()
        for feature in numerical_features:
            if (trial.suggest_float(
                    feature, 
                    feature_search_range[feature]['min'], 
                    feature_search_range[feature]['max']
                ) >= 0.5):
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
        )

        return cur_valid_score

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

    optimal_value = study.best_trial.value
    original_feature_number = len(all_features)

    logger.info(
        f"Final selected {len(optimal_features)} features after Optuna search, "
        f"select ratio: {len(optimal_features) / original_feature_number:.2f}"
    )
    logger.info(
        f"Final Valid {settings.OPTIMIZE_METRIC} after feature selection: {optimal_value:.6f}"
    )
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
    logger.info("Optimize model hyperparameters...")
    study = optuna.create_study(
        study_name="Model Hyperparameter Optimization",
        directions=["minimize"],
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
    )

    cur_train_fixed_config = settings.FIXED_TRAINING_CONFIG.copy()
    cur_train_fixed_config.update({"seed": seed})
    cur_train_fixed_config.update(
        {"num_threads": min(max(1, cpu_count()), settings.MAX_NUM_PROCESSES)}
    )
    if given_model_hyperparams_pth and Path(given_model_hyperparams_pth).exists():
        logger.info("Enqueuing trial with given hyperparameter as baseline.")
        given_model_hyperparams = json.load(open(given_model_hyperparams_pth, "r"))
        mhp_baseline = {
            "drop_rate": given_model_hyperparams["drop_rate"],
            "skip_drop": given_model_hyperparams["skip_drop"],
            "num_leaves": given_model_hyperparams["num_leaves"],
            "learning_rate": given_model_hyperparams["learning_rate"],
            "min_data_in_leaf": given_model_hyperparams["min_data_in_leaf"],
            "lambda_l1": given_model_hyperparams["lambda_l1"],
            "lambda_l2": given_model_hyperparams["lambda_l2"],
        }
        baseline = mhp_baseline.copy()
        baseline.update(cur_train_fixed_config)
        study.enqueue_trial(baseline)
    else:
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
        baseline.update(cur_train_fixed_config)
        study.enqueue_trial(baseline)
    
    mhp_search_range = {}
    mhp_search_amplitude = settings.HYPERPARAMETER_SEARCH_AMPLITUDE
    logger.info(
        f"Updating each hyperparameter search range based on given features "
        f"with amplitude: {mhp_search_amplitude}"
    )
    for mhp in mhp_baseline.keys():
        center_value = mhp_baseline[mhp]
        lower_bound = center_value * (1 - mhp_search_amplitude)
        upper_bound = center_value * (1 + mhp_search_amplitude)
        
        if mhp in ["num_leaves", "min_data_in_leaf"]:
            lower_bound = int(max(1, lower_bound))
            upper_bound = math.ceil(max(lower_bound + 1, upper_bound))
        else:
            lower_bound = max(0.0, lower_bound)
            if mhp in ["drop_rate", "skip_drop"]:
                upper_bound = min(1.0, upper_bound)
        
        mhp_search_range[mhp] = {"min": lower_bound, "max": upper_bound}

        logger.info(
            f"  - {mhp}: "
            f"[{mhp_search_range[mhp]['min']:.2f}, "
            f"{mhp_search_range[mhp]['max']:.2f}]"
        )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "drop_rate": trial.suggest_float(
                "drop_rate", 
                mhp_search_range["drop_rate"]["min"], 
                mhp_search_range["drop_rate"]["max"]
            ),
            "skip_drop": trial.suggest_float(
                "skip_drop", 
                mhp_search_range["skip_drop"]["min"], 
                mhp_search_range["skip_drop"]["max"]
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", 
                mhp_search_range["num_leaves"]["min"], 
                mhp_search_range["num_leaves"]["max"], 
                log=True
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 
                mhp_search_range["learning_rate"]["min"], 
                mhp_search_range["learning_rate"]["max"], 
                log=True
            ),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", 
                mhp_search_range["min_data_in_leaf"]["min"], 
                mhp_search_range["min_data_in_leaf"]["max"]
            ),
            "lambda_l1": trial.suggest_float(
                "lambda_l1", 
                mhp_search_range["lambda_l1"]["min"], 
                mhp_search_range["lambda_l1"]["max"], 
                log=True
            ),
            "lambda_l2": trial.suggest_float(
                "lambda_l2", 
                mhp_search_range["lambda_l2"]["min"], 
                mhp_search_range["lambda_l2"]["max"], 
                log=True
            ),
        }
        params.update(cur_train_fixed_config)

        _, cur_valid_score = model_training(
            X_train,
            y_train,
            X_validate,
            y_validate,
            categorical_features,
            params,
        )

        return cur_valid_score

    study.optimize(
        objective, 
        n_trials=settings.HYPERPARAMETER_SEARCH_LIMIT, 
        timeout=604800
    )

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
            logger.info(f"    - Value ({settings.OPTIMIZE_METRIC}): {trial.value:.6f}")
            logger.info("    - Params:")
            for param_name, param_value in trial.params.items():
                if isinstance(param_value, float):
                    logger.info(f"      - {param_name}: {param_value:.6f}")
                else:
                    logger.info(f"      - {param_name}: {param_value}")

    optimal_trial = study.best_trial
    optimal_params = optimal_trial.params.copy()
    optimal_value = optimal_trial.value
    optimal_params.update(cur_train_fixed_config)

    logger.info(f"Selected Optimal Trial Number: {optimal_trial.number}")
    logger.info(f"  Optimal Trial Value ({settings.OPTIMIZE_METRIC}): {optimal_value:.4f}")
    if verbose:
        logger.info(f"  Optimal Trial Parameters: {optimal_params}")

    logger.info("Hyperparameter optimization completed")

    return optimal_params
