import lightgbm as lgb
from lightgbm import register_logger
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import logging
from prettytable import PrettyTable
from collections import defaultdict
import math
from pathlib import Path
import json

from ml4investment.config import settings
from ml4investment.utils.data_loader import sample_training_data
from ml4investment.utils.utils import set_random_seed, id_to_stock_code, OptimalIterationLogger, get_detailed_static_result

logger = logging.getLogger(__name__)
register_logger(logger)


def model_training(X_train: pd.DataFrame, 
                   y_train: pd.Series, 
                   X_validate: pd.DataFrame,
                   y_validate: pd.Series, 
                   X_validate_dict: dict, 
                   y_validate_dict: dict,
                   categorical_features: list = None,
                   model_hyperparams: dict = None, 
                   target_stock_list: list = None,
                   optimize_predict_stocks: bool = True,
                   seed: int = 42,
                   verbose: bool = False) -> tuple[lgb.Booster, list]:
    """ Train the final model with optimized parameters """
    logger.info("Begin model training with optimized parameters")
    logger.info(model_hyperparams)
    metric_logger_cb = OptimalIterationLogger()
    final_model = lgb.train(
        model_hyperparams,
        train_set=lgb.Dataset(X_train, 
                              label=y_train, 
                              categorical_feature=categorical_features),
        valid_sets=[lgb.Dataset(X_validate, 
                                label=y_validate, 
                                categorical_feature=categorical_features)],
        num_boost_round=int(model_hyperparams['num_rounds']),
        callbacks=[
            lgb.log_evaluation(period=100),
            metric_logger_cb
        ]
    )
    final_model.best_iteration = metric_logger_cb.optimal_iteration
    logger.info(f"Final model training completed. Optimal iteration on validation set: {final_model.best_iteration} with lowest MAE: {metric_logger_cb.optimal_score}")
    
    valid_mae, predict_stock_list = validate_model(
        final_model, 
        X_validate, y_validate,
        X_validate_dict, y_validate_dict,
        target_stock_list=target_stock_list,
        optimize_predict_stocks=optimize_predict_stocks,
        verbose=verbose
    )
    
    logger.info("Model training completed")

    return final_model, predict_stock_list


def validate_model(model: lgb.Booster,
                   X_validate: pd.DataFrame,
                   y_validate: pd.Series,
                   X_validate_dict: dict, 
                   y_validate_dict: dict,
                   target_stock_list: list,
                   optimize_predict_stocks: bool, 
                   verbose: bool = False) -> tuple[float, list]:
    """ Validate the model on the validation dataset """
    logger.info("Starting model validation on the provided validation set.")

    stock_actual_gains_collect = defaultdict(lambda: {'total_gain': 1.0, 'sample_count': 0})

    preds = model.predict(X_validate, num_iteration=model.best_iteration)

    unique_stock_ids_in_val = X_validate['stock_id'].unique()
    for stock_id_val in unique_stock_ids_in_val:
        stock_code_val = id_to_stock_code(stock_id_val)

        stock_mask = (X_validate['stock_id'] == stock_id_val)
        stock_preds = preds[stock_mask]
        stock_y_val_numpy = y_validate[stock_mask].to_numpy()

        if stock_code_val in target_stock_list:
            # Only optimize predict stock list from target stock list
            positive_pred_mask = stock_preds > 0
            if np.any(positive_pred_mask):
                factors_to_multiply = 1 + stock_y_val_numpy[positive_pred_mask]
                current_stock_gain_prod = np.prod(factors_to_multiply)
                
                stock_actual_gains_collect[stock_code_val]['total_gain'] *= current_stock_gain_prod
                stock_actual_gains_collect[stock_code_val]['sample_count'] += len(stock_preds)
            else:
                stock_actual_gains_collect[stock_code_val]['sample_count'] += len(stock_preds)

    stock_avg_actual_gain_dict = {
        stock_code: data['total_gain'] ** (1 / data['sample_count'])
        for stock_code, data in stock_actual_gains_collect.items()
    }

    if optimize_predict_stocks:
        logger.info("Begin predict stocks optimization")
        logger.info(f"Using average actual gain as the predict stocks optimization metric with target number {settings.PREDICT_STOCK_NUMBER}")
        sorted_stock_avg_actual_gain_list = sorted(stock_avg_actual_gain_dict.items(), key=lambda item: item[1], reverse=True)
        predict_stock_list = [stock for stock, _ in sorted_stock_avg_actual_gain_list[:settings.PREDICT_STOCK_NUMBER]]
        logger.info(f"Selected {len(predict_stock_list)} stocks for prediction: {', '.join(predict_stock_list)}")
    else:
        logger.info("No predict stocks optimization. Using all target stocks as predict stocks")
        predict_stock_list = target_stock_list

    avg_mae, avg_mse = get_detailed_static_result(
        model=model,
        X_dict=X_validate_dict,
        y_dict=y_validate_dict,
        predict_stock_list=predict_stock_list,
        start_date=settings.VALIDATION_DATA_START_DATE,
        end_date=settings.VALIDATION_DATA_END_DATE,
        name="Validation",
        verbose=verbose
    )

    logger.info("Model validation completed.")

    return avg_mae, predict_stock_list


def optimize_data_sampling_proportion(X_train: pd.DataFrame,
                                      y_train: pd.Series,
                                      X_validate: pd.DataFrame,
                                      y_validate: pd.Series,
                                      target_sample_size: int, 
                                      categorical_features: list,
                                      model_hyperparams: dict,
                                      given_data_sampling_proportion_pth: str,
                                      seed: int = 42,
                                      verbose: bool = False) -> dict:
    """ Optimize data sampling proportion for training """
    train_months = sorted(list(X_train.index.tz_localize(None).to_period('M').astype(str).unique()))
    def objective(trial: optuna.Trial) -> float:
        month_proportion_dict = {}
        for month in train_months:
            month_proportion_dict[month] = trial.suggest_float(f'{month}', 0.0, 1.0)

        # Normalize the proportions to sum to 1
        total_proportion = sum(month_proportion_dict.values())
        month_proportion_dict = {month: proportion / total_proportion for month, proportion in month_proportion_dict.items()}
        
        cur_X_train, cur_y_train = sample_training_data(
            X_train, y_train, 
            sampling_proportion=month_proportion_dict, 
            target_sample_size=target_sample_size,
            seed=seed
        )
        
        cur_metric_logger_cb = OptimalIterationLogger()
        cur_model = lgb.train(
            model_hyperparams,
            train_set=lgb.Dataset(cur_X_train, 
                                  label=cur_y_train, 
                                  categorical_feature=categorical_features),
            valid_sets=[lgb.Dataset(X_validate, 
                                  label=y_validate, 
                                  categorical_feature=categorical_features)],
            num_boost_round=int(model_hyperparams['num_rounds']),
            callbacks=[
                lgb.log_evaluation(False),
                cur_metric_logger_cb
            ]
        )
        cur_model.best_iteration = cur_metric_logger_cb.optimal_iteration
        cur_valid_mae = cur_metric_logger_cb.optimal_score
        logger.info(f"Current trial model training completed. Optimal iteration on validation set: {cur_model.best_iteration} with lowest MAE: {cur_valid_mae}")
        
        return cur_valid_mae

    logger.info("Optimizing data sampling proportion...")
    study = optuna.create_study(
        study_name="Data Sampling Proportion Optimization",
        directions=["minimize"],
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
    )

    logger.info("Enqueuing trial with uniform sampling proportion as a baseline.")
    uniform_params = {month: 0.5 for month in train_months}
    study.enqueue_trial(uniform_params)

    if given_data_sampling_proportion_pth and Path(given_data_sampling_proportion_pth).exists():
        logger.info("Enqueuing trial with given sampling proportion as another baseline.")
        given_data_sampling_proportion = json.load(open(given_data_sampling_proportion_pth, 'r'))
        study.enqueue_trial(given_data_sampling_proportion)

    study.optimize(objective, n_trials=settings.DATA_SAMPLING_PROPORTION_SEARCH_LIMIT, timeout=604800)

    optimal_trial = study.best_trial
    optimal_params = optimal_trial.params.copy()
    optimal_valid_mae = optimal_trial.value
    
    logger.info(f"Selected Optimal Trial Number: {optimal_trial.number}")
    logger.info(f"  Optimal Trial Value (Valid MAE): {optimal_valid_mae:.4f}")
    if verbose:
        logger.info(f"  Optimal Trial Data Sampling Proportion: {optimal_params}")

    optimal_data_sampling_proportion = {}
    total_proportion = sum(optimal_params.values())
    for month in train_months:
        optimal_data_sampling_proportion[month] = optimal_params[month] / total_proportion
    
    if verbose:
        logger.info("Optimal data sampling proportion:")
        for month, proportion in optimal_data_sampling_proportion.items():
            logger.info(f"  {month}: {proportion*100:.2f}%")
            
    logger.info("Data sampling proportion optimization completed")
    
    return optimal_data_sampling_proportion


def optimize_model_features(X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_validate: pd.DataFrame,
                            y_validate: pd.Series,
                            categorical_features: list,
                            model_hyperparams: dict,
                            seed: int,
                            verbose: bool = False) -> list[list]:
    """ Optimize model features using Recursive Feature Elimination (RFE) """
    logger.info("Feature Optimization begins...")
    # train the model to get the initial feature importance and use it as baseline
    initial_metric_logger_cb = OptimalIterationLogger()
    optimal_model = lgb.train(
        model_hyperparams,
        train_set=lgb.Dataset(X_train, 
                              label=y_train, 
                              categorical_feature=categorical_features),
        valid_sets=[lgb.Dataset(X_validate, 
                                label=y_validate, 
                                categorical_feature=categorical_features)],
        num_boost_round=int(model_hyperparams['num_rounds']),
        callbacks=[
            lgb.log_evaluation(False),
            initial_metric_logger_cb
        ]
    )
    optimal_model.best_iteration = initial_metric_logger_cb.optimal_iteration
    original_valid_mae = initial_metric_logger_cb.optimal_score
    optimal_valid_mae = original_valid_mae
    importance = optimal_model.feature_importance(importance_type='gain', iteration=optimal_model.best_iteration)
    features = optimal_model.feature_name()
    sorted_feature_imp_tmp = sorted(zip(importance, features), reverse=True)
    feature_ranking = list(reversed([f for _, f in sorted_feature_imp_tmp]))
    optimal_features = feature_ranking.copy()

    original_feature_number = len(feature_ranking)
    feature_search_num = settings.FEATURE_SEARCH_LIMIT

    while(feature_search_num > 0):
        feature_search_num -= 1
        logger.info(f"Current feature search number: {feature_search_num}")

        feature_to_remove = feature_ranking[0]
        logger.info(f"Trying to remove feature: '{feature_to_remove}'")
        
        if feature_to_remove in settings.CATEGORICAL_FEATURES:
            logger.info(f"Skipping '{feature_to_remove}' feature removal")
            feature_ranking = feature_ranking[1:]
            continue
        
        candidate_features = [f for f in optimal_features if f != feature_to_remove]

        cur_X_train = X_train[candidate_features]
        cur_X_validate = X_validate[candidate_features]
        
        cur_metric_logger_cb = OptimalIterationLogger()
        cur_model = lgb.train(
            model_hyperparams,
            train_set=lgb.Dataset(cur_X_train,
                                  label=y_train,
                                  categorical_feature=categorical_features),
            valid_sets=[lgb.Dataset(cur_X_validate,
                                    label=y_validate,
                                    categorical_feature=categorical_features)],
            num_boost_round=model_hyperparams['num_rounds'],
            callbacks=[
                lgb.log_evaluation(False),
                cur_metric_logger_cb
            ]
        )
        cur_model.best_iteration = cur_metric_logger_cb.optimal_iteration
        cur_valid_mae = cur_metric_logger_cb.optimal_score
        logger.info(f"Current MAE after removing '{feature_to_remove}': {cur_valid_mae:.6f}")
        
        if cur_valid_mae <= optimal_valid_mae:
            logger.info(f"Removing '{feature_to_remove}' improved or kept performance.")
            optimal_valid_mae = cur_valid_mae
            optimal_features = candidate_features
            optimal_model = cur_model

            importance = optimal_model.feature_importance(importance_type='gain', iteration=optimal_model.best_iteration)
            features = optimal_model.feature_name()
            sorted_feature_imp_tmp = sorted(zip(importance, features), reverse=True)
            if verbose:
                logger.info(f'Top 10 features by gain importance: {sorted_feature_imp_tmp[:10]}')
                logger.info(f'Worst 10 features by gain importance: {sorted_feature_imp_tmp[-10:]}')
            feature_ranking = list(reversed([f for _, f in sorted_feature_imp_tmp]))

            if verbose:
                logger.info(f"Updated optimal features: {', '.join(optimal_features)}")
        else:
            logger.info(f"Removing '{feature_to_remove}' degraded performance. Skip it.")
            feature_ranking = feature_ranking[1:]
    
    logger.info(f"Final selected {len(optimal_features)} features after RFE, select ratio: {len(optimal_features) / original_feature_number:.2f}")
    logger.info(f"Final Valid MAE after feature selection: {optimal_valid_mae:.6f}, improvement: {original_valid_mae - optimal_valid_mae:.6f}")
    if verbose:
        logger.info(f"Optimal features: {optimal_features}")
    
    logger.info("Feature optimization completed")

    return optimal_features


def optimize_model_hyperparameters(X_train: pd.DataFrame, 
                                   y_train: pd.Series, 
                                   X_validate: pd.DataFrame,
                                   y_validate: pd.Series,
                                   categorical_features: list,
                                   given_model_hyperparams_pth: str,
                                   seed: int = 42,
                                   verbose: bool = False) -> tuple[dict]:
    """ Optimize model hyperparameters using Optuna """
    def objective(trial: optuna.Trial) -> float:
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'verbosity': -1,
            'boosting_type': 'dart',

            'drop_rate': trial.suggest_float('drop_rate', 0.05, 0.2),
            'skip_drop': trial.suggest_float('skip_drop', 0.3, 0.7),

            'num_leaves': trial.suggest_int('num_leaves', 15, 255, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),

            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True),

            'num_rounds': settings.NUM_ROUNDS,
            
            # For feature and data, we optimize them seperately
            'feature_fraction': 1.0,
            'bagging_freq': 0,
            'bagging_fraction': 1.0,
            
            # For feature and data, we optimize them seperately
            'feature_fraction': trial.suggest_float('feature_fraction', 1.0, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 1.0, 1.0),
            
            'seed': seed,
            'force_row_wise': True,
            'deterministic': True
        }

        cur_metric_logger_cb = OptimalIterationLogger()
        cur_model = lgb.train(
            params,
            train_set=lgb.Dataset(X_train,
                                  label=y_train,
                                  categorical_feature=categorical_features),
            valid_sets=[lgb.Dataset(X_validate,
                                    label=y_validate,
                                    categorical_feature=categorical_features)],
            num_boost_round=params['num_rounds'],
            callbacks=[
                lgb.log_evaluation(False),
                cur_metric_logger_cb
            ]
        )
        cur_model.best_iteration = cur_metric_logger_cb.optimal_iteration
        cur_valid_mae = cur_metric_logger_cb.optimal_score
        logger.info(f"Current trial model training completed. Optimal iteration on validation set: {cur_model.best_iteration} with lowest MAE: {cur_valid_mae}")

        return cur_valid_mae

    logger.info("Begin hyperparameter optimization")
    study = optuna.create_study(
        study_name="Model Hyperparameter Optimization",
        directions=["minimize"],
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
    )

    logger.info("Enqueuing trial with default hyperparameter as a baseline.")
    default_params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'verbosity': -1,
            'boosting_type': 'dart',

            'drop_rate': 0.1,
            'skip_drop': 0.5,

            'num_leaves': 31,
            'learning_rate': 0.1,
            'min_data_in_leaf': 20,

            'lambda_l1': 1e-8,
            'lambda_l2': 1e-8,

            'num_rounds': settings.NUM_ROUNDS,
            
            # For feature and data, we optimize them seperately
            'feature_fraction': 1.0,
            'bagging_freq': 0,
            'bagging_fraction': 1.0,
            
            'seed': seed,
            'force_row_wise': True,
            'deterministic': True
        }
    study.enqueue_trial(default_params)

    if given_model_hyperparams_pth and Path(given_model_hyperparams_pth).exists():
        logger.info("Enqueuing trial with given hyperparameter as another baseline.")
        given_model_hyperparams = json.load(open(given_model_hyperparams_pth, 'r'))
        study.enqueue_trial(given_model_hyperparams)

    study.optimize(objective, n_trials=settings.HYPERPARAMETER_SEARCH_LIMIT, timeout=604800)

    optimal_trial = study.best_trial
    optimal_params = optimal_trial.params.copy()
    optimal_valid_mae = optimal_trial.value
    
    optimal_params.update({
        'objective': 'regression_l1',
        'metric': 'mae',
        'verbosity': -1,
        'boosting_type': 'dart',

        'num_rounds': settings.NUM_ROUNDS,

        'feature_fraction': 1.0,
        'bagging_freq': 0,
        'bagging_fraction': 1.0,
        
        'seed': seed,
        'force_row_wise': True,
        'deterministic': True
    })
    logger.info(f"Selected Optimal Trial Number: {optimal_trial.number}")
    logger.info(f"  Optimal Trial Value (Valid MAE): {optimal_valid_mae:.4f}")
    if verbose:
        logger.info(f"  Optimal Trial Parameters: {optimal_params}")

    logger.info("Hyperparameter optimization completed")
    
    return optimal_params
