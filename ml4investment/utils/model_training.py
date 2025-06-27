import lightgbm as lgb
from lightgbm import register_logger
import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import logging
from prettytable import PrettyTable
from collections import defaultdict
import math

from ml4investment.config import settings
from ml4investment.utils.utils import id_to_stock_code

logger = logging.getLogger(__name__)
register_logger(logger)


def model_training(X_train: pd.DataFrame, 
                   y_train: pd.Series, 
                   X_validate: pd.DataFrame,
                   y_validate: pd.Series,
                   categorical_features: list = None,
                   model_hyperparams: dict = None, 
                   target_stock_list: list = None,
                   optimize_predict_stocks: bool = True,
                   seed: int = 42,
                   verbose: bool = False) -> tuple[lgb.Booster, dict, float, float, list, list]:
    """ Time series modeling training pipeline """
    def objective(trial: optuna.Trial) -> float:
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'verbosity': -1,
            'boosting_type': 'dart',

            'drop_rate': trial.suggest_float('drop_rate', 0.05, 0.2),
            'max_drop': trial.suggest_int('max_drop', 10, 40),
            'skip_drop': trial.suggest_float('skip_drop', 0.0, 0.6),

            'num_leaves': trial.suggest_int('num_leaves', 48, 256),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),

            'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 1e-1, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 1e-1, log=True),

            'max_bin': trial.suggest_int('max_bin', 128, 255),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 0.1),
            
            # For feature and data, we optimize them seperately
            'feature_fraction': trial.suggest_float('feature_fraction', 1.0, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 1.0, 1.0),
            
            'seed': seed,
            'force_row_wise': True,
            'deterministic': True
        }
        
        avg_mae, avg_sign_accuracy, stock_avg_actual_gain_dict = cross_validate(params, num_rounds=trial.suggest_int('num_rounds', 800, 2400))
        
        trial.set_user_attr("avg_sign_accuracy", avg_sign_accuracy)
        trial.set_user_attr("stock_avg_actual_gain_dict", stock_avg_actual_gain_dict)
        
        return avg_mae
    
    def cross_validate(params: dict, num_rounds: int) -> tuple[float, float, dict]:
        tscv = TimeSeriesSplit(n_splits=settings.N_SPLIT)
        target_preds_all = []
        target_y_val_all = []
        stock_actual_gains_collect = defaultdict(lambda: {'total_gain': 1.0, 'sample_count': 0})

        for fold, (train_idx, valid_idx) in enumerate(tscv.split(X_train)):
            cur_X_train, cur_X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            cur_y_train, cur_y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
            assert 'stock_id' in cur_X_train.columns
            assert 'stock_id' in cur_X_val.columns

            model = lgb.train(
                params,
                train_set=lgb.Dataset(cur_X_train,
                                      label=cur_y_train,
                                      categorical_feature=categorical_features),
                valid_sets=[lgb.Dataset(cur_X_val,
                                        label=cur_y_val,
                                        categorical_feature=categorical_features)],
                num_boost_round=num_rounds,
                callbacks=[
                    lgb.log_evaluation(False),
                ]
            )
            preds = model.predict(cur_X_val)

            unique_stock_ids_in_fold = cur_X_val['stock_id'].unique()
            for stock_id_val in unique_stock_ids_in_fold:
                stock_code_val = id_to_stock_code(stock_id_val)
                if stock_code_val in target_stock_list:
                    stock_mask = (cur_X_val['stock_id'] == stock_id_val)
                    stock_preds = preds[stock_mask]
                    stock_y_val_numpy = cur_y_val[stock_mask].to_numpy()

                    target_preds_all.extend(stock_preds)
                    target_y_val_all.extend(stock_y_val_numpy)

                    positive_pred_mask = stock_preds > 0
                    factors_to_multiply = 1 + stock_y_val_numpy[positive_pred_mask]
                    current_fold_total_gain = np.prod(factors_to_multiply)
                    
                    stock_actual_gains_collect[stock_code_val]['total_gain'] *= current_fold_total_gain
                    stock_actual_gains_collect[stock_code_val]['sample_count'] += len(stock_preds)

        stock_avg_actual_gain_dict = {
            stock_code: data['total_gain'] ** (1 / data['sample_count'])
            for stock_code, data in stock_actual_gains_collect.items()
        }

        avg_mae_overall = mean_absolute_error(target_y_val_all, target_preds_all) if target_y_val_all else 0.0
        avg_sign_acc_overall = (np.sign(np.array(target_preds_all)) == np.sign(np.array(target_y_val_all))).mean() if target_y_val_all else 0.0

        return avg_mae_overall, avg_sign_acc_overall, stock_avg_actual_gain_dict

    if model_hyperparams is None:
        logger.info("Begin hyperparameter optimization")
        study = optuna.create_study(
            directions=["minimize"],
            sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
        )
        study.optimize(objective, n_trials=settings.HYPERPARAMETER_SEARCH_LIMIT, timeout=1000000000)
        logger.info("Hyperparameter optimization completed")

        best_trial = study.best_trial
        best_params = best_trial.params.copy()
        best_mae = best_trial.value
        best_sign_accuracy = best_trial.user_attrs.get("avg_sign_accuracy", None)
        best_stock_avg_actual_gain_dict = best_trial.user_attrs.get("stock_avg_actual_gain_dict", {})
        
        best_params.update({
            'objective': 'regression_l1',
            'metric': 'mae',
            'verbosity': -1,
            'boosting_type': 'dart',

            'seed': seed,
            'force_row_wise': True,
            'deterministic': True
        })
        logger.info(f"Selected Best Trial Number: {best_trial.number}")
        logger.info(f"  Best Trial Value (MAE): {best_mae:.4f}")
        logger.info(f"  Best Trial Sign Accuracy: {best_sign_accuracy*100:.2f}%")
        logger.info(f"  Best Trial Stock Average Actual Gain: {best_stock_avg_actual_gain_dict}")
        if verbose:
            logger.info(f"  Best Trial Parameters: best_params")
    else:
        logger.info("Load input model hyperparameter")
        best_params = model_hyperparams.copy()
    
    logger.info("Begin model training with optimized parameters")
    final_model = lgb.train(
        best_params,
        train_set=lgb.Dataset(X_train, 
                              label=y_train, 
                              categorical_feature=categorical_features, 
                              free_raw_data=False),
        valid_sets=lgb.Dataset(X_validate, 
                              label=y_validate, 
                              categorical_feature=categorical_features, 
                              free_raw_data=False),
        num_boost_round=int(best_params['num_rounds'] / (1 - 1 / settings.N_SPLIT)),
        callbacks=[
            lgb.log_evaluation(period=100),
        ]
    )
    logger.info("Model training completed")
    
    valid_mae, valid_sign_accuracy, valid_stock_avg_actual_gain_dict = validate_model(
        final_model, X_validate, y_validate,
        target_stock_list=target_stock_list
    )

    if optimize_predict_stocks:
        logger.info("Begin predict stocks optimization")
        logger.info(f"Using average actual gain as the predict stocks optimization metric with target number {settings.PREDICT_STOCK_NUMBER}")
        sorted_stock_avg_actual_gain_list = sorted(valid_stock_avg_actual_gain_dict.items(), key=lambda item: item[1], reverse=True)
        predict_stock_list = [stock for stock, _ in sorted_stock_avg_actual_gain_list[:settings.PREDICT_STOCK_NUMBER]]
        logger.info(f"Selected {len(predict_stock_list)} stocks for prediction: {', '.join(predict_stock_list)}")
    else:
        logger.info("No predict stocks optimization. Using all target stocks as predict stocks")
        predict_stock_list = target_stock_list
    
    importance = final_model.feature_importance(importance_type='gain')
    features = final_model.feature_name()
    sorted_feature_imp = sorted(zip(importance, features), reverse=True)
    if verbose:
        logger.info(f'Top 10 features by gain importance: {sorted_feature_imp[:10]}')
        logger.info(f'Worst 10 features by gain importance: {sorted_feature_imp[-10:]}')
    
    return final_model, best_params, valid_mae, valid_sign_accuracy, sorted_feature_imp, predict_stock_list


def validate_model(model: lgb.Booster,
                   X_validate: pd.DataFrame,
                   y_validate: pd.Series,
                   target_stock_list: list) -> tuple[float, float, dict]:
    """ Validate the model on the validation dataset """
    logger.info("Starting model validation on the provided validation set.")

    target_preds_all = []
    target_y_val_all = []
    stock_actual_gains_collect = defaultdict(lambda: {'total_gain': 1.0, 'sample_count': 0})

    preds = model.predict(X_validate)

    unique_stock_ids_in_val = X_validate['stock_id'].unique()
    for stock_id_val in unique_stock_ids_in_val:
        stock_code_val = id_to_stock_code(stock_id_val)

        if stock_code_val in target_stock_list:
            stock_mask = (X_validate['stock_id'] == stock_id_val)
            stock_preds = preds[stock_mask]
            stock_y_val_numpy = y_validate[stock_mask].to_numpy()

            target_preds_all.extend(stock_preds)
            target_y_val_all.extend(stock_y_val_numpy)

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

    avg_mae_overall = mean_absolute_error(target_y_val_all, target_preds_all) if target_y_val_all else 0.0
    avg_sign_acc_overall = (np.sign(np.array(target_preds_all)) == np.sign(np.array(target_y_val_all))).mean() if target_y_val_all else 0.0

    logger.info("Model validation completed.")
    logger.info(f"Validation Overall Metrics - MAE: {avg_mae_overall:.4f} | Sign Accuracy: {avg_sign_acc_overall*100:.2f}%")

    return avg_mae_overall, avg_sign_acc_overall, stock_avg_actual_gain_dict


def optimize_data_sampling_proportion(X_train: pd.DataFrame,
                                      y_train: pd.Series,
                                      X_validate: pd.DataFrame,
                                      y_validate: pd.Series,
                                      categorical_features: list,
                                      model_hyperparams: dict,
                                      target_stock_list: list,
                                      seed: int = 42,
                                      verbose: bool = False) -> dict:
    """ Optimize data sampling proportion for training """
    logger.info("Optimizing data sampling proportion...")
    
    train_months = sorted(list(X_train.index.tz_localize(None).to_period('M').astype(str).unique()))
    month_mae_dict = {}
    
    for month in train_months:
        logger.info(f"Processing month: {month}")
        month_mask = (X_train.index.tz_localize(None).to_period('M') == month)
        X_train_month = X_train[month_mask]
        y_train_month = y_train[month_mask]

        _, _, mae, _, _, _ = model_training(
            X_train_month, y_train_month,
            X_validate, y_validate, 
            categorical_features=categorical_features,
            model_hyperparams=model_hyperparams,
            target_stock_list=target_stock_list,
            optimize_predict_stocks=False,
            seed=seed,
            verbose=verbose
        )

        month_mae_dict[month] = mae
    
    if verbose:
        logger.info("Monthly MAE values:")
        for month, mae in month_mae_dict.items():
            logger.info(f"  {month}: {mae:.6f}")
    
    worst_mae = max(month_mae_dict.values())
    sampling_temperature = settings.DATA_SAMPLING_TEMPERATURE
    month_performance_dict = {month: (worst_mae / mae - 1) for month, mae in month_mae_dict.items()}
    normalization_term = sum([math.exp(performance / sampling_temperature) for performance in month_performance_dict.values()])
    optimal_data_sampling_proportion = {month: math.exp(performance / sampling_temperature) / normalization_term for month, performance in month_performance_dict.items()}
    
    if verbose:
        logger.info("Optimal data sampling proportion:")
        for month, proportion in optimal_data_sampling_proportion.items():
            logger.info(f"  {month}: {proportion*100:.2f}%")
    
    return optimal_data_sampling_proportion


def optimize_model_features(X_train: pd.DataFrame,
                            y_train: pd.Series,
                            categorical_features: list,
                            model_hyperparams: dict,
                            target_stock_list: list,
                            optimize_predict_stocks: bool,
                            original_model: lgb.Booster,
                            original_sorted_feature_imp: list, 
                            original_features: list, 
                            original_mae: float, 
                            original_sign_accuracy: float,
                            seed: int,
                            verbose: bool = False) -> list[lgb.Booster, dict, list, list]:
    """ Optimize model features using Recursive Feature Elimination (RFE) """
    logger.info("Feature Optimization begins...")
    feature_ranking = list(reversed([f for _, f in original_sorted_feature_imp]))

    original_feature_number = len(original_features)
    feature_search_num = settings.FEATURE_SEARCH_LIMIT

    optimal_features = original_features.copy()
    optimal_mae = original_mae
    optimal_sign_accuracy = original_sign_accuracy
    optimal_model = original_model
    optimal_model_hyperparams = model_hyperparams.copy()
    optimal_predict_stock_list = []
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

        X_train_tmp = X_train[candidate_features]
        
        model_tmp, model_hyperparams_tmp, mae_tmp, sign_accuracy_tmp, sorted_feature_imp_tmp, predict_stock_list_tmp = model_training(
            X_train_tmp, y_train, 
            categorical_features=categorical_features,
            model_hyperparams=model_hyperparams,
            target_stock_list=target_stock_list,
            optimize_predict_stocks=optimize_predict_stocks,
            seed=seed,
            verbose=verbose
        )
        
        if mae_tmp <= optimal_mae:
            logger.info(f"Removing '{feature_to_remove}' improved or kept performance.")
            optimal_mae = mae_tmp
            optimal_sign_accuracy = sign_accuracy_tmp
            optimal_model = model_tmp
            optimal_model_hyperparams = model_hyperparams_tmp
            optimal_features = candidate_features
            optimal_predict_stock_list = predict_stock_list_tmp
            if verbose:
                logger.info(f"Updated optimal features: {', '.join(optimal_features)}")
            feature_ranking = list(reversed([f for _, f in sorted_feature_imp_tmp]))
        else:
            logger.info(f"Removing '{feature_to_remove}' degraded performance. Skip it.")
            feature_ranking = feature_ranking[1:]
    
    logger.info(f"Final selected {len(optimal_features)} features after RFE, select ratio: {len(optimal_features) / original_feature_number:.2f}")
    logger.info(f"Final MAE after feature selection: {optimal_mae:.6f}, improvement: {original_mae - optimal_mae:.6f}")
    logger.info(f"Final sign accuracy after feature selection: {optimal_sign_accuracy*100:.2f}%, improvement: {(optimal_sign_accuracy - original_sign_accuracy)*100:.2f}%")
    if verbose:
        logger.info(f"Optimal features: {', '.join(optimal_features)}")

    return optimal_model, optimal_model_hyperparams, optimal_features, optimal_predict_stock_list
