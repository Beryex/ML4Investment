import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def model_training(x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series) -> lgb.Booster:
    """ Time series modeling training pipeline """
    # 1. Data preparation for LightGBM
    train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    val_set = lgb.Dataset(x_val, label=y_val, reference=train_set, free_raw_data=False)
    
    # 2. Hyperparameter optimization with temporal validation
    def objective(trial: optuna.Trial) -> float:
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'verbosity': -1,
            'seed': 42,
            'boosting_type': 'dart',
            'drop_rate': trial.suggest_float('drop_rate', 0.05, 0.15),
            'max_drop': trial.suggest_int('max_drop', 10, 30),
            'skip_drop': trial.suggest_float('skip_drop', 0.4, 0.8),
            'num_leaves': trial.suggest_int('num_leaves', 16, 64),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
            'force_row_wise': True
        }
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for fold, (train_idx, valid_idx) in enumerate(tscv.split(x_train)):
            cur_x_train, cur_x_val = x_train.iloc[train_idx], x_train.iloc[valid_idx]
            cur_y_train, cur_y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
            
            model = lgb.train(
                params,
                train_set=lgb.Dataset(cur_x_train, label=cur_y_train),
                valid_sets=[lgb.Dataset(cur_x_val, label=cur_y_val)],
                num_boost_round=trial.suggest_int('num_rounds', 50, 300),
                callbacks=[
                    lgb.log_evaluation(False),
                ]
            )
            preds = model.predict(cur_x_val)
            scores.append(mean_absolute_error(cur_y_val, preds))
            
        return np.mean(scores)

    logger.info("Begin hyperparameter optimization")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, timeout=7200)
    logger.info("Hyperparameter optimization completed")
    
    # 3. Final model training with best parameters
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'regression_l1',
        'metric': 'mae',
        'verbosity': 1,
        'force_row_wise': True,  # Ensure reproducibility
        'xgboost_dart_mode': True
    })
    
    logger.info("Begin model training with optimized parameters")
    final_model = lgb.train(
        best_params,
        train_set=train_set,
        valid_sets=[val_set],
        num_boost_round=int(best_params['num_rounds']),
        callbacks=[
            lgb.log_evaluation(period=100),
            lgb.record_evaluation(eval_result={}),
        ]
    )
    logger.info("Model training completed")
    
    # 4. Model validation and feature analysis
    val_pred = final_model.predict(x_val)
    mae = mean_absolute_error(y_val, val_pred)
    sign_accuracy = (np.sign(val_pred) == np.sign(y_val.to_numpy())).mean() * 100
    logger.info(f"Model validation - MAE: {mae:.4f} | Sign Accuracy: {sign_accuracy:.2f}% | Features used: {len(final_model.feature_name())}")
    
    # Feature importance analysis
    importance = final_model.feature_importance(importance_type='gain')
    features = final_model.feature_name()
    sorted_imp = sorted(zip(importance, features), reverse=True)
    logger.info("Top features by gain:")
    for imp, name in sorted_imp[:10]:
        logger.info(f"{name}: {imp:.2f}")
    
    return final_model
