import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def model_training(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> lgb.Booster:
    """ Time series modeling training pipeline """
    # 1. Data preparation for LightGBM
    train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    test_set = lgb.Dataset(x_test, label=y_train, reference=train_set, free_raw_data=False)
    
    # 2. Hyperparameter optimization with temporal validation
    def objective(trial: optuna.Trial) -> float:
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'verbosity': -1,
            'seed': 42,
            'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart']),
            'num_leaves': trial.suggest_int('num_leaves', 16, 128),
            'learning_rate': trial.suggest_float('lr', 0.02, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        }
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for fold, (train_idx, valid_idx) in enumerate(tscv.split(x_train)):
            X_tr, X_val = x_train.iloc[train_idx], x_train.iloc[valid_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
            
            model = lgb.train(
                params,
                train_set=lgb.Dataset(X_tr, label=y_tr),
                valid_sets=[lgb.Dataset(X_val, label=y_val)],
                num_boost_round=2000,
                early_stopping_rounds=100,
                verbose_eval=False
            )
            preds = model.predict(X_val)
            scores.append(mean_absolute_error(y_val, preds))
            
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, timeout=3600)
    
    # 3. Final model training with best parameters
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'regression_l1',
        'metric': 'mae',
        'verbosity': 1,
        'deterministic': True  # Ensure reproducibility
    })
    
    final_model = lgb.train(
        best_params,
        train_set=train_set,
        valid_sets=[test_set],
        num_boost_round=2000,
        early_stopping_rounds=100,
        callbacks=[
            lgb.log_evaluation(period=100),
            lgb.record_evaluation(eval_result={}),
        ]
    )
    
    # 4. Model validation and feature analysis
    test_pred = final_model.predict(x_test)
    mae = mean_absolute_error(y_test, test_pred)
    logger.info(f"Model validation - Test MAE: {mae:.4f} | Features used: {len(final_model.feature_name())}")
    
    # Feature importance analysis
    importance = sorted(zip(final_model.feature_importance(), final_model.feature_name()), reverse=True)
    logger.info("Top predictive features:")
    for imp, name in importance[:10]:
        logger.info(f"{name}: {imp:.2f}")
    
    return final_model
