import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def model_training(x_train: pd.DataFrame, 
                   x_test: pd.DataFrame, 
                   y_train: pd.Series, 
                   y_test: pd.Series, 
                   categorical_features: list = None,
                   best_params: dict = None) -> lgb.Booster:
    """ Time series modeling training pipeline """
    # 1. Data preparation for LightGBM
    if 'stock_id' in x_train.columns:
        assert x_train['stock_id'].dtype.name == 'category'

    train_set = lgb.Dataset(x_train, 
                            label=y_train, 
                            categorical_feature=categorical_features, 
                            free_raw_data=False)
    test_set = lgb.Dataset(x_test, 
                           label=y_test, 
                           categorical_feature=categorical_features,
                           reference=train_set, 
                           free_raw_data=False)
    
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
            'num_leaves': trial.suggest_int('num_leaves', 32, 128),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200),
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
                train_set=lgb.Dataset(cur_x_train, 
                                      label=cur_y_train,
                                      categorical_feature=categorical_features),
                valid_sets=[lgb.Dataset(cur_x_val, 
                                        label=cur_y_val,
                                        categorical_feature=categorical_features)],
                num_boost_round=trial.suggest_int('num_rounds', 800, 2500),
                callbacks=[
                    lgb.log_evaluation(False),
                ]
            )
            preds = model.predict(cur_x_val)
            scores.append(mean_absolute_error(cur_y_val, preds))
            
        return np.mean(scores)

    if best_params is None:
        logger.info("Begin hyperparameter optimization")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100, timeout=7200)
        logger.info("Hyperparameter optimization completed")

        best_params = study.best_params.copy()
        best_params.update({
            'verbosity': 1,
            'force_row_wise': True,  # Ensure reproducibility
            'xgboost_dart_mode': True
        })
    else:
        logger.info("Load input hyperparameter")
        best_params = best_params.copy()
        best_params.setdefault('verbosity', 1)
        best_params.setdefault('force_row_wise', True)
        best_params.setdefault('xgboost_dart_mode', True)
    
    # 3. Final model training with best parameters    
    logger.info("Begin model training with optimized parameters")
    final_model = lgb.train(
        best_params,
        train_set=train_set,
        valid_sets=[test_set],
        num_boost_round=int(best_params['num_rounds'] * 1.5),
        callbacks=[
            lgb.log_evaluation(period=100),
            lgb.record_evaluation(eval_result={}),
        ]
    )
    logger.info("Model training completed")
    
    # 4. Model validation and feature analysis
    y_pred = final_model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    sign_accuracy = (np.sign(y_pred) == np.sign(y_test.to_numpy())).mean() * 100
    logger.info(f"Model validation - MAE: {mae:.4f} | Sign Accuracy: {sign_accuracy:.2f}% | Features used: {len(final_model.feature_name())}")
    
    # Feature importance analysis
    importance = final_model.feature_importance(importance_type='gain')
    features = final_model.feature_name()
    sorted_imp = sorted(zip(importance, features), reverse=True)
    logger.info("Top features by gain:")
    for imp, name in sorted_imp[:10]:
        logger.info(f"{name}: {imp:.2f}")
    
    return final_model
