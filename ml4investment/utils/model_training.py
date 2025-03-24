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
                   model_hyperparams: dict = None, 
                   seed: int = 42) -> lgb.Booster:
    """ Time series modeling training pipeline """
    train_set = lgb.Dataset(x_train, 
                            label=y_train, 
                            categorical_feature=categorical_features, 
                            free_raw_data=False)
    test_set = lgb.Dataset(x_test, 
                           label=y_test, 
                           categorical_feature=categorical_features,
                           reference=train_set, 
                           free_raw_data=False)
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            
            'objective': 'regression_l1',
            'metric': 'mae',
            'verbosity': -1,
            'boosting_type': 'dart',

            'drop_rate': trial.suggest_float('drop_rate', 0.05, 0.15),
            'max_drop': trial.suggest_int('max_drop', 10, 25),
            'skip_drop': trial.suggest_float('skip_drop', 0.3, 0.7),

            'num_leaves': trial.suggest_int('num_leaves', 40, 96),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 150),

            'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 1e-1, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 1e-1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),

            'bagging_freq': trial.suggest_int('bagging_freq', 3, 7),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.9),

            'max_bin': trial.suggest_int('max_bin', 128, 255),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 0.1),
            
            'seed': seed,
            'force_row_wise': True,
            'deterministic': True
        }
        
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
                num_boost_round=trial.suggest_int('num_rounds', 800, 2000),
                callbacks=[
                    lgb.log_evaluation(False),
                ]
            )
            preds = model.predict(cur_x_val)
            scores.append(mean_absolute_error(cur_y_val, preds))
            
        return np.mean(scores)

    if model_hyperparams is None:
        logger.info("Begin hyperparameter optimization")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10, timeout=7200)
        logger.info("Hyperparameter optimization completed")

        best_params = study.best_params.copy()
        best_params.update({
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,

            'objective': 'regression_l1',
            'metric': 'mae',
            'verbosity': -1,
            'boosting_type': 'dart',

            'seed': seed,
            'force_row_wise': True,
            'deterministic': True
        })
        logger.info(f"Optimized model parameters: {best_params}")

    else:
        logger.info("Load input model hyperparameter")
        best_params = model_hyperparams.copy()
     
    logger.info("Begin model training with optimized parameters")
    final_model = lgb.train(
        best_params,
        train_set=train_set,
        valid_sets=[test_set],
        num_boost_round=int(best_params['num_rounds']),
        callbacks=[
            lgb.log_evaluation(False),
        ]
    )
    logger.info("Model training completed")
    
    y_pred = final_model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    sign_accuracy = (np.sign(y_pred) == np.sign(y_test.to_numpy())).mean() * 100
    logger.info(f"Model validation - MAE: {mae:.4f} | Sign Accuracy: {sign_accuracy:.2f}% | Features used: {len(final_model.feature_name())}")
    
    importance = final_model.feature_importance(importance_type='gain')
    features = final_model.feature_name()
    sorted_imp = sorted(zip(importance, features), reverse=True)
    logger.info("Top features by gain:")
    for imp, name in sorted_imp[:10]:
        logger.info(f"{name}: {imp:.2f}")
    
    return final_model, best_params
