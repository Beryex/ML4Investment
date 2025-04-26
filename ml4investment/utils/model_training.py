import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import logging
from prettytable import PrettyTable

from ml4investment.config import settings
from ml4investment.utils.feature_engineering import id_to_stock_code

logger = logging.getLogger(__name__)


def model_training(x_train: pd.DataFrame, 
                   x_test: pd.DataFrame, 
                   y_train: pd.Series, 
                   y_test: pd.Series, 
                   target_stock_list: list,
                   categorical_features: list = None,
                   model_hyperparams: dict = None, 
                   use_mae: bool = False,
                   seed: int = 42,
                   verbose: bool = False) -> tuple[lgb.Booster, dict, dict, list, float]:
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
            'num_threads': settings.THREAD_NUM,
            
            'objective': 'regression_l1',
            'metric': 'mae',
            'verbosity': -1,
            'boosting_type': 'dart',

            'drop_rate': trial.suggest_float('drop_rate', 0.05, 0.2),
            'max_drop': trial.suggest_int('max_drop', 10, 50),
            'skip_drop': trial.suggest_float('skip_drop', 0.0, 0.6),

            'num_leaves': trial.suggest_int('num_leaves', 48, 128),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 100, 300),

            'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 1e-1, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 1e-1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),

            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),

            'max_bin': trial.suggest_int('max_bin', 128, 255),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 0.1),
            
            'seed': seed,
            'force_row_wise': True,
            'deterministic': True
        }
        
        tscv = TimeSeriesSplit(n_splits=settings.N_SPLIT)
        maes = []
        sign_accs = []
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
            maes.append(mean_absolute_error(cur_y_val, preds))
            sign_accs.append((np.sign(preds) == np.sign(cur_y_val.to_numpy())).mean())

        avg_mae = np.mean(maes)
        avg_sign_acc = np.mean(sign_accs)
            
        return avg_mae, avg_sign_acc

    if model_hyperparams is None:
        logger.info("Begin hyperparameter optimization")
        study = optuna.create_study(
            directions=["minimize", "maximize"],
            sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
        )
        study.optimize(objective, n_trials=settings.N_TRIALS, timeout=25000)
        logger.info("Hyperparameter optimization completed")

        pareto_trials = [t for t in study.best_trials if t.values[1] > settings.SIGN_ACCURACY_THRESHOLD]
        best_trial = min(pareto_trials, key=lambda t: t.values[0])
        best_params = best_trial.params.copy()
        
        best_params.update({
            'num_threads': settings.THREAD_NUM,

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
        best_params = model_hyperparams.copy()
     
    logger.info("Begin model training with optimized parameters")
    final_model = lgb.train(
        best_params,
        train_set=train_set,
        valid_sets=[test_set],
        num_boost_round=int(best_params['num_rounds'] / (1 - 1 / settings.N_SPLIT)),
        callbacks=[
            lgb.log_evaluation(False),
        ]
    )
    logger.info("Model training completed")
    
    y_pred = final_model.predict(x_test)
    overall_mae = mean_absolute_error(y_test, y_pred)
    overall_sign_accuracy = (np.sign(y_pred) == np.sign(y_test.to_numpy())).mean() * 100
    logger.info(f"Model validation - MAE: {overall_mae:.4f} | Sign Accuracy: {overall_sign_accuracy:.2f}% | Features used: {len(final_model.feature_name())}")
    
    importance = final_model.feature_importance(importance_type='gain')
    features = final_model.feature_name()
    sorted_feature_imp = sorted(zip(importance, features), reverse=True)
    if verbose:
        features_table = PrettyTable()
        features_table.field_names = ["Feature", "Importance"]
        logger.info("Top features by gain:")
        for imp, name in sorted_feature_imp:
            features_table.add_row([name, f"{imp:.2f}"], divider=True)
        logger.info(f'\n{features_table.get_string(title="Top features by gain")}')
    
    test_df = x_test.copy()
    test_df['y_true'] = y_test.values
    test_df['y_pred'] = y_pred
    test_df['sign_correct'] = (np.sign(test_df['y_true']) == np.sign(test_df['y_pred'])).astype(int)
    test_df['MAE'] = mean_absolute_error(test_df['y_true'], test_df['y_pred'])

    stock_sign_acc = (
        test_df.groupby('stock_id', observed=True)['sign_correct']
        .mean()
        .sort_values(ascending=False)
    )

    stock_MAE = (
        test_df.groupby('stock_id', observed=True)
        .apply(lambda df: mean_absolute_error(df['y_true'], df['y_pred']))
        .sort_values()
    )

    logger.info(f"Selecting predict stocks from target stocks: {target_stock_list}")
    predict_stocks = {"predict_stocks": []}
    if use_mae:
        logger.info("Using MAE for stock selection")
        for stock_id, mae in stock_MAE.items():
            stock = id_to_stock_code(stock_id)
            if mae < settings.MAE_THRESHOLD and stock in target_stock_list:
                predict_stocks["predict_stocks"].append(stock)
    else:
        logger.info("Using sign accuracy for stock selection")
        for stock_id, sign_accuracy in stock_sign_acc.items():
            stock = id_to_stock_code(stock_id)
            if sign_accuracy >= settings.SIGN_ACCURACY_THRESHOLD and stock in target_stock_list:
                predict_stocks["predict_stocks"].append(stock)
    logger.info(f"Suggested predict stocks: {predict_stocks['predict_stocks']}")
    
    return final_model, best_params, predict_stocks, sorted_feature_imp, overall_mae
