import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import logging
from prettytable import PrettyTable
from collections import defaultdict

from ml4investment.config import settings
from ml4investment.utils.utils import id_to_stock_code

logger = logging.getLogger(__name__)


def model_training(x_train: pd.DataFrame, 
                   y_train: pd.Series, 
                   categorical_features: list = None,
                   model_hyperparams: dict = None, 
                   target_stock_list: list = None,
                   optimize_predict_stocks: bool = True,
                   seed: int = 42,
                   verbose: bool = False) -> tuple[lgb.Booster, dict, float, float, dict, list]:
    """ Time series modeling training pipeline """
    train_set = lgb.Dataset(x_train, 
                            label=y_train, 
                            categorical_feature=categorical_features, 
                            free_raw_data=False)
    
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
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),

            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),

            'max_bin': trial.suggest_int('max_bin', 128, 255),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 0.1),
            
            'seed': seed,
            'force_row_wise': True,
            'deterministic': True
        }
        
        avg_mae, avg_sign_accuracy, avg_stock_maes, avg_stock_sign_accs = cross_validate(params, num_rounds=trial.suggest_int('num_rounds', 800, 2400))
        
        trial.set_user_attr("stock_maes", avg_stock_maes)
        trial.set_user_attr("stock_sign_accs", avg_stock_sign_accs)
        
        return avg_mae, avg_sign_accuracy
    
    def cross_validate(params: dict, num_rounds: int) -> tuple[float, float, dict, dict]:
        tscv = TimeSeriesSplit(n_splits=settings.N_SPLIT)
        target_preds_all = []
        target_y_val_all = []
        stock_maes_collect = defaultdict(list)
        stock_sign_accs_collect = defaultdict(list)

        for fold, (train_idx, valid_idx) in enumerate(tscv.split(x_train)):
            cur_x_train, cur_x_val = x_train.iloc[train_idx], x_train.iloc[valid_idx]
            cur_y_train, cur_y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
            assert 'stock_id' in cur_x_train.columns
            assert 'stock_id' in cur_x_val.columns

            model = lgb.train(
                params,
                train_set=lgb.Dataset(cur_x_train,
                                      label=cur_y_train,
                                      categorical_feature=categorical_features),
                valid_sets=[lgb.Dataset(cur_x_val,
                                        label=cur_y_val,
                                        categorical_feature=categorical_features)],
                num_boost_round=num_rounds,
                callbacks=[
                    lgb.log_evaluation(False),
                ]
            )
            preds = model.predict(cur_x_val)

            unique_stock_ids_in_fold = cur_x_val['stock_id'].unique()
            for stock_id_val in unique_stock_ids_in_fold:
                stock_code_val = id_to_stock_code(stock_id_val)
                if stock_code_val in target_stock_list:
                    stock_mask = (cur_x_val['stock_id'] == stock_id_val)
                    stock_preds = preds[stock_mask]
                    stock_y_val_numpy = cur_y_val[stock_mask].to_numpy()

                    target_preds_all.extend(stock_preds)
                    target_y_val_all.extend(stock_y_val_numpy)

                    stock_mae_fold_specific = mean_absolute_error(stock_y_val_numpy, stock_preds)
                    stock_sign_acc_fold_specific = (np.sign(stock_preds) == np.sign(stock_y_val_numpy)).mean()

                    stock_maes_collect[stock_code_val].append(stock_mae_fold_specific)
                    stock_sign_accs_collect[stock_code_val].append(stock_sign_acc_fold_specific)

        stock_maes_dict = {
            stock_code: np.mean(m_list) for stock_code, m_list in stock_maes_collect.items()
        }
        stock_sign_accs_dict = {
            stock_code: np.mean(sa_list) for stock_code, sa_list in stock_sign_accs_collect.items()
        }

        avg_mae_overall = mean_absolute_error(target_y_val_all, target_preds_all) if target_y_val_all else 0.0
        avg_sign_acc_overall = (np.sign(target_preds_all) == np.sign(np.array(target_y_val_all))).mean() if target_y_val_all else 0.0

        return avg_mae_overall, avg_sign_acc_overall, stock_maes_dict, stock_sign_accs_dict

    if model_hyperparams is None:
        logger.info("Begin hyperparameter optimization")
        study = optuna.create_study(
            directions=["minimize", "maximize"],
            sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
        )
        study.optimize(objective, n_trials=settings.HYPERPARAMETER_SEARCH_LIMIT, timeout=1000000000)
        logger.info("Hyperparameter optimization completed")

        pareto_trials = [t for t in study.best_trials if t.values[0] < settings.MAE_THRESHOLD]
        best_trial = max(pareto_trials, key=lambda t: t.values[1])
        best_params = best_trial.params.copy()
        best_mae = best_trial.values[0]
        best_sign_accuracy = best_trial.values[1]
        best_stock_maes = best_trial.user_attrs.get("stock_maes", {})
        best_stock_sign_accs = best_trial.user_attrs.get("stock_sign_accs", {})
        
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
        logger.info(f"  Parameters: {best_params}")
        logger.info(f"  Metrics:")
        logger.info(f"    MAE (avg over folds): {best_mae:.4f}")
        logger.info(f"    Sign Accuracy (avg over folds): {best_sign_accuracy:.2f}%")

    else:
        best_params = model_hyperparams.copy()
        logger.info(f"Begin validation with provided hyperparameters")
        best_mae, best_sign_accuracy, best_stock_maes, best_stock_sign_accs = cross_validate(best_params, num_rounds=best_params['num_rounds'])
        logger.info(f"Validation completed")
        logger.info(f"Validation Overall Metrics - MAE: {best_mae:.4f} | Sign Accuracy: {best_sign_accuracy*100:.2f}%")
    
    logger.info("Begin model training with optimized parameters")
    final_model = lgb.train(
        best_params,
        train_set=train_set,
        valid_sets=[],
        num_boost_round=int(best_params['num_rounds'] / (1 - 1 / settings.N_SPLIT)),
        callbacks=[
            lgb.log_evaluation(False),
        ]
    )
    logger.info("Model training completed")

    if optimize_predict_stocks:
        logger.info("Begin predict stocks optimization")
        logger.info(f"Using sign accuracy as the predict stocks optimization metric with target number {settings.PREDICT_STOCK_NUMBER}")
        best_stock_sign_accs = sorted(best_stock_sign_accs.items(), key=lambda item: item[1], reverse=True)
        predict_stock_list = [stock for stock, acc in best_stock_sign_accs[:settings.PREDICT_STOCK_NUMBER]]
    else:
        logger.info("No predict stocks optimization. Using all target stocks as predict stocks")
        predict_stock_list = target_stock_list
    
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
    
    return final_model, best_params, best_mae, best_sign_accuracy, sorted_feature_imp, predict_stock_list
