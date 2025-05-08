import argparse
import logging
import pandas as pd
import json
import os
import pickle
from argparse import Namespace

from ml4investment.config import settings
from ml4investment.utils.utils import set_random_seed
from ml4investment.utils.logging import configure_logging
from ml4investment.utils.feature_engineering import calculate_features, process_features_for_train
from ml4investment.utils.model_training import model_training

configure_logging(env="prod", file_name="train.log")
logger = logging.getLogger("ml4investment.train")


def train(train_stock_list: list, 
          fetched_data: dict, 
          seed: int,
          args: Namespace):
    """ Train model based on the given stocks """
    logger.info(f"Start training model given stocks: {train_stock_list}")
    logger.info(f"Current trading time: {pd.Timestamp.now(tz='America/New_York')}")

    train_data = {}
    train_data_start_date = settings.TRAINING_DATA_START_DATE
    train_data_end_date = settings.TRAINING_DATA_END_DATE
    logger.info(f"Load input fetched data, starting from {train_data_start_date} to {train_data_end_date}")
    for stock in train_stock_list:
        train_data[stock] = fetched_data[stock].loc[train_data_start_date:train_data_end_date]

    daily_features_data = calculate_features(train_data)

    X_train, y_train, process_feature_config = process_features_for_train(daily_features_data, test_number=settings.TEST_DAY_NUMBER, seed=seed)
    parent_dir = os.path.dirname(args.save_process_feature_config_pth)
    os.makedirs(parent_dir, exist_ok=True)
    with open(args.save_process_feature_config_pth, 'wb') as f:
        pickle.dump(process_feature_config, f)
    logger.info(f"Processing features configuration saved to {args.save_process_feature_config_pth}")

    if not args.optimize_model_features:
        logger.info(f"Load input selected features")
        selected_features = json.load(open(args.features_pth, 'r'))["features"]
        X_train = X_train[selected_features]

    logger.info(f"Oldest date in training data: {X_train.index.min()}")
    logger.info(f"Newest date in training data: {X_train.index.max()}")
    logger.info(f"Total processed samples: {X_train.shape[0]}")
    logger.info(f"Number of features: {X_train.shape[1]}")

    if args.optimize_model_hyperparameters:
        logger.info("Optimize model hyperparameters from the scratch")
        model_hyperparams = None
    else:
        logger.info("Load input model hyperparameter")
        model_hyperparams = json.load(open(args.model_hyperparams_pth, 'r'))
    
    optimal_model, optimal_model_hyperparams, optimal_mae, optimal_sign_accuracy, sorted_feature_imp, optimal_target_stock_list = model_training(
        X_train, y_train, 
        categorical_features=['stock_id', 'stock_sector'],
        model_hyperparams=model_hyperparams,
        optimize_target_stocks=args.optimize_target_stocks,
        seed=seed,
        verbose=args.verbose
    )
    optimal_features = list(X_train.columns)

    if args.optimize_model_features:
        logger.info("Feature Optimization begins...")
        feature_ranking = list(reversed([f for _, f in sorted_feature_imp]))

        original_feature_number = len(optimal_features)
        original_mae = optimal_mae
        original_sign_accuracy = optimal_sign_accuracy

        for feature_to_remove in feature_ranking:
            if feature_to_remove == 'stock_id':
                logger.info("Skipping 'stock_id' feature removal")
                continue
            if feature_to_remove == 'stock_sector':
                logger.info("Skipping 'stock_sector' feature removal")
                continue
            
            candidate_features = [f for f in optimal_features if f != feature_to_remove]

            X_train_tmp = X_train[candidate_features]

            categorical_features_tmp = []
            if 'stock_id' in X_train_tmp.columns:
                categorical_features_tmp.append('stock_id')
            if 'stock_sector' in X_train_tmp.columns:
                categorical_features_tmp.append('stock_sector')
            
            model_tmp, model_hyperparams_tmp, mae_tmp, sign_accuracy_tmp, sorted_feature_imp_tmp, target_stock_list_tmp = model_training(
                X_train_tmp, y_train, 
                categorical_features=categorical_features_tmp,
                model_hyperparams=model_hyperparams,
                optimize_target_stocks=args.optimize_target_stocks,
                seed=seed,
                verbose=args.verbose
            )
            
            if sign_accuracy_tmp >= optimal_sign_accuracy:
                logger.info(f"Removing '{feature_to_remove}' improved or kept performance.")
                optimal_mae = mae_tmp
                optimal_sign_accuracy = sign_accuracy_tmp
                optimal_model = model_tmp
                optimal_model_hyperparams = model_hyperparams_tmp
                optimal_features = candidate_features
                optimal_target_stock_list = target_stock_list_tmp
                if args.verbose:
                    logger.info(f"Updated optimal features: {', '.join(optimal_features)}")
                
            else:
                logger.info(f"Removing '{feature_to_remove}' degraded performance.")
        
        logger.info(f"Final selected {len(optimal_features)} features after RFE, select ratio: {len(optimal_features) / original_feature_number:.2f}")
        if args.verbose:
            logger.info(f"Optimal features: {', '.join(optimal_features)}")
        logger.info(f"Final MAE after feature selection: {optimal_mae:.6f}, improvement: {original_mae - optimal_mae:.6f}")
        logger.info(f"Final sign accuracy after feature selection: {optimal_sign_accuracy*100:.2f}%, improvement: {(optimal_sign_accuracy - original_sign_accuracy)*100:.2f}%")

    os.makedirs(os.path.dirname(args.save_model_pth), exist_ok=True)
    optimal_model.save_model(args.save_model_pth)
    logger.info(f"Model saved to {args.save_model_pth}")

    if args.optimize_model_features:
        os.makedirs(os.path.dirname(args.save_features_pth), exist_ok=True)
        optimal_features = {"features": optimal_features}
        with open(args.save_features_pth, 'w') as f:
            json.dump(optimal_features, f, indent=4)
        logger.info(f"Optimized features saved to {args.save_features_pth}")

    if args.optimize_model_hyperparameters:
        os.makedirs(os.path.dirname(args.save_model_hyperparams_pth), exist_ok=True)
        with open(args.save_model_hyperparams_pth, 'w') as f:
            json.dump(optimal_model_hyperparams, f, indent=4)
        logger.info(f"Optimized model hyperparameters saved to {args.save_model_hyperparams_pth}")

    target_stocks = {"target_stocks": optimal_target_stock_list}
    with open(args.save_target_stocks_pth, 'w') as f:
        json.dump(target_stocks, f, indent=4)
    logger.info(f"Target stocks saved to {args.save_target_stocks_pth}")
    
    logger.info("Training process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_stocks", type=str, default='config/train_stocks.json')
    parser.add_argument("--fetched_data_pth", "-fdp", type=str, default='data/fetched_data.pkl')
    parser.add_argument("--optimize_model_hyperparameters", "-omhp", action='store_true', default=False)
    parser.add_argument("--model_hyperparams_pth", "-mhpp", type=str, default='data/prod_model_hyperparams.json')
    parser.add_argument("--optimize_model_features", "-omf", action='store_true', default=False)
    parser.add_argument("--features_pth", "-fp", type=str, default='data/prod_model_features.json')
    parser.add_argument("--optimize_target_stocks", "-ots", action='store_true', default=False)

    parser.add_argument("--save_process_feature_config_pth", "-spfcp", type=str, default='data/prod_process_feature_config.pkl')
    parser.add_argument("--save_model_pth", "-smp", type=str, default='data/prod_model.model')
    parser.add_argument("--save_features_pth", "-sfp", type=str, default='data/prod_model_features.json')
    parser.add_argument("--save_model_hyperparams_pth", "-smhpp", type=str, default='data/prod_model_hyperparams.json')
    parser.add_argument("--save_target_stocks_pth", type=str, default='config/target_stocks.json')

    parser.add_argument("--verbose", "-v", action='store_true', default=False)
    parser.add_argument("--seed", "-s", type=int, default=42)

    args = parser.parse_args()
    
    seed = args.seed
    set_random_seed(seed)

    train_stock_list = json.load(open(args.train_stocks, 'r'))["train_stocks"]
    fetched_data = pickle.load(open(args.fetched_data_pth, 'rb'))

    train(train_stock_list, fetched_data, seed, args)
