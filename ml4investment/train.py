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
from ml4investment.utils.model_training import model_training, optimize_model_features

configure_logging(env="train", file_name="train.log")
logger = logging.getLogger("ml4investment.train")


def train(train_stock_list: list, 
          target_stock_list: list,
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
    
    optimal_model, optimal_model_hyperparams, optimal_mae, optimal_sign_accuracy, sorted_feature_imp, optimal_predict_stock_list = model_training(
        X_train, y_train, 
        categorical_features=settings.CATEGORICAL_FEATURES,
        model_hyperparams=model_hyperparams,
        target_stock_list=target_stock_list,
        optimize_predict_stocks=args.optimize_predict_stocks,
        seed=seed,
        verbose=args.verbose
    )
    optimal_features = list(X_train.columns)

    if args.optimize_model_features:
        optimal_model, optimal_model_hyperparams, optimal_features, optimal_predict_stock_list = optimize_model_features(
            X_train, y_train, 
            categorical_features=settings.CATEGORICAL_FEATURES,
            model_hyperparams=model_hyperparams,
            target_stock_list=target_stock_list,
            optimize_predict_stocks=args.optimize_predict_stocks,
            original_model=optimal_model,
            original_sorted_feature_imp=sorted_feature_imp,
            original_features=optimal_features,
            original_mae=optimal_mae,
            original_sign_accuracy=optimal_sign_accuracy,
            seed=seed,
            verbose=args.verbose
        )

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

    predict_stocks = {"predict_stocks": optimal_predict_stock_list}
    os.makedirs(os.path.dirname(args.save_predict_stocks_pth), exist_ok=True)
    with open(args.save_predict_stocks_pth, 'w') as f:
        json.dump(predict_stocks, f, indent=4)
    logger.info(f"Target stocks saved to {args.save_predict_stocks_pth}")
    
    logger.info("Training process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_stocks", type=str, default='config/train_stocks.json')
    parser.add_argument("--target_stocks", type=str, default='config/target_stocks.json')
    parser.add_argument("--fetched_data_pth", "-fdp", type=str, default='data/fetched_data.pkl')
    parser.add_argument("--optimize_model_hyperparameters", "-omhp", action='store_true', default=False)
    parser.add_argument("--model_hyperparams_pth", "-mhpp", type=str, default='data/prod_model_hyperparams.json')
    parser.add_argument("--optimize_model_features", "-omf", action='store_true', default=False)
    parser.add_argument("--features_pth", "-fp", type=str, default='data/prod_model_features.json')
    parser.add_argument("--optimize_predict_stocks", "-ops", action='store_true', default=False)

    parser.add_argument("--save_process_feature_config_pth", "-spfcp", type=str, default='data/prod_process_feature_config.pkl')
    parser.add_argument("--save_model_pth", "-smp", type=str, default='data/prod_model.model')
    parser.add_argument("--save_features_pth", "-sfp", type=str, default='data/prod_model_features.json')
    parser.add_argument("--save_model_hyperparams_pth", "-smhpp", type=str, default='data/prod_model_hyperparams.json')
    parser.add_argument("--save_predict_stocks_pth", type=str, default='config/predict_stocks.json')

    parser.add_argument("--verbose", "-v", action='store_true', default=False)
    parser.add_argument("--seed", "-s", type=int, default=42)

    args = parser.parse_args()
    
    seed = args.seed
    set_random_seed(seed)

    train_stock_list = json.load(open(args.train_stocks, 'r'))["train_stocks"]
    target_stock_list = json.load(open(args.target_stocks, 'r'))["target_stocks"]
    fetched_data = pickle.load(open(args.fetched_data_pth, 'rb'))

    train(train_stock_list, target_stock_list, fetched_data, seed, args)
