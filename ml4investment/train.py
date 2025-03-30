import argparse
import logging
import pandas as pd
import json
import os
import pickle
from argparse import Namespace

from ml4investment.config import settings
from ml4investment.utils.seed import set_random_seed
from ml4investment.utils.data_loader import fetch_trading_day_data, merge_fetched_data, get_target_stocks
from ml4investment.utils.logging import configure_logging
from ml4investment.utils.feature_engineering import calculate_features, process_features_for_train
from ml4investment.utils.model_training import model_training

configure_logging(env="prod", file_name="train.log")
logger = logging.getLogger("ml4investment.train")


def train(train_stock_list: list, 
          target_stock_list: list,
          fetched_data: dict, 
          model_hyperparams: dict, 
          seed: int,
          args: Namespace):
    """ Train model based on the given stocks """
    logger.info(f"Start training model given stocks: {train_stock_list}")
    logger.info(f"Current trading time: {pd.Timestamp.now(tz='America/New_York')}")

    if target_stock_list is None:
        target_stock_list = get_target_stocks(train_stock_list)
        target_stocks = {"target_stocks": target_stock_list}
        with open(args.save_target_stocks_pth, 'w') as f:
            json.dump(target_stocks, f)
        logger.info(f"Target stocks saved to {args.save_target_stocks_pth}")
    else:
        logger.info(f"Load input target stocks")

    if fetched_data is None:
        fetched_data = fetch_trading_day_data(train_stock_list, period = settings.TRAIN_DAYS)

        """ Fetch new data and merge with previous saved data """
        if os.path.exists(args.save_fetched_data_pth):
            logger.info(f"Loading previously saved data from {args.save_fetched_data_pth}")
            with open(args.save_fetched_data_pth, 'rb') as f:
                existing_data = pickle.load(f)
        else:
            logger.info("No previous data found. Starting fresh.")
            existing_data = {}
        
        merged_data = merge_fetched_data(existing_data, fetched_data)
        with open(args.save_fetched_data_pth, 'wb') as f:
            pickle.dump(merged_data, f)
        logger.info(f"Fetched data saved to {args.save_fetched_data_pth}")
    else:
        logger.info(f"Load input fetched data")

    daily_features_data = calculate_features(fetched_data)

    X_train, X_test, y_train, y_test, process_features_config_data = process_features_for_train(daily_features_data, test_number=settings.TEST_DAY_NUMBER)
    parent_dir = os.path.dirname(args.save_process_feature_config_pth)
    os.makedirs(parent_dir, exist_ok=True)
    with open(args.save_process_feature_config_pth, 'wb') as f:
        pickle.dump(process_features_config_data, f)
    logger.info(f"Processing features configuration saved to {args.save_process_feature_config_pth}")

    logger.info(f"Oldest date in training data: {X_train.index.min()}")
    logger.info(f"Newest date in training data: {X_train.index.max()}")
    logger.info(f"Oldest date in testing data: {X_test.index.min()}")
    logger.info(f"Newest date in testing data: {X_test.index.max()}")
    logger.info(f"Total processed samples: {X_train.shape[0] + X_test.shape[0]}")
    logger.info(f"Number of features: {X_train.shape[1]}")

    if model_hyperparams is None:
        optimize = True
    else:
        logger.info("Load input model hyperparameter")
        optimize = False

    model, model_hyperparams, predict_stocks = model_training(
        X_train, X_test, y_train, y_test, 
        target_stock_list,
        categorical_features=['stock_id'],
        model_hyperparams=model_hyperparams,
        seed=seed
    )

    if optimize:
        os.makedirs(os.path.dirname(args.save_model_hyperparams_pth), exist_ok=True)
        with open(args.save_model_hyperparams_pth, 'w') as f:
            json.dump(model_hyperparams, f, indent=4)
        logger.info(f"Optimized model hyperparameters saved to {args.save_model_hyperparams_pth}")

    model.save_model(args.save_model_pth)
    logger.info(f"Model saved to {args.save_model_pth}")

    with open(args.save_predict_stocks_pth, 'w') as f:
        json.dump(predict_stocks, f)
    logger.info(f"Predict stocks saved to {args.save_predict_stocks_pth}")

    logger.info("Training process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_stocks", type=str, default='config/train_stocks.json')
    parser.add_argument("--get_target_stocks_from_scratch", "-gtsfs", action='store_true', default=False)
    parser.add_argument("--target_stocks", type=str, default='config/target_stocks.json')
    parser.add_argument("--fetch_data_from_scratch", "-fdfs", action='store_true', default=False)
    parser.add_argument("--fetched_data_pth", "-fdp", type=str, default='data/fetched_data.pkl')
    parser.add_argument("--optimize_from_scratch", "-ofs", action='store_true', default=False)
    parser.add_argument("--model_hyperparams_pth", "-mhpp", type=str, default='config/prod_model_hyperparams_optimal.json')

    parser.add_argument("--save_target_stocks_pth", type=str, default='config/target_stocks.json')
    parser.add_argument("--save_predict_stocks_pth", type=str, default='config/predict_stocks.json')
    parser.add_argument("--save_fetched_data_pth", "-sfdp", type=str, default='data/fetched_data.pkl')
    parser.add_argument("--save_process_feature_config_pth", "-spfcp", type=str, default='data/prod_process_feature_config.pkl')
    parser.add_argument("--save_model_pth", "-smp", type=str, default='data/prod_model.model')
    parser.add_argument("--save_model_hyperparams_pth", "-smhpp", type=str, default='config/prod_model_hyperparams.json')

    parser.add_argument("--seed", "-s", type=int, default=42)

    args = parser.parse_args()
    
    seed = args.seed
    set_random_seed(seed)

    train_stock_list = json.load(open(args.train_stocks, 'r'))["train_stocks"]
    target_stock_list = None if args.get_target_stocks_from_scratch else json.load(open(args.target_stocks, 'rb'))["target_stocks"]
    fetched_data = None if args.fetch_data_from_scratch else pickle.load(open(args.fetched_data_pth, 'rb'))
    model_hyperparams = None if args.optimize_from_scratch else json.load(open(args.model_hyperparams_pth, 'r'))

    train(train_stock_list, target_stock_list, fetched_data, model_hyperparams, seed, args)
