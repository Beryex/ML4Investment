import argparse
import json
import logging
import pickle

import lightgbm as lgb
import pandas as pd
import wandb
from wandb.sdk.wandb_run import Run

from ml4investment.config.global_settings import settings
from ml4investment.utils.feature_engineering import (
    calculate_features,
    process_features_for_backtest,
)
from ml4investment.utils.logging import configure_logging, setup_wandb
from ml4investment.utils.model_predicting import get_detailed_static_result
from ml4investment.utils.utils import set_random_seed

configure_logging(env="backtest", file_name="backtest.log")
logger = logging.getLogger("ml4investment.backtest")


def backtest(
    run: Run,
    train_stock_list: list,
    predict_stock_list: list,
    fetched_data: dict,
    process_feature_config: dict,
    selected_features: dict,
    model: lgb.Booster,
    seed: int,
):
    """Backtest the model performance for the given stocks for the last week"""
    logger.info(f"Start backtesting based on the given stocks: {predict_stock_list}")
    logger.info(f"Current trading time: {pd.Timestamp.now(tz='America/New_York')}")
    set_random_seed(seed)

    backtest_data = {}
    train_data_start_date = settings.TRAINING_DATA_START_DATE
    logger.info("Load input fetched data")
    for stock in train_stock_list:
        backtest_data[stock] = fetched_data[stock].loc[train_data_start_date:]

    daily_features_data = calculate_features(backtest_data)

    X_backtest_dict, y_backtest_dict, backtest_day_number = (
        process_features_for_backtest(
            daily_features_data, process_feature_config, predict_stock_list
        )
    )

    for i in range(backtest_day_number):
        for stock, data in X_backtest_dict[i].items():
            X_backtest_dict[i][stock] = data[selected_features]

    backtest_oldest_dates = {
        X_backtest.index.min() for X_backtest in X_backtest_dict[0].values()
    }
    if len(backtest_oldest_dates) != 1:
        logger.error("Oldest backtest date mismatched")
        raise ValueError("Oldest backtest date mismatched")
    backtest_oldest_date = backtest_oldest_dates.pop()

    backtest_newest_dates = {
        X_backtest.index.max()
        for X_backtest in X_backtest_dict[backtest_day_number - 1].values()
    }
    if len(backtest_newest_dates) != 1:
        logger.error("Newest backtest date mismatched")
        raise ValueError("Newest backtest date mismatched")
    backtest_newest_date = backtest_newest_dates.pop()

    logger.info(f"Oldest date in backtest data: {backtest_oldest_date}")
    logger.info(f"Newest date in backtest data: {backtest_newest_date}")

    feature_nums = {
        len(list(X_predict.columns)) for X_predict in X_backtest_dict[0].values()
    }
    if len(feature_nums) != 1:
        logger.error(f"Feature number mismatched: {feature_nums}")
        raise ValueError(f"Feature number mismatched: {feature_nums}")
    feature_num = feature_nums.pop()
    logger.info(f"Number of features: {feature_num}")

    (
        backtest_mae,
        backtest_mse,
        backtest_sign_acc,
        backtest_precision,
        backtest_recall,
        backtest_f1,
        backtest_gain,
    ) = get_detailed_static_result(
        model=model,
        X_dict=X_backtest_dict,
        y_dict=y_backtest_dict,
        predict_stock_list=predict_stock_list,
        start_date=backtest_oldest_date,
        end_date=backtest_newest_date,
        name="Backtest Overall",
        verbose=args.verbose,
    )

    wandb.log(
        {
            "backtest_mae": backtest_mae,
            "backtest_mse": backtest_mse,
            "backtest_sign_acc": backtest_sign_acc,
            "backtest_precision": backtest_precision,
            "backtest_recall": backtest_recall,
            "backtest_f1": backtest_f1,
            "backtest_gain": backtest_gain,
        }
    )

    run.finish()

    logger.info("Backtesting process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_stocks", "-ts", type=str, default="config/train_stocks.json"
    )
    parser.add_argument(
        "--predict_stocks", "-ps", type=str, default="config/predict_stocks.json"
    )
    parser.add_argument(
        "--fetched_data_pth", "-fdp", type=str, default="data/fetched_data.pkl"
    )

    parser.add_argument(
        "--process_feature_config_pth",
        "-pfcp",
        type=str,
        default="data/prod_process_feature_config.pkl",
    )
    parser.add_argument(
        "--features_pth", "-fp", type=str, default="data/prod_model_features.json"
    )
    parser.add_argument("--model_pth", "-mp", type=str, default="data/prod_model.model")

    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--seed", "-s", type=int, default=settings.SEED)

    args = parser.parse_args()

    train_stock_list = json.load(open(args.train_stocks, "r"))["train_stocks"]
    predict_stock_list = json.load(open(args.predict_stocks, "r"))["predict_stocks"]
    fetched_data = pickle.load(open(args.fetched_data_pth, "rb"))
    process_feature_config = pickle.load(open(args.process_feature_config_pth, "rb"))
    selected_features = json.load(open(args.features_pth, "r"))["features"]
    model = lgb.Booster(model_file=args.model_pth)
    seed = args.seed

    run = setup_wandb(config=vars(args))

    backtest(
        run,
        train_stock_list,
        predict_stock_list,
        fetched_data,
        process_feature_config,
        selected_features,
        model,
        seed,
    )
