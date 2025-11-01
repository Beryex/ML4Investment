import argparse
import json
import logging
import pickle

import lightgbm as lgb
import pandas as pd
import wandb
from wandb.sdk.wandb_run import Run

from ml4investment.config.global_settings import settings
from ml4investment.utils.feature_calculating import calculate_features
from ml4investment.utils.feature_processing import process_features_for_backtest
from ml4investment.utils.logging import configure_logging, setup_wandb
from ml4investment.utils.utils import get_detailed_static_result, set_random_seed


def backtest(
    run: Run,
    train_stock_list: list,
    predict_stock_list: list,
    fetched_data_df: pd.DataFrame,
    process_feature_config: dict,
    selected_features: dict,
    model: lgb.Booster,
    seed: int,
):
    """Backtest the model performance for the given stocks for the last week"""
    logger.info(f"Start backtesting based on the given stocks: {predict_stock_list}")
    logger.info(f"Current trading time: {pd.Timestamp.now(tz='America/New_York')}")
    set_random_seed(seed)

    train_data_start_date = settings.TRAINING_DATA_START_DATE
    logger.info(f"Load input fetched data, starting from {train_data_start_date}")
    train_data_df = fetched_data_df[fetched_data_df["stock_code"].isin(train_stock_list)]
    backtest_data_df = train_data_df.loc[train_data_start_date:]

    daily_features_df = calculate_features(backtest_data_df)

    X_backtest, y_backtest = process_features_for_backtest(
        daily_features_df, process_feature_config, predict_stock_list
    )

    X_backtest = X_backtest[selected_features]

    start_date = X_backtest.index.min()
    end_date = X_backtest.index.max()
    logger.info(f"Oldest date in backtest data: {start_date}")
    logger.info(f"Newest date in backtest data: {end_date}")
    logger.info(f"Total processed samples in backtest data: {X_backtest.shape[0]}")
    logger.info(f"Number of features in backtest data: {X_backtest.shape[1]}")

    (
        backtest_day_number,
        backtest_mae,
        backtest_mse,
        backtest_sign_acc,
        backtest_precision,
        backtest_recall,
        backtest_f1,
        backtest_average_daily_gain,
        backtest_overall_gain,
        backtest_annualized_sharpe_ratio,
        backtest_max_drawdown,
        sorted_stocks,
    ) = get_detailed_static_result(
        model=model,
        X=X_backtest,
        y=y_backtest,
        predict_stock_list=predict_stock_list,
        name="Backtest",
        verbose=args.verbose,
    )

    wandb.log(
        {
            "backtest_day_number": backtest_day_number,
            "backtest_mae": backtest_mae,
            "backtest_mse": backtest_mse,
            "backtest_sign_acc": backtest_sign_acc,
            "backtest_precision": backtest_precision,
            "backtest_recall": backtest_recall,
            "backtest_f1": backtest_f1,
            "backtest_average_daily_gain": backtest_average_daily_gain,
            "backtest_overall_gain": backtest_overall_gain,
            "backtest_annualized_sharpe_ratio": backtest_annualized_sharpe_ratio,
            "backtest_max_drawdown": backtest_max_drawdown,
        }
    )

    run.finish()

    logger.info("Backtesting process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_stocks", "-ts", type=str, default="config/train_stocks.json")
    parser.add_argument("--predict_stocks", "-ps", type=str, default="data/predict_stocks.json")
    parser.add_argument(
        "--fetched_data_pth", "-fdp", type=str, default="data/fetched_data.parquet"
    )

    parser.add_argument(
        "--process_feature_config_pth",
        "-pfcp",
        type=str,
        default="data/prod_process_feature_config.pkl",
    )
    parser.add_argument("--features_pth", "-fp", type=str, default="data/prod_features.json")
    parser.add_argument("--model_pth", "-mp", type=str, default="data/prod_model.model")

    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--seed", "-s", type=int, default=settings.SEED)

    args = parser.parse_args()

    configure_logging(env="backtest", file_name="backtest.log")
    logger = logging.getLogger("ml4investment.backtest")

    train_stock_list = json.load(open(args.train_stocks, "r"))["train_stocks"]
    predict_stock_list = json.load(open(args.predict_stocks, "r"))["predict_stocks"]
    fetched_data_df = pd.read_parquet(args.fetched_data_pth)
    process_feature_config = pickle.load(open(args.process_feature_config_pth, "rb"))
    selected_features = json.load(open(args.features_pth, "r"))["features"]
    model = lgb.Booster(model_file=args.model_pth)
    seed = args.seed

    run = setup_wandb(config=vars(args))

    backtest(
        run,
        train_stock_list,
        predict_stock_list,
        fetched_data_df,
        process_feature_config,
        selected_features,
        model,
        seed,
    )
