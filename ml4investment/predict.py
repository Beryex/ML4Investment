import argparse
import json
import logging
import math
import pickle

import lightgbm as lgb
import pandas as pd
import wandb
from prettytable import PrettyTable
from wandb.sdk.wandb_run import Run

from ml4investment.config.global_settings import settings
from ml4investment.utils.feature_engineering import (
    calculate_features,
    process_features_for_predict,
)
from ml4investment.utils.logging import configure_logging, setup_wandb
from ml4investment.utils.model_predicting import (
    get_predict_top_stocks_and_weights,
    model_predict,
    perform_schwab_trade,
)
from ml4investment.utils.utils import set_random_seed, setup_schwab_client


def predict(
    run: Run,
    train_stock_list: list,
    predict_stock_list: list,
    fetched_data: dict,
    process_feature_config: dict,
    selected_features: dict,
    model: lgb.Booster,
    seed: int,
):
    """Predict the optimal stock with the highest price change for the given stocks"""
    logger.info(f"Start predict the given stocks: {predict_stock_list}")
    logger.info(f"Current trading time: {pd.Timestamp.now(tz='America/New_York')}")
    set_random_seed(seed)

    predict_data = {}
    train_data_start_date = settings.TRAINING_DATA_START_DATE
    for stock in train_stock_list:
        predict_data[stock] = fetched_data[stock].loc[train_data_start_date:]
    logger.info("Load input fetched data")

    daily_features_data = calculate_features(predict_data)

    X_predict_dict = process_features_for_predict(daily_features_data, process_feature_config)

    for stock, data in X_predict_dict.items():
        X_predict_dict[stock] = data[selected_features]

    predict_dates = {str(X_predict.index[0]) for X_predict in X_predict_dict.values()}
    if len(predict_dates) != 1:
        logger.error(f"Predict date mismatched: {predict_dates}")
        raise ValueError(f"Predict date mismatched: {predict_dates}")
    predict_date = predict_dates.pop()
    logger.info(f"Predicting based on data on {predict_date}")

    feature_nums = {len(list(X_predict.columns)) for X_predict in X_predict_dict.values()}
    if len(feature_nums) != 1:
        logger.error(f"Feature number mismatched: {feature_nums}")
        raise ValueError(f"Feature number mismatched: {feature_nums}")
    feature_num = feature_nums.pop()
    logger.info(f"Number of features: {feature_num}")

    predictions = {}
    for stock in predict_stock_list:
        predictions[stock] = model_predict(model, X_predict_dict[stock])

    field_names = [
        "Stock",
        "Open Price Change Predict",
        "Recommended Weight",
        "Recommended Investment Value",
        "Last Price",
        "Recommended Buy in number",
    ]
    predict_table = PrettyTable()
    predict_table.field_names = field_names
    wandb_table = wandb.Table(columns=field_names)

    sorted_stock_gain_prediction = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    predict_top_stock_and_weights_list = get_predict_top_stocks_and_weights(
        sorted_stock_gain_prediction
    )
    actual_number_selected = len(predict_top_stock_and_weights_list)

    total_balance = client.account_details(account_hash, fields="positions").json()[
        "securitiesAccount"
    ]["currentBalances"]["equity"]

    stock_quotes = client.quotes(symbols=predict_stock_list, fields="quote").json()
    stock_last_prices = {
        stock: quote["quote"]["lastPrice"] for stock, quote in stock_quotes.items()
    }

    stock_to_buy_in: dict[str, int] = {}
    if actual_number_selected == 0:
        logger.info("No stocks were recommended today (no positive predicted returns)")
    else:
        logger.info(f"Give recommendation based on total investment value: ${total_balance:.2f}")

        for stock, weight in predict_top_stock_and_weights_list:
            recommended_investment_value = total_balance * weight
            recommended_buy_in_number = math.floor(
                recommended_investment_value / stock_last_prices[stock]
            )
            stock_to_buy_in[stock] = recommended_buy_in_number

            row = [
                stock,
                f"{predictions[stock]:+.2%}",
                f"{weight:.2%}",
                f"${recommended_investment_value:.2f}",
                f"${stock_last_prices[stock]:.2f}",
                recommended_buy_in_number,
            ]
            predict_table.add_row(row, divider=True)
            wandb_table.add_data(*row)

    if args.verbose:
        for stock, pred in sorted_stock_gain_prediction[actual_number_selected:]:
            row = [
                stock,
                f"{pred:+.2%}",
                "0",
                "0",
                f"${stock_last_prices[stock]:.2f}",
                0,
            ]
            predict_table.add_row(row, divider=True)
            wandb_table.add_data(*row)

    if args.verbose or actual_number_selected > 0:
        logger.info(
            f"\n{
                predict_table.get_string(
                    title=f'Suggested top {actual_number_selected} stocks to buy:'
                )
            }"
        )

    wandb.log({"daily_predictions": wandb_table})

    if args.perform_trading:
        perform_schwab_trade(client, account_hash, stock_to_buy_in)

    run.finish()

    logger.info("Prediction process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_stocks", "-ts", type=str, default="config/train_stocks.json")
    parser.add_argument("--predict_stocks", "-ps", type=str, default="config/predict_stocks.json")
    parser.add_argument("--fetched_data_pth", "-fdp", type=str, default="data/fetched_data.pkl")

    parser.add_argument(
        "--process_feature_config_pth",
        "-pfcp",
        type=str,
        default="data/prod_process_feature_config.pkl",
    )
    parser.add_argument("--features_pth", "-fp", type=str, default="data/prod_features.json")
    parser.add_argument("--model_pth", "-mp", type=str, default="data/prod_model.model")

    parser.add_argument("--perform_trading", "-pt", action="store_true", default=False)

    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--seed", "-s", type=int, default=settings.SEED)

    args = parser.parse_args()

    configure_logging(env="predict", file_name="predict.log")
    logger = logging.getLogger("ml4investment.predict")

    train_stock_list = json.load(open(args.train_stocks, "r"))["train_stocks"]
    predict_stock_list = json.load(open(args.predict_stocks, "r"))["predict_stocks"]
    fetched_data = pickle.load(open(args.fetched_data_pth, "rb"))
    process_feature_config = pickle.load(open(args.process_feature_config_pth, "rb"))
    selected_features = json.load(open(args.features_pth, "r"))["features"]
    model = lgb.Booster(model_file=args.model_pth)
    seed = args.seed

    run = setup_wandb(config=vars(args))

    client, account_hash = setup_schwab_client()

    predict(
        run,
        train_stock_list,
        predict_stock_list,
        fetched_data,
        process_feature_config,
        selected_features,
        model,
        seed,
    )
