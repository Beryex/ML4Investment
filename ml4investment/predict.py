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
from ml4investment.utils.utils import (
    id_to_stock_code,
    perform_schwab_trade,
    set_random_seed,
    setup_schwab_client,
)


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

    X_predict = process_features_for_predict(
        daily_features_data, process_feature_config, predict_stock_list
    )

    X_predict = X_predict[selected_features]

    assert X_predict.index.min() == X_predict.index.max()
    logger.info(f"Predicting based on data on {X_predict.index.min()}")
    logger.info(f"Total processed samples in predict data: {X_predict.shape[0]}")
    logger.info(f"Number of features in predict data: {X_predict.shape[1]}")

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

    predictions = model.predict(X_predict, num_iteration=model.best_iteration)

    total_balance = client.account_details(account_hash, fields="positions").json()[
        "securitiesAccount"
    ]["currentBalances"]["equity"]

    results_df = pd.DataFrame(index=X_predict.index)
    results_df["stock_code"] = X_predict["stock_id"].map(id_to_stock_code)
    results_df["prediction"] = predictions
    results_df["last_price"] = 0.0
    sorted_results = results_df.sort_values("prediction", ascending=False)
    stock_quotes = client.quotes(symbols=predict_stock_list, fields="quote").json()
    stock_last_prices = {
        stock: quote["quote"]["lastPrice"] for stock, quote in stock_quotes.items()
    }
    sorted_results["last_price"] = sorted_results["stock_code"].map(stock_last_prices).astype(float)

    recommended_df = (
        sorted_results[sorted_results["prediction"] > 0]
        .head(settings.NUMBER_OF_STOCKS_TO_BUY)
        .copy()
    )

    if recommended_df.empty:
        logger.info("No stocks were recommended today (no positive predicted returns)")
        recommended_df["weight"] = 0.0
        stock_to_buy_in = {}
    else:
        logger.info(f"Give recommendation based on total investment value: ${total_balance:.2f}")
        total_pred_sum = recommended_df["prediction"].sum()
        recommended_df["weight"] = recommended_df["prediction"] / total_pred_sum

        recommended_df["invest_value"] = total_balance * recommended_df["weight"]
        recommended_df["shares_to_buy"] = (
            recommended_df["invest_value"] / recommended_df["last_price"]
        ).apply(math.floor)
        stock_to_buy_in = recommended_df.set_index("stock_code")["shares_to_buy"].to_dict()

        for _, row in recommended_df.iterrows():
            table_row = [
                row["stock_code"],
                f"{row['prediction']:+.2%}",
                f"{row['weight']:.2%}",
                f"${row['invest_value']:.2f}",
                f"${row['last_price']:.2f}",
                row["shares_to_buy"],
            ]
            predict_table.add_row(table_row, divider=True)
            wandb_table.add_data(*table_row)

    if args.verbose:
        rejected_stocks = sorted_results[
            ~sorted_results["stock_code"].isin(recommended_df["stock_code"])
        ]
        for _, row in rejected_stocks.iterrows():
            table_row = [
                row["stock_code"],
                f"{row['prediction']:+.2%}",
                "0.00%",
                "$0.00",
                f"${row['last_price']:.2f}",
                0,
            ]
            predict_table.add_row(table_row, divider=True)
            wandb_table.add_data(*table_row)

    if args.verbose or not recommended_df.empty:
        title = f"Suggested top {len(recommended_df)} stocks to buy:"
        logger.info(f"\n{predict_table.get_string(title=title)}")

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
