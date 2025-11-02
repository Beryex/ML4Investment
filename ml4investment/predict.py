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
from ml4investment.utils.feature_calculating import calculate_features
from ml4investment.utils.feature_processing import process_features_for_predict
from ml4investment.utils.logging import configure_logging, setup_wandb
from ml4investment.utils.model_predicting import get_stocks_portfolio
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
    fetched_data_df: pd.DataFrame,
    process_feature_config: dict,
    selected_features: dict,
    model: lgb.Booster,
    seed: int,
):
    """Predict the optimal stock with the highest price change for the given stocks"""
    logger.info(f"Start predict the given stocks: {predict_stock_list}")
    logger.info(f"Current trading time: {pd.Timestamp.now(tz='America/New_York')}")
    set_random_seed(seed)

    train_data_start_date = settings.TRAINING_DATA_START_DATE
    logger.info(f"Load input fetched data, starting from {train_data_start_date}")
    train_data_df = fetched_data_df[fetched_data_df["stock_code"].isin(train_stock_list)]
    predict_data_df = train_data_df.loc[train_data_start_date:]

    daily_features_data = calculate_features(predict_data_df)

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
        "Recommended Number",
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
    sorted_results["last_price"] = (
        sorted_results["stock_code"].map(stock_last_prices).astype(float)
    )

    logger.info(f"Selecting stocks using strategy: {settings.STOCK_SELECTION_STRATEGY}")
    recommended_df = get_stocks_portfolio(sorted_results)

    if recommended_df.empty:
        logger.info("No stocks were recommended today (strategy produced no candidates)")
        stock_to_execute = {}
    else:
        logger.info(f"Give recommendation based on total investment value: ${total_balance:.2f}")

        recommended_df["invest_value"] = total_balance * recommended_df["weight"]
        recommended_df["shares_to_execute"] = (
            recommended_df["invest_value"] / recommended_df["last_price"]
        ).apply(math.floor)

        stock_to_execute = (
            recommended_df.set_index("stock_code")[["shares_to_execute", "action"]]
            .to_dict("index")
        )

        for _, row in recommended_df.iterrows():
            table_row = [
                f"{row['stock_code']} ({row['action']})",
                f"{row['prediction']:+.2%}",
                f"{row['weight']:.2%}",
                f"${row['invest_value']:.2f}",
                f"${row['last_price']:.2f}",
                row["shares_to_execute"],
            ]
            predict_table.add_row(table_row, divider=True)
            wandb_table.add_data(*table_row)

    if args.verbose:
        rejected_stocks = sorted_results[
            ~sorted_results["stock_code"].isin(recommended_df["stock_code"])
        ]
        for _, row in rejected_stocks.iterrows():
            table_row = [
                f"{row['stock_code']} (SKIP)",
                f"{row['prediction']:+.2%}",
                "0.00%",
                "$0.00",
                f"${row['last_price']:.2f}",
                0,
            ]
            predict_table.add_row(table_row, divider=True)
            wandb_table.add_data(*table_row)

    if args.verbose or not recommended_df.empty:
        title = f"Suggested top {len(recommended_df)} positions:"
        logger.info(f"\n{predict_table.get_string(title=title)}")

    wandb.log({"daily_predictions": wandb_table})

    if args.perform_trading:
        actionable_plan: dict[str, dict[str, int | str]] = {}
        for stock, info in stock_to_execute.items():
            shares = int(info.get("shares_to_execute", 0))
            if shares <= 0:
                continue
            actionable_plan[str(stock)] = {
                "shares_to_execute": shares,
                "action": str(info.get("action", "")),
            }
        if not actionable_plan:
            logger.info("No trades to execute after filtering zero-share positions.")
        else:
            perform_schwab_trade(client, account_hash, actionable_plan)

    run.finish()

    logger.info("Prediction process completed.")


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

    parser.add_argument("--perform_trading", "-pt", action="store_true", default=False)

    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--seed", "-s", type=int, default=settings.SEED)

    args = parser.parse_args()

    configure_logging(env="predict", file_name="predict.log")
    logger = logging.getLogger("ml4investment.predict")

    train_stock_list = json.load(open(args.train_stocks, "r"))["train_stocks"]
    predict_stock_list = json.load(open(args.predict_stocks, "r"))["predict_stocks"]
    fetched_data_df = pd.read_parquet(args.fetched_data_pth)
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
        fetched_data_df,
        process_feature_config,
        selected_features,
        model,
        seed,
    )
