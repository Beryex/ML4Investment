import argparse
import json
import logging
import math
import pickle
from typing import Any

import lightgbm as lgb
import pandas as pd
import wandb
from prettytable import PrettyTable
from wandb.sdk.wandb_run import Run

from ml4investment.config.global_settings import settings
from ml4investment.utils.feature_calculating import calculate_features
from ml4investment.utils.feature_processing import process_features_for_predict
from ml4investment.utils.logging import configure_logging, setup_wandb
from ml4investment.utils.model_predicting import get_prev_actual_ranking, get_stocks_portfolio
from ml4investment.utils.utils import (
    id_to_stock_code,
    perform_schwab_trade,
    set_random_seed,
    setup_schwab_client,
)

logger = logging.getLogger(__name__)


def _load_json_file(path: str) -> dict[str, Any]:
    """Load a JSON file from disk.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON as a dictionary.
    """
    with open(path, "r") as file_handle:
        return json.load(file_handle)


def _load_prediction_data(
    fetched_data_df: pd.DataFrame, train_stock_list: list[str]
) -> pd.DataFrame:
    """Filter fetched data for prediction window.

    Args:
        fetched_data_df: Raw intraday data.
        train_stock_list: Stock codes used for training.

    Returns:
        Filtered intraday DataFrame for prediction.
    """
    train_data_start_date = settings.TRAINING_DATA_START_DATE
    logger.info("Load input fetched data, starting from %s", train_data_start_date)
    train_data_df = fetched_data_df[fetched_data_df["stock_code"].isin(train_stock_list)]
    return train_data_df.loc[train_data_start_date:]


def _prepare_prediction_features(
    predict_data_df: pd.DataFrame,
    process_feature_config: dict[str, Any],
    predict_stock_list: list[str],
    selected_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate and process features for prediction.

    Args:
        predict_data_df: Intraday data for prediction.
        process_feature_config: Feature processing configuration.
        predict_stock_list: Stock codes to include.
        selected_features: Feature list selected during training.

    Returns:
        Tuple of (X_predict, daily_features_data).
    """
    daily_features_data = calculate_features(predict_data_df)

    X_predict = process_features_for_predict(
        daily_features_data, process_feature_config, predict_stock_list
    )

    X_predict = X_predict[selected_features]

    return X_predict, daily_features_data


def _log_prediction_summary(X_predict: pd.DataFrame) -> None:
    """Log summary information for prediction dataset.

    Args:
        X_predict: Prediction feature DataFrame.
    """
    assert X_predict.index.min() == X_predict.index.max()
    logger.info("Predicting based on data on %s", X_predict.index.min())
    logger.info("Total processed samples in predict data: %s", X_predict.shape[0])
    logger.info("Number of features in predict data: %s", X_predict.shape[1])


def _init_prediction_tables() -> tuple[PrettyTable, wandb.Table, list[str]]:
    """Initialize prediction output tables.

    Returns:
        Tuple of (pretty_table, wandb_table, field_names).
    """
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
    return predict_table, wandb_table, field_names


def _get_account_balance(client: Any, account_hash: str) -> float:
    """Fetch current account equity balance.

    Args:
        client: Schwab client.
        account_hash: Account hash string.

    Returns:
        Current equity balance.
    """
    return client.account_details(account_hash, fields="positions").json()["securitiesAccount"][
        "currentBalances"
    ]["equity"]


def _build_results_df(
    X_predict: pd.DataFrame, predictions: pd.Series | list[float]
) -> pd.DataFrame:
    """Build prediction results DataFrame.

    Args:
        X_predict: Prediction feature DataFrame.
        predictions: Model predictions.

    Returns:
        DataFrame with stock_code, prediction, and last_price columns.
    """
    results_df = pd.DataFrame(index=X_predict.index)
    results_df["stock_code"] = X_predict["stock_id"].map(id_to_stock_code)
    results_df["prediction"] = predictions
    results_df["last_price"] = 0.0
    return results_df


def _fetch_last_prices(client: Any, predict_stock_list: list[str]) -> dict[str, float]:
    """Fetch latest prices from Schwab.

    Args:
        client: Schwab client.
        predict_stock_list: Stock codes to query.

    Returns:
        Mapping of stock codes to last price.
    """
    stock_quotes = client.quotes(symbols=predict_stock_list, fields="quote").json()
    return {
        stock: quote["quote"]["lastPrice"]
        for stock, quote in stock_quotes.items()
        if "quote" in quote
    }


def _attach_last_prices(
    sorted_results: pd.DataFrame, stock_last_prices: dict[str, float]
) -> pd.DataFrame:
    """Attach last prices to the sorted results DataFrame.

    Args:
        sorted_results: Results sorted by prediction.
        stock_last_prices: Last price mapping.

    Returns:
        Updated results DataFrame.
    """
    sorted_results["last_price"] = (
        sorted_results["stock_code"].map(stock_last_prices).astype(float)
    )
    return sorted_results


def _build_recommendations(
    sorted_results: pd.DataFrame, daily_features_data: pd.DataFrame
) -> pd.DataFrame:
    """Build recommended portfolio based on predictions and momentum.

    Args:
        sorted_results: Results sorted by prediction.
        daily_features_data: Historical features data.

    Returns:
        Recommended portfolio DataFrame.
    """
    logger.info("Selecting stocks using strategy: %s", settings.STOCK_SELECTION_STRATEGY)
    logger.info("Using momentum weight: %.2f", settings.STOCK_SELECTION_MOMENTUM)
    current_ts = sorted_results.index.max()
    prev_actuals = get_prev_actual_ranking(
        stock_codes=sorted_results["stock_code"],
        historical_df=daily_features_data,
        current_ts=current_ts,
        actual_col="Target",
    )
    return get_stocks_portfolio(sorted_results, prev_actuals=prev_actuals)


def _populate_recommendation_tables(
    recommended_df: pd.DataFrame,
    predict_table: PrettyTable,
    wandb_table: wandb.Table,
    total_balance: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, int | str]]]:
    """Populate recommendation tables and build execution payload.

    Args:
        recommended_df: Recommended portfolio DataFrame.
        predict_table: PrettyTable instance to populate.
        wandb_table: W&B table to populate.
        total_balance: Total account equity.

    Returns:
        Tuple of (updated_recommended_df, execution_payload).
    """
    if recommended_df.empty:
        logger.info("No stocks were recommended today (strategy produced no candidates)")
        return recommended_df, {}

    logger.info("Give recommendation based on total investment value: $%.2f", total_balance)

    recommended_df = recommended_df.copy()
    recommended_df["invest_value"] = total_balance * recommended_df["weight"]
    recommended_df["shares_to_execute"] = (
        recommended_df["invest_value"] / recommended_df["last_price"]
    ).apply(math.floor)

    stock_to_execute = recommended_df.set_index("stock_code")[
        ["shares_to_execute", "action"]
    ].to_dict("index")

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

    return recommended_df, stock_to_execute


def _populate_rejected_rows(
    sorted_results: pd.DataFrame,
    recommended_df: pd.DataFrame,
    predict_table: PrettyTable,
    wandb_table: wandb.Table,
) -> None:
    """Populate rows for rejected stocks when verbose is enabled.

    Args:
        sorted_results: Results sorted by prediction.
        recommended_df: Recommended portfolio DataFrame.
        predict_table: PrettyTable instance.
        wandb_table: W&B table instance.
    """
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


def _log_prediction_table(predict_table: PrettyTable, recommended_df: pd.DataFrame) -> None:
    """Log prediction table output.

    Args:
        predict_table: PrettyTable instance.
        recommended_df: Recommended portfolio DataFrame.
    """
    title = f"Suggested top {len(recommended_df)} positions:"
    logger.info("\n%s", predict_table.get_string(title=title))


def _execute_trades_if_enabled(
    stock_to_execute: dict[str, dict[str, int | str]],
    client: Any,
    account_hash: str,
    args: argparse.Namespace,
) -> None:
    """Execute Schwab trades based on recommendation output.

    Args:
        stock_to_execute: Mapping of stock to execution details.
        client: Schwab client.
        account_hash: Account hash string.
        args: CLI arguments.
    """
    if not args.perform_trading:
        return

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
        return

    perform_schwab_trade(client, account_hash, actionable_plan)


def predict(
    run: Run,
    train_stock_list: list[str],
    predict_stock_list: list[str],
    fetched_data_df: pd.DataFrame,
    process_feature_config: dict[str, Any],
    selected_features: list[str],
    model: lgb.Booster,
    seed: int,
    client: Any,
    account_hash: str,
    args: argparse.Namespace,
) -> None:
    """Predict the optimal stock with the highest price change for the given stocks.

    Args:
        run: Active W&B run.
        train_stock_list: Stock codes used for training.
        predict_stock_list: Stock codes eligible for prediction.
        fetched_data_df: Intraday fetched data.
        process_feature_config: Feature processing configuration.
        selected_features: Feature list selected during training.
        model: Trained LightGBM model.
        seed: Random seed.
        client: Schwab client.
        account_hash: Account hash string.
        args: CLI arguments.
    """
    logger.info("Start predict the given stocks: %s", predict_stock_list)
    logger.info("Current trading time: %s", pd.Timestamp.now(tz="America/New_York"))
    set_random_seed(seed)

    predict_data_df = _load_prediction_data(fetched_data_df, train_stock_list)
    X_predict, daily_features_data = _prepare_prediction_features(
        predict_data_df,
        process_feature_config,
        predict_stock_list,
        selected_features,
    )

    _log_prediction_summary(X_predict)

    predict_table, wandb_table, _ = _init_prediction_tables()

    predictions = model.predict(X_predict, num_iteration=model.best_iteration)

    total_balance = _get_account_balance(client, account_hash)

    results_df = _build_results_df(X_predict, predictions)
    sorted_results = results_df.sort_values("prediction", ascending=False)
    stock_last_prices = _fetch_last_prices(client, predict_stock_list)
    sorted_results = _attach_last_prices(sorted_results, stock_last_prices)

    recommended_df = _build_recommendations(sorted_results, daily_features_data)

    recommended_df, stock_to_execute = _populate_recommendation_tables(
        recommended_df, predict_table, wandb_table, total_balance
    )

    if args.verbose:
        _populate_rejected_rows(sorted_results, recommended_df, predict_table, wandb_table)

    if args.verbose or not recommended_df.empty:
        _log_prediction_table(predict_table, recommended_df)

    wandb.log({"daily_predictions": wandb_table})

    _execute_trades_if_enabled(stock_to_execute, client, account_hash, args)

    run.finish()

    logger.info("Prediction process completed.")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for prediction CLI."""
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

    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    configure_logging(env="predict", file_name="predict.log")
    logger = logging.getLogger("ml4investment.predict")

    train_stock_list = _load_json_file(args.train_stocks)["train_stocks"]
    predict_stock_list = _load_json_file(args.predict_stocks)["predict_stocks"]
    fetched_data_df = pd.read_parquet(args.fetched_data_pth)
    process_feature_config = pickle.load(open(args.process_feature_config_pth, "rb"))
    selected_features = _load_json_file(args.features_pth)["features"]
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
        client,
        account_hash,
        args,
    )
