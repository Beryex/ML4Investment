import argparse
import json
import logging
import os
from typing import Any

import pandas as pd

from ml4investment.config.global_settings import settings
from ml4investment.utils.data_loader import (
    fetch_data_from_schwab,
    get_available_stocks,
    get_stock_sector_id_mapping,
    load_local_data,
    merge_fetched_data,
)
from ml4investment.utils.logging import configure_logging
from ml4investment.utils.utils import setup_schwab_client

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


def _save_json_file(path: str, payload: dict[str, Any]) -> None:
    """Save a dictionary payload to JSON.

    Args:
        path: JSON file path.
        payload: Data to write.
    """
    with open(path, "w") as file_handle:
        json.dump(payload, file_handle, indent=4)


def _load_available_stock_list(args: argparse.Namespace) -> list[str]:
    """Load or fetch the available stock list.

    Args:
        args: CLI arguments.

    Returns:
        List of available stock symbols.
    """
    if args.get_available_stocks:
        return get_available_stocks()

    return _load_json_file(args.available_stocks)["available_stocks"]


def _fetch_latest_data(args: argparse.Namespace, available_stocks: list[str]) -> pd.DataFrame:
    """Fetch latest data based on CLI flags.

    Args:
        args: CLI arguments.
        available_stocks: Stock symbols to fetch.

    Returns:
        Fetched intraday data.
    """
    if args.load_local_data:
        logger.info("Load local data from %s for the given stocks", args.local_data_pth)
        return load_local_data(available_stocks, base_dir=args.local_data_pth)

    logger.info("Fetch data from Schwab for the given stocks")
    client, _account_hash = setup_schwab_client()
    return fetch_data_from_schwab(client, available_stocks)


def _load_existing_data(path: str) -> pd.DataFrame:
    """Load previously saved data if present.

    Args:
        path: Saved parquet path.

    Returns:
        Existing data DataFrame or empty DataFrame.
    """
    if os.path.exists(path):
        logger.info("Loading previously saved data from %s", path)
        return pd.read_parquet(path)

    logger.info("No previous data found. Starting fresh.")
    return pd.DataFrame()


def _log_fetch_stats(prefix: str, df: pd.DataFrame) -> None:
    """Log summary statistics for fetched data.

    Args:
        prefix: Log prefix label.
        df: DataFrame to summarize.
    """
    logger.info(prefix)
    logger.info("  Number of stocks: %s", df["stock_code"].nunique())
    logger.info("  Number of data points: %s", len(df))
    logger.info("  Overall earliest data timestamp: %s", df.index.min())
    logger.info("  Overall latest data timestamp: %s", df.index.max())


def _save_fetched_data(path: str, merged_data_df: pd.DataFrame) -> None:
    """Save merged fetched data to parquet.

    Args:
        path: Output parquet path.
        merged_data_df: Merged data DataFrame.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    merged_data_df.to_parquet(path)
    logger.info("Fetched data saved to %s", path)


def _save_available_stocks(path: str, stocks: list[str]) -> None:
    """Save the available stocks list.

    Args:
        path: Output JSON path.
        stocks: List of stock symbols.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _save_json_file(path, {"available_stocks": stocks})
    logger.info("Available stocks saved to %s", path)


def _save_sector_mapping(stocks: list[str]) -> None:
    """Fetch and save the stock sector id mapping.

    Args:
        stocks: List of stock symbols.
    """
    stock_sectors_id_mapping = get_stock_sector_id_mapping(stocks)
    os.makedirs(os.path.dirname(settings.STOCK_SECTOR_ID_MAP_PTH), exist_ok=True)
    _save_json_file(settings.STOCK_SECTOR_ID_MAP_PTH, stock_sectors_id_mapping)
    logger.info("Stock sectors id mapping saved to %s", settings.STOCK_SECTOR_ID_MAP_PTH)


def fetch_data(args: argparse.Namespace) -> None:
    """Fetch data for the given stocks.

    Args:
        args: CLI arguments.
    """
    logger.info("Current trading time: %s", pd.Timestamp.now(tz="America/New_York"))

    available_stocks_list = _load_available_stock_list(args)
    logger.info("Start fetching data for given stocks: %s...", available_stocks_list[:100])

    fetched_data_df = _fetch_latest_data(args, available_stocks_list)

    existing_data_df = _load_existing_data(args.save_fetched_data_pth)
    merged_data_df = merge_fetched_data(existing_data_df, fetched_data_df)

    _log_fetch_stats("--- Stats for fetched data ---", fetched_data_df)
    _log_fetch_stats(
        "--- Stats for overall data after merging with exist data ---", merged_data_df
    )

    _save_fetched_data(args.save_fetched_data_pth, merged_data_df)

    if args.get_available_stocks:
        _save_available_stocks(args.save_available_stocks_pth, available_stocks_list)

    if args.get_stock_sector_id_mapping:
        _save_sector_mapping(available_stocks_list)

    logger.info("Fetching data completed.")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for fetch_data CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--available_stocks", type=str, default="config/available_stocks.json")
    parser.add_argument(
        "--get_available_stocks",
        "-gas",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_available_stocks_pth", type=str, default="config/available_stocks.json"
    )
    parser.add_argument("--load_local_data", "-lld", action="store_true", default=False)
    parser.add_argument("--local_data_pth", "-ldp", type=str)
    parser.add_argument(
        "--fetched_data_pth", "-fdp", type=str, default="data/fetched_data.parquet"
    )
    parser.add_argument(
        "--get_stock_sector_id_mapping",
        "-gssim",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_fetched_data_pth", "-sfdp", type=str, default="data/fetched_data.parquet"
    )

    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    configure_logging(env="fetch_data", file_name="fetch_data.log")
    logger = logging.getLogger("ml4investment.fetch_data")

    fetch_data(args)
