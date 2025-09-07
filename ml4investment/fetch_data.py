import argparse
import json
import logging
import os

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


def fetch_data():
    """Fetch data for the given stocks"""
    logger.info(f"Current trading time: {pd.Timestamp.now(tz='America/New_York')}")

    """ Update available stocks if needed """
    if args.get_available_stocks:
        available_stocks_list = get_available_stocks()
    else:
        available_stocks_list = json.load(open(args.available_stocks, "r"))["available_stocks"]
    logger.info(f"Start fetching data for given stocks: {available_stocks_list[:100]}...")

    """ Fetch new data """
    if args.load_local_data:
        logger.info(f"Load local data from {args.local_data_pth} for the given stocks")
        fetched_data_df = load_local_data(available_stocks_list, base_dir=args.local_data_pth)
    else:
        logger.info("Fetch data from Schwab for the given stocks")
        client, account_hash = setup_schwab_client()
        fetched_data_df = fetch_data_from_schwab(client, available_stocks_list)

    """ Merge with previous saved data """
    if os.path.exists(args.save_fetched_data_pth):
        logger.info(f"Loading previously saved data from {args.save_fetched_data_pth}")
        existing_data_df = pd.read_parquet(args.save_fetched_data_pth)
    else:
        logger.info("No previous data found. Starting fresh.")
        existing_data_df = pd.DataFrame()
    merged_data_df = merge_fetched_data(existing_data_df, fetched_data_df)

    logger.info("--- Stats for fetched data ---")
    logger.info(f"  Number of stocks: {fetched_data_df['stock_code'].nunique()}")
    logger.info(f"  Number of data points: {len(fetched_data_df)}")
    logger.info(f"  Overall earliest data timestamp: {fetched_data_df.index.min()}")
    logger.info(f"  Overall latest data timestamp: {fetched_data_df.index.max()}")

    logger.info("--- Stats for overall data after merging with exist data ---")
    logger.info(f"  Number of stocks: {merged_data_df['stock_code'].nunique()}")
    logger.info(f"  Number of data points: {len(merged_data_df)}")
    logger.info(f"  Overall earliest data timestamp: {merged_data_df.index.min()}")
    logger.info(f"  Overall latest data timestamp: {merged_data_df.index.max()}")

    os.makedirs(os.path.dirname(args.save_fetched_data_pth), exist_ok=True)
    merged_data_df.to_parquet(args.save_fetched_data_pth)
    logger.info(f"Fetched data saved to {args.save_fetched_data_pth}")

    if args.get_available_stocks:
        available_stocks = {"available_stocks": available_stocks_list}
        os.makedirs(os.path.dirname(args.save_available_stocks_pth), exist_ok=True)
        with open(args.save_available_stocks_pth, "w") as f:
            json.dump(available_stocks, f, indent=4)
        logger.info(f"Available stocks saved to {args.save_available_stocks_pth}")

    if args.get_stock_sector_id_mapping:
        stock_sectors_id_mapping = get_stock_sector_id_mapping(available_stocks_list)
        os.makedirs(os.path.dirname(settings.STOCK_SECTOR_ID_MAP_PTH), exist_ok=True)
        with open(settings.STOCK_SECTOR_ID_MAP_PTH, "w") as f:
            json.dump(stock_sectors_id_mapping, f, indent=4)
        logger.info(f"Stock sectors id mapping saved to {settings.STOCK_SECTOR_ID_MAP_PTH}")

    logger.info("Fetching data completed.")


if __name__ == "__main__":
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

    args = parser.parse_args()

    configure_logging(env="fetch_data", file_name="fetch_data.log")
    logger = logging.getLogger("ml4investment.fetch_data")

    fetch_data()
