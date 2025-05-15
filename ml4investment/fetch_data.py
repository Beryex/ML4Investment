import argparse
import logging
import pandas as pd
import json
import os
import pickle
from argparse import Namespace

from ml4investment.config import settings
from ml4investment.utils.data_loader import fetch_data_from_yfinance, load_local_data, merge_fetched_data, generate_stock_sectors_id_mapping
from ml4investment.utils.logging import configure_logging

configure_logging(env="fetch_data", file_name="fetch_data.log")
logger = logging.getLogger("ml4investment.fetch_data")


def fetch_data(train_stock_list: list,
               args: Namespace):
    """ Fetch data for the given stocks """
    logger.info(f"Start fetching data for given stocks: {train_stock_list}")
    logger.info(f"Current trading time: {pd.Timestamp.now(tz='America/New_York')}")

    """ Fetch new data """
    if args.load_local_data:
        logger.info(f"Load local data from {args.local_data_pth} for the given stocks")
        fetched_data = load_local_data(train_stock_list, base_dir=args.local_data_pth, check_valid=True)
    else:
        logger.info(f"Fetch data from yfinance for the given stocks")
        fetched_data = fetch_data_from_yfinance(train_stock_list, period=settings.TRAIN_DAYS)

    """ Merge with previous saved data """
    if os.path.exists(args.save_fetched_data_pth):
        logger.info(f"Loading previously saved data from {args.save_fetched_data_pth}")
        with open(args.save_fetched_data_pth, 'rb') as f:
            existing_data = pickle.load(f)
    else:
        logger.info("No previous data found. Starting fresh.")
        existing_data = {}
    
    merged_data, _ = merge_fetched_data(existing_data, fetched_data)
    with open(args.save_fetched_data_pth, 'wb') as f:
        pickle.dump(merged_data, f)
    logger.info(f"Fetched data saved to {args.save_fetched_data_pth}")

    logger.info(f"--- Stats for fetched data ---")
    logger.info(f"  Number of stocks: {len(fetched_data)}")
    logger.info(f"  Number of data points: {sum(len(df) for df in fetched_data.values())}")
    logger.info(f"  Overall earliest data timestamp: {min(df.index.min() for df in fetched_data.values())}")
    logger.info(f"  Overall latest data timestamp: {max(df.index.max() for df in fetched_data.values())}")
        
    logger.info(f"--- Stats for overall data after merging with exist data ---")
    logger.info(f"  Number of stocks: {len(merged_data)}")
    logger.info(f"  Number of data points: {sum(len(df) for df in merged_data.values())}")
    logger.info(f"  Overall earliest data timestamp: {min(df.index.min() for df in merged_data.values())}")
    logger.info(f"  Overall latest data timestamp: {max(df.index.max() for df in merged_data.values())}")

    if args.generate_stock_sector_id_mapping:
        stock_sectors_id_mapping = generate_stock_sectors_id_mapping(train_stock_list)
        logger.info(f"Stock sectors id mapping: {stock_sectors_id_mapping}")

    logger.info("Fetching data completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_stocks", type=str, default='config/train_stocks.json')
    parser.add_argument("--load_local_data", "-lld", action='store_true', default=False)
    parser.add_argument("--local_data_pth", "-ldp", type=str)
    parser.add_argument("--fetched_data_pth", "-fdp", type=str, default='data/fetched_data.pkl')
    parser.add_argument("--generate_stock_sector_id_mapping", "-gssim", action='store_true', default=False)
    
    parser.add_argument("--save_fetched_data_pth", "-sfdp", type=str, default='data/fetched_data.pkl')

    args = parser.parse_args()

    train_stock_list = json.load(open(args.train_stocks, 'r'))["train_stocks"]
    
    fetch_data(train_stock_list, args)
