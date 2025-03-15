import argparse
import logging
import pandas as pd
from prettytable import PrettyTable

from ml4investment.utils.seed import set_random_seed
from ml4investment.utils.logging import configure_logging
from ml4investment.predict_single_stock import predict_single_stock

configure_logging(env="prod", file_name="predict_optimal_stock.log")
logger = logging.getLogger("ml4investment.test")


def predict_optimal_stock(stock_list, seed):
    """ Predict the optimal stock with the highest price change for the given stocks """
    logger.info(f"Start predicting stocks: {stock_list}")
    logger.info(f"Current trading time: {pd.Timestamp.now(tz='America/New_York')}")
    set_random_seed(seed)

    results_table = PrettyTable()
    results_table.field_names = ["Stock", "Predicted Price Change"]
    
    optimal_ratio = float('-inf')
    optimal_stock = ""
    for stock in stock_list:
        predict_ratio = predict_single_stock(stock, seed)
        results_table.add_row([stock, predict_ratio], divider=True)
        if predict_ratio > optimal_ratio:
            optimal_ratio = predict_ratio
            optimal_stock = stock

    logger.info(results_table.get_string(title=f"Predict price changes for stocks: {stock_list}"))
    logger.info(f"Suggested optimal stock: {optimal_stock} with predicted price change: {optimal_ratio:+.2%}")
    logger.info("Prediction process completed.")

    return optimal_stock, predict_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", "-ss", type=str, default="AMZN,MSFT,AAPL,GOOGL,TSLA,META,INTC,AMD,NVDA,BABA,TCEHY,JD")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    stock_list = args.stocks.split(",")
    seed = args.seed

    predict_optimal_stock(stock_list, seed)
