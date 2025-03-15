import argparse
import logging
import pandas as pd

from ml4investment.utils.seed import set_random_seed
from ml4investment.utils.data_loader import fetch_trading_day_data
from ml4investment.utils.logging import configure_logging
from ml4investment.utils.feature_engineering import calculate_features, process_features
from ml4investment.utils.model_training import model_training
from ml4investment.utils.model_predicting import model_predict

configure_logging(env="prod", file_name="predict_single_stock.log")
logger = logging.getLogger("ml4investment.test")


def predict_single_stock(stock, seed):
    """ Predict the price change for the given stock """
    logger.info(f"Start predicting stock: {stock}")
    logger.info(f"Current trading time: {pd.Timestamp.now(tz='America/New_York')}")
    set_random_seed(seed)

    # 1. Data fetching
    fetched_data = fetch_trading_day_data(stock)

    # 2. Feature engineering
    daily_features_data = calculate_features(fetched_data)
    X_train, X_test, y_train, y_test, x_predict = process_features(daily_features_data)

    # 3. Model training
    model = model_training(X_train, X_test, y_train, y_test)

    # 4. Model prediction
    predict_ratio = model_predict(model, x_predict)

    logger.info(f"Predicted price change for {stock}: {predict_ratio:+.2%}")
    logger.info("Prediction process completed.")

    return predict_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", type=str, default="MSFT")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    stock = args.stock
    seed = args.seed

    predict_single_stock(stock, seed)
