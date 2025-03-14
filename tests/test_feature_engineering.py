import logging

from ml4investment.utils.data_loader import fetch_trading_day_data
from ml4investment.utils.feature_engineering import calculate_features, process_features
from ml4investment.utils.logging import configure_logging

configure_logging(env="test", file_name="test_feature_engineering.log")
logger = logging.getLogger("ml4investment.test")


def main():
    """ Test the feature engineering functionality """
    calculated_features = test_calculate_features()
    X_train, X_test, y_train, y_test, x_predict = test_process_features(calculated_features)


def test_calculate_features():
    stock = 'AAPL' 
    fetched_data = fetch_trading_day_data(stock)
    logger.info(f"Fetched trading day data: {fetched_data}")
    calculated_features = calculate_features(fetched_data)
    logger.info(f"Calculated features and corresponding values: {calculated_features}")
    return calculated_features


def test_process_features(calculated_features):
    X_train, X_test, y_train, y_test, x_predict = process_features(calculated_features)
    logger.info(f"Processed features and corresponding values for X_train: {X_train}")
    logger.info(f"Processed features and corresponding values for X_test: {X_test}")
    logger.info(f"Processed features and corresponding values for y_train: {y_train}")
    logger.info(f"Processed features and corresponding values for y_test: {y_test}")
    logger.info(f"Processed features and corresponding values for x_predict: {x_predict}")
    return X_train, X_test, y_train, y_test, x_predict

if __name__ == "__main__":
    main()
