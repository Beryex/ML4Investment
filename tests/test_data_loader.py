import logging

from ml4investment.utils.data_loader import fetch_trading_day_data
from ml4investment.utils.logging import configure_logging

configure_logging(env="test", file_name="test_feature_engineering.log")
logger = logging.getLogger("ml4investment.test")


def main():
    """ Test data load functionality """
    test_fetch_trading_day_data()


def test_fetch_trading_day_data():
    stock = 'AAPL' 
    fetched_data = fetch_trading_day_data(stock)
    logger.info(f"Fetched trading day data: {fetched_data}")


if __name__ == "__main__":
    main()