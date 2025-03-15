import logging

from ml4investment.utils.logging import configure_logging
from ml4investment.utils.data_loader import fetch_trading_day_data

configure_logging(env="test", file_name="test_logging.log")
logger = logging.getLogger("ml4investment.test")


def main():
    """ Test the logging configuration and functionality """
    test_logging_within_current_files()
    test_logging_within_otherfiles()


def test_logging_within_current_files():
    """ Expect all 3 messages shown """
    logger.info("This is an info message")
    logger.debug("This is a debug message")
    logger.error("This is an error message")


def test_logging_within_otherfiles():
    """ Expect to see the stock number, stock features and timezone in the log """
    stock = 'AAPL' 
    fetch_trading_day_data(stock)


if __name__ == "__main__":
    main()
