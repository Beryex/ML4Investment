import logging
import os
import random
import time

import numpy as np
import requests.exceptions
import schwabdev
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducible usage"""
    logger.info(f"Set random seed: {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def stock_code_to_id(stock_code: str) -> int:
    """Change the stock string to the sum of ASCII value of each char within the stock code"""
    return sum(ord(c) * 256**i for i, c in enumerate(reversed(stock_code)))


def id_to_stock_code(code_id: int) -> str:
    """Change the stock id to the string of stock code"""
    chars = []
    while code_id > 0:
        ascii_val = code_id % 256
        chars.append(chr(ascii_val))
        code_id //= 256
    return "".join(reversed(chars))


class OptimalIterationCallback:
    """Callback to record the optimal iteration during training."""

    def __init__(self, eval_set_idx: int = 0, metric: str = "l1"):
        self.eval_set_idx = eval_set_idx
        self.metric = metric
        self.optimal_score = float("inf")
        self.optimal_iteration = -1

    def __call__(self, env):
        """The callback logic."""
        current_score = env.evaluation_result_list[self.eval_set_idx][2]

        if current_score < self.optimal_score:
            self.optimal_score = current_score
            self.optimal_iteration = env.iteration


def OptimalIterationLogger(eval_set_idx: int = 0, metric: str = "l1"):
    """Factory function to create an OptimalIterationCallback instance."""
    return OptimalIterationCallback(eval_set_idx, metric)


def setup_schwab_client() -> tuple[schwabdev.Client, str]:
    """Setup Schwab client with API keys from environment variables."""
    load_dotenv()
    APP_KEY = os.getenv("SCHWAB_APP_KEY")
    APP_SECRET = os.getenv("SCHWAB_SECRET")
    CALLBACK_URL = os.getenv("SCHWAB_CALLBACK_URL")
    assert all([APP_KEY, APP_SECRET, CALLBACK_URL]), (
        "Please set SCHWAB_APP_KEY, SCHWAB_SECRET, and SCHWAB_CALLBACK_URL in your .env file."
    )
    assert isinstance(CALLBACK_URL, str), "CALLBACK_URL must be a string."
    client = schwabdev.Client(APP_KEY, APP_SECRET, CALLBACK_URL)
    linked_accounts = client.account_linked().json()
    account_hash = linked_accounts[0].get("hashValue")
    return client, account_hash


def get_schwab_formatted_order(symbol: str, instruction: str, quantity: int) -> dict:
    """Format the order details for Schwab API."""
    return {
        "orderType": "MARKET",
        "session": "NORMAL",
        "duration": "DAY",
        "orderStrategyType": "SINGLE",
        "orderLegCollection": [
            {
                "instruction": instruction,
                "quantity": quantity,
                "instrument": {"symbol": symbol, "assetType": "EQUITY"},
            }
        ],
    }


def retry_api_call(func, max_retries=5, base_delay=2.0):
    """Retry API call with exponential backoff"""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except (
            requests.exceptions.JSONDecodeError,
            requests.exceptions.RequestException,
            ConnectionError,
        ) as e:
            if attempt == max_retries:
                raise e
            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            logger.warning(
                f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}. Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
