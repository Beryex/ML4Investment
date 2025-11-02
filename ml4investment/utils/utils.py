import datetime
import logging
import os
import random
import time
from collections import defaultdict
from typing import cast

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests.exceptions
import schwabdev
import shap
from dotenv import load_dotenv
from prettytable import PrettyTable
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from ml4investment.config.global_settings import settings

logger = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducible usage"""
    logger.info(f"Set random seed: {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _coerce_stock_id(value):
    if isinstance(value, (np.generic,)):
        return int(value.item())
    return int(value)


def stock_code_to_id(stock_code: str) -> int:
    """Change the stock string to the sum of ASCII value of each char within the stock code"""
    return int(sum(ord(c) * 256**i for i, c in enumerate(reversed(stock_code))))


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
            self.optimal_iteration = env.iteration + 1  # +1 because iteration is zero-based


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


def perform_schwab_trade(
    client: schwabdev.Client,
    account_hash: str,
    stock_to_execute: dict[str, dict[str, int | str]],
) -> None:
    """Execute required long/short trades on Schwab via API."""
    now_et = pd.Timestamp.now(tz="America/New_York")
    if now_et.weekday() < 5:
        # Define trading hours
        start_time = datetime.time(9, 30)
        end_time = datetime.time(16, 0)

        if start_time <= now_et.time() < end_time:
            logger.warning(
                "Trading can only be executed except market hours to avoid Day Trader Pattern. "
                "No trading executed!"
            )
            return

    logger.info("Performing Schwab trade...")
    account_orders = client.account_orders(
        account_hash,
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(days=7),  # 7 days is sufficient for daily usage, hardcoded here
        datetime.datetime.now(datetime.timezone.utc),
    ).json()

    account_positions = {}
    positions_payload = client.account_details(account_hash, fields="positions").json()[
        "securitiesAccount"
    ].get("positions", [])
    for position in positions_payload:
        symbol = position["instrument"]["symbol"]
        long_qty = int(position.get("longQuantity", 0))
        short_qty = int(position.get("shortQuantity", 0))
        account_positions[symbol] = long_qty - short_qty
    logger.info(f"Current account positions: ")
    for stock, qty in account_positions.items():
        logger.info(f"  {stock}: {qty} share(s)")

    logger.info("Canceling previous active orders...")
    opening_orders = [
        order for order in account_orders if order.get("status") in settings.OPENING_STATUS
    ]

    for order in opening_orders:
        order_detail = order["orderLegCollection"][0]
        logger.info(
            f"Canceling order: {order_detail['instruction']} {order_detail['quantity']} "
            f"share(s) of {order_detail['instrument']['symbol']}"
        )
        client.order_cancel(account_hash, order["orderId"])
    logger.info("All previous active orders canceled")
    
    logger.info("Placing new orders...")

    target_positions: dict[str, int] = {}
    for stock, info in stock_to_execute.items():
        action = str(info.get("action", "")).upper()
        shares = int(info.get("shares_to_execute", 0))
        if shares < 0:
            logger.warning("Negative share request for %s, skipping", stock)
            continue
        if action not in {"BUY_LONG", "SELL_SHORT"}:
            logger.warning("Unknown action %s for %s, skipping", action, stock)
            continue
        target_positions[stock] = -shares if action == "SELL_SHORT" else shares

    orders: list[tuple[str, str, int]] = []
    unchanged: list[tuple[str, int]] = []

    all_symbols = set(account_positions.keys()) | set(target_positions.keys())

    for stock in sorted(all_symbols):
        current = int(account_positions.get(stock, 0))
        desired = int(target_positions.get(stock, 0))

        if current == desired:
            if current != 0:
                unchanged.append((stock, current))
            continue

        if current > 0:
            if desired >= 0:
                diff = desired - current
                if diff > 0:
                    orders.append((stock, "BUY", diff))
                elif diff < 0:
                    orders.append((stock, "SELL", -diff))
            else:
                orders.append((stock, "SELL", current))
                short_target = -desired
                if short_target > 0:
                    orders.append((stock, "SELL_SHORT", short_target))
        else:
            if desired <= 0:
                diff = desired - current
                if diff > 0:
                    orders.append((stock, "BUY_TO_COVER", diff))
                elif diff < 0:
                    orders.append((stock, "SELL_SHORT", -diff))
            else:
                if current < 0:
                    orders.append((stock, "BUY_TO_COVER", -current))
                if desired > 0:
                    orders.append((stock, "BUY", desired))

    if unchanged:
        logger.info(
            "No change for existing stock %s",
            ", ".join(f"{symbol} ({qty} share(s))" for symbol, qty in unchanged),
        )

    for stock, instruction, qty in orders:
        qty_int = int(qty)
        if qty_int <= 0:
            continue
        logger.info("%s %d share(s) of %s", instruction.replace("_", " "), qty_int, stock)
        formatted_order = get_schwab_formatted_order(stock, instruction, qty_int)
        client.order_place(account_hash, formatted_order)

    logger.info("All new orders placed")


def retry_api_call(func, max_retries=10, base_delay=2.0):
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
                f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
