import datetime
import logging
import os
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import requests.exceptions
import schwabdev
from dotenv import load_dotenv

from ml4investment.config.global_settings import settings

logger = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducible usage."""
    logger.info("Set random seed: %s", seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _coerce_stock_id(value: Any) -> int:
    """Coerce stock ID to integer."""
    if isinstance(value, (np.generic,)):
        return int(value.item())
    return int(value)


def stock_code_to_id(stock_code: str) -> int:
    """Change the stock string to the sum of ASCII value of each char within the stock code."""
    return int(sum(ord(c) * 256**i for i, c in enumerate(reversed(stock_code))))


def id_to_stock_code(code_id: int) -> str:
    """Change the stock id to the string of stock code."""
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

    def __call__(self, env) -> None:
        """The callback logic."""
        current_score = env.evaluation_result_list[self.eval_set_idx][2]

        if current_score < self.optimal_score:
            self.optimal_score = current_score
            self.optimal_iteration = env.iteration + 1  # +1 because iteration is zero-based


def OptimalIterationLogger(eval_set_idx: int = 0, metric: str = "l1") -> OptimalIterationCallback:
    """Factory function to create an OptimalIterationCallback instance."""
    return OptimalIterationCallback(eval_set_idx, metric)


def setup_schwab_client() -> tuple[schwabdev.Client, str]:
    """Setup Schwab client with API keys from environment variables."""
    load_dotenv()
    app_key = os.getenv("SCHWAB_APP_KEY")
    app_secret = os.getenv("SCHWAB_SECRET")
    callback_url = os.getenv("SCHWAB_CALLBACK_URL")
    assert all([app_key, app_secret, callback_url]), (
        "Please set SCHWAB_APP_KEY, SCHWAB_SECRET, and SCHWAB_CALLBACK_URL in your .env file."
    )
    assert isinstance(callback_url, str), "CALLBACK_URL must be a string."
    client = schwabdev.Client(app_key, app_secret, callback_url)
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


def _is_market_hours(now_et: pd.Timestamp) -> bool:
    """Check if current time is within standard trading hours.

    Args:
        now_et: Current timestamp in Eastern Time.

    Returns:
        True if now is within market hours.
    """
    if now_et.weekday() >= 5:
        return False

    start_time = datetime.time(9, 30)
    end_time = datetime.time(16, 0)
    return start_time <= now_et.time() < end_time


def _get_account_orders(client: schwabdev.Client, account_hash: str) -> list[dict[str, Any]]:
    """Fetch recent account orders.

    Args:
        client: Schwab client.
        account_hash: Account hash string.

    Returns:
        List of order dictionaries.
    """
    return client.account_orders(
        account_hash,
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7),
        datetime.datetime.now(datetime.timezone.utc),
    ).json()


def _get_account_positions(client: schwabdev.Client, account_hash: str) -> dict[str, int]:
    """Fetch current account positions.

    Args:
        client: Schwab client.
        account_hash: Account hash string.

    Returns:
        Mapping of stock symbol to net position.
    """
    account_positions: dict[str, int] = {}
    positions_payload = (
        client.account_details(account_hash, fields="positions")
        .json()["securitiesAccount"]
        .get("positions", [])
    )
    for position in positions_payload:
        symbol = position["instrument"]["symbol"]
        long_qty = int(position.get("longQuantity", 0))
        short_qty = int(position.get("shortQuantity", 0))
        account_positions[symbol] = long_qty - short_qty
    return account_positions


def _log_account_positions(account_positions: dict[str, int]) -> None:
    """Log current account positions.

    Args:
        account_positions: Mapping of stock symbol to net position.
    """
    logger.info("Current account positions: ")
    for stock, qty in account_positions.items():
        logger.info("  %s: %d share(s)", stock, qty)


def _cancel_open_orders(
    client: schwabdev.Client, account_hash: str, account_orders: list[dict[str, Any]]
) -> None:
    """Cancel open orders in the account.

    Args:
        client: Schwab client.
        account_hash: Account hash string.
        account_orders: List of order payloads.
    """
    logger.info("Canceling previous active orders...")
    opening_orders = [
        order for order in account_orders if order.get("status") in settings.OPENING_STATUS
    ]

    for order in opening_orders:
        order_detail = order["orderLegCollection"][0]
        logger.info(
            "Canceling order: %s %s share(s) of %s",
            order_detail["instruction"],
            order_detail["quantity"],
            order_detail["instrument"]["symbol"],
        )
        client.order_cancel(account_hash, order["orderId"])
    logger.info("All previous active orders canceled")


def _build_target_positions(stock_to_execute: dict[str, dict[str, int | str]]) -> dict[str, int]:
    """Build target positions from execution payload.

    Args:
        stock_to_execute: Mapping of stock to execution details.

    Returns:
        Mapping of stock symbol to desired net position.
    """
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
    return target_positions


def _build_orders(
    account_positions: dict[str, int], target_positions: dict[str, int]
) -> tuple[list[tuple[str, str, int]], list[tuple[str, int]]]:
    """Build order list to move from current to target positions.

    Args:
        account_positions: Current account positions.
        target_positions: Desired net positions.

    Returns:
        Tuple of (orders, unchanged positions).
    """
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

    return orders, unchanged


def _log_unchanged_positions(unchanged: list[tuple[str, int]]) -> None:
    """Log unchanged positions.

    Args:
        unchanged: List of unchanged position tuples.
    """
    if unchanged:
        logger.info(
            "No change for existing stock %s",
            ", ".join(f"{symbol} ({qty} share(s))" for symbol, qty in unchanged),
        )


def _place_orders(
    client: schwabdev.Client, account_hash: str, orders: list[tuple[str, str, int]]
) -> None:
    """Place orders using the Schwab API.

    Args:
        client: Schwab client.
        account_hash: Account hash string.
        orders: List of (stock, instruction, quantity) tuples.
    """
    for stock, instruction, qty in orders:
        qty_int = int(qty)
        if qty_int <= 0:
            continue
        logger.info("%s %d share(s) of %s", instruction.replace("_", " "), qty_int, stock)
        formatted_order = get_schwab_formatted_order(stock, instruction, qty_int)
        client.order_place(account_hash, formatted_order)


def perform_schwab_trade(
    client: schwabdev.Client,
    account_hash: str,
    stock_to_execute: dict[str, dict[str, int | str]],
) -> None:
    """Execute required long/short trades on Schwab via API."""
    now_et = pd.Timestamp.now(tz="America/New_York")
    if _is_market_hours(now_et):
        logger.warning(
            "Trading can only be executed except market hours to avoid Day Trader Pattern. "
            "No trading executed!"
        )
        return

    logger.info("Performing Schwab trade...")
    account_orders = _get_account_orders(client, account_hash)
    account_positions = _get_account_positions(client, account_hash)

    _log_account_positions(account_positions)
    _cancel_open_orders(client, account_hash, account_orders)

    logger.info("Placing new orders...")

    target_positions = _build_target_positions(stock_to_execute)
    orders, unchanged = _build_orders(account_positions, target_positions)
    _log_unchanged_positions(unchanged)
    _place_orders(client, account_hash, orders)

    logger.info("All new orders placed")


def retry_api_call(func, max_retries: int = 10, base_delay: float = 2.0):
    """Retry API call with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except (
            requests.exceptions.JSONDecodeError,
            requests.exceptions.RequestException,
            ConnectionError,
        ) as exc:
            if attempt == max_retries:
                raise exc
            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            logger.warning(
                "API call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt + 1,
                max_retries + 1,
                str(exc),
                delay,
            )
            time.sleep(delay)
