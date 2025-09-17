import datetime
import logging
import os
import random
import time
from collections import defaultdict
from typing import cast

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests.exceptions
import schwabdev
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


def get_detailed_static_result(
    model: lgb.Booster,
    X: pd.DataFrame,
    y: pd.Series,
    predict_stock_list: list,
    name: str = "",
    verbose: bool = True,
) -> tuple[int, float, float, float, float, float, float, float, float, float, float, list[str]]:
    """Display detailed static result of the model predictions"""
    preds = model.predict(X, num_iteration=model.best_iteration)
    assert isinstance(preds, np.ndarray)

    results_df = X[["stock_id"]].copy()
    results_df["y_actual"] = y
    results_df["prediction"] = preds

    """Compute overall metrics"""
    mae_overall = mean_absolute_error(results_df["y_actual"], results_df["prediction"])
    mse_overall = mean_squared_error(results_df["y_actual"], results_df["prediction"])
    sign_acc_overall = float(
        (np.sign(results_df["y_actual"]) == np.sign(results_df["prediction"])).mean()
    )

    binary_y_true = (results_df["y_actual"] > 0).astype(int)
    binary_y_pred = (results_df["prediction"] > 0).astype(int)
    precision_overall = float(precision_score(binary_y_true, binary_y_pred, zero_division=0))
    recall_overall = float(recall_score(binary_y_true, binary_y_pred, zero_division=0))
    f1_overall = float(f1_score(binary_y_true, binary_y_pred, zero_division=0))

    """Compute stock-level metrics"""
    stock_metrics = defaultdict(dict)

    for stock_id, stock_df in results_df.groupby("stock_id", observed=True):
        stock_code = id_to_stock_code(int(stock_id))  # type: ignore
        if stock_code not in predict_stock_list:
            continue

        y_true_stock = stock_df["y_actual"]
        y_pred_stock = stock_df["prediction"]

        stock_metrics[stock_code]["mae"] = mean_absolute_error(y_true_stock, y_pred_stock)
        stock_metrics[stock_code]["mse"] = mean_squared_error(y_true_stock, y_pred_stock)
        stock_metrics[stock_code]["sign_acc"] = (
            np.sign(y_true_stock) == np.sign(y_pred_stock)
        ).mean()

        b_y_true = (y_true_stock > 0).astype(int)
        b_y_pred = (y_pred_stock > 0).astype(int)
        stock_metrics[stock_code]["precision"] = precision_score(
            b_y_true, b_y_pred, zero_division=0
        )
        stock_metrics[stock_code]["recall"] = recall_score(b_y_true, b_y_pred, zero_division=0)
        stock_metrics[stock_code]["f1"] = f1_score(b_y_true, b_y_pred, zero_division=0)

        positive_mask = y_pred_stock > 0
        if positive_mask.any():
            gain_factors = 1 + y_true_stock[positive_mask]
            overall_gain = gain_factors.prod()
        else:
            overall_gain = 1.0
        stock_metrics[stock_code]["overall_gain"] = overall_gain
        stock_metrics[stock_code]["avg_daily_gain"] = cast(float, overall_gain) ** (
            1 / len(stock_df)
        )

    """Compute daily-level metrics"""
    gain_actual = 1.0
    daily_results_table_data = []
    daily_returns_list = []
    k = settings.NUMBER_OF_STOCKS_TO_BUY

    unique_days = results_df.index.unique()
    day_number = len(unique_days)

    for date, daily_df in results_df.groupby(results_df.index):
        daily_df_filtered = daily_df[
            daily_df["stock_id"].map(id_to_stock_code).isin(predict_stock_list)
        ]

        positive_preds_df = daily_df_filtered[daily_df_filtered["prediction"] > 0]

        top_predicted = positive_preds_df.sort_values("prediction", ascending=False).head(k)

        if top_predicted.empty:
            daily_gain_predict = 1.0
            daily_gain_actual = 1.0
        else:
            total_prediction_sum = top_predicted["prediction"].sum()
            top_predicted["weights"] = top_predicted["prediction"] / total_prediction_sum
            daily_gain_predict = (
                (1 + top_predicted["prediction"]) * top_predicted["weights"]
            ).sum()
            daily_gain_actual = ((1 + top_predicted["y_actual"]) * top_predicted["weights"]).sum()

        daily_returns_list.append(daily_gain_actual - 1)

        top_actual = daily_df_filtered.sort_values("y_actual", ascending=False).head(k)

        if top_actual.empty:
            daily_gain_optimal = 1.0
        else:
            daily_gain_optimal = (1 + top_actual["y_actual"]).mean()

        gain_actual *= daily_gain_actual

        daily_results_table_data.append(
            {
                "day": date.strftime("%Y-%m-%d"),
                "daily_gain_predict": daily_gain_predict,
                "daily_gain_actual": daily_gain_actual,
                "daily_gain_optimal": daily_gain_optimal,
                "predict_optimal_stocks": [
                    id_to_stock_code(sid) for sid in top_predicted["stock_id"]
                ],
                "actual_optimal_stocks": [id_to_stock_code(sid) for sid in top_actual["stock_id"]],
                "cumulative_gain": gain_actual,
            }
        )

    average_daily_gain = gain_actual ** (1 / day_number) if day_number > 0 else 1.0

    daily_returns_np = np.array(daily_returns_list)
    if np.std(daily_returns_np) > 0:
        daily_sharpe_ratio = np.mean(daily_returns_np) / np.std(daily_returns_np)
        annualized_sharpe_ratio = daily_sharpe_ratio * np.sqrt(settings.TRADING_DAYS_PER_YEAR)
    else:
        annualized_sharpe_ratio = 0.0

    cumulative_returns = np.cumprod(1 + daily_returns_np)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

    sorted_stocks = sorted(
        stock_metrics.keys(),
        key=lambda s: stock_metrics[s][settings.PREDICT_STOCK_OPTIMIZE_METRIC],
        reverse=True,
    )

    start_date = X.index.min()
    end_date = X.index.max()

    if verbose:
        stock_static_table = PrettyTable()
        stock_static_table.field_names = [
            "Stock",
            "MAE",
            "MSE",
            "Sign Acc",
            "Precision",
            "Recall",
            "F1",
            "Avg Daily Gain",
            "Overall Gain",
        ]
        for stock in sorted_stocks:
            metrics = stock_metrics[stock]
            stock_static_table.add_row(
                [
                    stock,
                    f"{metrics['mae']:.7f}",
                    f"{metrics['mse']:.7f}",
                    f"{metrics['sign_acc'] * 100:.2f}%",
                    f"{metrics['precision'] * 100:.2f}%",
                    f"{metrics['recall'] * 100:.2f}%",
                    f"{metrics['f1'] * 100:.2f}%",
                    f"{metrics['avg_daily_gain']:+.4%}",
                    f"{metrics['overall_gain']:+.2%}",
                ],
                divider=True,
            )
        title_str = f"{name} Stock-level Static Result from {start_date} to {end_date}"
        logger.info(f"\n{stock_static_table.get_string(title=title_str)}")

        daily_static_table = PrettyTable()
        daily_static_table.field_names = [
            "Day",
            "Predict Daily Gain",
            "Actual Daily Gain",
            "Optimal Daily Gain",
            "Predicted Stocks",
            "Optimal Stocks",
            "Cumulative Gain",
        ]
        for res in daily_results_table_data:
            daily_static_table.add_row(
                [
                    res["day"],
                    f"{res['daily_gain_predict']:+.2%}",
                    f"{res['daily_gain_actual']:+.2%}",
                    f"{res['daily_gain_optimal']:+.2%}",
                    res["predict_optimal_stocks"],
                    res["actual_optimal_stocks"],
                    f"{res['cumulative_gain']:+.2%}",
                ],
                divider=True,
            )
        title_str = f"{name} Daily-level Static Result from {start_date} to {end_date}"
        logger.info(f"\n{daily_static_table.get_string(title=title_str)}")

    overall_static_table = PrettyTable()
    overall_static_table.field_names = [
        "Trading Days",
        "MAE",
        "MSE",
        "Sign Acc",
        "Precision",
        "Recall",
        "F1",
        "Avg Daily Gain",
        "Overall Gain",
        "Annualized Sharpe Ratio",
        "Max Drawdown",
    ]
    overall_static_table.add_row(
        [
            f"{day_number}",
            f"{mae_overall:.7f}",
            f"{mse_overall:.7f}",
            f"{sign_acc_overall * 100:.2f}%",
            f"{precision_overall * 100:.2f}%",
            f"{recall_overall * 100:.2f}%",
            f"{f1_overall * 100:.2f}%",
            f"{average_daily_gain:+.4%}",
            f"{gain_actual:+.2%}",
            f"{annualized_sharpe_ratio:.3f}",
            f"{max_drawdown:.2%}",
        ],
        divider=True,
    )
    title_str = f"{name} Overall Static Result from {start_date} to {end_date}"
    logger.info(f"\n{overall_static_table.get_string(title=title_str)}")

    return (
        day_number,
        mae_overall,
        mse_overall,
        sign_acc_overall,
        precision_overall,
        recall_overall,
        f1_overall,
        average_daily_gain,
        gain_actual,
        annualized_sharpe_ratio,
        max_drawdown,
        sorted_stocks,
    )


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
    client: schwabdev.Client, account_hash: str, stock_to_buy_in: dict
) -> None:
    """Execute all required trading on schwab via api"""
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
    try:
        account_positions = {
            position["instrument"]["symbol"]: position["longQuantity"]
            for position in client.account_details(account_hash, fields="positions").json()[
                "securitiesAccount"
            ]["positions"]
        }
    except KeyError:
        account_positions = {}

    all_stocks_involved = list(account_positions.keys() | stock_to_buy_in.keys())
    for stock in all_stocks_involved:
        if stock in stock_to_buy_in and stock not in account_positions:
            logger.info(f"New stock: Buy {stock_to_buy_in[stock]} share(s) of {stock}")
            formatted_order = get_schwab_formatted_order(stock, "BUY", stock_to_buy_in[stock])
            client.order_place(account_hash, formatted_order)
        elif stock in account_positions and stock not in stock_to_buy_in:
            logger.info(f"Existing stock: Sell {account_positions[stock]} share(s) of {stock}")
            formatted_order = get_schwab_formatted_order(stock, "SELL", account_positions[stock])
            client.order_place(account_hash, formatted_order)
        else:
            if stock_to_buy_in[stock] == account_positions[stock]:
                logger.info(
                    f"Existing stock: No change for {stock}, "
                    f"already holding {account_positions[stock]} share(s)"
                )
            elif stock_to_buy_in[stock] > account_positions[stock]:
                logger.info(
                    f"Existing stock: Buy {stock_to_buy_in[stock] - account_positions[stock]} "
                    f"additional share(s) of {stock}"
                )
                formatted_order = get_schwab_formatted_order(
                    stock, "BUY", stock_to_buy_in[stock] - account_positions[stock]
                )
                client.order_place(account_hash, formatted_order)
            else:
                logger.info(
                    f"Existing stock: Sell {account_positions[stock] - stock_to_buy_in[stock]} "
                    f"share(s) of {stock}"
                )
                formatted_order = get_schwab_formatted_order(
                    stock, "SELL", account_positions[stock] - stock_to_buy_in[stock]
                )
                client.order_place(account_hash, formatted_order)

    logger.info("All new orders placed")


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
                f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
