import logging
import os
import random

import lightgbm as lgb
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from ml4investment.config.global_settings import settings
from ml4investment.utils.model_predicting import (
    get_predict_top_stocks_and_weights,
    model_predict,
)

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


def update_gains(
    sorted_stock_gain_prediction: list,
    sorted_stock_gain_actual: list,
    y_predict_dict: dict,
    y_dict: dict,
    gain_predict: float,
    gain_actual: float,
    gain_optimal: float,
    backtest_day_index: int,
    number_of_stock_to_buy: int = 2,
) -> tuple[float, float, float, float, float, float, list, list]:
    """Update the backtest gains based on the current day's predictions and actual results"""
    predict_top_stock_and_weights_list = get_predict_top_stocks_and_weights(
        sorted_stock_gain_prediction
    )
    daily_number_of_stock_to_buy = len(predict_top_stock_and_weights_list)
    predict_optimal_stocks = [item[0] for item in predict_top_stock_and_weights_list]

    actual_optimal_stocks = [
        item[0] for item in sorted_stock_gain_actual[:number_of_stock_to_buy]
    ]
    weight_optimal = 1.0 / min(number_of_stock_to_buy, len(actual_optimal_stocks))
    daily_gain_optimal = sum(
        (1 + y_dict[backtest_day_index][s]) * weight_optimal
        for s in actual_optimal_stocks
    )
    gain_optimal *= daily_gain_optimal

    if daily_number_of_stock_to_buy == 0:
        return (
            gain_predict,
            gain_actual,
            gain_optimal,
            1.0,
            1.0,
            daily_gain_optimal,
            [],
            actual_optimal_stocks,
        )

    daily_gain_predict = sum(
        (1 + y_predict_dict[backtest_day_index][s]) * w
        for (s, w) in predict_top_stock_and_weights_list
    )
    gain_predict *= daily_gain_predict

    daily_gain_actual = sum(
        (1 + y_dict[backtest_day_index][s]) * w
        for (s, w) in predict_top_stock_and_weights_list
    )
    gain_actual *= daily_gain_actual

    return (
        gain_predict,
        gain_actual,
        gain_optimal,
        daily_gain_predict,
        daily_gain_actual,
        daily_gain_optimal,
        predict_optimal_stocks,
        actual_optimal_stocks,
    )


def get_detailed_static_result(
    model: lgb.Booster,
    X_dict: dict,
    y_dict: dict,
    predict_stock_list: list,
    start_date: str,
    end_date: str,
    name: str = "",
    verbose: bool = True,
) -> tuple[float, float]:
    """Display detailed static result of the model predictions within a single function structure"""
    assert len(X_dict) == len(y_dict), "Length of X_dict and y_dict must be the same"
    day_number = len(X_dict)

    target_y_list = []
    predict_y_list = []
    y_predict_dict = {}

    for i in range(day_number):
        y_predict_dict[i] = {}
        for stock in predict_stock_list:
            prediction = model_predict(model, X_dict[i][stock])

            y_predict_dict[i][stock] = prediction
            target_y_list.append(y_dict[i][stock])
            predict_y_list.append(prediction)

    gain_predict, gain_actual, gain_optimal = 1.0, 1.0, 1.0
    daily_results_for_table = []

    for i in range(day_number):
        sorted_stock_gain_prediction = sorted(
            y_predict_dict[i].items(), key=lambda x: x[1], reverse=True
        )
        sorted_stock_gain_actual = sorted(
            y_dict[i].items(), key=lambda x: x[1], reverse=True
        )

        (
            gain_predict,
            gain_actual,
            gain_optimal,
            daily_gain_predict,
            daily_gain_actual,
            daily_gain_optimal,
            cur_optimal_stocks,
            cur_actual_optimal_stocks,
        ) = update_gains(
            sorted_stock_gain_prediction,
            sorted_stock_gain_actual,
            y_predict_dict,
            y_dict,
            gain_predict,
            gain_actual,
            gain_optimal,
            backtest_day_index=i,
            number_of_stock_to_buy=settings.NUMBER_OF_STOCKS_TO_BUY,
        )

        cur_day = {str(stock.index[0]) for stock in X_dict[i].values()}.pop()

        daily_results_for_table.append(
            {
                "day": cur_day,
                "daily_gain_predict": daily_gain_predict,
                "daily_gain_actual": daily_gain_actual,
                "daily_gain_optimal": daily_gain_optimal,
                "predict_optimal_stocks": cur_optimal_stocks,
                "actual_optimal_stocks": cur_actual_optimal_stocks,
                "cumulative_gain_actual": gain_actual,
            }
        )

    mae_overall = mean_absolute_error(target_y_list, predict_y_list)
    mse_overall = mean_squared_error(target_y_list, predict_y_list)

    target_y_np = np.array(target_y_list)
    predict_y_np = np.array(predict_y_list)

    sign_acc_overall = (np.sign(predict_y_np) == np.sign(target_y_np)).mean()

    binary_y_true = (target_y_np > 0).astype(int)
    binary_y_pred = (predict_y_np > 0).astype(int)

    precision_overall = precision_score(binary_y_true, binary_y_pred, zero_division=0)
    recall_overall = recall_score(binary_y_true, binary_y_pred, zero_division=0)
    f1_overall = f1_score(binary_y_true, binary_y_pred, zero_division=0)

    if verbose:
        detailed_table = PrettyTable()
        detailed_table.field_names = [
            "Day",
            "Predict daily gain",
            "Actual daily gain",
            "Optimal daily gain",
            "Predict optimal stock to buy",
            "Actual optimal stock to buy",
            "Overall Actual Gain",
        ]
        for res in daily_results_for_table:
            pred_stocks_str = [
                f"{s} ({settings.STOCK_SECTOR_ID_MAP.get(s, 'N/A')})"
                for s in res["predict_optimal_stocks"]
            ]
            actual_stocks_str = [
                f"{s} ({settings.STOCK_SECTOR_ID_MAP.get(s, 'N/A')})"
                for s in res["actual_optimal_stocks"]
            ]

            row = [
                res["day"],
                f"{res['daily_gain_predict']:+.2%}",
                f"{res['daily_gain_actual']:+.2%}",
                f"{res['daily_gain_optimal']:+.2%}",
                pred_stocks_str,
                actual_stocks_str,
                f"{res['cumulative_gain_actual']:+.2%}",
            ]
            detailed_table.add_row(row, divider=True)

        detailed_table.add_row(
            [
                "Overall",
                f"{gain_predict:+.2%}",
                f"{gain_actual:+.2%}",
                f"{gain_optimal:+.2%}",
                "N/A",
                "N/A",
                "N/A",
            ],
            divider=True,
        )
        logger.info(
            f"\n{detailed_table.get_string(title=f'Detailed Backtest Result from {start_date} to {end_date}')}"
        )

    else:
        backtest_table = PrettyTable()
        backtest_table.field_names = [
            "Backtest Trading Day Number",
            "Predict overall gain",
            "Actual overall gain",
            "Optimal overall gain",
            "Efficiency",
        ]
        efficiency = (gain_actual / gain_optimal) if gain_optimal != 0 else 0
        backtest_table.add_row(
            [
                day_number,
                f"{gain_predict:+.2%}",
                f"{gain_actual:+.2%}",
                f"{gain_optimal:+.2%}",
                f"{efficiency:.2%}",
            ],
            divider=True,
        )
        logger.info(
            f"\n{backtest_table.get_string(title=f'Backtest Result from {start_date} to {end_date}')}"
        )

    detailed_static_table = PrettyTable()
    detailed_static_table.field_names = [
        "MAE",
        "MSE",
        "Sign Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Actual Gain",
    ]
    row = [
        f"{mae_overall:.7f}",
        f"{mse_overall:.7f}",
        f"{sign_acc_overall * 100:.2f}%",
        f"{precision_overall * 100:.2f}%",
        f"{recall_overall * 100:.2f}%",
        f"{f1_overall * 100:.2f}%",
        f"{gain_actual:+.2%}",
    ]
    detailed_static_table.add_row(row, divider=True)
    logger.info(
        f"\n{detailed_static_table.get_string(title=f'{name} Detailed Static Result')}"
    )

    return mae_overall, mse_overall


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
