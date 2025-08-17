import datetime
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
import schwabdev
from prettytable import PrettyTable
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from ml4investment.config.global_settings import settings
from ml4investment.utils.utils import get_schwab_formatted_order

logger = logging.getLogger(__name__)


def model_predict(model: lgb.Booster, x_predict: pd.DataFrame) -> float:
    """Prediction function with validation checks"""
    expected_features = model.feature_name()
    received_features = x_predict.columns.tolist()

    if len(expected_features) != len(received_features):
        missing = set(expected_features) - set(received_features)
        extra = set(received_features) - set(expected_features)
        logger.error(f"Feature dimension mismatch. Missing: {missing}, Extra: {extra}")
        raise ValueError("Feature mismatch between training and prediction data")

    if not np.all(expected_features == received_features):
        x_predict = x_predict.reindex(columns=expected_features)
        logger.warning("Feature order mismatch detected, auto-corrected column order")

    if not x_predict.columns.equals(pd.Index(expected_features)):
        x_predict = x_predict.reindex(columns=expected_features)

    if not isinstance(x_predict, pd.DataFrame):
        logger.error(f"Invalid input type: {type(x_predict)}, expected DataFrame")
        raise TypeError("Input must be pandas DataFrame")

    try:
        X = x_predict.values.reshape(1, -1)
        y_pred = model.predict(X, num_iteration=model.best_iteration)
        assert isinstance(y_pred, np.ndarray)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise RuntimeError("Prediction error occurred") from e

    prediction = float(y_pred[0])

    return prediction


def get_predict_top_stocks_and_weights(sorted_stock_gain_prediction: list) -> list:
    """Get the top stocks and their recommended weights for investment"""
    top_stocks = [item for item in sorted_stock_gain_prediction if item[1] > 0][
        : settings.NUMBER_OF_STOCKS_TO_BUY
    ]
    actual_number_selected = len(top_stocks)

    if actual_number_selected == 0:
        return []

    predicted_returns = [value for _, value in top_stocks]
    total_pred = sum(predicted_returns)
    weights = [ret / total_pred for ret in predicted_returns]

    predict_top_stock_and_weights_list = [
        (stock, weight) for (stock, _), weight in zip(top_stocks, weights)
    ]

    return predict_top_stock_and_weights_list


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
) -> tuple[float, float, float, float, float, float, float]:
    """Display detailed static result of the model predictions within a single function structure"""
    assert len(X_dict) == len(y_dict), "Length of X_dict and y_dict must be the same"
    day_number = len(X_dict)

    target_y_list = []
    predict_y_list = []
    y_predict_dict = {}

    for i in range(day_number):
        y_predict_dict[i] = {}
        for stock in predict_stock_list:
            if stock not in X_dict[i]:
                logger.warning(
                    f"Stock {stock} not found in prediction data for day {i}"
                )
                continue

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

    return (
        mae_overall,
        mse_overall,
        float(sign_acc_overall),
        float(precision_overall),
        float(recall_overall),
        float(f1_overall),
        gain_actual,
    )


def perform_schwab_trade(
    client: schwabdev.Client, account_hash: str, stock_to_buy_in: dict
) -> None:
    """Execute all required trading on schwab via api"""
    logger.info("Performing Schwab trade...")
    account_orders = client.account_orders(
        account_hash,
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(
            days=7
        ),  # 7 days is sufficient for daily usage, hardcoded here
        datetime.datetime.now(datetime.timezone.utc),
    ).json()

    logger.info("Canceling previous active orders...")
    opening_orders = [
        order
        for order in account_orders
        if order.get("status") in settings.OPENING_STATUS
    ]

    for order in opening_orders:
        order_detail = order["orderLegCollection"][0]
        logger.info(
            f"{order_detail['instruction']} {order_detail['quantity']} share(s) of {order_detail['instrument']['symbol']}"
        )
        client.order_cancel(account_hash, order["orderId"])
    logger.info("All previous active orders canceled")

    logger.info("Placing new orders...")
    account_positions = {
        position["instrument"]["symbol"]: position["longQuantity"]
        for position in client.account_details(account_hash, fields="positions").json()[
            "securitiesAccount"
        ]["positions"]
    }

    all_stocks_involved = list(account_positions.keys() | stock_to_buy_in.keys())
    for stock in all_stocks_involved:
        if stock in stock_to_buy_in and stock not in account_positions:
            logger.info(f"New stock: Buy {stock_to_buy_in[stock]} share(s) of {stock}")
            formatted_order = get_schwab_formatted_order(
                stock, "BUY", stock_to_buy_in[stock]
            )
            client.order_place(account_hash, formatted_order)
        elif stock in account_positions and stock not in stock_to_buy_in:
            logger.info(
                f"Existing stock: Sell {account_positions[stock]} share(s) of {stock}"
            )
            formatted_order = get_schwab_formatted_order(
                stock, "SELL", account_positions[stock]
            )
            client.order_place(account_hash, formatted_order)
        else:
            if stock_to_buy_in[stock] == account_positions[stock]:
                logger.info(
                    f"Existing stock: No change for {stock}, already holding {account_positions[stock]} share(s)"
                )
            elif stock_to_buy_in[stock] > account_positions[stock]:
                logger.info(
                    f"Existing stock: Buy {stock_to_buy_in[stock] - account_positions[stock]} additional share(s) of {stock}"
                )
                formatted_order = get_schwab_formatted_order(
                    stock, "BUY", stock_to_buy_in[stock] - account_positions[stock]
                )
                client.order_place(account_hash, formatted_order)
            else:
                logger.info(
                    f"Existing stock: Sell {account_positions[stock] - stock_to_buy_in[stock]} share(s) of {stock}"
                )
                formatted_order = get_schwab_formatted_order(
                    stock, "SELL", account_positions[stock] - stock_to_buy_in[stock]
                )
                client.order_place(account_hash, formatted_order)
    logger.info("All new orders placed")
