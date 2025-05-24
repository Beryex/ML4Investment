import os
import random
import numpy as np
import logging

from ml4investment.config import settings

logger = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """ Set random seed for reproducible usage """
    logger.info(f"Set random seed: {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def stock_code_to_id(stock_code: str) -> int:
    """ Change the stock string to the sum of ASCII value of each char within the stock code """
    return sum(ord(c) * 256 ** i for i, c in enumerate(reversed(stock_code)))


def id_to_stock_code(code_id: int) -> str:
    """  Change the stock id to the string of stock code """
    chars = []
    while code_id > 0:
        ascii_val = code_id % 256
        chars.append(chr(ascii_val))
        code_id //= 256
    return ''.join(reversed(chars))


def update_backtest_gains(
    sorted_stock_gain_backtest_prediction: list, 
    sorted_stock_gain_backtest_actual: list, 
    y_predict_dict: dict, 
    y_backtest_dict: dict, 
    gain_predict: float, 
    gain_actual: float, 
    gain_optimal: float, 
    backtest_day_index: int, 
    number_of_stock_to_buy: int = 2 
) -> tuple[float, float, float, float, float, float, list, list]:
    """ Update the backtest gains based on the current day's predictions and actual results """
    cur_optimal_stocks = [item[0] for item in sorted_stock_gain_backtest_prediction[:number_of_stock_to_buy]]
    cur_optimal_stocks = [s for s in cur_optimal_stocks if y_predict_dict[backtest_day_index][s] > 0]
    daily_number_of_stock_to_buy = len(cur_optimal_stocks)

    actual_optimal_stocks = [item[0] for item in sorted_stock_gain_backtest_actual[:number_of_stock_to_buy]]
    weight_optimal = 1.0 / min(number_of_stock_to_buy, len(actual_optimal_stocks))
    daily_gain_optimal = sum((1 + y_backtest_dict[backtest_day_index][s]) * weight_optimal for s in actual_optimal_stocks)
    gain_optimal *= daily_gain_optimal

    if daily_number_of_stock_to_buy == 0:
        return gain_predict, gain_actual, gain_optimal, 1.0, 1.0, daily_gain_optimal, [], actual_optimal_stocks
    
    predicted_returns = [y_predict_dict[backtest_day_index][s] for s in cur_optimal_stocks]
    total_pred_sum = sum(predicted_returns)
    weights = [pred / total_pred_sum for pred in predicted_returns]

    daily_gain_predict = sum((1 + y_predict_dict[backtest_day_index][s]) * w for s, w in zip(cur_optimal_stocks, weights))
    daily_gain_actual  = sum((1 + y_backtest_dict[backtest_day_index][s]) * w for s, w in zip(cur_optimal_stocks, weights))

    gain_predict *= daily_gain_predict
    gain_actual  *= daily_gain_actual

    return gain_predict, gain_actual, gain_optimal, daily_gain_predict, daily_gain_actual, daily_gain_optimal, cur_optimal_stocks, actual_optimal_stocks
    