import logging

from ml4investment.utils.model_predicting import get_predict_top_stocks_and_weights

logger = logging.getLogger(__name__)


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
    predict_top_stock_and_weights_list = get_predict_top_stocks_and_weights(sorted_stock_gain_backtest_prediction)
    daily_number_of_stock_to_buy = len(predict_top_stock_and_weights_list)
    predict_optimal_stocks = [item[0] for item in predict_top_stock_and_weights_list]

    actual_optimal_stocks = [item[0] for item in sorted_stock_gain_backtest_actual[:number_of_stock_to_buy]]
    weight_optimal = 1.0 / min(number_of_stock_to_buy, len(actual_optimal_stocks))
    daily_gain_optimal = sum((1 + y_backtest_dict[backtest_day_index][s]) * weight_optimal for s in actual_optimal_stocks)
    gain_optimal *= daily_gain_optimal

    if daily_number_of_stock_to_buy == 0:
        return gain_predict, gain_actual, gain_optimal, 1.0, 1.0, daily_gain_optimal, [], actual_optimal_stocks

    daily_gain_predict = sum((1 + y_predict_dict[backtest_day_index][s]) * w for (s, w) in predict_top_stock_and_weights_list)
    gain_predict *= daily_gain_predict

    daily_gain_actual  = sum((1 + y_backtest_dict[backtest_day_index][s]) * w for (s, w) in predict_top_stock_and_weights_list)
    gain_actual  *= daily_gain_actual

    return gain_predict, gain_actual, gain_optimal, daily_gain_predict, daily_gain_actual, daily_gain_optimal, predict_optimal_stocks, actual_optimal_stocks
