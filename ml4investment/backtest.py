import argparse
import logging
import pandas as pd
from prettytable import PrettyTable
import json
import pickle
import lightgbm as lgb

from ml4investment.utils.seed import set_random_seed
from ml4investment.utils.data_loader import fetch_trading_day_data
from ml4investment.utils.logging import configure_logging
from ml4investment.utils.feature_engineering import calculate_features, process_features_for_backtest
from ml4investment.utils.model_predicting import model_predict

configure_logging(env="prod", file_name="backtest.log")
logger = logging.getLogger("ml4investment.backtest")


def predict(predict_stock_list: list, process_feature_config_pth: str, model_pth: str, seed: int):
    """ Backtest the model performance for the given stocks for the last week """
    logger.info(f"Start backtesting based on the given stocks: {predict_stock_list}")
    logger.info(f"Current trading time: {pd.Timestamp.now(tz='America/New_York')}")
    set_random_seed(seed)

    fetched_data = fetch_trading_day_data(predict_stock_list, period = '1mo')

    daily_features_data = calculate_features(fetched_data)

    with open(process_feature_config_pth, 'rb') as f:
        process_feature_config = pickle.load(f)
    logger.info(f"Load processing features configuration from {process_feature_config_pth}")
    X_backtest_dict, y_backtest_dict, backtest_day_number = process_features_for_backtest(daily_features_data, process_feature_config)

    backtest_oldest_dates = {X_backtest.index.min() for X_backtest in X_backtest_dict[0].values()}
    if len(backtest_oldest_dates) != 1:
        logger.error("Oldest backtest date mismatched")
        raise ValueError("Oldest backtest date mismatched")
    backtest_oldest_date = backtest_oldest_dates.pop()

    backtest_newest_dates = {X_backtest.index.max() for X_backtest in X_backtest_dict[backtest_day_number - 1].values()}
    if len(backtest_newest_dates) != 1:
        logger.error("Oldest backtest date mismatched")
        raise ValueError("Oldest backtest date mismatched")
    backtest_newest_date = backtest_newest_dates.pop()

    logger.info(f"Oldest date in backtest data: {backtest_oldest_date}")
    logger.info(f"Newest date in backtest data: {backtest_newest_date}")

    model = lgb.Booster(model_file=model_pth)
    logger.info(f"Load LGB model from {model_pth}")
    
    backtest_results = {}
    for i in range(backtest_day_number):
        backtest_results[i] = {}
        for stock in predict_stock_list:
            backtest_results[i][stock] = model_predict(model, X_backtest_dict[i][stock])

    results_table = PrettyTable()
    results_table.field_names = [
        "Day", 
        "Predict First Optimal Stock", 
        "Predict First Optimal Price Change", 
        "Actual First Optimal Price Change", 
        "Predict Second Optimal Stock", 
        "Predict Second Optimal Price Change", 
        "Actual Second Optimal Price Change"
    ]
    gain_predict = 1
    gain_actual = 1
    for i in range(backtest_day_number):
        sorted_stock_gain_backtest_prediction = sorted(backtest_results[i].items(), key=lambda x: x[1], reverse=True)
        cur_first_optimal_stock = sorted_stock_gain_backtest_prediction[0][0]
        cur_second_optimal_stock = sorted_stock_gain_backtest_prediction[1][0]
        if backtest_results[i][cur_first_optimal_stock] > 0 and backtest_results[i][cur_second_optimal_stock] > 0:
            gain_predict = 0.5 * gain_predict * (1 + backtest_results[i][cur_first_optimal_stock]) + 0.5 * gain_predict * (1 + backtest_results[i][cur_second_optimal_stock])
            gain_actual = 0.5 * gain_actual * (1 + y_backtest_dict[i][cur_first_optimal_stock]) + 0.5 * gain_actual * (1 + y_backtest_dict[i][cur_second_optimal_stock])
        cur_day = str(X_backtest_dict[i][cur_first_optimal_stock].index[0])
        row = [
            cur_day, 
            cur_first_optimal_stock, 
            f"{backtest_results[i][cur_first_optimal_stock]:+.2%}", 
            f"{y_backtest_dict[i][cur_first_optimal_stock]:+.2%}", 
            cur_second_optimal_stock, 
            f"{backtest_results[i][cur_second_optimal_stock]:+.2%}", 
            f"{y_backtest_dict[i][cur_second_optimal_stock]:+.2%}"
        ]
        results_table.add_row(row, divider=True)
    logger.info(f'\n{results_table.get_string(title=f"Backtest price changes for stocks")}')

    logger.info(f"Backtesting for last {backtest_day_number} days: Predict overall gain {gain_predict:+.2%}, Actual overall gain: {gain_actual:+.2%}")

    logger.info("Backtesting process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_stocks", "-ps", type=str, default='config/predict_stocks.json')

    parser.add_argument("--process_feature_config_pth", "-pfcp", type=str, default='data/prod_process_feature_config.pkl')
    parser.add_argument("--model_pth", "-mp", type=str, default='data/prod_model.model')

    parser.add_argument("--seed", "-s", type=int, default=42)

    args = parser.parse_args()

    predict_stock_list = json.load(open(args.predict_stocks, 'r'))["predict_stocks"]
    process_feature_config_pth = args.process_feature_config_pth
    model_pth = args.model_pth
    seed = args.seed

    predict(predict_stock_list, process_feature_config_pth, model_pth, seed)
