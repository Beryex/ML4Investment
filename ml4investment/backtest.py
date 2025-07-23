import argparse
import logging
import pandas as pd
from prettytable import PrettyTable
import json
import pickle
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error

from ml4investment.config import settings
from ml4investment.utils.utils import set_random_seed
from ml4investment.utils.logging import configure_logging
from ml4investment.utils.feature_engineering import calculate_features, process_features_for_backtest
from ml4investment.utils.model_predicting import model_predict
from ml4investment.utils.model_backtesting import update_backtest_gains

configure_logging(env="backtest", file_name="backtest.log")
logger = logging.getLogger("ml4investment.backtest")


def backtest(train_stock_list:list, predict_stock_list: list, fetched_data: dict, process_feature_config: dict, selected_features: dict, model: lgb.Booster, seed: int):
    """ Backtest the model performance for the given stocks for the last week """
    logger.info(f"Start backtesting based on the given stocks: {predict_stock_list}")
    logger.info(f"Current trading time: {pd.Timestamp.now(tz='America/New_York')}")
    set_random_seed(seed)

    backtest_data = {}
    train_data_start_date = settings.TRAINING_DATA_START_DATE
    for stock in train_stock_list:
        backtest_data[stock] = fetched_data[stock].loc[train_data_start_date:]
    logger.info(f"Load input fetched data")

    daily_features_data = calculate_features(backtest_data)

    X_backtest_dict, y_backtest_dict, backtest_day_number = process_features_for_backtest(daily_features_data, process_feature_config, predict_stock_list)

    for i in range(backtest_day_number):
        for stock, data in X_backtest_dict[i].items():
            X_backtest_dict[i][stock] = data[selected_features]

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

    feature_nums = {len(list(X_predict.columns)) for X_predict in X_backtest_dict[0].values()}
    if len(feature_nums) != 1:
        logger.error(f"Feature number mismatched: {feature_nums}")
        raise ValueError(f"Feature number mismatched: {feature_nums}")
    feature_num = feature_nums.pop()
    logger.info(f"Number of features: {feature_num}")
    
    y_predict_dict = {}
    for i in range(backtest_day_number):
        y_predict_dict[i] = {}
        for stock in predict_stock_list:
            y_predict_dict[i][stock] = model_predict(model, X_backtest_dict[i][stock])

    detailed_backtest_table = PrettyTable()
    detailed_backtest_table.field_names = [
        "Day", 
        "Predict daily gain", 
        "Actual daily gain", 
        "Optimal daily gain", 
        "Predict optimal stock to buy",
        "Actual optimal stock to buy",
        "Overall Actual Gain"
    ]
    gain_predict = 1
    gain_actual = 1
    gain_optimal = 1
    for i in range(backtest_day_number):
        sorted_stock_gain_backtest_prediction = sorted(y_predict_dict[i].items(), key=lambda x: x[1], reverse=True)
        sorted_stock_gain_backtest_actual = sorted(y_backtest_dict[i].items(), key=lambda x: x[1], reverse=True)
        gain_predict, gain_actual, gain_optimal, daily_gain_predict, daily_gain_actual, daily_gain_optimal, cur_optimal_stocks, cur_actual_optimal_stocks = update_backtest_gains(
            sorted_stock_gain_backtest_prediction,
            sorted_stock_gain_backtest_actual,
            y_predict_dict,
            y_backtest_dict,
            gain_predict,
            gain_actual,
            gain_optimal,
            backtest_day_index = i,
            number_of_stock_to_buy = settings.NUMBER_OF_STOCKS_TO_BUY
        )
        cur_day = {str(stock.index[0]) for stock in X_backtest_dict[i].values()}.pop()
        cur_optimal_stocks_with_sector = []
        for stock in cur_optimal_stocks:
            cur_optimal_stocks_with_sector.append(f"{stock} ({settings.STOCK_SECTOR_ID_MAP[stock]})")
        cur_actual_optimal_stocks_with_sector = []
        for stock in cur_actual_optimal_stocks:
            cur_actual_optimal_stocks_with_sector.append(f"{stock} ({settings.STOCK_SECTOR_ID_MAP[stock]})")
        
        row = [
            cur_day, 
            f"{daily_gain_predict:+.2%}",
            f"{daily_gain_actual:+.2%}",
            f"{daily_gain_optimal:+.2%}", 
            cur_optimal_stocks_with_sector,
            cur_actual_optimal_stocks_with_sector,
            f"{gain_actual:+.2%}"
        ]
        detailed_backtest_table.add_row(row, divider=True)
    detailed_backtest_table.add_row(["Overall", f"{gain_predict:+.2%}", f"{gain_actual:+.2%}", f"{gain_optimal:+.2%}", "N/A", "N/A", "N/A"], divider=True)

    backtest_table = PrettyTable()
    backtest_table.field_names = [
        "Backtest Trading Day Number", 
        "Predict overall gain", 
        "Actual overall gain", 
        "Optimal overall gain", 
        "Efficiency"
    ]
    backtest_table.add_row([backtest_day_number, f"{gain_predict:+.2%}", f"{gain_actual:+.2%}", f"{gain_optimal:+.2%}", f"{(gain_actual/gain_optimal):.2%}"], divider=True)

    if args.verbose:
        logger.info(f"\n{detailed_backtest_table.get_string(title=f'Detailed Backtest Result from {backtest_oldest_date} to {backtest_newest_date}')}")
    else:
        logger.info(f"\n{backtest_table.get_string(title=f'Backtest Result from {backtest_oldest_date} to {backtest_newest_date}')}")

    all_predictions = []
    all_actuals = []
    for i in range(backtest_day_number):
        for stock in predict_stock_list:
            all_predictions.append(y_predict_dict[i][stock])
            all_actuals.append(y_backtest_dict[i][stock])
    all_actuals_np = np.array(all_actuals)
    all_predictions_np = np.array(all_predictions)

    overall_mae = mean_absolute_error(all_actuals_np, all_predictions_np)
    sign_matches = (np.sign(all_predictions_np) == np.sign(all_actuals_np))
    overall_sign_accuracy = np.mean(sign_matches)
    logger.info(f"Backtest Overall Metrics - MAE: {overall_mae:.4f} | Sign Accuracy: {overall_sign_accuracy*100:.2f}%")

    logger.info("Backtesting process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_stocks", "-ts", type=str, default='config/train_stocks.json')
    parser.add_argument("--predict_stocks", "-ps", type=str, default='config/predict_stocks.json')
    parser.add_argument("--fetched_data_pth", "-fdp", type=str, default='data/fetched_data.pkl')

    parser.add_argument("--process_feature_config_pth", "-pfcp", type=str, default='data/prod_process_feature_config.pkl')
    parser.add_argument("--features_pth", "-fp", type=str, default='data/prod_model_features.json')
    parser.add_argument("--model_pth", "-mp", type=str, default='data/prod_model.model')

    parser.add_argument("--verbose", "-v", action='store_true', default=False)
    parser.add_argument("--seed", "-s", type=int, default=42)

    args = parser.parse_args()

    train_stock_list = json.load(open(args.train_stocks, 'r'))["train_stocks"]
    predict_stock_list = json.load(open(args.predict_stocks, 'r'))["predict_stocks"]
    fetched_data = pickle.load(open(args.fetched_data_pth, 'rb'))
    process_feature_config = pickle.load(open(args.process_feature_config_pth, 'rb'))
    selected_features =json.load(open(args.features_pth, 'r'))["features"]
    model = lgb.Booster(model_file=args.model_pth)
    seed = args.seed

    backtest(train_stock_list, predict_stock_list, fetched_data, process_feature_config, selected_features, model, seed)
