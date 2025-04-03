import argparse
import logging
import pandas as pd
from prettytable import PrettyTable
import json
import pickle
import lightgbm as lgb

from ml4investment.config import settings
from ml4investment.utils.utils import set_random_seed
from ml4investment.utils.data_loader import fetch_trading_day_data
from ml4investment.utils.logging import configure_logging
from ml4investment.utils.feature_engineering import calculate_features, process_features_for_predict
from ml4investment.utils.model_predicting import model_predict

configure_logging(env="prod", file_name="predict.log")
logger = logging.getLogger("ml4investment.predict")


def predict(train_stock_list: list, predict_stock_list: list, fetched_data: dict, process_feature_config_pth: str, model_pth: str, seed: int):
    """ Predict the optimal stock with the highest price change for the given stocks """
    logger.info(f"Start predict the given stocks: {predict_stock_list}")
    logger.info(f"Current trading time: {pd.Timestamp.now(tz='America/New_York')}")
    set_random_seed(seed)

    predict_data = {}
    for stock in train_stock_list:
        predict_data[stock] = fetched_data[stock]
    logger.info(f"Load input fetched data")

    daily_features_data = calculate_features(predict_data)
    
    with open(process_feature_config_pth, 'rb') as f:
        process_feature_config = pickle.load(f)
    logger.info(f"Load processing features configuration from {process_feature_config_pth}")
    X_predict_dict = process_features_for_predict(daily_features_data, process_feature_config)
    
    predict_dates = {str(X_predict.index[0]) for X_predict in X_predict_dict.values()}
    if len(predict_dates) != 1:
        logger.error(f"Predict date mismatched: {predict_dates}")
        raise ValueError(f"Predict date mismatched: {predict_dates}")
    predict_date = predict_dates.pop()
    logger.info(f"Predicting based on data on {predict_date}")

    model = lgb.Booster(model_file=model_pth)
    logger.info(f"Load LGB model from {model_pth}")
    
    predictions = {}
    for stock in predict_stock_list:
        today_pred = model_predict(model, X_predict_dict[stock])
        predictions[stock] = today_pred
    
    results_table = PrettyTable()
    field_names = ["Stock"]
    field_names.append("Open_Price_Change_Predict")
    results_table.field_names = field_names
    for stock in sorted(predictions, key=predictions.get, reverse=True):
        row = [stock, f"{predictions[stock]:+.2%}"]
        results_table.add_row(row, divider=True)
    logger.info(f'\n{results_table.get_string(title="Predict price changes for stocks")}')

    sorted_stock_gain_prediction = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_stocks = [item for item in sorted_stock_gain_prediction if item[1] > 0][:settings.NUMBER_OF_STOCKS_TO_BUY]
    actual_number_selected = len(top_stocks)

    if actual_number_selected == 0:
        logger.info("No stocks were recommended today (no positive predicted returns).")
        return
    else:
        predicted_returns = [value for _, value in top_stocks]
        total_pred = sum(predicted_returns)
        weights = [ret / total_pred for ret in predicted_returns]

        logger.info(f"Suggested top {actual_number_selected} stocks to buy:")
        for (stock, pred), weight in zip(top_stocks, weights):
            logger.info(f"  - {stock:>6} | predicted change: {pred:+.2%} | recommended weight: {weight:.2%}")

    logger.info("Prediction process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_stocks", "-ts", type=str, default='config/train_stocks.json')
    parser.add_argument("--predict_stocks", "-ps", type=str, default='config/predict_stocks.json')
    parser.add_argument("--fetched_data_pth", "-fdp", type=str, default='data/fetched_data.pkl')

    parser.add_argument("--process_feature_config_pth", "-pfcp", type=str, default='data/prod_process_feature_config.pkl')
    parser.add_argument("--model_pth", "-mp", type=str, default='data/prod_model.model')

    parser.add_argument("--seed", "-s", type=int, default=42)

    args = parser.parse_args()

    train_stock_list = json.load(open(args.train_stocks, 'r'))["train_stocks"]
    predict_stock_list = json.load(open(args.predict_stocks, 'r'))["predict_stocks"]
    fetched_data = pickle.load(open(args.fetched_data_pth, 'rb'))
    process_feature_config_pth = args.process_feature_config_pth
    model_pth = args.model_pth
    seed = args.seed

    predict(train_stock_list, predict_stock_list, fetched_data, process_feature_config_pth, model_pth, seed)
