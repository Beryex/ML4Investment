import argparse
import logging
import pandas as pd
from prettytable import PrettyTable

from ml4investment.utils.seed import set_random_seed
from ml4investment.utils.data_loader import fetch_trading_day_data
from ml4investment.utils.logging import configure_logging
from ml4investment.utils.feature_engineering import calculate_features, process_features
from ml4investment.utils.model_training import model_training
from ml4investment.utils.model_predicting import model_predict

configure_logging(env="prod", file_name="predict_optimal_stock.log")
logger = logging.getLogger("ml4investment.test")


def predict_optimal_stock(stock_list, seed):
    """ Predict the optimal stock with the highest price change for the given stocks """
    logger.info(f"Start predicting stocks: {stock_list}")
    logger.info(f"Current trading time: {pd.Timestamp.now(tz='America/New_York')}")
    set_random_seed(seed)

    # 1. Combine all stock data
    global_X_train, global_X_test = [], []
    global_y_train, global_y_test = [], []
    x_predict_dict = {}
    
    for stock in stock_list:
        fetched_data = fetch_trading_day_data(stock)

        daily_features_data = calculate_features(fetched_data)
        cur_X_train, cur_X_test, cur_y_train, cur_y_test, cur_x_predict = process_features(daily_features_data)

        cur_X_train['stock'] = stock
        cur_X_test['stock'] = stock
        cur_x_predict['stock'] = stock

        global_X_train.append(cur_X_train)
        global_X_test.append(cur_X_test)
        global_y_train.append(cur_y_train)
        global_y_test.append(cur_y_test)
        x_predict_dict[stock] = cur_x_predict
    
    X_train = pd.concat(global_X_train)
    X_test = pd.concat(global_X_test)
    y_train = pd.concat(global_y_train)
    y_test = pd.concat(global_y_test)

    X_train = pd.get_dummies(X_train, columns=['stock'])
    X_test = pd.get_dummies(X_test, columns=['stock'])
    for stock in x_predict_dict:
        x_predict_dict[stock] = pd.get_dummies(x_predict_dict[stock], columns=['stock'])
    
    # 2. Train the model based on all stock data
    model = model_training(X_train, X_test, y_train, y_test)
    
    # 3. Predict each stock using the trained model
    results_table = PrettyTable()
    results_table.field_names = ["Stock", "Predicted Price Change"]
    
    optimal_ratio = float('-inf')
    optimal_stock = ""
    for stock, x_predict in x_predict_dict.items():
        predict_ratio = model_predict(model, x_predict)
        results_table.add_row([stock, predict_ratio], divider=True)
        if predict_ratio > optimal_ratio:
            optimal_ratio = predict_ratio
            optimal_stock = stock

    logger.info(results_table.get_string(title=f"Predict price changes for stocks: {stock_list}"))
    logger.info(f"Suggested optimal stock: {optimal_stock} with predicted price change: {optimal_ratio:+.2%}")
    logger.info("Prediction process completed.")

    return optimal_stock, optimal_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", "-ss", type=str, default="AMZN,MSFT,AAPL,GOOGL,TSLA,META,INTC,AMD,NVDA,BABA,TCEHY,JD")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    stock_list = args.stocks.split(",")
    seed = args.seed

    predict_optimal_stock(stock_list, seed)
