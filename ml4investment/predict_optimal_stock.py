import argparse
import logging
import pandas as pd
from prettytable import PrettyTable
from sklearn.preprocessing import RobustScaler

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
    stock_id_map = {stock: stock_code_to_id(stock) for stock in stock_list}
    if len(stock_id_map.values()) != len(set(stock_id_map.values())):
        logger.error(f"Stock mapping mismatch. Stock id number: {len(stock_id_map.values())}, Stock number: {len(set(stock_id_map.values()))}")
        raise ValueError("Stock mapping mismatch.")
    all_stock_ids = list(stock_id_map.values())
    cat_type = pd.CategoricalDtype(categories=all_stock_ids)

    global_X_train, global_X_test = [], []
    global_y_train, global_y_test = [], []
    x_predict_dict = {}
    
    for stock in stock_list:
        fetched_data = fetch_trading_day_data(stock)

        daily_features_data = calculate_features(fetched_data)
        cur_X_train, cur_X_test, cur_y_train, cur_y_test, cur_x_predict = process_features(daily_features_data)

        stock_id = stock_id_map[stock]
        cur_X_train['stock_id'] = stock_id
        cur_X_test['stock_id'] = stock_id
        cur_x_predict['stock_id'] = stock_id

        cur_X_train['stock_id'] = cur_X_train['stock_id'].astype(cat_type)
        cur_X_test['stock_id'] = cur_X_test['stock_id'].astype(cat_type)
        cur_x_predict['stock_id'] = cur_x_predict['stock_id'].astype(cat_type)

        global_X_train.append(cur_X_train)
        global_X_test.append(cur_X_test)
        global_y_train.append(cur_y_train)
        global_y_test.append(cur_y_test)
        x_predict_dict[stock] = cur_x_predict
    
    X_train = pd.concat(global_X_train)
    X_test = pd.concat(global_X_test)
    y_train = pd.concat(global_y_train)
    y_test = pd.concat(global_y_test)
    
    # 2. Train the model based on all stock data
    model = model_training(X_train, X_test, y_train, y_test, categorical_features=['stock_id'])
    
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


def stock_code_to_id(stock_code: str) -> int:
    """ Change the stock string to the sum of ASCII value of each char within the stock code """
    return sum(ord(c) for c in stock_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", "-ss", type=str, default="AMZN,MSFT,AAPL,GOOGL,TSLA,META,INTC,AMD,NVDA,BABA,TCEHY,JD")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    stock_list = args.stocks.split(",")
    seed = args.seed

    predict_optimal_stock(stock_list, seed)
