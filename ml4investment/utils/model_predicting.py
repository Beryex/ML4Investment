import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml4investment.config.global_settings import settings

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
