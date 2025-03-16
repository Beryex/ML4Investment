import lightgbm as lgb
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def model_predict(model: lgb.Booster, x_predict: pd.DataFrame) -> float:
    """ Prediction function with validation checks """
    # 1. Feature validation
    if 'stock_id' in x_predict.columns:
        assert x_predict['stock_id'].dtype.name == 'category'
        
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

    # 2. Data type consistency check
    if not isinstance(x_predict, pd.DataFrame):
        logger.error(f"Invalid input type: {type(x_predict)}, expected DataFrame")
        raise TypeError("Input must be pandas DataFrame")
        
    # 3. Prediction execution
    try:
        # Convert to 2D array even for single sample
        X = x_predict.values.reshape(1, -1)
        y_pred = model.predict(X, num_iteration=model.best_iteration)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise RuntimeError("Prediction error occurred") from e

    # 4. Output formatting
    prediction = round(float(y_pred[0]), 4)  # Align with target precision
    
    # 5. Prediction metadata logging
    logger.info(f"Generated prediction: {prediction}")
    
    return prediction
