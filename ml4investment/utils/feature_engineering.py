import pandas as pd
import numpy as np
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import pandas_market_calendars as mcal
import logging

logger = logging.getLogger(__name__)


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Process 1h OHLCV data to create daily features and prediction target """
    # 1. Preserve core price-volume columns
    price_volume = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[price_volume].copy()
    
    # 2. Generate intraday technical indicators
    # Price momentum features
    df['Returns_1h'] = df['Close'].pct_change()
    df['MA6'] = df['Close'].rolling(6).mean()
    df['RSI_14'] = _calculate_rsi(df['Close'], 14)
    
    # Volatility features
    df['ATR_14'] = _calculate_atr(df, 14)  # Average True Range
    df['Bollinger_Width'] = (df['Close'].rolling(20).std() / df['MA6']) * 100
    
    # Volume-based features
    df['Volume_Spike'] = df['Volume'] / df['Volume'].rolling(24).mean()
    df['Intraday_Range'] = (df['High'] - df['Low']) / df['Open']
    df['OBV'] = _calculate_obv(df)
    
    # Time-session features
    morning_mask = df.index.time < pd.to_datetime('12:00').time()
    df['Morning_Volume_Ratio'] = (df[morning_mask]['Volume'].rolling(3).sum() / 
                                  df['Volume'].rolling(6).sum())
    df['Afternoon_Return'] = (df['Close'].pct_change()
                             .between_time('13:00', '15:00')
                             .rolling(2).mean())
    
    # Price-volume relationship
    df['Price_Volume_Correlation'] = df['Close'].rolling(6).corr(df['Volume'])
    
    # 3. Aggregate to daily timeframe
    aggregation_rules = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min', 
        'Close': 'last',
        'Volume': 'sum',
        'Returns_1h': ['mean', 'std'],
        'MA6': 'last',
        'RSI_14': 'last',
        'ATR_14': 'mean',
        'Bollinger_Width': 'last',
        'Volume_Spike': 'max',
        'Intraday_Range': 'last',
        'Morning_Volume_Ratio': 'last',
        'Afternoon_Return': 'last',
        'OBV': 'last',
        'Price_Volume_Correlation': 'mean'
    }
    
    nyse = mcal.get_calendar('NYSE')
    trading_day = CustomBusinessDay(calendar=nyse)
    daily_df = df.resample(trading_day).agg(aggregation_rules)
    daily_df.columns = ['_'.join(col).strip() for col in daily_df.columns]
    
    # 4. Create lagged features
    lag_mapping = {
        'RSI_14': 'RSI_14_last',
        'MA6': 'MA6_last',
        'Volume_Spike': 'Volume_Spike_max'
    }
    for base_feature, aggregated_col in lag_mapping.items():
        for lag in [1, 2, 3]:
            daily_df[f'{base_feature}_lag{lag}'] = daily_df[aggregated_col].shift(lag)
    
    # 5. Define prediction target
    daily_df['Target'] = (daily_df['Open_first'].shift(-1) - daily_df['Open_first']).div(daily_df['Open_first']).round(4)
    
    return daily_df


def process_features(daily_df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Process Data, including washing, removing Nan, scaling and spliting """
    # 1. Isolate prediction sample (most recent observation)
    x_predict = daily_df.iloc[[-1]].copy()
    main_data = daily_df.iloc[:-1].copy()

    # 2. Core data cleansing
    # Remove samples with missing targets (preserve prediction row)
    main_data = main_data.dropna(subset=['Target'])
    
    # 3. Feature-target segmentation
    feature_cols = [col for col in main_data.columns if col != 'Target']
    X = main_data[feature_cols]
    y = main_data['Target']
    logger.info(f"Total processed samples: {X.shape[0]}")
    logger.info(f"Number of features: {X.shape[1]}")
    
    # 4. Data processing pipeline
    # 4.1 Missing value imputation (time-aware forward fill and then backward fill if necessary) 
    X = X.ffill().bfill()
    assert X.isnull().sum().sum() == 0, "Training data contains missing values"
    last_valid_values = X.iloc[-1]
    
    # 4.2 Winsorization (5%-95% quantile clipping)
    quantiles = X.quantile([0.05, 0.95])
    X = X.clip(lower=quantiles.xs(0.05), upper=quantiles.xs(0.95), axis=1)
    
    # 4.3 Robust scaling (median/IQR normalization)
    scaler = RobustScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns.tolist(), index=X.index)
    
    # 5. Prediction data processing
    # 5.1 Feature alignment
    x_predict = x_predict.reindex(columns=feature_cols)
    
    # 5.2 Consistent pipeline application
    # Imputation using training parameters
    x_predict = x_predict[feature_cols].ffill().fillna(last_valid_values)
    assert x_predict.isnull().sum().sum() == 0, "Predicting data contains missing values"
    
    # Apply training-based quantile clipping
    x_predict = x_predict.clip(lower=quantiles.xs(0.05), upper=quantiles.xs(0.95), axis=1)
    
    # Scale with pre-fit scaler
    x_predict = pd.DataFrame(scaler.transform(x_predict), columns=x_predict.columns.tolist(), index=x_predict.index)
    
    # 6. Temporal data partitioning
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    assert X_train.index[-1] < X_test.index[0], "Temporal leakage detected"
    assert X_test.index[-1] < x_predict.index[0], "Prediction time violation"
    
    return X_train, X_test, y_train, y_test, x_predict


def _calculate_rsi(series: pd.Series, window: int) -> pd.Series:
    """ Compute Relative Strength Index (RSI) """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _calculate_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """ Calculate Average True Range (ATR) """
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    
    true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return true_range.rolling(window).mean()


def _calculate_obv(df: pd.DataFrame) -> pd.Series:
    """ Compute On-Balance Volume (OBV) """
    price_change = df['Close'].diff()
    return (np.sign(price_change) * df['Volume']).cumsum()
