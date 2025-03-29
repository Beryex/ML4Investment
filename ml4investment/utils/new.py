import pandas as pd
import numpy as np
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import pandas_market_calendars as mcal
import logging

pd.set_option('future.no_silent_downcasting', True)

logger = logging.getLogger(__name__)


def calculate_features(df_dict: dict) -> dict:
    """ Process 1h OHLCV data to create daily features and prediction target """
    daily_dict = {}
    price_volume = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    for stock, df in df_dict.items():
        df = df[price_volume].copy()
        
        df['Returns_1h'] = df['Close'].pct_change()
        df['MA7'] = df['Close'].rolling(7).mean()
        df['RSI_91'] = _calculate_rsi(df['Close'], 91)
        
        df['ATR_91'] = _calculate_atr(df, 91)
        df['Bollinger_Width'] = (df['Close'].rolling(21).std() / df['Close'].rolling(21).mean()) * 100
        
        df['Volume_Spike'] = df['Volume'] / df['Volume'].rolling(21).mean()
        df['Intraday_Range'] = (df['High'] - df['Low']) / df['Open']
        df['OBV'] = _calculate_obv(df)
        
        morning_mask = df.index.time < pd.to_datetime('12:00').time()
        df['Morning_Volume_Ratio'] = (df[morning_mask]['Volume'].rolling(3).sum() / 
                                    df['Volume'].rolling(7).sum())
        df['Afternoon_Return'] = (df['Close'].pct_change()
                                .between_time('13:00', '16:00')
                                .rolling(3).mean())
        df['Late_Afternoon_Volume_Ratio'] = (
            df.between_time('14:30', '16:00')['Volume'].rolling(2).sum() /
            df['Volume'].rolling(7).sum()
        )
        
        df['Price_Volume_Correlation'] = df['Close'].rolling(7).corr(df['Volume'])

        rolling_max = df['Close'].rolling(14).max()
        rolling_min = df['Close'].rolling(14).min()
        df['Relative_Price_Position'] = (df['Close'] - rolling_min) / (rolling_max - rolling_min + 1e-6)

        close_prices = df['Close']
        fast_ema = close_prices.ewm(span=12 * 7, adjust=False).mean()
        slow_ema = close_prices.ewm(span=26 * 7, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        df['MACD_line'] = macd_line
        df['MACD_signal'] = macd_line.ewm(span=9 * 7, adjust=False).mean()
        df['MACD_hist'] = fast_ema - slow_ema

        high = df['High'].rolling(14).max()
        low = df['Low'].rolling(14).min()
        k = (df['Close'] - low) / (high - low) * 100
        df['Stochastic_K'] = k
        df['Stochastic_D'] = k.rolling(3).mean()

        df['Return_abs'] = df['Close'].pct_change().abs()
        df['Gap'] = df['Open'].pct_change()
        df['Price_Efficiency'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'])

        close_hour_mask = df.index.hour >= 15
        df['Closing_Vol_Ratio'] = df[close_hour_mask]['Volume'].sum() / df['Volume'].rolling(7).sum()

        df['OFI'] = (df['Close'].diff() * df['Volume']).rolling(21).mean()
        
        aggregation_rules = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min', 
            'Close': ['last', 'std'],
            'Volume': ['sum', 'mean', 'std'],
            'Returns_1h': ['mean', 'std'],
            'MA7': 'last',
            'RSI_91': 'last',
            'ATR_91': 'mean',
            'Bollinger_Width': 'last',
            'Volume_Spike': 'max',
            'Intraday_Range': 'last',
            'Morning_Volume_Ratio': 'last',
            'Afternoon_Return': 'last',
            'Late_Afternoon_Volume_Ratio': 'last',
            'OBV': 'last',
            'Price_Volume_Correlation': 'mean',
            'Relative_Price_Position': 'last',
            'MACD_line': 'last',
            'MACD_signal': 'last',
            'MACD_hist': 'last',
            'Stochastic_K': 'last',
            'Stochastic_D': 'last',
            'Return_abs': 'sum',
            'Gap': 'last',
            'Price_Efficiency': 'mean',
            'Closing_Vol_Ratio': 'last',
            'OFI': ['max', 'last', lambda x: np.sign(x).sum()]
        }
        
        nyse = mcal.get_calendar('NYSE')
        trading_day = CustomBusinessDay(calendar=nyse)
        daily_df = df.resample(trading_day).agg(aggregation_rules)
        daily_df.columns = ['_'.join(col).strip() for col in daily_df.columns]
        
        daily_df['Return_1d'] = daily_df['Close_last'].pct_change(1, fill_method=None)
        daily_df['Return_3d'] = daily_df['Close_last'].pct_change(3, fill_method=None)
        daily_df['Return_5d'] = daily_df['Close_last'].pct_change(5, fill_method=None)

        daily_df['Volatility_5d'] = daily_df['Close_last'].rolling(5).std()
        daily_df['Volatility_10d'] = daily_df['Close_last'].rolling(10).std()

        daily_df['Momentum_to_Volatility'] = daily_df['Return_1d'] / (daily_df['Volatility_5d'] + 1e-6)
        daily_df['Volume_MA_ratio'] = daily_df['Volume_sum'] / daily_df['Volume_sum'].rolling(5).mean()

        daily_df['Weekday'] = daily_df.index.dayofweek
        daily_df['Is_Friday'] = (daily_df['Weekday'] == 4).astype(int)
        daily_df['Is_Monday'] = (daily_df['Weekday'] == 0).astype(int)

        lag_features = {
            'RSI_91_last': [1, 2, 3, 5],
            'MACD_hist_last': [1, 3],
            'Closing_Vol_Ratio_last': [1]
        }

        for base, lags in lag_features.items():
            for lag in lags:
                daily_df[f'{base}_lag{lag}'] = daily_df[base].shift(lag)
        
        all_returns = pd.DataFrame({cur_stock: df['Return_1d'] for cur_stock, df in daily_dict.items()})
        rank_df = all_returns.rank(axis=1, pct=True)
        for cur_stock in daily_dict:
            daily_dict[cur_stock]['Relative_Return_Rank'] = rank_df[cur_stock]
        
        daily_df['Vol_Price_Synergy'] = daily_df['Volume_sum'] * daily_df['Returns_1h_mean']
        daily_df['Risk_Adj_Momentum'] = daily_df['Return_5d'] / daily_df['ATR_91_mean']
        
        daily_df['Target'] = (daily_df['Open_first'].shift(-2) - daily_df['Open_first'].shift(-1)) / daily_df['Open_first'].shift(-1)
        
        now = pd.Timestamp.now(tz='America/New_York')
        trading_hours = nyse.schedule(start_date=now.date(), end_date=now.date())
        
        if not trading_hours.empty:
            market_open = trading_hours.iloc[0]['market_open']
            market_close = trading_hours.iloc[0]['market_close']
            if market_open <= now <= market_close:
                daily_df = daily_df.drop(daily_df.index[-1])
        
        daily_dict[stock] = daily_df
    
    return daily_dict


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


def process_features_for_train(daily_dict: dict, test_number: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """ Process Data, including washing, removing Nan, scaling and spliting for training purpose """
    process_features_config_data = {}
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []

    stock_id_map = {stock: stock_code_to_id(stock) for stock in daily_dict.keys()}
    if len(stock_id_map.values()) != len(set(stock_id_map.values())):
        logger.error(f"Stock mapping mismatch. Stock id number: {len(stock_id_map.values())}, Stock number: {len(set(stock_id_map.values()))}")
        raise ValueError("Stock mapping mismatch.")
    all_stock_ids = sorted(set(stock_id_map.values()))
    cat_type = pd.CategoricalDtype(categories=all_stock_ids)
    process_features_config_data['cat_type'] = cat_type
    
    for stock, df in daily_dict.items():
        df = df.sort_index().ffill()
        
        split_idx = len(df) - test_number
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        feature_cols = [col for col in df.columns if col != 'Target']
        target_col = 'Target'
        X_train_stock = train_df[feature_cols]
        y_train_stock = train_df[target_col]
        X_test_stock = test_df[feature_cols]
        y_test_stock = test_df[target_col]
        
        X_train_ffilled = X_train_stock.ffill().bfill()
        last_train_values = X_train_ffilled.iloc[-1]
        assert X_train_ffilled.isnull().sum().sum() == 0, f"Training data contains missing values for stock {stock}"
        
        X_test_ffilled = X_test_stock.fillna(last_train_values).ffill()
        assert X_test_ffilled.isnull().sum().sum() == 0, f"Testing data contains missing values for stock {stock}"
        
        quantiles = X_train_ffilled.quantile([0.05, 0.95])
        lower_bound = quantiles.xs(0.05)
        upper_bound = quantiles.xs(0.95)
        X_train_clipped = X_train_ffilled.clip(lower_bound, upper_bound, axis=1)
        X_test_clipped = X_test_ffilled.clip(lower_bound, upper_bound, axis=1)
        
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_clipped),
            columns=feature_cols,
            index=X_train_clipped.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_clipped),
            columns=feature_cols,
            index=X_test_clipped.index
        )
        
        stock_id = stock_id_map[stock]
        X_train_scaled['stock_id'] = stock_id
        X_test_scaled['stock_id'] = stock_id
        X_train_scaled['stock_id'] = X_train_scaled['stock_id'].astype(cat_type)
        X_test_scaled['stock_id'] = X_test_scaled['stock_id'].astype(cat_type)
        
        X_train_list.append(X_train_scaled)
        X_test_list.append(X_test_scaled)
        y_train_list.append(y_train_stock)
        y_test_list.append(y_test_stock)

        process_features_config_data[stock] = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'scaler': scaler
        }
    
    X_train = pd.concat(X_train_list)
    X_test = pd.concat(X_test_list)
    y_train = pd.concat(y_train_list)
    y_test = pd.concat(y_test_list)
    
    if X_train.index.max() >= X_test.index.min():
        logger.error("Temporal leakage detected in combined dataset")
        raise ValueError("Temporal leakage detected in combined dataset")
    
    return X_train, X_test, y_train, y_test, process_features_config_data


def process_features_for_predict(daily_dict: dict, config_data: dict) -> dict:
    """ Process Data, including washing, removing Nan, scaling and spliting for prediction purpose """
    X_predict_dict = {}
    cat_type = config_data['cat_type']

    for stock, df in daily_dict.items():
        cur_config_data = config_data[stock]
        lower_bound = cur_config_data['lower_bound']
        upper_bound = cur_config_data['upper_bound']
        scaler = cur_config_data['scaler']

        df = df.sort_index().ffill()

        feature_cols = [col for col in df.columns if col != 'Target']
        X_predict_stock = df[feature_cols]

        X_predict_ffilled = X_predict_stock.ffill().bfill()
        assert X_predict_ffilled.isnull().sum().sum() == 0, f"Prediction data contains missing values for stock {stock}"

        X_predict_clipped = X_predict_ffilled.clip(lower_bound, upper_bound, axis=1)

        X_predict_scaled = pd.DataFrame(
            scaler.transform(X_predict_clipped),
            columns=feature_cols,
            index=X_predict_clipped.index
        )

        stock_id = stock_code_to_id(stock)
        X_predict_scaled['stock_id'] = stock_id
        X_predict_scaled['stock_id'] = X_predict_scaled['stock_id'].astype(cat_type)

        X_predict = X_predict_scaled.iloc[[-1]].copy()
        X_predict_dict[stock] = X_predict

    return X_predict_dict


def process_features_for_backtest(daily_dict: dict, config_data: dict) -> tuple[dict, dict, int]:
    """ Process Data, including washing, removing Nan, scaling and spliting for backtest purpose """
    X_backtest_dict = {}
    y_backtest_dict = {}
    cat_type = config_data['cat_type']
    
    for stock, df in daily_dict.items():
        daily_dict[stock] = df.dropna(subset=['Target'])
    backtest_day_numbers = {df.shape[0] for df in daily_dict.values()}
    if len(backtest_day_numbers) != 1:
        logger.error("Backtest day number mismatched")
        raise ValueError("Backtest day number mismatched")
    backtest_day_number = backtest_day_numbers.pop()

    for i in range(backtest_day_number):
        X_backtest_dict[i] = {}
        y_backtest_dict[i] = {}

    for stock, df in daily_dict.items():
        cur_config_data = config_data[stock]
        lower_bound = cur_config_data['lower_bound']
        upper_bound = cur_config_data['upper_bound']
        scaler = cur_config_data['scaler']

        df = df.sort_index().ffill()

        feature_cols = [col for col in df.columns if col != 'Target']
        target_col = 'Target'
        X_backtest_stock = df[feature_cols]
        y_backtest_stock = df[target_col]

        X_backtest_ffilled = X_backtest_stock.ffill().bfill()
        assert X_backtest_ffilled.isnull().sum().sum() == 0, f"Prediction data contains missing values for stock {stock}"

        X_backtest_clipped = X_backtest_ffilled.clip(lower_bound, upper_bound, axis=1)

        X_backtest_scaled = pd.DataFrame(
            scaler.transform(X_backtest_clipped),
            columns=feature_cols,
            index=X_backtest_clipped.index
        )

        stock_id = stock_code_to_id(stock)
        X_backtest_scaled['stock_id'] = stock_id
        X_backtest_scaled['stock_id'] = X_backtest_scaled['stock_id'].astype(cat_type)

        for i in range(backtest_day_number):
            X_backtest_dict[i][stock] = X_backtest_scaled.iloc[[i]]
            y_backtest_dict[i][stock] = y_backtest_stock.iloc[i]

    return X_backtest_dict, y_backtest_dict, backtest_day_number


def stock_code_to_id(stock_code: str) -> int:
    """ Change the stock string to the sum of ASCII value of each char within the stock code """
    return sum(ord(c) * 256 ** i for i, c in enumerate(reversed(stock_code)))
