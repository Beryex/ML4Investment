import pandas as pd
import numpy as np
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.preprocessing import RobustScaler
import pandas_market_calendars as mcal
import logging
from tqdm import tqdm
from sklearn.utils import shuffle

from ml4investment.config import settings

pd.set_option('future.no_silent_downcasting', True)

logger = logging.getLogger(__name__)


def calculate_features(df_dict: dict) -> dict:
    """Process 1h OHLCV data to create daily features and prediction target."""
    daily_dict = {}
    price_volume = ['Open', 'High', 'Low', 'Close', 'Volume']

    with tqdm(df_dict.items(), desc="Calculate features") as pbar:
        for stock, df in pbar:
            pbar.set_postfix({'stock': stock}, refresh=True)
            df = df[price_volume].copy()

            # === Basic Calculations ===
            df['Returns_1h'] = df['Close'].pct_change()
            df['Gap'] = df['Open'].pct_change()
            df['Return_abs'] = df['Close'].pct_change().abs()
            df['Intraday_Range'] = (df['High'] - df['Low']) / df['Open']
            df['Price_Efficiency'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'])

            # === Technical Indicators ===
            df['RSI_91'] = _calculate_rsi(df['Close'], settings.DATA_PER_DAY * 14)
            df['ATR_91'] = _calculate_atr(df, settings.DATA_PER_DAY * 14)
            df['Bollinger_Width'] = (df['Close'].rolling(settings.DATA_PER_DAY * 3).std() / df['Close'].rolling(settings.DATA_PER_DAY * 3).mean()) * 100
            df['OBV'] = _calculate_obv(df)

            # === Price & Volume Interaction ===
            df['Volume_Spike'] = df['Volume'] / df['Volume'].rolling(settings.DATA_PER_DAY * 3).mean()
            df['Price_Volume_Correlation'] = df['Close'].rolling(settings.DATA_PER_DAY).corr(df['Volume'])

            # === Flow & Direction ===
            df['Flow_Intensity'] = (np.sign(df['Close'].diff()) * df['Volume']) / df['Volume'].rolling(21).mean()
            df['Flow_Intensity'] = df['Flow_Intensity'].rolling(3).mean()
            df['Direction_Consistency'] = (np.sign(df['Close'].diff()) == np.sign(df['Close'].diff().shift(1))).rolling(3).sum()

            # === Momentum Indicators ===
            df['Momentum_Accel'] = df['Close'].pct_change(3) - df['Close'].pct_change(settings.DATA_PER_DAY)
            df['Vol_Ratio'] = df['Close'].rolling(settings.DATA_PER_DAY).std() / df['Close'].rolling(settings.DATA_PER_DAY * 5).std()

            # === VWAP ===
            df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
            df['VWAP_Deviation'] = (df['Close'] - df['VWAP']) / df['VWAP']

            # === MACD ===
            close_prices = df['Close']
            fast_ema = close_prices.ewm(span=settings.DATA_PER_DAY * 12, adjust=False).mean()
            slow_ema = close_prices.ewm(span=settings.DATA_PER_DAY * 26, adjust=False).mean()
            macd_line = fast_ema - slow_ema
            df['MACD_line'] = macd_line
            df['MACD_signal'] = macd_line.ewm(span=settings.DATA_PER_DAY * 9, adjust=False).mean()

            # === Stochastic Oscillator ===
            high_rolling = df['High'].rolling(settings.DATA_PER_DAY * 2).max()
            low_rolling = df['Low'].rolling(settings.DATA_PER_DAY * 2).min()
            k = (df['Close'] - low_rolling) / (high_rolling - low_rolling + 1e-6) * 100
            df['Stochastic_K'] = k
            df['Stochastic_D'] = k.rolling(3).mean()

            # === Relative Price Position ===
            rolling_max = df['Close'].rolling(settings.DATA_PER_DAY * 2).max()
            rolling_min = df['Close'].rolling(settings.DATA_PER_DAY * 2).min()
            df['Relative_Price_Position'] = (df['Close'] - rolling_min) / (rolling_max - rolling_min + 1e-6)

            # === Intraday Features ===
            morning_mask = df.index.time < pd.to_datetime('12:00').time()
            df['Morning_Volume_Ratio'] = (df[morning_mask]['Volume'].rolling(3).sum() / df['Volume'].rolling(settings.DATA_PER_DAY).sum())
            df['Afternoon_Return'] = df['Close'].pct_change().between_time('13:00', '16:00').rolling(3).mean()
            df['Late_Afternoon_Volume_Ratio'] = df.between_time('14:30', '16:00')['Volume'].rolling(2).sum() / df['Volume'].rolling(settings.DATA_PER_DAY).sum()
            df['Late_Momentum'] = df.between_time('13:30', '16:00')['Close'].pct_change().rolling(2).mean()
            df['End_Volume_Ratio'] = df.between_time('14:30', '15:30')['Volume'].sum() / df['Volume'].rolling(settings.DATA_PER_DAY).sum()

            # === Divergence ===
            price_up = df['Close'].pct_change() > 0
            volume_down = df['Volume'].pct_change() < 0
            df['Divergence'] = (price_up & volume_down).astype(int)

            # === Daily Aggregation ===
            aggregation_rules = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min', 
                'Close': ['last', 'std'],
                'Volume': ['sum', 'std'],
                'Returns_1h': ['mean', 'std'],
                'RSI_91': 'last',
                'ATR_91': 'mean',
                'Bollinger_Width': 'last',
                'Volume_Spike': 'max',
                'Intraday_Range': 'last',
                'Morning_Volume_Ratio': 'last',
                'Afternoon_Return': 'last',
                'Late_Afternoon_Volume_Ratio': 'last',
                'Late_Momentum': 'last',
                'End_Volume_Ratio': 'last',
                'OBV': 'last',
                'Price_Volume_Correlation': 'mean',
                'Relative_Price_Position': 'last',
                'MACD_line': 'last',
                'MACD_signal': 'last',
                'Stochastic_K': 'last',
                'Stochastic_D': 'last',
                'Return_abs': 'sum',
                'Gap': 'last',
                'Price_Efficiency': 'mean',
                'Flow_Intensity': 'last',
                'Direction_Consistency': 'mean',
                'VWAP_Deviation': ['mean', 'last'],
                'Momentum_Accel': 'last',
                'Vol_Ratio': 'last',
                'Divergence': ['sum']
            }

            nyse = mcal.get_calendar('NYSE')
            trading_day = CustomBusinessDay(calendar=nyse)
            daily_df = df.resample(trading_day).agg(aggregation_rules)
            daily_df.columns = ['_'.join(col).strip() for col in daily_df.columns]

            # === Daily Targets ===
            daily_df['Return_1d'] = daily_df['Close_last'].pct_change(1, fill_method=None)
            daily_df['Return_3d'] = daily_df['Close_last'].pct_change(3, fill_method=None)
            daily_df['Return_5d'] = daily_df['Close_last'].pct_change(5, fill_method=None)

            daily_df['Volatility_5d'] = daily_df['Close_last'].rolling(5).std()
            daily_df['Volatility_Change'] = daily_df['Volatility_5d'].diff()
            daily_df['Volatility_10d'] = daily_df['Close_last'].rolling(10).std()
            daily_df['Volatility_Skew'] = daily_df['Volatility_5d'] / (daily_df['Volatility_10d'] + 1e-6)
            daily_df['Momentum_to_Volatility'] = daily_df['Return_1d'] / (daily_df['Volatility_5d'] + 1e-6)
            daily_df['Volume_MA_ratio'] = daily_df['Volume_sum'] / daily_df['Volume_sum'].rolling(5).mean()
            daily_df['MA5_Deviation'] = daily_df['Close_last'] / daily_df['Close_last'].rolling(5).mean() - 1
            daily_df['MA20_Deviation'] = daily_df['Close_last'] / daily_df['Close_last'].rolling(20).mean() - 1
            daily_df['Weekday'] = daily_df.index.dayofweek
            daily_df['Relative_Rank_Change'] = daily_df['Relative_Price_Position_last'] - daily_df['Relative_Price_Position_last'].shift(3)
            daily_df['OC_Momentum'] = (daily_df['Close_last'] - daily_df['Open_first']) / daily_df['Open_first']
            daily_df['Close_Momentum_Slope'] = (
                daily_df['Close_last'].rolling(5).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
            )

            # === Lag Features ===
            lag_features = {'RSI_91_last': [1, 2, 3, 5]}
            for base, lags in lag_features.items():
                for lag in lags:
                    daily_df[f'{base}_lag{lag}'] = daily_df[base].shift(lag)

            daily_df['Vol_Price_Synergy'] = daily_df['Volume_sum'] * daily_df['Returns_1h_mean']
            daily_df['Risk_Adj_Momentum'] = daily_df['Return_5d'] / daily_df['ATR_91_mean']
            daily_df['Target'] = (daily_df['Open_first'].shift(-2) - daily_df['Open_first'].shift(-1)) / daily_df['Open_first'].shift(-1)

            # === Trend Signals ===
            daily_df['MACD_Diff'] = daily_df['MACD_line_last'] - daily_df['MACD_signal_last']
            daily_df['MACD_Direction'] = daily_df['MACD_Diff'].diff() > 0
            daily_df['Trend_Continuation_3d'] = ((daily_df['Return_1d'] > 0) & 
                                                (daily_df['Return_3d'] > 0) & 
                                                (daily_df['Return_5d'] > 0)).astype(int)
            daily_df['Trend_Reversal'] = ((daily_df['Return_1d'] * daily_df['Return_3d']) < 0).astype(int)

            # === Price/Volume Divergence Extended ===
            daily_df['Volume_Change'] = daily_df['Volume_sum'].pct_change()
            daily_df['Price_vs_Volume_Diff'] = daily_df['Return_1d'] - daily_df['Volume_Change']

            # === Time & Cycle Position ===
            daily_df['Month'] = daily_df.index.month
            daily_df['Day_of_Month'] = daily_df.index.day
            daily_df['Week_of_Quarter'] = ((daily_df.index.isocalendar().week - 1) % 13 + 1)

            # === Opening Deviation ===
            daily_df['Open_vs_Close1d'] = daily_df['Open_first'] / daily_df['Close_last'].shift(1) - 1
            daily_df['Open_vs_MA5'] = daily_df['Open_first'] / daily_df['Close_last'].rolling(5).mean() - 1
            daily_df['Open_Volatility_3d'] = daily_df['Open_first'].rolling(3).std()
            daily_df['Open_Momentum_3d'] = daily_df['Open_first'].pct_change(3, fill_method=None)

            # === Range Compression & Expansion ===
            daily_df['Range_Std_5d'] = (daily_df['High_max'] - daily_df['Low_min']).rolling(5).std()
            daily_df['Range_to_Volume'] = daily_df['Range_Std_5d'] / daily_df['Volume_sum'].rolling(5).mean()

            # === Volume Acceleration ===
            daily_df['Volume_Accel'] = daily_df['Volume_sum'].diff(1) - daily_df['Volume_sum'].diff(2)
            daily_df['Volume_Shock_5d'] = daily_df['Volume_sum'].rolling(5).std() / daily_df['Volume_sum'].rolling(5).mean()
            daily_df['Volume_Trend_5d'] = (
                daily_df['Volume_sum'].rolling(5).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
            )

            # === Flow Strength ===
            daily_df['Net_Flow'] = daily_df['OBV_last'].diff()
            daily_df['Flow_to_Volume'] = daily_df['Net_Flow'] / (daily_df['Volume_sum'] + 1e-6)

            # === Trend Normalization ===
            daily_df['Normalized_Trend_Strength'] = daily_df['Return_5d'] / (daily_df['Volatility_5d'] + 1e-6)

            # === Remove Incomplete Trading Day ===
            now = pd.Timestamp.now(tz='America/New_York')
            trading_hours = nyse.schedule(start_date=now.date(), end_date=now.date())
            if not trading_hours.empty:
                market_open = trading_hours.iloc[0]['market_open']
                market_close = trading_hours.iloc[0]['market_close']
                if market_open <= now <= market_close:
                    daily_df = daily_df.drop(daily_df.index[-1])

            daily_dict[stock] = daily_df

        # === Cross-Stock Feature: After All Stocks Are Done ===
        all_returns = pd.DataFrame({
            stock: df['Return_1d'] for stock, df in daily_dict.items()
        })
        rank_df = all_returns.rank(axis=1, pct=True)
        
        for stock in daily_dict:
            daily_dict[stock]['Relative_Return_Rank'] = rank_df[stock]

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


def process_features_for_train(daily_dict: dict, test_number: int = 63, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """ Process Data, including washing, removing Nan, scaling and spliting for training purpose """
    process_feature_config = {}
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []

    for stock, df in daily_dict.items():
        daily_dict[stock] = df.dropna(subset=['Target'])

    stock_id_map = {}
    with tqdm(daily_dict.keys(), desc="Calculate stocks data") as pbar:
        for stock in pbar:
            pbar.set_postfix({'stock': stock}, refresh=True)
            stock_id_map[stock] = stock_code_to_id(stock)

    if len(stock_id_map.values()) != len(set(stock_id_map.values())):
        logger.error(f"Stock mapping mismatch. Stock id number: {len(stock_id_map.values())}, Stock number: {len(set(stock_id_map.values()))}")
        raise ValueError("Stock mapping mismatch.")
    all_stock_ids = sorted(set(stock_id_map.values()))
    cat_type = pd.CategoricalDtype(categories=all_stock_ids)
    process_feature_config['cat_type'] = cat_type
    process_feature_config['stock_id_map'] = stock_id_map

    stock_sector_id_map = settings.STOCK_SECTOR_ID_MAP
    
    all_sector_ids = sorted(set(stock_sector_id_map.values()))
    cat_sector_type = pd.CategoricalDtype(categories=all_sector_ids)
    process_feature_config['cat_sector_type'] = cat_sector_type
    process_feature_config['stock_sector_id_map'] = stock_sector_id_map

    with tqdm(daily_dict.items(), desc="Process features for train") as pbar:
        for stock, df in pbar:
            pbar.set_postfix({'stock': stock}, refresh=True)
            
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

            boolean_cols = X_train_ffilled.select_dtypes(include='bool').columns.tolist()
            numerical_cols = [col for col in X_train_ffilled.columns if col not in boolean_cols]

            quantiles = X_train_ffilled[numerical_cols].quantile([0.05, 0.95])
            lower_bound = quantiles.xs(0.05)
            upper_bound = quantiles.xs(0.95)

            X_train_clipped = X_train_ffilled.copy()
            X_test_clipped = X_test_ffilled.copy()

            X_train_clipped[numerical_cols] = X_train_ffilled[numerical_cols].clip(lower_bound, upper_bound, axis=1)
            X_test_clipped[numerical_cols] = X_test_ffilled[numerical_cols].clip(lower_bound, upper_bound, axis=1)

            scaler = RobustScaler()
            X_train_scaled_numerical = pd.DataFrame(
                scaler.fit_transform(X_train_clipped[numerical_cols]),
                columns=numerical_cols,
                index=X_train_clipped.index
            )
            X_test_scaled_numerical = pd.DataFrame(
                scaler.transform(X_test_clipped[numerical_cols]),
                columns=numerical_cols,
                index=X_test_clipped.index
            )

            X_train_scaled = pd.concat([X_train_scaled_numerical, X_train_clipped[boolean_cols]], axis=1)
            X_test_scaled = pd.concat([X_test_scaled_numerical, X_test_clipped[boolean_cols]], axis=1)

            stock_id = stock_id_map[stock]
            X_train_scaled['stock_id'] = stock_id
            X_test_scaled['stock_id'] = stock_id
            X_train_scaled['stock_id'] = X_train_scaled['stock_id'].astype(cat_type)
            X_test_scaled['stock_id'] = X_test_scaled['stock_id'].astype(cat_type)

            sector_id = stock_sector_id_map[stock]
            X_train_scaled['stock_sector'] = sector_id
            X_test_scaled['stock_sector'] = sector_id
            X_train_scaled['stock_sector'] = X_train_scaled['stock_sector'].astype(cat_sector_type)
            X_test_scaled['stock_sector'] = X_test_scaled['stock_sector'].astype(cat_sector_type)

            X_train_list.append(X_train_scaled)
            X_test_list.append(X_test_scaled)
            y_train_list.append(y_train_stock)
            y_test_list.append(y_test_stock)

            process_feature_config[stock] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'scaler': scaler
            }

    X_train = pd.concat(X_train_list)
    X_test = pd.concat(X_test_list)
    y_train = pd.concat(y_train_list)
    y_test = pd.concat(y_test_list)

    X_train, y_train = shuffle(X_train, y_train, random_state=seed)

    if X_train.index.max() >= X_test.index.min():
        logger.error("Temporal leakage detected in combined dataset")
        raise ValueError("Temporal leakage detected in combined dataset")

    return X_train, X_test, y_train, y_test, process_feature_config


def process_features_for_predict(daily_dict: dict, config_data: dict) -> dict:
    """ Process Data, including washing, removing Nan, scaling and spliting for prediction purpose """
    X_predict_dict = {}

    cat_type = config_data['cat_type']
    stock_id_map = config_data['stock_id_map']
    cat_sector_type = config_data['cat_sector_type']
    stock_sector_id_map = config_data['stock_sector_id_map']

    with tqdm(daily_dict.items(), desc="Process features for predict") as pbar:
        for stock, df in pbar:
            cur_config_data = config_data[stock]
            lower_bound = cur_config_data['lower_bound']
            upper_bound = cur_config_data['upper_bound']
            scaler = cur_config_data['scaler']

            df = df.sort_index().ffill()

            feature_cols = [col for col in df.columns if col != 'Target']
            X_predict_stock = df[feature_cols]

            X_predict_ffilled = X_predict_stock.ffill().bfill()
            assert X_predict_ffilled.isnull().sum().sum() == 0, f"Prediction data contains missing values for stock {stock}"

            boolean_cols = X_predict_ffilled.select_dtypes(include='bool').columns.tolist()
            numerical_cols = [col for col in X_predict_ffilled.columns if col not in boolean_cols]

            X_predict_clipped = X_predict_ffilled.copy()

            X_predict_clipped[numerical_cols] = X_predict_ffilled[numerical_cols].clip(lower_bound, upper_bound, axis=1)

            X_predict_scaled_numerical = pd.DataFrame(
                scaler.transform(X_predict_clipped[numerical_cols]),
                columns=numerical_cols,
                index=X_predict_clipped.index
            )

            X_predict_scaled = pd.concat([X_predict_scaled_numerical, X_predict_clipped[boolean_cols]], axis=1)

            stock_id = stock_id_map[stock]
            X_predict_scaled['stock_id'] = stock_id
            X_predict_scaled['stock_id'] = X_predict_scaled['stock_id'].astype(cat_type)

            sector_id = stock_sector_id_map[stock]
            X_predict_scaled['stock_sector'] = sector_id
            X_predict_scaled['stock_sector'] = X_predict_scaled['stock_sector'].astype(cat_sector_type)

            X_predict = X_predict_scaled.iloc[[-1]].copy()
            X_predict_dict[stock] = X_predict

    return X_predict_dict


def process_features_for_backtest(daily_dict: dict, config_data: dict, predict_stock_list: list) -> tuple[dict, dict, int]:
    """ Process Data, including washing, removing Nan, scaling and spliting for backtest purpose """
    X_backtest_dict = {}
    y_backtest_dict = {}

    cat_type = config_data['cat_type']
    stock_id_map = config_data['stock_id_map']
    cat_sector_type = config_data['cat_sector_type']
    stock_sector_id_map = config_data['stock_sector_id_map']
    
    for stock, df in daily_dict.items():
        daily_dict[stock] = df.dropna(subset=['Target'])
    
    for stock in daily_dict.keys():
        daily_dict[stock] = daily_dict[stock].tail(settings.TEST_DAY_NUMBER)

    backtest_day_numbers = {df.shape[0] for df in daily_dict.values()}
    if len(backtest_day_numbers) != 1:
        logger.error("Backtest day number mismatched")
        raise ValueError("Backtest day number mismatched")
    backtest_day_number = backtest_day_numbers.pop()

    for i in range(backtest_day_number):
        X_backtest_dict[i] = {}
        y_backtest_dict[i] = {}

    with tqdm(predict_stock_list, desc="Process features for backtest") as pbar:
        for stock in pbar:
            pbar.set_postfix({'stock': stock,}, refresh=True)

            df = daily_dict[stock]
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

            boolean_cols = X_backtest_ffilled.select_dtypes(include='bool').columns.tolist()
            numerical_cols = [col for col in X_backtest_ffilled.columns if col not in boolean_cols]

            X_backtest_clipped = X_backtest_ffilled.copy()

            X_backtest_clipped[numerical_cols] = X_backtest_ffilled[numerical_cols].clip(lower_bound, upper_bound, axis=1)

            X_backtest_scaled_numerical = pd.DataFrame(
                scaler.transform(X_backtest_clipped[numerical_cols]),
                columns=numerical_cols,
                index=X_backtest_clipped.index
            )

            X_backtest_scaled = pd.concat([X_backtest_scaled_numerical, X_backtest_clipped[boolean_cols]], axis=1)

            stock_id = stock_id_map[stock]
            X_backtest_scaled['stock_id'] = stock_id
            X_backtest_scaled['stock_id'] = X_backtest_scaled['stock_id'].astype(cat_type)

            sector_id = stock_sector_id_map[stock]
            X_backtest_scaled['stock_sector'] = sector_id
            X_backtest_scaled['stock_sector'] = X_backtest_scaled['stock_sector'].astype(cat_sector_type)

            for i in range(backtest_day_number):
                X_backtest_dict[i][stock] = X_backtest_scaled.iloc[[i]]
                y_backtest_dict[i][stock] = y_backtest_stock.iloc[i]

    return X_backtest_dict, y_backtest_dict, backtest_day_number


def stock_code_to_id(stock_code: str) -> int:
    """ Change the stock string to the sum of ASCII value of each char within the stock code """
    return sum(ord(c) * 256 ** i for i, c in enumerate(reversed(stock_code)))

def id_to_stock_code(code_id: int) -> str:
    """  Change the stock id to the string of stock code """
    chars = []
    while code_id > 0:
        ascii_val = code_id % 256
        chars.append(chr(ascii_val))
        code_id //= 256
    return ''.join(reversed(chars))
