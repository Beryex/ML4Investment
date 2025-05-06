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

EPSILON = 1e-9

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
            df['Returns_1h'] = df['Close'].pct_change(fill_method=None)
            df['Return_abs'] = df['Returns_1h'].abs()
            df['Intraday_Range'] = (df['High'] - df['Low']) / df['Open']
            df['Price_Efficiency'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + EPSILON)
            df['High_Low_Ratio'] = df['High'] / (df['Low'] + EPSILON)
            df['Close_Open_Ratio'] = df['Close'] / (df['Open'] + EPSILON)
            df['Body_Range_Ratio'] = (df['Close'] - df['Open']).abs() / (df['High'] - df['Low'] + EPSILON)
            df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / (df['High'] - df['Low'] + EPSILON)
            df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / (df['High'] - df['Low'] + EPSILON)
            df['Hourly_Volatility'] = df['Returns_1h'].rolling(settings.DATA_PER_DAY).std()
            df['Hourly_Open_Gap'] = df['Open'].pct_change(fill_method=None) 

            # === Technical Indicators ===
            df['RSI_91'] = _calculate_rsi(df['Close'], settings.DATA_PER_DAY * 14)
            df['ATR_91'] = _calculate_atr(df['High'], df['Low'], df['Close'], settings.DATA_PER_DAY * 14)
            df['Bollinger_Width'] = (df['Close'].rolling(settings.DATA_PER_DAY * 3).std() / df['Close'].rolling(settings.DATA_PER_DAY * 3).mean()) * 100
            df['OBV'] = _calculate_obv(df)

            df['RSI_7'] = _calculate_rsi(df['Close'], settings.DATA_PER_DAY)
            df['RSI_21'] = _calculate_rsi(df['Close'], settings.DATA_PER_DAY * 3)
            df['ATR_7'] = _calculate_atr(df['High'], df['Low'], df['Close'], settings.DATA_PER_DAY)
            df['ATR_21'] = _calculate_atr(df['High'], df['Low'], df['Close'], settings.DATA_PER_DAY * 3)
            df['MA_7'] = df['Close'].rolling(settings.DATA_PER_DAY).mean()
            df['MA_21'] = df['Close'].rolling(settings.DATA_PER_DAY * 3).mean()
            df['MA_Diff_7_21'] = df['MA_7'] - df['MA_21']
            df['Close_vs_MA7'] = df['Close'] / (df['MA_7'] + EPSILON) - 1
            df['Close_vs_MA21'] = df['Close'] / (df['MA_21'] + EPSILON) - 1

            df['ADX_14h'] = _calculate_adx(df['High'], df['Low'], df['Close'], timeperiod=14)
            df['CCI_14h'] =  _calculate_cci(df['High'], df['Low'], df['Close'], timeperiod=14)
            df['MFI_14h'] = _calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
            upper, middle, lower = _calculate_bbands(df['Close'], timeperiod=20, nbdev=2)
            df['BB_Upper'] = upper
            df['BB_Lower'] = lower
            df['BB_Middle'] = middle
            df['Close_vs_BB'] = (df['Close'] - df['BB_Middle']) / (df['BB_Upper'] - df['BB_Lower'] + EPSILON)

            # === Price & Volume Interaction ===
            df['Volume_Spike'] = df['Volume'] / (df['Volume'].rolling(settings.DATA_PER_DAY * 3).mean() + EPSILON)
            df['Price_Volume_Correlation'] = df['Close'].rolling(settings.DATA_PER_DAY).corr(df['Volume'])
            df['Volume_Change_Ratio'] = df['Volume'].pct_change(fill_method=None)
            df['Volume_Change_Ratio'] = df['Volume_Change_Ratio'].replace([np.inf, -np.inf], np.nan)
            df['Signed_Volume'] = np.sign(df['Returns_1h']) * df['Volume']
            df['Signed_Volume_MA7'] = df['Signed_Volume'].rolling(settings.DATA_PER_DAY).mean()

            # === Flow & Direction ===
            df['Flow_Intensity'] = (np.sign(df['Close'].diff()) * df['Volume']) / (df['Volume'].rolling(21).mean())
            df['Flow_Intensity'] = df['Flow_Intensity'].rolling(3).mean()
            df['Direction_Consistency'] = (np.sign(df['Close'].diff()) == np.sign(df['Close'].diff().shift(1))).rolling(3).sum()

            # === Momentum Indicators ===
            df['Momentum_Accel'] = df['Close'].pct_change(3, fill_method=None) - df['Close'].pct_change(settings.DATA_PER_DAY, fill_method=None)
            df['Vol_Ratio'] = df['Close'].rolling(settings.DATA_PER_DAY).std() / (df['Close'].rolling(settings.DATA_PER_DAY * 5).std() + EPSILON)
            df['ROC_1h'] = df['Close'].pct_change(1, fill_method=None)
            df['ROC_3h'] = df['Close'].pct_change(3,fill_method=None)
            df['ROC_7h'] = df['Close'].pct_change(settings.DATA_PER_DAY, fill_method=None)

            # === VWAP ===
            df['Date'] = df.index.date # 确保 Date 列存在
            vwap_daily = df.groupby('Date').apply(lambda x: (x['Volume'] * (x['High'] + x['Low'] + x['Close']) / 3).cumsum() / (x['Volume'].cumsum() + EPSILON))
            vwap_daily = vwap_daily.reset_index(level=0, drop=True)
            df['VWAP_Daily'] = vwap_daily
            df['VWAP_Cumulative'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / (df['Volume'].cumsum() + EPSILON)
            df['VWAP_Deviation_Daily'] = (df['Close'] - df['VWAP_Daily']) / (df['VWAP_Daily'] + EPSILON)
            df['VWAP_Deviation_Cumulative'] = (df['Close'] - df['VWAP_Cumulative']) / (df['VWAP_Cumulative'] + EPSILON)

            # === MACD ===
            close_prices = df['Close']
            macd_line, macd_signal, macd_hist = _calculate_MACD(close_prices)
            df['MACD_line'] = macd_line
            df['MACD_signal'] = macd_signal
            df['MACD_hist'] = macd_hist

            # === Stochastic Oscillator ===
            high_rolling = df['High'].rolling(settings.DATA_PER_DAY * 2).max()
            low_rolling = df['Low'].rolling(settings.DATA_PER_DAY * 2).min()
            k = (df['Close'] - low_rolling) / (high_rolling - low_rolling + EPSILON) * 100
            df['Stochastic_K'] = k
            df['Stochastic_D'] = k.rolling(3).mean()

            # === Relative Price Position ===
            rolling_max = df['Close'].rolling(settings.DATA_PER_DAY * 2).max()
            rolling_min = df['Close'].rolling(settings.DATA_PER_DAY * 2).min()
            df['Relative_Price_Position'] = (df['Close'] - rolling_min) / (rolling_max - rolling_min + EPSILON)

            # === Intraday Features ===
            df['Hour_Index'] = df.index.hour * 100 + df.index.minute 
            df['Time_Portion'] = df['Hour_Index'].apply(lambda x: 0 if x < 1130 else (1 if x < 1430 else 2))
            
            morning_mask = df['Hour_Index'] < 1230 
            df['Morning_Volume_Ratio'] = (df['Volume'].where(morning_mask).rolling(3).sum() / (df['Volume'].rolling(settings.DATA_PER_DAY).sum() + EPSILON))

            afternoon_mask = df['Hour_Index'] >= 1330
            df['Afternoon_Return'] = df['Returns_1h'].where(afternoon_mask).rolling(3).mean()
            
            late_afternoon_mask = df['Hour_Index'] >= 1430
            df['Late_Afternoon_Volume_Ratio'] = df['Volume'].where(late_afternoon_mask).rolling(2).sum() / (df['Volume'].rolling(settings.DATA_PER_DAY).sum() + EPSILON)
            df['Late_Momentum'] = df['Returns_1h'].where(late_afternoon_mask).rolling(2).mean()

            end_mask = (df['Hour_Index'] >= 1430) & (df['Hour_Index'] <= 1530)
            end_volume_sum_daily = df['Volume'].where(end_mask, 0).groupby(df.index.date).transform('sum')
            total_volume_daily = df['Volume'].groupby(df.index.date).transform('sum')
            df['End_Volume_Ratio'] = end_volume_sum_daily / (total_volume_daily + EPSILON)

            df['First_Hour_Return'] = df.groupby('Date')['Returns_1h'].transform('first')
            df['Last_Hour_Return'] = df.groupby('Date')['Returns_1h'].transform('last')
            df['First_Hour_Volume_Ratio'] = df.groupby('Date')['Volume'].transform('first') / (df.groupby('Date')['Volume'].transform('sum') + EPSILON)
            df['Last_Hour_Volume_Ratio'] = df.groupby('Date')['Volume'].transform('last') / (df.groupby('Date')['Volume'].transform('sum') + EPSILON)
            
            # === Divergence ===
            price_up = df['Close'].pct_change(fill_method=None) > 0
            volume_down = df['Volume'].pct_change(fill_method=None) < 0
            df['Price_Up_Vol_Down_Div'] = (price_up & volume_down).astype(int)

            price_down = df['Close'].pct_change(fill_method=None) < 0
            volume_up = df['Volume'].pct_change(fill_method=None) > 0
            df['Price_Down_Vol_Up_Div'] = (price_down & volume_up).astype(int)

            # === Daily Aggregation ===
            aggregation_rules = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min', 
                'Close': ['last', 'std', 'mean'],
                'Volume': ['sum', 'std', 'mean'],
                'Returns_1h': ['mean', 'std', 'median', 'skew'],
                'Hourly_Open_Gap': ['mean', 'std'],
                'Return_abs': ['sum', 'mean'],
                'Intraday_Range': ['last', 'mean'],
                'Price_Efficiency': ['mean', 'last'],
                'High_Low_Ratio': ['mean', 'last'],
                'Close_Open_Ratio': ['mean', 'last'],
                'Body_Range_Ratio': ['mean', 'last'],
                'Upper_Shadow': ['mean', 'last'],
                'Lower_Shadow': ['mean', 'last'],
                'Hourly_Volatility': ['mean', 'last'], 

                'RSI_91': 'last',
                'ATR_91': 'mean', 
                'Bollinger_Width': 'last',
                'OBV': 'last',
                'RSI_7': 'last', 
                'RSI_21': 'last', 
                'ATR_7': 'mean', 
                'ATR_21': 'mean', 
                'MA_7': 'last', 
                'MA_21': 'last',
                'MA_Diff_7_21': 'last',
                'Close_vs_MA7': 'last',
                'Close_vs_MA21': 'last',
                
                'ADX_14h': 'last',
                'CCI_14h': 'last',
                'MFI_14h': 'last',
                'Close_vs_BB': 'last',

                'Volume_Spike': ['max', 'mean'],
                'Price_Volume_Correlation': ['mean', 'last'],
                'Volume_Change_Ratio': ['mean', 'std'],
                'Signed_Volume_MA7': ['last', 'mean'],

                'Flow_Intensity': ['last', 'mean'],
                'Direction_Consistency': ['mean', 'last'],

                'Momentum_Accel': ['last', 'mean'],
                'Vol_Ratio': ['last', 'mean'],
                'ROC_3h': ['last', 'mean'],
                'ROC_7h': ['last', 'mean'],

                'VWAP_Deviation_Daily': ['mean', 'last', 'std'],
                'VWAP_Deviation_Cumulative': ['mean', 'last'], 
                
                'MACD_line': 'last',
                'MACD_signal': 'last',
                'MACD_hist': 'last',

                'Stochastic_K': 'last',
                'Stochastic_D': 'last',

                'Relative_Price_Position': ['last', 'mean'],

                'Hour_Index': ['first', 'last'],
                'Time_Portion': 'last',
                'Morning_Volume_Ratio': ['last', 'mean'], 
                'Afternoon_Return': ['last', 'mean'], 
                'Late_Afternoon_Volume_Ratio': ['last', 'mean'],
                'Late_Momentum': ['last', 'mean'],
                'End_Volume_Ratio': ['last'],
                'First_Hour_Return': 'first',
                'Last_Hour_Return': 'last',
                'First_Hour_Volume_Ratio': 'first',
                'Last_Hour_Volume_Ratio': 'last',

                'Price_Up_Vol_Down_Div': ['sum', 'mean'],
                'Price_Down_Vol_Up_Div': ['sum', 'mean'],

                'Price_Efficiency': ['mean', 'std']
            }

            nyse = mcal.get_calendar('NYSE')
            unique_dates_idx = pd.to_datetime(pd.Series(df.index.normalize().date).unique())
            
            schedule = nyse.schedule(
                start_date=unique_dates_idx.min(),
                end_date=unique_dates_idx.max()
            )
            correct_trading_dates = schedule.index.date
            buggy_trading_day_offset = CustomBusinessDay(calendar=nyse)     # here will introduce holidays caused by bug in CustomBusinessDay, so we add these fix code below
            daily_df_aggregated_potentially_incorrect = df.resample(buggy_trading_day_offset).agg(aggregation_rules)
            index_dates_array = daily_df_aggregated_potentially_incorrect.index.normalize().date
            correct_rows_mask = np.isin(index_dates_array, correct_trading_dates)
            daily_df_filtered = daily_df_aggregated_potentially_incorrect[correct_rows_mask]
            daily_df = daily_df_filtered.ffill()
            daily_df.columns = ['_'.join(col).strip() for col in daily_df.columns]

            # === Daily Targets ===
            prev_close_daily = daily_df['Close_last'].shift(1) 
            daily_df['Overnight_Gap_Daily'] = (daily_df['Open_first'] - prev_close_daily) / (prev_close_daily + EPSILON)

            daily_df['Return_1d'] = daily_df['Close_last'].pct_change(1, fill_method=None)
            daily_df['Return_3d'] = daily_df['Close_last'].pct_change(3, fill_method=None)
            daily_df['Return_5d'] = daily_df['Close_last'].pct_change(5, fill_method=None)
            daily_df['Return_10d'] = daily_df['Close_last'].pct_change(10, fill_method=None)
            daily_df['Return_20d'] = daily_df['Close_last'].pct_change(20, fill_method=None)

            daily_df['Volatility_5d'] = daily_df['Return_1d'].rolling(5).std()
            daily_df['Volatility_Change'] = daily_df['Volatility_5d'].diff()
            daily_df['Volatility_10d'] = daily_df['Return_1d'].rolling(10).std()
            daily_df['Volatility_20d'] = daily_df['Return_1d'].rolling(20).std()
            daily_df['Volatility_Skew'] = daily_df['Volatility_5d'] / (daily_df['Volatility_10d'] + EPSILON)
            daily_df['Momentum_to_Volatility'] = daily_df['Return_1d'] / (daily_df['Volatility_5d'] + EPSILON)
            daily_df['Volume_MA_ratio'] = daily_df['Volume_sum'] / (daily_df['Volume_sum'].rolling(5).mean() + EPSILON)
            daily_df['MA5_Deviation'] = daily_df['Close_last'] / (daily_df['Close_last'].rolling(5).mean() + EPSILON) - 1
            daily_df['MA20_Deviation'] = daily_df['Close_last'] / (daily_df['Close_last'].rolling(20).mean() + EPSILON) - 1

            daily_df['MA10'] = daily_df['Close_last'].rolling(10).mean()
            daily_df['MA50'] = daily_df['Close_last'].rolling(50).mean()
            daily_df['MA10_Deviation'] = daily_df['Close_last'] / (daily_df['MA10'] + EPSILON) - 1
            daily_df['MA50_Deviation'] = daily_df['Close_last'] / (daily_df['MA50'] + EPSILON) - 1
            daily_df['MA5_vs_MA20'] = daily_df['Close_last'].rolling(5).mean() / (daily_df['Close_last'].rolling(20).mean() + EPSILON) - 1 

            daily_df['Weekday'] = daily_df.index.dayofweek
            daily_df['Relative_Rank_Change'] = daily_df['Relative_Price_Position_last'].diff(3)
            daily_df['OC_Momentum'] = (daily_df['Close_last'] - daily_df['Open_first']) / (daily_df['Open_first'] + EPSILON)
            daily_df['Close_Momentum_Slope'] = (
                daily_df['Close_last'].rolling(5).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
            )

            # create a new copy to improve memory efficiency
            daily_df = daily_df.copy()

            daily_df['RSI_14d'] = _calculate_rsi(daily_df['Close_last'], 14)
            close_prices_daily = daily_df['Close_last']
            macd_line_daily, macd_signal_daily, macd_hist_daily = _calculate_MACD(close_prices_daily)
            daily_df['MACD_line_daily'] = macd_line_daily
            daily_df['MACD_signal_daily'] = macd_signal_daily
            daily_df['MACD_hist_daily'] = macd_hist_daily
            slowk, slowd = _calculate_stoch(daily_df['High_max'], daily_df['Low_min'], daily_df['Close_last'], fastk_period=14, slowk_period=3, slowd_period=3)
            daily_df['Stochastic_K_daily'] = slowk
            daily_df['Stochastic_D_daily'] = slowd
            daily_df['ATR_14d'] = _calculate_atr(daily_df['High_max'], daily_df['Low_min'], daily_df['Close_last'], 14)
            daily_df['ADX_14d'] = _calculate_adx(daily_df['High_max'], daily_df['Low_min'], daily_df['Close_last'], timeperiod=14)
            daily_df['CCI_14d'] = _calculate_cci(daily_df['High_max'], daily_df['Low_min'], daily_df['Close_last'], timeperiod=14)
            daily_df['MFI_14d'] = _calculate_mfi(daily_df['High_max'], daily_df['Low_min'], daily_df['Close_last'], daily_df['Volume_sum'], timeperiod=14)
            upper_d, middle_d, lower_d = _calculate_bbands(daily_df['Close_last'], timeperiod=20, nbdev=2)
            daily_df['BB_Upper_daily'] = upper_d
            daily_df['BB_Lower_daily'] = lower_d
            daily_df['BB_Middle_daily'] = middle_d
            daily_df['Close_vs_BB_daily'] = (daily_df['Close_last'] - middle_d) / (upper_d - lower_d + EPSILON)

            # create a new copy to improve memory efficiency
            daily_df = daily_df.copy()

            # === Lag Features ===
            lag_features_config = {
                'Return_1d': [1, 2, 3, 5, 10],
                'Volatility_5d': [1, 2, 3, 5],
                'Volume_sum': [1, 2, 3, 5],
                'Volume_MA_ratio': [1, 2, 3],
                'OC_Momentum': [1, 2, 3],
                'Overnight_Gap_Daily': [1, 2, 3],
                'ATR_91_mean': [1, 2, 3, 5],
                'RSI_91_last': [1, 2, 3, 5],
                'MACD_hist_last': [1, 2, 3],
                'Stochastic_K_last': [1, 2],
                'Relative_Price_Position_last': [1, 2, 3],
                'VWAP_Deviation_Daily_last': [1, 2],
                'RSI_14d': [1, 2, 3],
                'MACD_hist_daily': [1, 2, 3],
                'Stochastic_K_daily': [1, 2],
            }

            for base, lags in lag_features_config.items():
                for lag in lags:
                    daily_df[f'{base}_lag{lag}'] = daily_df[base].shift(lag)

            daily_df['Vol_Price_Synergy'] = daily_df['Volume_sum'] * daily_df['Returns_1h_mean']
            daily_df['Risk_Adj_Momentum_5d'] = daily_df['Return_5d'] / (daily_df['ATR_14d'] + EPSILON)
            daily_df['Risk_Adj_Momentum_1d'] = daily_df['Return_1d'] / (daily_df['ATR_14d'] + EPSILON)

            # === Trend Signals ===
            daily_df['MACD_Diff'] = daily_df['MACD_line_last'] - daily_df['MACD_signal_last']
            daily_df['MACD_Direction'] = (daily_df['MACD_Diff'].diff() > 0).astype(int)
            daily_df['Trend_Continuation_3d'] = ((daily_df['Return_1d'] > 0) & 
                                                 (daily_df['Return_1d'].shift(1) > 0) &
                                                 (daily_df['Return_1d'].shift(2) > 0)).astype(int)
            daily_df['Trend_Reversal'] = ((daily_df['Return_1d'] * daily_df['Return_1d'].shift(1)) < 0).astype(int)

            # create a new copy to improve memory efficiency
            daily_df = daily_df.copy()

            # === Price/Volume Divergence Extended ===
            daily_df['Volume_Change'] = daily_df['Volume_sum'].pct_change(fill_method=None)
            daily_df['Price_vs_Volume_Diff'] = daily_df['Return_1d'] - daily_df['Volume_Change']
            daily_df['OBV_last_diff'] = daily_df['OBV_last'].diff()
            daily_df['OBV_Price_Momentum'] = daily_df['OBV_last_diff'] * daily_df['Return_1d']
            daily_df['OBV_Price_Divergence'] = ((daily_df['OBV_last_diff'] > 0) & (daily_df['Return_1d'] < 0)) | \
                                               ((daily_df['OBV_last_diff'] < 0) & (daily_df['Return_1d'] > 0)).astype(int)

            # === Time & Cycle Position ===
            daily_df['Month'] = daily_df.index.month
            daily_df['Day_of_Month'] = daily_df.index.day
            daily_df['Day_of_Year'] = daily_df.index.dayofyear
            daily_df['Week_of_Year'] = daily_df.index.isocalendar().week.astype(int)
            daily_df['Week_of_Quarter'] = ((daily_df['Week_of_Year'] - 1) % 13 + 1)
            
            # === Opening Deviation ===
            daily_df['Open_vs_MA5'] = daily_df['Open_first'] / (daily_df['Close_last'].rolling(5).mean().shift(1) + EPSILON) - 1
            daily_df['Open_vs_MA20'] = daily_df['Open_first'] / (daily_df['Close_last'].rolling(20).mean().shift(1) + EPSILON) - 1
            daily_df['Open_Volatility_3d'] = daily_df['Open_first'].pct_change(fill_method=None).rolling(3).std()
            daily_df['Open_Momentum_3d'] = daily_df['Open_first'].pct_change(3, fill_method=None)

            # === Range Compression & Expansion ===
            daily_df['Daily_Range'] = daily_df['High_max'] - daily_df['Low_min']
            daily_df['Range_Change'] = daily_df['Daily_Range'].pct_change(fill_method=None)
            daily_df['Range_MA5'] = daily_df['Daily_Range'].rolling(5).mean()
            daily_df['Range_vs_MA5'] = daily_df['Daily_Range'] / (daily_df['Range_MA5'] + EPSILON)
            daily_df['Range_Std_5d'] = daily_df['Daily_Range'].rolling(5).std()
            daily_df['Range_to_Volume'] = daily_df['Daily_Range'] / (daily_df['Volume_sum'] + EPSILON)

            # === Volume Acceleration ===
            daily_df['Volume_Accel'] = daily_df['Volume_sum'].diff(1) - daily_df['Volume_sum'].diff(2)
            daily_df['Volume_Shock_5d'] = daily_df['Volume_sum'].rolling(5).std() / (daily_df['Volume_sum'].rolling(5).mean() + EPSILON)
            daily_df['Volume_Trend_5d'] = (
                daily_df['Volume_sum'].rolling(5).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
            )

            # === Flow Strength ===
            daily_df['Flow_to_Volume'] = daily_df['OBV_last_diff'] / (daily_df['Volume_sum'] + EPSILON)
            mfm = ((daily_df['Close_last'] - daily_df['Low_min']) - (daily_df['High_max'] - daily_df['Close_last'])) / (daily_df['High_max'] - daily_df['Low_min'] + EPSILON)
            mfv = mfm * daily_df['Volume_sum']
            daily_df['CMF_20d'] = mfv.rolling(20).sum() / (daily_df['Volume_sum'].rolling(20).sum() + EPSILON)

            # === Trend Normalization ===
            daily_df['Normalized_Trend_Strength'] = daily_df['Return_5d'] / (daily_df['Volatility_5d'] + EPSILON)
            daily_df['Normalized_Trend_Strength_20d'] = daily_df['Return_20d'] / (daily_df['Volatility_20d'] + EPSILON)
            daily_df['Normalized_OC_Momentum'] = daily_df['OC_Momentum'] / (daily_df['Volatility_5d'] + EPSILON)
            
            # === Target Calculation ===
            daily_df['Target'] = (daily_df['Open_first'].shift(-2) - daily_df['Open_first'].shift(-1)) / (daily_df['Open_first'].shift(-1) + EPSILON)

            # create a new copy to improve memory efficiency
            daily_df = daily_df.copy()

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
        all_data_panel = pd.concat(daily_dict, axis=1)
    
        all_returns = all_data_panel.loc[:, pd.IndexSlice[:, 'Return_1d']].droplevel(1, axis=1)
        rank_return_1d = all_returns.rank(axis=1, pct=True)
        
        features_to_rank = [
            'Volatility_5d', 
            'Volume_MA_ratio', 
            'MA5_Deviation', 
            'OC_Momentum', 
            'Overnight_Gap_Daily',
            'CMF_20d'
        ]
        ranks = {}
        for feature in features_to_rank:
            feature_data = all_data_panel.loc[:, pd.IndexSlice[:, feature]]
            if not feature_data.empty:
                feature_data = feature_data.droplevel(1, axis=1)
                ranks[f'Rank_{feature}'] = feature_data.rank(axis=1, pct=True)

        for stock in daily_dict.keys():
            daily_dict[stock]['Relative_Return_Rank'] = rank_return_1d[stock]
            for rank_name, rank_df in ranks.items():
                daily_dict[stock][rank_name] = rank_df[stock]

        return daily_dict


def _calculate_MACD(series: pd.DataFrame) -> pd.Series:
    fast_ema = series.ewm(span=12, adjust=False).mean()
    slow_ema = series.ewm(span=26, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    return macd_line, macd_signal, macd_hist

def _calculate_rsi(series: pd.Series, window: int) -> pd.Series:
    """ Compute Relative Strength Index (RSI) """
    delta = series.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    
    rs = avg_gain / (avg_loss + EPSILON)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """ Calculate Average True Range (ATR) """
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    
    true_range = pd.DataFrame({'hl': hl, 'hc': hc, 'lc': lc}).max(axis=1)
    atr = true_range.ewm(com=window - 1, min_periods=window).mean()
    return atr


def _calculate_obv(df: pd.DataFrame) -> pd.Series:
    """ Compute On-Balance Volume (OBV) """
    price_change = df['Close'].diff()
    signed_volume = np.sign(price_change).fillna(0) * df['Volume']
    return signed_volume.cumsum()


def _calculate_stoch(high: pd.Series, low: pd.Series, close: pd.Series, 
                     fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3):
    """Calculates Slow Stochastic Oscillator (%K and %D) using pandas."""

    min_required_len = fastk_period + slowk_period + slowd_period - 2 
    if high.isna().all() or low.isna().all() or close.isna().all() or high.count() < min_required_len:
         # Return NaNs if input is unsuitable or too short
         nan_series = pd.Series(np.nan, index=high.index)
         return nan_series, nan_series

    lowest_low = low.rolling(window=fastk_period, min_periods=fastk_period).min()
    highest_high = high.rolling(window=fastk_period, min_periods=fastk_period).max()

    range_high_low = highest_high - lowest_low
    fastk = 100.0 * (close - lowest_low) / (range_high_low + EPSILON)
    
    slowk = fastk.rolling(window=slowk_period, min_periods=slowk_period).mean()

    slowd = slowk.rolling(window=slowd_period, min_periods=slowd_period).mean()

    return slowk, slowd


def _calculate_bbands(close: pd.Series, timeperiod: int = 20, nbdev: float = 2.0):
    """Calculates Bollinger Bands."""
    if close.isna().all() or close.count() < timeperiod:
        # Return series of NaNs if input is all NaN or too short
        nan_series = pd.Series(np.nan, index=close.index)
        return nan_series, nan_series, nan_series
        
    middle = close.rolling(window=timeperiod, min_periods=timeperiod).mean()
    std_dev = close.rolling(window=timeperiod, min_periods=timeperiod).std()
    upper = middle + (nbdev * std_dev)
    lower = middle - (nbdev * std_dev)
    return upper, middle, lower


def _calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14):
    """Calculates Commodity Channel Index (CCI)."""
    if high.isna().all() or low.isna().all() or close.isna().all() or high.count() < timeperiod:
        return pd.Series(np.nan, index=high.index)

    tp = (high + low + close) / 3
    tp_sma = tp.rolling(window=timeperiod, min_periods=timeperiod).mean()
    
    # Calculate mean deviation using rolling apply (can be slow on large data)
    # Ensure raw=True for potential performance benefit, handle potential all-NaN windows
    mean_dev = tp.rolling(window=timeperiod, min_periods=timeperiod).apply(
        lambda x: np.mean(np.abs(x - np.nanmean(x))) if not np.isnan(x).all() else np.nan, 
        raw=True
    )
    
    cci = (tp - tp_sma) / (0.015 * mean_dev + EPSILON) 
    return cci


def _calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, timeperiod: int = 14):
    """Calculates Money Flow Index (MFI)."""
    if high.isna().all() or low.isna().all() or close.isna().all() or volume.isna().all() or high.count() < timeperiod:
         return pd.Series(np.nan, index=high.index)

    tp = (high + low + close) / 3
    rmf = tp * volume
    tp_diff = tp.diff(1)

    pmf = rmf.where(tp_diff > 0, 0.0).rolling(window=timeperiod, min_periods=timeperiod).sum()
    nmf = rmf.where(tp_diff < 0, 0.0).rolling(window=timeperiod, min_periods=timeperiod).sum()

    mfr = pmf / (nmf + EPSILON)
    mfi = 100.0 - (100.0 / (1.0 + mfr))
    
    mfi[nmf == 0] = 100.0

    return mfi


def _calculate_atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int):
    """Helper for ATR with Wilder smoothing (alpha=1/N)."""
    if high.isna().all() or low.isna().all() or close.isna().all() or high.count() < timeperiod + 1: # Need T+1 for diff
        return pd.Series(np.nan, index=high.index)
        
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    atr = tr.ewm(alpha=1.0/timeperiod, adjust=False, min_periods=timeperiod).mean() 
    return atr


def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14):
    """Calculates Average Directional Movement Index (ADX)."""
    if high.isna().all() or low.isna().all() or close.isna().all() or high.count() < timeperiod * 2: # ADX needs longer lookback
        return pd.Series(np.nan, index=high.index)

    move_up = high.diff(1)
    move_down = -low.diff(1) # low_prev - low
    
    plus_dm = pd.Series(np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=high.index).fillna(0)
    minus_dm = pd.Series(np.where((move_down > move_up) & (move_down > 0), move_down, 0.0), index=high.index).fillna(0)

    atr = _calculate_atr_wilder(high, low, close, timeperiod)

    plus_dm_smooth = plus_dm.ewm(alpha=1.0/timeperiod, adjust=False, min_periods=timeperiod).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1.0/timeperiod, adjust=False, min_periods=timeperiod).mean()

    plus_di = 100.0 * (plus_dm_smooth / (atr + EPSILON))
    minus_di = 100.0 * (minus_dm_smooth / (atr + EPSILON))

    di_diff = abs(plus_di - minus_di)
    di_sum = plus_di + minus_di
    
    dx = 100.0 * (di_diff / (di_sum + EPSILON)) 
    dx[di_sum < EPSILON] = 0.0

    adx = dx.ewm(alpha=1.0/timeperiod, adjust=False, min_periods=timeperiod).mean()
    
    return adx


def process_features_for_train(daily_dict: dict, test_number: int = 63, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """ Process Data, including washing, removing Nan, scaling and spliting for training purpose """
    process_feature_config = {}
    X_train_list, y_train_list = [], []

    for stock, df in daily_dict.items():
        daily_dict[stock] = df.dropna(subset=['Target'])

    stock_id_map = {}
    for stock in daily_dict.keys():
        stock_id_map[stock] = stock_code_to_id(stock)
    if len(stock_id_map.values()) != len(set(stock_id_map.values())):
        logger.error(f"Stock mapping mismatch. Stock id number: {len(stock_id_map.values())}, Stock number: {len(set(stock_id_map.values()))}")
        raise ValueError("Stock mapping mismatch.")
    process_feature_config['stock_id_map'] = stock_id_map

    all_stock_ids = sorted(set(stock_id_map.values()))
    cat_stock_id_type = pd.CategoricalDtype(categories=all_stock_ids)
    process_feature_config['cat_stock_id_type'] = cat_stock_id_type

    stock_sector_id_map = settings.STOCK_SECTOR_ID_MAP
    process_feature_config['stock_sector_id_map'] = stock_sector_id_map
    
    all_sector_ids = sorted(set(stock_sector_id_map.values()))
    cat_sector_id_type = pd.CategoricalDtype(categories=all_sector_ids)
    process_feature_config['cat_sector_id_type'] = cat_sector_id_type

    with tqdm(daily_dict.items(), desc="Process features for train") as pbar:
        for stock, df in pbar:
            pbar.set_postfix({'stock': stock}, refresh=True)

            feature_cols = [col for col in df.columns if col != 'Target']
            target_col = 'Target'
            X_train_stock = df[feature_cols]
            y_train_stock = df[target_col]

            # only train data has NAN caused by lookingback
            X_train_dropna = X_train_stock.dropna()
            y_train_stock = y_train_stock.loc[X_train_dropna.index]
            
            assert X_train_dropna.isnull().sum().sum() == 0, f"Training data contains missing values for stock {stock}"

            boolean_cols = X_train_dropna.select_dtypes(include='bool').columns.tolist()
            numerical_cols = [col for col in X_train_dropna.columns if col not in boolean_cols]

            quantiles = X_train_dropna[numerical_cols].quantile([0.05, 0.95])
            lower_bound = quantiles.xs(0.05)
            upper_bound = quantiles.xs(0.95)

            X_train_clipped = X_train_dropna.copy()

            X_train_clipped[numerical_cols] = X_train_dropna[numerical_cols].clip(lower_bound, upper_bound, axis=1)

            scaler = RobustScaler()
            X_train_scaled_numerical = pd.DataFrame(
                scaler.fit_transform(X_train_clipped[numerical_cols]),
                columns=numerical_cols,
                index=X_train_clipped.index
            )

            X_train_scaled = pd.concat([X_train_scaled_numerical, X_train_clipped[boolean_cols]], axis=1)

            stock_id = stock_id_map[stock]
            X_train_scaled['stock_id'] = stock_id
            X_train_scaled['stock_id'] = X_train_scaled['stock_id'].astype(cat_stock_id_type)

            sector_id = stock_sector_id_map[stock]
            X_train_scaled['stock_sector'] = sector_id
            X_train_scaled['stock_sector'] = X_train_scaled['stock_sector'].astype(cat_sector_id_type)

            X_train_list.append(X_train_scaled)
            y_train_list.append(y_train_stock)

            process_feature_config[stock] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'scaler': scaler
            }

    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)

    X_train, y_train = shuffle(X_train, y_train, random_state=seed)

    return X_train, y_train, process_feature_config


def process_features_for_predict(daily_dict: dict, config_data: dict) -> dict:
    """ Process Data, including washing, removing Nan, scaling and spliting for prediction purpose """
    X_predict_dict = {}

    stock_id_map = config_data['stock_id_map']
    cat_stock_id_type = config_data['cat_stock_id_type']
    stock_sector_id_map = config_data['stock_sector_id_map']
    cat_sector_id_type = config_data['cat_sector_id_type']

    with tqdm(daily_dict.items(), desc="Process features for predict") as pbar:
        for stock, df in pbar:
            cur_config_data = config_data[stock]
            lower_bound = cur_config_data['lower_bound']
            upper_bound = cur_config_data['upper_bound']
            scaler = cur_config_data['scaler']

            feature_cols = [col for col in df.columns if col != 'Target']
            X_predict_stock = df[feature_cols]

            X_predict_dropna = X_predict_stock.dropna()
            
            assert X_predict_dropna.isnull().sum().sum() == 0, f"Prediction data contains missing values for stock {stock}"

            boolean_cols = X_predict_dropna.select_dtypes(include='bool').columns.tolist()
            numerical_cols = [col for col in X_predict_dropna.columns if col not in boolean_cols]

            X_predict_clipped = X_predict_dropna.copy()

            X_predict_clipped[numerical_cols] = X_predict_dropna[numerical_cols].clip(lower_bound, upper_bound, axis=1)

            X_predict_scaled_numerical = pd.DataFrame(
                scaler.transform(X_predict_clipped[numerical_cols]),
                columns=numerical_cols,
                index=X_predict_clipped.index
            )

            X_predict_scaled = pd.concat([X_predict_scaled_numerical, X_predict_clipped[boolean_cols]], axis=1)

            stock_id = stock_id_map[stock]
            X_predict_scaled['stock_id'] = stock_id
            X_predict_scaled['stock_id'] = X_predict_scaled['stock_id'].astype(cat_stock_id_type)

            sector_id = stock_sector_id_map[stock]
            X_predict_scaled['stock_sector'] = sector_id
            X_predict_scaled['stock_sector'] = X_predict_scaled['stock_sector'].astype(cat_sector_id_type)

            X_predict = X_predict_scaled.iloc[[-1]].copy()
            X_predict_dict[stock] = X_predict

    return X_predict_dict


def process_features_for_backtest(daily_dict: dict, config_data: dict, target_stock_list: list) -> tuple[dict, dict, int]:
    """ Process Data, including washing, removing Nan, scaling and spliting for backtest purpose """
    X_backtest_dict = {}
    y_backtest_dict = {}

    stock_id_map = config_data['stock_id_map']
    cat_stock_id_type = config_data['cat_stock_id_type']
    stock_sector_id_map = config_data['stock_sector_id_map']
    cat_sector_id_type = config_data['cat_sector_id_type']
    
    for stock, df in daily_dict.items():
        daily_dict[stock] = df.dropna(subset=['Target'])
    
    for stock in daily_dict.keys():
        daily_dict[stock] = daily_dict[stock].tail(settings.TEST_DAY_NUMBER)

    backtest_day_numbers = {df.shape[0] for df in daily_dict.values()}
    if len(backtest_day_numbers) != 1:
        logger.error(f"Backtest day number mismatched: {backtest_day_numbers}")
        raise ValueError(f"Backtest day number mismatched: {backtest_day_numbers}")
    backtest_day_number = backtest_day_numbers.pop()

    for i in range(backtest_day_number):
        X_backtest_dict[i] = {}
        y_backtest_dict[i] = {}

    with tqdm(target_stock_list, desc="Process features for backtest") as pbar:
        for stock in pbar:
            pbar.set_postfix({'stock': stock,}, refresh=True)

            df = daily_dict[stock]
            cur_config_data = config_data[stock]
            lower_bound = cur_config_data['lower_bound']
            upper_bound = cur_config_data['upper_bound']
            scaler = cur_config_data['scaler']

            feature_cols = [col for col in df.columns if col != 'Target']
            target_col = 'Target'
            X_backtest_stock = df[feature_cols]
            y_backtest_stock = df[target_col]

            X_backtest_dropna = X_backtest_stock.dropna()
            y_backtest_stock = y_backtest_stock.loc[X_backtest_dropna.index]
            
            assert X_backtest_dropna.isnull().sum().sum() == 0, f"Prediction data contains missing values for stock {stock}"

            boolean_cols = X_backtest_dropna.select_dtypes(include='bool').columns.tolist()
            numerical_cols = [col for col in X_backtest_dropna.columns if col not in boolean_cols]

            X_backtest_clipped = X_backtest_dropna.copy()

            X_backtest_clipped[numerical_cols] = X_backtest_dropna[numerical_cols].clip(lower_bound, upper_bound, axis=1)

            X_backtest_scaled_numerical = pd.DataFrame(
                scaler.transform(X_backtest_clipped[numerical_cols]),
                columns=numerical_cols,
                index=X_backtest_clipped.index
            )

            X_backtest_scaled = pd.concat([X_backtest_scaled_numerical, X_backtest_clipped[boolean_cols]], axis=1)

            stock_id = stock_id_map[stock]
            X_backtest_scaled['stock_id'] = stock_id
            X_backtest_scaled['stock_id'] = X_backtest_scaled['stock_id'].astype(cat_stock_id_type)

            sector_id = stock_sector_id_map[stock]
            X_backtest_scaled['stock_sector'] = sector_id
            X_backtest_scaled['stock_sector'] = X_backtest_scaled['stock_sector'].astype(cat_sector_id_type)

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
