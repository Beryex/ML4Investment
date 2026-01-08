import logging
from multiprocessing import Pool, cpu_count
from typing import Any

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from pandas.tseries.offsets import CustomBusinessDay
from tqdm import tqdm

from ml4investment.config.global_settings import settings
from ml4investment.utils.utils import stock_code_to_id

pd.set_option("future.no_silent_downcasting", True)

EPSILON = 1e-9

logger = logging.getLogger(__name__)


def _calculate_MACD(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram.

    Args:
        series: Close price series.

    Returns:
        Tuple of (macd_line, macd_signal, macd_hist) as Series.
    """
    fast_ema = series.ewm(span=12, adjust=False).mean()
    slow_ema = series.ewm(span=26, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    return macd_line, macd_signal, macd_hist


def _calculate_rsi(series: pd.Series, window: int) -> pd.Series:
    """Compute Relative Strength Index (RSI).

    Args:
        series: Price series.
        window: RSI window length.

    Returns:
        RSI values as a Series.
    """
    delta = series.diff().astype(float)
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / (avg_loss + EPSILON)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Calculate Average True Range (ATR).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        window: ATR window length.

    Returns:
        ATR values as a Series.
    """
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()

    true_range = pd.DataFrame({"hl": hl, "hc": hc, "lc": lc}).max(axis=1)
    atr = true_range.ewm(com=window - 1, min_periods=window).mean()
    return atr


def _calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Compute On-Balance Volume (OBV).

    Args:
        df: DataFrame containing Close and Volume columns.

    Returns:
        OBV values as a Series.
    """
    price_change = df["Close"].diff().fillna(0)
    signed_volume = np.sign(price_change) * df["Volume"]
    return signed_volume.cumsum()  # type: ignore


def _calculate_stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fastk_period: int = 14,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Calculate Slow Stochastic Oscillator (%K and %D).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        fastk_period: Fast %K window.
        slowk_period: Slow %K smoothing window.
        slowd_period: Slow %D smoothing window.

    Returns:
        Tuple of (slowk, slowd) Series.
    """
    min_required_len = fastk_period + slowk_period + slowd_period - 2
    if (
        high.isna().all()
        or low.isna().all()
        or close.isna().all()
        or high.count() < min_required_len
    ):
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
    """Calculate Bollinger Bands.

    Args:
        close: Close price series.
        timeperiod: Rolling window length.
        nbdev: Standard deviation multiplier.

    Returns:
        Tuple of (upper, middle, lower) band Series.
    """
    if close.isna().all() or close.count() < timeperiod:
        nan_series = pd.Series(np.nan, index=close.index)
        return nan_series, nan_series, nan_series

    middle = close.rolling(window=timeperiod, min_periods=timeperiod).mean()
    std_dev = close.rolling(window=timeperiod, min_periods=timeperiod).std()
    upper = middle + (nbdev * std_dev)
    lower = middle - (nbdev * std_dev)
    return upper, middle, lower


def _calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14):
    """Calculate Commodity Channel Index (CCI).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        timeperiod: Rolling window length.

    Returns:
        CCI values as a Series.
    """
    if high.isna().all() or low.isna().all() or close.isna().all() or high.count() < timeperiod:
        return pd.Series(np.nan, index=high.index)

    tp = (high + low + close) / 3
    tp_sma = tp.rolling(window=timeperiod, min_periods=timeperiod).mean()

    mean_dev = tp.rolling(window=timeperiod, min_periods=timeperiod).apply(
        lambda x: np.mean(np.abs(x - np.nanmean(x))) if not np.isnan(x).all() else np.nan,
        raw=True,
    )

    cci = (tp - tp_sma) / (0.015 * mean_dev + EPSILON)
    return cci


def _calculate_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    timeperiod: int = 14,
) -> pd.Series:
    """Calculate Money Flow Index (MFI).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series.
        timeperiod: Rolling window length.

    Returns:
        MFI values as a Series.
    """
    if (
        high.isna().all()
        or low.isna().all()
        or close.isna().all()
        or volume.isna().all()
        or high.count() < timeperiod
    ):
        return pd.Series(np.nan, index=high.index)

    tp = (high + low + close) / 3
    rmf = tp * volume
    tp_diff = tp.diff(1).astype(float)

    pmf = rmf.where(tp_diff > 0, 0.0).rolling(window=timeperiod, min_periods=timeperiod).sum()
    nmf = rmf.where(tp_diff < 0, 0.0).rolling(window=timeperiod, min_periods=timeperiod).sum()

    mfr = pmf / (nmf + EPSILON)
    mfi = 100.0 - (100.0 / (1.0 + mfr))

    mfi[nmf == 0] = 100.0

    return mfi


def _calculate_atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int):
    """Calculate ATR with Wilder smoothing (alpha=1/N).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        timeperiod: Rolling window length.

    Returns:
        ATR values as a Series.
    """
    if (
        high.isna().all()
        or low.isna().all()
        or close.isna().all()
        or high.count() < timeperiod + 1
    ):
        return pd.Series(np.nan, index=high.index)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)

    atr = tr.ewm(alpha=1.0 / timeperiod, adjust=False, min_periods=timeperiod).mean()
    return atr


def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14):
    """Calculate Average Directional Movement Index (ADX).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        timeperiod: Rolling window length.

    Returns:
        ADX values as a Series.
    """
    if (
        high.isna().all()
        or low.isna().all()
        or close.isna().all()
        or high.count() < timeperiod * 2
    ):
        return pd.Series(np.nan, index=high.index)

    move_up = high.diff(1).astype(float)
    move_down = -low.diff(1).astype(float)

    plus_dm = pd.Series(
        np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=high.index
    ).fillna(0)
    minus_dm = pd.Series(
        np.where((move_down > move_up) & (move_down > 0), move_down, 0.0),
        index=high.index,
    ).fillna(0)

    atr = _calculate_atr_wilder(high, low, close, timeperiod)

    plus_dm_smooth = plus_dm.ewm(
        alpha=1.0 / timeperiod, adjust=False, min_periods=timeperiod
    ).mean()
    minus_dm_smooth = minus_dm.ewm(
        alpha=1.0 / timeperiod, adjust=False, min_periods=timeperiod
    ).mean()

    plus_di = 100.0 * (plus_dm_smooth / (atr + EPSILON))
    minus_di = 100.0 * (minus_dm_smooth / (atr + EPSILON))

    di_diff = abs(plus_di - minus_di)
    di_sum = plus_di + minus_di

    dx = 100.0 * (di_diff / (di_sum + EPSILON))
    dx[di_sum < EPSILON] = 0.0

    adx = dx.ewm(alpha=1.0 / timeperiod, adjust=False, min_periods=timeperiod).mean()

    return adx


def _prepare_intraday_frame(task: pd.DataFrame) -> pd.DataFrame:
    """Extract intraday OHLCV columns for a single stock.

    Args:
        task: Intraday DataFrame containing multiple feature columns.

    Returns:
        DataFrame with OHLCV columns and a DatetimeIndex.
    """
    price_volume = ["stock_code", "Open", "High", "Low", "Close", "Volume"]
    df = task[price_volume].copy()
    assert isinstance(df.index, pd.DatetimeIndex)
    return df


def _add_identifier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add stock and sector identifiers.

    Args:
        df: Intraday DataFrame with stock_code column.

    Returns:
        DataFrame with stock_id and sector_id columns.
    """
    df["stock_id"] = stock_code_to_id(df["stock_code"].iloc[0])
    df["sector_id"] = settings.STOCK_SECTOR_ID_MAP.get(df["stock_code"].iloc[0], -1)
    return df


def _add_basic_calculations(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic intraday return and range features.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with basic return and range features.
    """
    df["Returns_1h"] = df["Close"].pct_change(fill_method=None)
    df["Return_abs"] = df["Returns_1h"].abs()
    df["Intraday_Range"] = (df["High"] - df["Low"]) / df["Open"]
    range_high_low = df["High"] - df["Low"]
    df["Price_Efficiency"] = (df["Close"] - df["Open"]) / (range_high_low + EPSILON)
    df["High_Low_Ratio"] = df["High"] / (df["Low"] + EPSILON)
    df["Close_Open_Ratio"] = df["Close"] / (df["Open"] + EPSILON)
    df["Body_Range_Ratio"] = (df["Close"] - df["Open"]).abs() / (range_high_low + EPSILON)
    df["Upper_Shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / (
        range_high_low + EPSILON
    )
    df["Lower_Shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / (
        range_high_low + EPSILON
    )
    df["Hourly_Volatility"] = df["Returns_1h"].rolling(settings.DATA_PER_DAY).std()
    df["Hourly_Open_Gap"] = df["Open"].pct_change(fill_method=None)
    return df


def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add intraday technical indicator features.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with technical indicators.
    """
    df["RSI_91"] = _calculate_rsi(df["Close"], settings.DATA_PER_DAY * 14)
    df["ATR_91"] = _calculate_atr(df["High"], df["Low"], df["Close"], settings.DATA_PER_DAY * 14)
    df["Bollinger_Width"] = (
        df["Close"].rolling(settings.DATA_PER_DAY * 3).std()
        / df["Close"].rolling(settings.DATA_PER_DAY * 3).mean()
    ) * 100
    df["OBV"] = _calculate_obv(df)

    df["RSI_7"] = _calculate_rsi(df["Close"], settings.DATA_PER_DAY)
    df["RSI_21"] = _calculate_rsi(df["Close"], settings.DATA_PER_DAY * 3)
    df["ATR_7"] = _calculate_atr(df["High"], df["Low"], df["Close"], settings.DATA_PER_DAY)
    df["ATR_21"] = _calculate_atr(df["High"], df["Low"], df["Close"], settings.DATA_PER_DAY * 3)
    df["MA_7"] = df["Close"].rolling(settings.DATA_PER_DAY).mean()
    df["MA_21"] = df["Close"].rolling(settings.DATA_PER_DAY * 3).mean()
    df["MA_Diff_7_21"] = df["MA_7"] - df["MA_21"]
    df["Close_vs_MA7"] = df["Close"] / (df["MA_7"] + EPSILON) - 1
    df["Close_vs_MA21"] = df["Close"] / (df["MA_21"] + EPSILON) - 1

    df["ADX_14h"] = _calculate_adx(df["High"], df["Low"], df["Close"], timeperiod=14)
    df["CCI_14h"] = _calculate_cci(df["High"], df["Low"], df["Close"], timeperiod=14)
    df["MFI_14h"] = _calculate_mfi(df["High"], df["Low"], df["Close"], df["Volume"], timeperiod=14)
    upper, middle, lower = _calculate_bbands(df["Close"], timeperiod=20, nbdev=2)
    df["BB_Upper"] = upper
    df["BB_Lower"] = lower
    df["BB_Middle"] = middle
    df["Close_vs_BB"] = (df["Close"] - df["BB_Middle"]) / (
        df["BB_Upper"] - df["BB_Lower"] + EPSILON
    )
    return df


def _add_price_volume_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Add price/volume interaction features.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with price/volume interaction features.
    """
    df["Volume_Spike"] = df["Volume"] / (
        df["Volume"].rolling(settings.DATA_PER_DAY * 3).mean() + EPSILON
    )
    df["Price_Volume_Correlation"] = df["Close"].rolling(settings.DATA_PER_DAY).corr(df["Volume"])
    df["Price_Volume_Correlation"] = df["Price_Volume_Correlation"].replace(
        [np.inf, -np.inf], np.nan
    )
    df["Price_Volume_Correlation"] = df["Price_Volume_Correlation"].fillna(0)
    df["Volume_Change_Ratio"] = df["Volume"].pct_change(fill_method=None)
    df["Volume_Change_Ratio"] = df["Volume_Change_Ratio"].replace([np.inf, -np.inf], np.nan)
    df["Signed_Volume"] = np.sign(df["Returns_1h"]) * df["Volume"]
    df["Signed_Volume_MA7"] = df["Signed_Volume"].rolling(settings.DATA_PER_DAY).mean()
    return df


def _add_flow_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add flow intensity and direction consistency features.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with flow and direction features.
    """
    df["Flow_Intensity"] = (np.sign(df["Close"].diff()) * df["Volume"]) / (
        df["Volume"].rolling(21).mean()
    )
    df["Flow_Intensity"] = df["Flow_Intensity"].rolling(3).mean()
    df["Direction_Consistency"] = (
        (np.sign(df["Close"].diff()) == np.sign(df["Close"].diff().shift(1))).rolling(3).sum()
    )
    return df


def _add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum and volatility ratio features.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with momentum indicators.
    """
    df["Momentum_Accel"] = df["Close"].pct_change(3, fill_method=None) - df["Close"].pct_change(
        settings.DATA_PER_DAY, fill_method=None
    )
    df["Vol_Ratio"] = df["Close"].rolling(settings.DATA_PER_DAY).std() / (
        df["Close"].rolling(settings.DATA_PER_DAY * 5).std() + EPSILON
    )
    df["ROC_1h"] = df["Close"].pct_change(1, fill_method=None)
    df["ROC_3h"] = df["Close"].pct_change(3, fill_method=None)
    df["ROC_7h"] = df["Close"].pct_change(settings.DATA_PER_DAY, fill_method=None)
    return df


def _add_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a plain date column derived from the DatetimeIndex.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with a Date column.
    """
    idx = pd.DatetimeIndex(df.index)
    df["Date"] = idx.date
    return df


def _add_vwap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily and cumulative VWAP features.

    Args:
        df: Intraday OHLCV DataFrame with Date column.

    Returns:
        DataFrame with VWAP features.
    """
    price_vol_product = df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3
    volume_series = df["Volume"]
    cum_price_vol = price_vol_product.groupby(df["Date"]).cumsum()
    cum_vol = volume_series.groupby(df["Date"]).cumsum()
    df["VWAP_Daily"] = cum_price_vol / (cum_vol + EPSILON)
    df["VWAP_Cumulative"] = (
        df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3
    ).cumsum() / (df["Volume"].cumsum() + EPSILON)
    df["VWAP_Deviation_Daily"] = (df["Close"] - df["VWAP_Daily"]) / (df["VWAP_Daily"] + EPSILON)
    df["VWAP_Deviation_Cumulative"] = (df["Close"] - df["VWAP_Cumulative"]) / (
        df["VWAP_Cumulative"] + EPSILON
    )
    return df


def _add_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add MACD features for intraday close prices.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with MACD features.
    """
    close_prices = df["Close"]
    macd_line, macd_signal, macd_hist = _calculate_MACD(close_prices)
    df["MACD_line"] = macd_line
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist
    return df


def _add_stochastic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add stochastic oscillator features.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with stochastic %K and %D.
    """
    high_rolling = df["High"].rolling(settings.DATA_PER_DAY * 2).max()
    low_rolling = df["Low"].rolling(settings.DATA_PER_DAY * 2).min()
    k = (df["Close"] - low_rolling) / (high_rolling - low_rolling + EPSILON) * 100
    df["Stochastic_K"] = k
    df["Stochastic_D"] = k.rolling(3).mean()
    return df


def _add_relative_price_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add relative price position within recent range.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with Relative_Price_Position feature.
    """
    rolling_max = df["Close"].rolling(settings.DATA_PER_DAY * 2).max()
    rolling_min = df["Close"].rolling(settings.DATA_PER_DAY * 2).min()
    df["Relative_Price_Position"] = (df["Close"] - rolling_min) / (
        rolling_max - rolling_min + EPSILON
    )
    return df


def _add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-of-day intraday features.

    Args:
        df: Intraday OHLCV DataFrame with Date column.

    Returns:
        DataFrame with intraday time and session features.
    """
    idx = pd.DatetimeIndex(df.index)
    df["Hour_Index"] = idx.hour * 100 + idx.minute
    df["Time_Portion"] = df["Hour_Index"].apply(
        lambda x: 0 if x < 1130 else (1 if x < 1430 else 2)
    )

    morning_mask = df["Hour_Index"] < 1230
    df["Morning_Volume_Ratio"] = df["Volume"].where(morning_mask).rolling(3).sum() / (
        df["Volume"].rolling(settings.DATA_PER_DAY).sum() + EPSILON
    )

    afternoon_mask = df["Hour_Index"] >= 1330
    df["Afternoon_Return"] = df["Returns_1h"].where(afternoon_mask).rolling(3).mean()

    late_afternoon_mask = df["Hour_Index"] >= 1430
    df["Late_Afternoon_Volume_Ratio"] = df["Volume"].where(late_afternoon_mask).rolling(
        2
    ).sum() / (df["Volume"].rolling(settings.DATA_PER_DAY).sum() + EPSILON)
    df["Late_Momentum"] = df["Returns_1h"].where(late_afternoon_mask).rolling(2).mean()

    end_mask = (df["Hour_Index"] >= 1430) & (df["Hour_Index"] <= 1530)
    end_volume_sum_daily = df["Volume"].where(end_mask, 0).groupby(df["Date"]).transform("sum")
    total_volume_daily = df["Volume"].groupby(df["Date"]).transform("sum")
    df["End_Volume_Ratio"] = end_volume_sum_daily / (total_volume_daily + EPSILON)

    df["First_Hour_Return"] = df.groupby("Date")["Returns_1h"].transform("first")
    df["Last_Hour_Return"] = df.groupby("Date")["Returns_1h"].transform("last")
    df["First_Hour_Volume_Ratio"] = df.groupby("Date")["Volume"].transform("first") / (
        df.groupby("Date")["Volume"].transform("sum") + EPSILON
    )
    df["Last_Hour_Volume_Ratio"] = df.groupby("Date")["Volume"].transform("last") / (
        df.groupby("Date")["Volume"].transform("sum") + EPSILON
    )
    return df


def _add_divergence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price/volume divergence flags.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with divergence features.
    """
    price_up = df["Close"].pct_change(fill_method=None) > 0
    volume_down = df["Volume"].pct_change(fill_method=None) < 0
    df["Price_Up_Vol_Down_Div"] = (price_up & volume_down).astype(int)

    price_down = df["Close"].pct_change(fill_method=None) < 0
    volume_up = df["Volume"].pct_change(fill_method=None) > 0
    df["Price_Down_Vol_Up_Div"] = (price_down & volume_up).astype(int)
    return df


def _build_daily_aggregation_rules() -> dict[str, Any]:
    """Build aggregation rules for intraday-to-daily rollups.

    Returns:
        Mapping from column names to aggregation functions.
    """
    return {
        "stock_code": "first",
        "stock_id": "first",
        "sector_id": "first",
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": ["last", "std", "mean"],
        "Volume": ["sum", "std", "mean"],
        "Returns_1h": ["mean", "std", "median", "skew"],
        "Hourly_Open_Gap": ["mean", "std"],
        "Return_abs": ["sum", "mean"],
        "Intraday_Range": ["last", "mean"],
        "Price_Efficiency": ["mean", "last", "std"],
        "High_Low_Ratio": ["mean", "last"],
        "Close_Open_Ratio": ["mean", "last"],
        "Body_Range_Ratio": ["mean", "last"],
        "Upper_Shadow": ["mean", "last"],
        "Lower_Shadow": ["mean", "last"],
        "Hourly_Volatility": ["mean", "last"],
        "RSI_91": "last",
        "ATR_91": "mean",
        "Bollinger_Width": "last",
        "OBV": "last",
        "RSI_7": "last",
        "RSI_21": "last",
        "ATR_7": "mean",
        "ATR_21": "mean",
        "MA_7": "last",
        "MA_21": "last",
        "MA_Diff_7_21": "last",
        "Close_vs_MA7": "last",
        "Close_vs_MA21": "last",
        "ADX_14h": "last",
        "CCI_14h": "last",
        "MFI_14h": "last",
        "Close_vs_BB": "last",
        "Volume_Spike": ["max", "mean"],
        "Price_Volume_Correlation": ["mean", "last"],
        "Volume_Change_Ratio": ["mean", "std"],
        "Signed_Volume_MA7": ["last", "mean"],
        "Flow_Intensity": ["last", "mean"],
        "Direction_Consistency": ["mean", "last"],
        "Momentum_Accel": ["last", "mean"],
        "Vol_Ratio": ["last", "mean"],
        "ROC_3h": ["last", "mean"],
        "ROC_7h": ["last", "mean"],
        "VWAP_Deviation_Daily": ["mean", "last", "std"],
        "VWAP_Deviation_Cumulative": ["mean", "last"],
        "MACD_line": "last",
        "MACD_signal": "last",
        "MACD_hist": "last",
        "Stochastic_K": "last",
        "Stochastic_D": "last",
        "Relative_Price_Position": ["last", "mean"],
        "Hour_Index": ["first", "last"],
        "Time_Portion": "last",
        "Morning_Volume_Ratio": ["last", "mean"],
        "Afternoon_Return": ["last", "mean"],
        "Late_Afternoon_Volume_Ratio": ["last", "mean"],
        "Late_Momentum": ["last", "mean"],
        "End_Volume_Ratio": ["last"],
        "First_Hour_Return": "first",
        "Last_Hour_Return": "last",
        "First_Hour_Volume_Ratio": "first",
        "Last_Hour_Volume_Ratio": "last",
        "Price_Up_Vol_Down_Div": ["sum", "mean"],
        "Price_Down_Vol_Up_Div": ["sum", "mean"],
    }


def _get_nyse_calendar() -> mcal.MarketCalendar:
    """Load the NYSE trading calendar."""
    return mcal.get_calendar("NYSE")


def _aggregate_intraday_to_daily(df: pd.DataFrame, nyse: mcal.MarketCalendar) -> pd.DataFrame:
    """Aggregate intraday features to daily features aligned with trading days.

    Args:
        df: Intraday feature DataFrame.
        nyse: NYSE market calendar.

    Returns:
        Daily aggregated DataFrame with flattened column names.
    """
    idx = pd.DatetimeIndex(df.index)
    aggregation_rules = _build_daily_aggregation_rules()
    unique_dates_idx = pd.to_datetime(pd.Series(idx.normalize().date).unique())

    schedule = nyse.schedule(start_date=unique_dates_idx.min(), end_date=unique_dates_idx.max())
    assert isinstance(schedule.index, pd.DatetimeIndex)

    correct_trading_dates = schedule.index.date
    # here will introduce holidays caused by bug in CustomBusinessDay,
    # so we add these fix code below
    buggy_trading_day_offset = CustomBusinessDay(calendar=nyse)  # type: ignore
    daily_df_aggregated_potentially_incorrect = df.resample(buggy_trading_day_offset).agg(
        aggregation_rules  # type: ignore
    )
    index_dates_array = daily_df_aggregated_potentially_incorrect.index.normalize().date  # type: ignore
    correct_rows_mask = np.isin(index_dates_array, correct_trading_dates)
    daily_df_filtered = daily_df_aggregated_potentially_incorrect[correct_rows_mask]
    daily_df = daily_df_filtered.ffill()
    daily_df.columns = ["_".join(col).strip() for col in daily_df.columns]

    rename_map = {
        "stock_code_first": "stock_code",
        "stock_id_first": "stock_id",
        "sector_id_first": "sector_id",
    }
    daily_df.rename(columns=rename_map, inplace=True)
    daily_df["stock_id"] = daily_df["stock_id"].astype("Int64")
    daily_df["sector_id"] = daily_df["sector_id"].astype("Int64")
    return daily_df


def _add_daily_return_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add daily gap and return features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with daily return features.
    """
    prev_close_daily = daily_df["Close_last"].shift(1)
    daily_df["Overnight_Gap_Daily"] = (daily_df["Open_first"] - prev_close_daily) / (
        prev_close_daily + EPSILON
    )

    daily_df["Return_1d"] = daily_df["Close_last"].pct_change(1, fill_method=None)
    daily_df["Return_3d"] = daily_df["Close_last"].pct_change(3, fill_method=None)
    daily_df["Return_5d"] = daily_df["Close_last"].pct_change(5, fill_method=None)
    daily_df["Return_10d"] = daily_df["Close_last"].pct_change(10, fill_method=None)
    daily_df["Return_20d"] = daily_df["Close_last"].pct_change(20, fill_method=None)
    return daily_df


def _add_daily_volatility_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add daily volatility and deviation features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with volatility-related features.
    """
    daily_df["Volatility_5d"] = daily_df["Return_1d"].rolling(5).std()
    daily_df["Volatility_Change"] = daily_df["Volatility_5d"].diff()
    daily_df["Volatility_10d"] = daily_df["Return_1d"].rolling(10).std()
    daily_df["Volatility_20d"] = daily_df["Return_1d"].rolling(20).std()
    daily_df["Volatility_Skew"] = daily_df["Volatility_5d"] / (
        daily_df["Volatility_10d"] + EPSILON
    )
    daily_df["Momentum_to_Volatility"] = daily_df["Return_1d"] / (
        daily_df["Volatility_5d"] + EPSILON
    )
    daily_df["Volume_MA_ratio"] = daily_df["Volume_sum"] / (
        daily_df["Volume_sum"].rolling(5).mean() + EPSILON
    )
    daily_df["MA5_Deviation"] = (
        daily_df["Close_last"] / (daily_df["Close_last"].rolling(5).mean() + EPSILON) - 1
    )
    daily_df["MA20_Deviation"] = (
        daily_df["Close_last"] / (daily_df["Close_last"].rolling(20).mean() + EPSILON) - 1
    )
    return daily_df


def _add_daily_moving_average_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add daily moving average and ratio features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with moving average features.
    """
    daily_df["MA10"] = daily_df["Close_last"].rolling(10).mean()
    daily_df["MA50"] = daily_df["Close_last"].rolling(50).mean()
    daily_df["MA10_Deviation"] = daily_df["Close_last"] / (daily_df["MA10"] + EPSILON) - 1
    daily_df["MA50_Deviation"] = daily_df["Close_last"] / (daily_df["MA50"] + EPSILON) - 1
    daily_df["MA5_vs_MA20"] = (
        daily_df["Close_last"].rolling(5).mean()
        / (daily_df["Close_last"].rolling(20).mean() + EPSILON)
        - 1
    )
    return daily_df


def _add_daily_momentum_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add daily momentum and weekday position features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with daily momentum and weekday features.
    """
    assert isinstance(daily_df.index, pd.DatetimeIndex)
    daily_df["Weekday"] = daily_df.index.dayofweek
    daily_df["Relative_Rank_Change"] = daily_df["Relative_Price_Position_last"].diff(3)
    daily_df["OC_Momentum"] = (daily_df["Close_last"] - daily_df["Open_first"]) / (
        daily_df["Open_first"] + EPSILON
    )
    daily_df["Close_Momentum_Slope"] = (
        daily_df["Close_last"]
        .rolling(5)
        .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
    )
    return daily_df


def _add_daily_technical_indicators(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add daily technical indicator features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with daily technical indicators.
    """
    daily_df["RSI_14d"] = _calculate_rsi(daily_df["Close_last"], 14)
    close_prices_daily = daily_df["Close_last"]
    macd_line_daily, macd_signal_daily, macd_hist_daily = _calculate_MACD(close_prices_daily)
    daily_df["MACD_line_daily"] = macd_line_daily
    daily_df["MACD_signal_daily"] = macd_signal_daily
    daily_df["MACD_hist_daily"] = macd_hist_daily
    slowk, slowd = _calculate_stoch(
        daily_df["High_max"],
        daily_df["Low_min"],
        daily_df["Close_last"],
        fastk_period=14,
        slowk_period=3,
        slowd_period=3,
    )
    daily_df["Stochastic_K_daily"] = slowk
    daily_df["Stochastic_D_daily"] = slowd
    daily_df["ATR_14d"] = _calculate_atr(
        daily_df["High_max"], daily_df["Low_min"], daily_df["Close_last"], 14
    )
    daily_df["ADX_14d"] = _calculate_adx(
        daily_df["High_max"], daily_df["Low_min"], daily_df["Close_last"], timeperiod=14
    )
    daily_df["CCI_14d"] = _calculate_cci(
        daily_df["High_max"], daily_df["Low_min"], daily_df["Close_last"], timeperiod=14
    )
    daily_df["MFI_14d"] = _calculate_mfi(
        daily_df["High_max"],
        daily_df["Low_min"],
        daily_df["Close_last"],
        daily_df["Volume_sum"],
        timeperiod=14,
    )
    upper_d, middle_d, lower_d = _calculate_bbands(daily_df["Close_last"], timeperiod=20, nbdev=2)
    daily_df["BB_Upper_daily"] = upper_d
    daily_df["BB_Lower_daily"] = lower_d
    daily_df["BB_Middle_daily"] = middle_d
    daily_df["Close_vs_BB_daily"] = (daily_df["Close_last"] - middle_d) / (
        upper_d - lower_d + EPSILON
    )
    return daily_df


def _add_lag_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged features for selected daily metrics.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with lagged features.
    """
    lag_features_config = {
        "Return_1d": [1, 2, 3, 5, 10],
        "Volatility_5d": [1, 2, 3, 5],
        "Volume_sum": [1, 2, 3, 5],
        "Volume_MA_ratio": [1, 2, 3],
        "OC_Momentum": [1, 2, 3],
        "Overnight_Gap_Daily": [1, 2, 3],
        "ATR_91_mean": [1, 2, 3, 5],
        "RSI_91_last": [1, 2, 3, 5],
        "MACD_hist_last": [1, 2, 3],
        "Stochastic_K_last": [1, 2],
        "Relative_Price_Position_last": [1, 2, 3],
        "VWAP_Deviation_Daily_last": [1, 2],
        "RSI_14d": [1, 2, 3],
        "MACD_hist_daily": [1, 2, 3],
        "Stochastic_K_daily": [1, 2],
    }

    for base, lags in lag_features_config.items():
        for lag in lags:
            daily_df[f"{base}_lag{lag}"] = daily_df[base].shift(lag)

    return daily_df


def _add_synergy_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add feature combinations for risk-adjusted momentum and synergy.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with synergy features.
    """
    daily_df["Vol_Price_Synergy"] = daily_df["Volume_sum"] * daily_df["Returns_1h_mean"]
    daily_df["Risk_Adj_Momentum_5d"] = daily_df["Return_5d"] / (daily_df["ATR_14d"] + EPSILON)
    daily_df["Risk_Adj_Momentum_1d"] = daily_df["Return_1d"] / (daily_df["ATR_14d"] + EPSILON)
    return daily_df


def _add_trend_signals(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add daily trend signal features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with trend signal features.
    """
    daily_df["MACD_Diff"] = daily_df["MACD_line_last"] - daily_df["MACD_signal_last"]
    daily_df["MACD_Direction"] = (daily_df["MACD_Diff"].diff().astype(float) > 0).astype(int)
    daily_df["Trend_Continuation_3d"] = (
        (daily_df["Return_1d"] > 0)
        & (daily_df["Return_1d"].shift(1) > 0)
        & (daily_df["Return_1d"].shift(2) > 0)
    ).astype(int)
    daily_df["Trend_Reversal"] = (
        (daily_df["Return_1d"] * daily_df["Return_1d"].shift(1)) < 0
    ).astype(int)
    return daily_df


def _add_price_volume_divergence_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add extended price/volume divergence features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with divergence features.
    """
    daily_df["Volume_Change"] = daily_df["Volume_sum"].pct_change(fill_method=None)
    daily_df["Price_vs_Volume_Diff"] = daily_df["Return_1d"] - daily_df["Volume_Change"]
    daily_df["OBV_last_diff"] = daily_df["OBV_last"].diff()
    daily_df["OBV_Price_Momentum"] = daily_df["OBV_last_diff"] * daily_df["Return_1d"]
    daily_df["OBV_Price_Divergence"] = (
        (daily_df["OBV_last_diff"] > 0) & (daily_df["Return_1d"] < 0)
    ) | ((daily_df["OBV_last_diff"] < 0) & (daily_df["Return_1d"] > 0)).astype(int)
    return daily_df


def _add_time_cycle_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based time and cycle position features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with time cycle features.
    """
    assert isinstance(daily_df.index, pd.DatetimeIndex)
    daily_df["Month"] = daily_df.index.month
    daily_df["Day_of_Month"] = daily_df.index.day
    daily_df["Day_of_Year"] = daily_df.index.dayofyear
    daily_df["Week_of_Year"] = daily_df.index.isocalendar().week.astype(int)
    daily_df["Week_of_Quarter"] = (daily_df["Week_of_Year"] - 1) % 13 + 1
    return daily_df


def _add_opening_deviation_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add opening deviation and momentum features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with opening deviation features.
    """
    daily_df["Open_vs_MA5"] = (
        daily_df["Open_first"] / (daily_df["Close_last"].rolling(5).mean().shift(1) + EPSILON) - 1
    )
    daily_df["Open_vs_MA20"] = (
        daily_df["Open_first"] / (daily_df["Close_last"].rolling(20).mean().shift(1) + EPSILON) - 1
    )
    daily_df["Open_Volatility_3d"] = (
        daily_df["Open_first"].pct_change(fill_method=None).rolling(3).std()
    )
    daily_df["Open_Momentum_3d"] = daily_df["Open_first"].pct_change(3, fill_method=None)
    return daily_df


def _add_range_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add daily range compression and expansion features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with range-related features.
    """
    daily_df["Daily_Range"] = daily_df["High_max"] - daily_df["Low_min"]
    daily_df["Daily_Range_for_pct_change"] = daily_df["Daily_Range"].replace(0, np.nan)
    daily_df["Daily_Range_for_pct_change"] = daily_df["Daily_Range_for_pct_change"].ffill()
    daily_df["Daily_Range_for_pct_change"] = daily_df["Daily_Range_for_pct_change"].fillna(EPSILON)
    daily_df["Range_Change"] = daily_df["Daily_Range_for_pct_change"].pct_change(fill_method=None)
    daily_df["Range_MA5"] = daily_df["Daily_Range"].rolling(5).mean()
    daily_df["Range_vs_MA5"] = daily_df["Daily_Range"] / (daily_df["Range_MA5"] + EPSILON)
    daily_df["Range_Std_5d"] = daily_df["Daily_Range"].rolling(5).std()
    daily_df["Range_to_Volume"] = daily_df["Daily_Range"] / (daily_df["Volume_sum"] + EPSILON)
    return daily_df


def _add_volume_acceleration_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add volume acceleration and trend features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with volume acceleration features.
    """
    daily_df["Volume_Accel"] = daily_df["Volume_sum"].diff(1) - daily_df["Volume_sum"].diff(2)
    daily_df["Volume_Shock_5d"] = daily_df["Volume_sum"].rolling(5).std() / (
        daily_df["Volume_sum"].rolling(5).mean() + EPSILON
    )
    daily_df["Volume_Trend_5d"] = (
        daily_df["Volume_sum"]
        .rolling(5)
        .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
    )
    return daily_df


def _add_flow_strength_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add flow strength and money flow features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with flow strength features.
    """
    daily_df["Flow_to_Volume"] = daily_df["OBV_last_diff"] / (daily_df["Volume_sum"] + EPSILON)
    mfm = (
        (daily_df["Close_last"] - daily_df["Low_min"])
        - (daily_df["High_max"] - daily_df["Close_last"])
    ) / (daily_df["High_max"] - daily_df["Low_min"] + EPSILON)
    mfv = mfm * daily_df["Volume_sum"]
    daily_df["CMF_20d"] = mfv.rolling(20).sum() / (
        daily_df["Volume_sum"].rolling(20).sum() + EPSILON
    )
    return daily_df


def _add_trend_normalization_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add trend normalization features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with normalized trend features.
    """
    daily_df["Normalized_Trend_Strength"] = daily_df["Return_5d"] / (
        daily_df["Volatility_5d"] + EPSILON
    )
    daily_df["Normalized_Trend_Strength_20d"] = daily_df["Return_20d"] / (
        daily_df["Volatility_20d"] + EPSILON
    )
    daily_df["Normalized_OC_Momentum"] = daily_df["OC_Momentum"] / (
        daily_df["Volatility_5d"] + EPSILON
    )
    return daily_df


def _add_target_column(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add prediction target column.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with Target column.
    """
    daily_df["Target"] = (daily_df["Open_first"].shift(-2) - daily_df["Open_first"].shift(-1)) / (
        daily_df["Open_first"].shift(-1) + EPSILON
    )
    return daily_df


def _drop_incomplete_trading_day(
    daily_df: pd.DataFrame, nyse: mcal.MarketCalendar
) -> pd.DataFrame:
    """Drop the last day if the current trading session is incomplete.

    Args:
        daily_df: Daily aggregated DataFrame.
        nyse: NYSE market calendar.

    Returns:
        DataFrame with incomplete last trading day removed.
    """
    now = pd.Timestamp.now(tz="America/New_York")
    trading_hours = nyse.schedule(start_date=now.date(), end_date=now.date())
    if not trading_hours.empty:
        market_open = trading_hours.iloc[0]["market_open"]
        market_close = trading_hours.iloc[0]["market_close"]
        if market_open <= now <= market_close:
            if not daily_df.empty and daily_df.index[-1].date() == now.date():
                daily_df = daily_df.drop(daily_df.index[-1])

    return daily_df


def _add_cross_stock_rank_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-stock ranking features after daily aggregation.

    Args:
        daily_df: Combined daily DataFrame for all stocks.

    Returns:
        DataFrame with ranking features added.
    """
    daily_df["Relative_Return_Rank"] = daily_df.groupby(level=0)["Return_1d"].rank(pct=True)

    features_to_rank = [
        "Volatility_5d",
        "Volume_MA_ratio",
        "MA5_Deviation",
        "OC_Momentum",
        "Overnight_Gap_Daily",
        "CMF_20d",
    ]
    for feature in features_to_rank:
        if feature in daily_df.columns:
            daily_df[f"Rank_{feature}"] = daily_df.groupby(level=0)[feature].rank(pct=True)
    return daily_df


def _embed_etf_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Embed ETF features into stock rows for downstream modeling.

    Args:
        daily_df: Combined daily DataFrame for all stocks.

    Returns:
        DataFrame with ETF feature columns merged into stock rows.
    """
    etf_list = [
        stock for stock in daily_df["stock_code"].unique() if stock in settings.SELECTIVE_ETF
    ]

    features_to_embed_from_etfs = [
        "Return_1d",
        "Return_5d",
        "Volatility_5d",
        "RSI_14d",
        "MACD_hist_daily",
        "Volume_MA_ratio",
        "ADX_14d",
        "CMF_20d",
    ]

    etf_data = daily_df[daily_df["stock_code"].isin(etf_list)]
    etf_data_indexed = etf_data.set_index("stock_code", append=True)
    etf_features_subset = etf_data_indexed[features_to_embed_from_etfs]
    etf_features_wide = etf_features_subset.unstack("stock_code")
    etf_features_wide.columns = [
        f"{stock}_{feature}" for feature, stock in etf_features_wide.columns
    ]

    stocks_only_df = daily_df[~daily_df["stock_code"].isin(etf_list)].copy()

    final_df = pd.merge(
        stocks_only_df, etf_features_wide, left_index=True, right_index=True, how="left"
    )

    return final_df


def calculate_one_stock_features(task: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily features for one stock.

    Args:
        task: Intraday OHLCV DataFrame for a single stock.

    Returns:
        Daily feature DataFrame indexed by date.
    """
    df = _prepare_intraday_frame(task)
    df = _add_identifier_features(df)
    df = _add_basic_calculations(df)
    df = _add_technical_indicators(df)
    df = _add_price_volume_interaction(df)
    df = _add_flow_direction_features(df)
    df = _add_momentum_indicators(df)
    df = _add_date_column(df)
    df = _add_vwap_features(df)
    df = _add_macd_features(df)
    df = _add_stochastic_features(df)
    df = _add_relative_price_position(df)
    df = _add_intraday_features(df)
    df = _add_divergence_features(df)

    nyse = _get_nyse_calendar()
    daily_df = _aggregate_intraday_to_daily(df, nyse)

    daily_df = _add_daily_return_features(daily_df)
    daily_df = _add_daily_volatility_features(daily_df)
    daily_df = _add_daily_moving_average_features(daily_df)
    daily_df = _add_daily_momentum_features(daily_df)

    # create a new copy to improve memory efficiency
    daily_df = daily_df.copy()

    daily_df = _add_daily_technical_indicators(daily_df)

    # create a new copy to improve memory efficiency
    daily_df = daily_df.copy()

    daily_df = _add_lag_features(daily_df)
    daily_df = _add_synergy_features(daily_df)
    daily_df = _add_trend_signals(daily_df)

    # create a new copy to improve memory efficiency
    daily_df = daily_df.copy()

    daily_df = _add_price_volume_divergence_features(daily_df)
    daily_df = _add_time_cycle_features(daily_df)
    daily_df = _add_opening_deviation_features(daily_df)
    daily_df = _add_range_features(daily_df)
    daily_df = _add_volume_acceleration_features(daily_df)
    daily_df = _add_flow_strength_features(daily_df)
    daily_df = _add_trend_normalization_features(daily_df)
    daily_df = _add_target_column(daily_df)

    # create a new copy to improve memory efficiency
    daily_df = daily_df.copy()

    daily_df = _drop_incomplete_trading_day(daily_df, nyse)

    return daily_df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Process 1h OHLCV data to create daily features and prediction target.

    Args:
        df: Intraday OHLCV DataFrame containing multiple stocks.

    Returns:
        Combined daily feature DataFrame with cross-stock features.
    """
    calculated_features_list = []
    tasks = [group for _, group in df.groupby("stock_code")]
    num_processes = min(max(1, cpu_count()), settings.MAX_NUM_PROCESSES)

    with Pool(processes=num_processes) as pool:
        results_iterator = pool.imap(calculate_one_stock_features, tasks)
        pbar = tqdm(results_iterator, total=len(tasks), desc="Calculating features")

        for daily_df in pbar:
            calculated_features_list.append(daily_df)

    daily_df_combined = pd.concat(calculated_features_list, axis=0)
    daily_df_combined.sort_index(inplace=True)

    daily_df_combined = _add_cross_stock_rank_features(daily_df_combined)
    final_df = _embed_etf_features(daily_df_combined)

    return final_df
