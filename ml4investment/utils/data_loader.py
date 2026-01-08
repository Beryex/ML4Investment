import datetime
import logging
import math
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import schwabdev
import yfinance as yf
from sklearn.utils import shuffle
from tqdm import tqdm

from ml4investment.config.global_settings import settings
from ml4investment.utils.utils import retry_api_call, stock_code_to_id

logger = logging.getLogger(__name__)


def _load_stock_source() -> pd.DataFrame:
    """Load the available stock source CSV.

    Returns:
        DataFrame containing stock symbols and market cap data.
    """
    logger.info("Reading from %s...", settings.AVAILABLE_STOCK_SOURCE)
    df = pd.read_csv(settings.AVAILABLE_STOCK_SOURCE)
    logger.info("Read %d stocks from the source", len(df))
    return df


def _filter_by_market_cap(df: pd.DataFrame) -> list[str]:
    """Filter available stocks by market capitalization.

    Args:
        df: DataFrame containing market cap data.

    Returns:
        List of stock symbols meeting the market cap threshold.
    """
    logger.info("Filtering stocks with market cap > %s", settings.MIN_MARKET_CAP)
    available_stocks_list = df[df["marketcap"] > settings.MIN_MARKET_CAP]["Symbol"].tolist()
    logger.info("Filtered down to %d stocks", len(available_stocks_list))
    return available_stocks_list


def _append_selective_etfs(stocks: list[str]) -> list[str]:
    """Append selective ETFs to the stock list.

    Args:
        stocks: List of stock symbols.

    Returns:
        Updated list of stock symbols.
    """
    logger.info(
        "Add selective ETF symbols to the available stocks list: %s", settings.SELECTIVE_ETF
    )
    stocks.extend(settings.SELECTIVE_ETF)
    return stocks


def get_available_stocks() -> list[str]:
    """Fetch the top market capitalization of USA companies."""
    df = _load_stock_source()
    available_stocks_list = _filter_by_market_cap(df)
    return _append_selective_etfs(available_stocks_list)


def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Filter a DataFrame to standard market hours.

    Args:
        df: Intraday raw data.

    Returns:
        Filtered DataFrame containing market hours only.
    """
    market_open = datetime.time(9, 30)
    market_close = datetime.time(16, 0)
    return df.between_time(market_open, market_close)


def _validate_datetime_index(df: pd.DataFrame, stock: str, msg: list[str]) -> bool:
    """Validate DataFrame index type.

    Args:
        df: DataFrame to validate.
        stock: Stock symbol for logging.
        msg: Message list for warnings.

    Returns:
        True if index is DatetimeIndex, False otherwise.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        msg.append(f"Stock {stock}: Index is not DatetimeIndex after filtering. Skip it")
        return False
    return True


def _adjust_closing_auction(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust the closing auction timestamp for resampling consistency.

    Args:
        df: Intraday DataFrame.

    Returns:
        DataFrame with adjusted closing auction timestamps.
    """
    market_close = datetime.time(16, 0)
    closing_auction_mask = df.index.time == market_close
    current_indices = df.index[closing_auction_mask]
    new_indices = current_indices - datetime.timedelta(minutes=1)
    index_updates = pd.Series(new_indices, index=current_indices).to_dict()
    df.rename(index=index_updates, inplace=True)
    return df


def _resample_ohlcv(
    df: pd.DataFrame, stock: str, interval_mins: int, msg: list[str]
) -> pd.DataFrame | None:
    """Resample raw OHLCV data to the configured interval.

    Args:
        df: Raw OHLCV DataFrame.
        stock: Stock symbol for logging.
        interval_mins: Resampling interval in minutes.
        msg: Message list for warnings.

    Returns:
        Resampled DataFrame or None if resample fails.
    """
    ohlcv_rules = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    try:
        df = df.resample(f"{interval_mins}min").agg(ohlcv_rules)  # type: ignore
    except (TypeError, ValueError) as exc:
        msg.append(f"Stock {stock}: Resample failed: {str(exc)}. Skip it")
        return None
    df.dropna(inplace=True)

    df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
    return df


def _normalize_index_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize index dates to remove time and timezone.

    Args:
        dates: DatetimeIndex to normalize.

    Returns:
        Normalized DatetimeIndex without timezone.
    """
    dates = dates.normalize()
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    return dates


def _get_trading_days(nyse: mcal.MarketCalendar, start: pd.Timestamp, end: pd.Timestamp) -> set:
    """Get trading days set between start and end.

    Args:
        nyse: NYSE market calendar.
        start: Start date.
        end: End date.

    Returns:
        Set of normalized trading dates.
    """
    schedule_in_range = nyse.schedule(start_date=start, end_date=end)
    assert isinstance(schedule_in_range.index, pd.DatetimeIndex)
    schedule_index = schedule_in_range.index.normalize()
    if schedule_index.tz is not None:
        schedule_index = schedule_index.tz_localize(None)
    return set(schedule_index)


def _filter_non_trading_dates(df: pd.DataFrame, stock: str, msg: list[str]) -> pd.DataFrame | None:
    """Remove data for non-trading dates.

    Args:
        df: Intraday DataFrame.
        stock: Stock symbol for logging.
        msg: Message list for warnings.

    Returns:
        Filtered DataFrame or None if empty after filtering.
    """
    nyse = mcal.get_calendar("NYSE")
    assert isinstance(df.index, pd.DatetimeIndex)
    unique_dates = _normalize_index_dates(df.index)
    unique_dates = unique_dates.unique()

    trading_days_set = _get_trading_days(nyse, unique_dates.min(), unique_dates.max())

    non_trading_dates_mask = ~unique_dates.isin(trading_days_set)
    non_trading_dates_found = unique_dates[non_trading_dates_mask]

    if not non_trading_dates_found.empty:
        last_non_trading_date = non_trading_dates_found.max().date()
        msg.append(
            f"Stock {stock}: Found non-trading date {last_non_trading_date}. "
            f"Removing data on or before this date."
        )
        df = df[df.index.date > last_non_trading_date]

    if df.empty:
        msg.append(f"Stock {stock}: No data remains after removing non-trading dates. Skip it")
        return None

    return df


def _filter_missing_trading_dates(
    df: pd.DataFrame, stock: str, msg: list[str]
) -> pd.DataFrame | None:
    """Remove data before missing trading dates.

    Args:
        df: Intraday DataFrame.
        stock: Stock symbol for logging.
        msg: Message list for warnings.

    Returns:
        Filtered DataFrame or None if empty after filtering.
    """
    nyse = mcal.get_calendar("NYSE")
    assert isinstance(df.index, pd.DatetimeIndex)
    unique_dates = _normalize_index_dates(df.index).unique()
    unique_dates_set = set(unique_dates)

    all_trading_days_in_range = _get_trading_days(nyse, unique_dates.min(), unique_dates.max())

    missing_dates = all_trading_days_in_range - unique_dates_set
    if missing_dates:
        last_missing_date = max(missing_dates).date()
        msg.append(
            f"Stock {stock}: Found missing trading date up to {last_missing_date}. "
            f"Removing data on or before this date."
        )
        df = df[df.index.date > last_missing_date]

    if df.empty:
        msg.append(
            f"Stock {stock}: No data remains after removing days preceding data gaps. Skip it"
        )
        return None

    return df


def _filter_invalid_start_times(
    df: pd.DataFrame, stock: str, msg: list[str]
) -> pd.DataFrame | None:
    """Remove data before days with invalid start times.

    Args:
        df: Intraday DataFrame.
        stock: Stock symbol for logging.
        msg: Message list for warnings.

    Returns:
        Filtered DataFrame or None if empty after filtering.
    """
    market_open = datetime.time(9, 30)
    assert isinstance(df.index, pd.DatetimeIndex)
    daily_first_timestamps = df.groupby(df.index.date).apply(lambda group: group.index.min())
    invalid_start_mask = daily_first_timestamps.dt.time != market_open
    invalid_start_days = daily_first_timestamps[invalid_start_mask]

    if not invalid_start_days.empty:
        last_invalid_date = invalid_start_days.index.max()
        msg.append(
            f"Stock {stock}: Found day {last_invalid_date} with invalid start time. "
            f"Removing data on or before this date."
        )
        df = df[df.index.date > last_invalid_date]

    if df.empty:
        msg.append(
            f"Stock {stock}: No valid data remains after removing days "
            f"with incorrect start times. Skip it"
        )
        return None

    return df


def _pad_missing_intraday(
    df: pd.DataFrame, stock: str, interval_mins: int, msg: list[str]
) -> pd.DataFrame:
    """Pad missing intraday rows within each day.

    Args:
        df: Intraday DataFrame.
        stock: Stock symbol for logging.
        interval_mins: Resampling interval in minutes.
        msg: Message list for warnings.

    Returns:
        Padded DataFrame.
    """
    assert isinstance(df.index, pd.DatetimeIndex)
    daily_counts = df.groupby(df.index.date).size()
    short_days = daily_counts[daily_counts < settings.DATA_PER_DAY]

    if short_days.empty:
        return df

    padding_rows = []
    for day, actual_count in short_days.items():
        msg.append(
            f"Stock {stock}: Found day {day} with {actual_count} data points. "
            f"Padding to {settings.DATA_PER_DAY} points."
        )
        num_to_pad = settings.DATA_PER_DAY - actual_count

        last_row_of_day = df[df.index.date == day].iloc[-1]
        last_close = last_row_of_day["Close"]
        last_timestamp = last_row_of_day.name

        for i in range(num_to_pad):
            new_timestamp = last_timestamp + pd.Timedelta(minutes=(i + 1) * interval_mins)
            padding_rows.append(
                {
                    "datetime": new_timestamp,
                    "Open": last_close,
                    "High": last_close,
                    "Low": last_close,
                    "Close": last_close,
                    "Volume": 1,
                }
            )

    if padding_rows:
        padding_df = pd.DataFrame(padding_rows).set_index("datetime")
        df = pd.concat([df, padding_df])
        df.sort_index(inplace=True)

    return df


def _validate_daily_counts(df: pd.DataFrame, stock: str, msg: list[str]) -> bool:
    """Validate daily counts after padding.

    Args:
        df: Intraday DataFrame.
        stock: Stock symbol for logging.
        msg: Message list for warnings.

    Returns:
        True if counts are valid, False otherwise.
    """
    assert isinstance(df.index, pd.DatetimeIndex)
    daily_counts = df.groupby(df.index.date).size()
    incorrect_counts = daily_counts[daily_counts != settings.DATA_PER_DAY]
    if not incorrect_counts.empty:
        msg.append(
            f"Found days with incorrect data point counts after padding for stock {stock}. Skip it"
        )
        return False
    return True


def process_raw_data(
    cur_raw_df: pd.DataFrame, stock: str, interval_mins: int
) -> tuple[pd.DataFrame | None, list[str]]:
    """Process raw data for a single stock."""
    msg: list[str] = []
    cur_processed_df = cur_raw_df.copy()

    cur_processed_df = _filter_market_hours(cur_processed_df)
    if cur_processed_df.empty:
        msg.append(f"Stock {stock}: No data within market hours. Skip it")
        return None, msg

    if not _validate_datetime_index(cur_processed_df, stock, msg):
        return None, msg

    cur_processed_df = _adjust_closing_auction(cur_processed_df)

    cur_processed_df = _resample_ohlcv(cur_processed_df, stock, interval_mins, msg)
    if cur_processed_df is None or cur_processed_df.empty:
        return None, msg

    filtered_df = _filter_non_trading_dates(cur_processed_df, stock, msg)
    if filtered_df is None:
        return None, msg

    filtered_df = _filter_missing_trading_dates(filtered_df, stock, msg)
    if filtered_df is None:
        return None, msg

    filtered_df = _filter_invalid_start_times(filtered_df, stock, msg)
    if filtered_df is None:
        return None, msg

    filtered_df = _pad_missing_intraday(filtered_df, stock, interval_mins, msg)

    if not _validate_daily_counts(filtered_df, stock, msg):
        return None, msg

    return filtered_df[["Open", "High", "Low", "Close", "Volume"]], msg


def fetch_one_stock_from_schwab(
    task: tuple[str, schwabdev.Client, datetime.datetime, datetime.datetime, int],
) -> tuple[str, pd.DataFrame | None, list[str]]:
    """Fetch trading day data for one stock from Schwab."""
    stock, client, start_date, end_date, interval_mins = task

    try:
        cur_raw_data = retry_api_call(
            lambda: client.price_history(
                symbol=stock,
                startDate=start_date,
                endDate=end_date,
                frequencyType="minute",
                frequency=1,
                needExtendedHoursData=True,
            ).json()
        )
    except Exception as exc:
        return (
            stock,
            None,
            [f"Failed to fetch data for stock {stock} after retries: {str(exc)}. Skip it"],
        )

    cur_raw_df = pd.DataFrame(cur_raw_data["candles"])
    if cur_raw_df is None or cur_raw_df.empty:
        return stock, None, [f"Stock: {stock}. contains no data. Skip it."]

    cur_raw_df["datetime"] = (
        pd.to_datetime(cur_raw_df["datetime"] / 1000, unit="s")
        .dt.tz_localize("UTC")
        .dt.tz_convert("America/New_York")
    )
    cur_raw_df.set_index("datetime", inplace=True)
    cur_raw_df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )

    cur_processed_df, msg = process_raw_data(cur_raw_df, stock, interval_mins)

    return stock, cur_processed_df, msg


def fetch_data_from_schwab(
    client: schwabdev.Client,
    stocks: list[str],
    period_days: int = settings.FETCH_PERIOD_DAYS,
    interval_mins: int = settings.DATA_INTERVAL_MINS,
) -> pd.DataFrame:
    """Fetch trading day data from Schwab."""
    end_date = datetime.datetime.now(datetime.timezone.utc)
    start_date = end_date - datetime.timedelta(days=period_days)
    fetched_data_list = []

    tasks = [(stock, client, start_date, end_date, interval_mins) for stock in stocks]
    num_processes = min(max(1, cpu_count()), settings.MAX_NUM_PROCESSES)

    with Pool(processes=num_processes) as pool:
        results_iterator = pool.imap(fetch_one_stock_from_schwab, tasks)
        pbar = tqdm(
            results_iterator,
            total=len(tasks),
            desc="Fetching and validating stocks data",
        )

        for stock, daily_df, msg in pbar:
            pbar.set_postfix({"stock": stock}, refresh=True)
            for message in msg:
                logger.warning(message)
            if daily_df is None:
                continue

            daily_df["stock_code"] = stock

            fetched_data_list.append(daily_df)

    fetched_data_df = pd.concat(fetched_data_list)

    return fetched_data_df


def _resolve_local_date_range() -> tuple[pd.Timestamp, pd.Timestamp, int, int]:
    """Resolve local data date range and year bounds.

    Returns:
        Tuple of (start_date, end_date, earliest_year, latest_year).
    """
    start_date = pd.to_datetime(settings.TRAINING_DATA_START_DATE)
    if start_date.tzinfo is None:
        start_date = start_date.tz_localize("America/New_York")
    else:
        start_date = start_date.tz_convert("America/New_York")
    earliest_year = start_date.year

    end_date_value = settings.TESTING_DATA_END_DATE
    end_date_is_date_only = (
        isinstance(end_date_value, str) and " " not in end_date_value and "T" not in end_date_value
    )
    if end_date_value is not None:
        end_date = pd.to_datetime(end_date_value)
    else:
        end_date = pd.Timestamp.now(tz="America/New_York")
    if end_date.tzinfo is None:
        end_date = end_date.tz_localize("America/New_York")
    else:
        end_date = end_date.tz_convert("America/New_York")
    if end_date_value is not None and end_date_is_date_only:
        end_date = end_date.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    latest_year = end_date.year

    logger.info("Filtering data between %s and %s", start_date, end_date)

    return start_date, end_date, earliest_year, latest_year


def _load_stock_csv_from_paths(
    stock: str,
    base_dir_path: Path,
    earliest_year: int,
    latest_year: int,
    pbar: tqdm,
) -> list[pd.DataFrame]:
    """Load CSV files for a stock from available paths.

    Args:
        stock: Stock symbol.
        base_dir_path: Base directory path.
        earliest_year: Earliest year to scan.
        latest_year: Latest year to scan.
        pbar: Progress bar instance.

    Returns:
        List of DataFrames loaded for the stock.
    """
    data_list: list[pd.DataFrame] = []
    file_name = f"{stock}.csv"
    file_path = base_dir_path / file_name

    try:
        if not file_path.exists():
            raise FileNotFoundError
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        data_list.append(df)
        return data_list
    except FileNotFoundError:
        pass

    if earliest_year <= 0:
        return data_list

    missing_year = None
    for year in range(earliest_year, latest_year + 1):
        pbar.set_postfix({"stock": stock, "year": year}, refresh=True)
        year_str = str(year)
        year_path = base_dir_path / year_str / file_name

        try:
            df_year = pd.read_csv(year_path, index_col=0, parse_dates=True)
            data_list.append(df_year)
            missing_year = False
        except FileNotFoundError:
            if missing_year is not None:
                logger.warning("Missing year data for %s in year %s. Skip it", stock, year)
                missing_year = True
                break

    if missing_year is not None and missing_year:
        return []

    return data_list


def _coerce_local_dataframe(
    df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """Coerce local dataframe index and filter by date range.

    Args:
        df: Raw DataFrame.
        start_date: Start date for filtering.
        end_date: End date for filtering.

    Returns:
        Filtered DataFrame with timezone aligned.
    """
    df = df.sort_index()
    df = df.tz_localize("America/New_York", ambiguous="infer")
    return df.loc[start_date:end_date]


def load_local_data(
    stocks: list[str], base_dir: str, interval_mins: int = settings.DATA_INTERVAL_MINS
) -> pd.DataFrame:
    """Load the local data for the given stocks."""
    logger.info("Loading local data")
    fetched_data_list = []

    base_dir_path = Path(base_dir)
    if not base_dir_path.is_dir():
        logger.error("Base directory %s does not exist", base_dir_path)
        raise ValueError(f"Base directory {base_dir_path} does not exist")

    start_date, end_date, earliest_year, latest_year = _resolve_local_date_range()

    with tqdm(stocks, desc="Fetch stocks data") as pbar:
        for stock in pbar:
            data_list = _load_stock_csv_from_paths(
                stock, base_dir_path, earliest_year, latest_year, pbar
            )

            if not data_list:
                logger.warning("No data found for %s in local files. Skip it", stock)
                continue

            cur_raw_df = pd.concat(data_list)
            cur_raw_df = _coerce_local_dataframe(cur_raw_df, start_date, end_date)
            if cur_raw_df.empty:
                logger.warning(
                    "No data found for %s within the specified date range. Skip it", stock
                )
                continue

            cur_processed_df, msg = process_raw_data(cur_raw_df, stock, interval_mins)

            for message in msg:
                logger.warning(message)

            if cur_processed_df is None:
                continue

            cur_processed_df["stock_code"] = stock
            cur_processed_df.index.name = "datetime"

            fetched_data_list.append(cur_processed_df)

    fetched_data_df = pd.concat(fetched_data_list)

    return fetched_data_df


def merge_fetched_data(existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    """Merge newly fetched data with previously saved data."""
    if new_data.empty:
        logger.info("New data is empty, no merge needed.")
        return existing_data

    if existing_data.empty:
        logger.info("Existing data is empty, using new data directly.")
        return new_data

    logger.info(
        "Starting merge: Merging data for %d stocks into existing data for %d stocks.",
        new_data["stock_code"].nunique(),
        existing_data["stock_code"].nunique(),
    )
    original_len = len(existing_data)

    combined_df = pd.concat([existing_data, new_data])

    index_name = combined_df.index.name

    combined_df = combined_df.reset_index()

    merged_df = combined_df.drop_duplicates(subset=["stock_code", index_name], keep="last")

    merged_df = merged_df.set_index(index_name).sort_index()

    logger.info(
        "Merge complete. Total rows changed: %d -> %d. Total stocks after merge: %d.",
        original_len,
        len(merged_df),
        merged_df["stock_code"].nunique(),
    )

    return merged_df


def sample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_proportion: dict[str, float],
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sample training data based on the given sampling proportion."""
    logger.info("Sampling training data based on the given proportions...")
    sampled_X_train_list: list[pd.DataFrame] = []
    sampled_y_train_list: list[pd.Series] = []

    y_train.name = "Target"

    combined_df = pd.concat([X_train, y_train], axis=1)

    for stock, proportion in sampling_proportion.items():
        stock_mask = X_train["stock_id"] == stock_code_to_id(stock)

        cur_stock_combined_orig = combined_df[stock_mask]

        if cur_stock_combined_orig.empty:
            logger.warning(
                "Stock '%s' specified in sampling_proportion has no data in X_train. Skip it.",
                stock,
            )
            continue

        cur_stock_sample_number = math.floor(len(cur_stock_combined_orig) * proportion)

        if cur_stock_sample_number == 0:
            logger.info(
                "Stock '%s' target sample number is 0. Skipping sampling for this stock.",
                stock,
            )
            continue

        sampled_stock_combined = cur_stock_combined_orig.sample(
            n=cur_stock_sample_number, random_state=seed, replace=False
        )

        sampled_stock_X_train = sampled_stock_combined.drop(columns=[y_train.name])
        sampled_stock_y_train = sampled_stock_combined[y_train.name]

        sampled_X_train_list.append(sampled_stock_X_train)
        sampled_y_train_list.append(sampled_stock_y_train)

    if not sampled_X_train_list:
        return pd.DataFrame(), pd.Series(dtype="object")

    final_X_train_sampled = pd.concat(sampled_X_train_list)
    final_y_train_sampled = pd.concat(sampled_y_train_list)

    shuffled = shuffle(final_X_train_sampled, final_y_train_sampled, random_state=seed)

    assert shuffled is not None, "Shuffle returned None unexpectedly"
    shuffled_X, shuffled_y = shuffled
    assert isinstance(shuffled_X, pd.DataFrame)
    assert isinstance(shuffled_y, pd.Series)

    logger.info("Successfully sampled training data. New training set size: %d", len(shuffled_X))
    return shuffled_X, shuffled_y


def get_stock_sector_id_mapping(available_stocks: list) -> dict[str, int]:
    """Get train stocks across different sectors with minimum market cap."""
    info_list = []
    with tqdm(available_stocks, desc="Fetch stocks data") as pbar:
        for stock in pbar:
            pbar.set_postfix(
                {
                    "stock": stock,
                },
                refresh=True,
            )

            stock_info = yf.Ticker(stock).info
            sector = stock_info.get("sector", "Others")
            market_cap = stock_info.get("marketCap", 0)
            info_list.append({"symbol": stock, "sector": sector, "market_cap": market_cap})

    grouped_by_sector = defaultdict(list)
    for stock_data in info_list:
        grouped_by_sector[stock_data["sector"]].append(stock_data)

    stock_sectors_id_mapping = {}
    for sector, stocks_in_sector in grouped_by_sector.items():
        sorted_stocks = sorted(
            stocks_in_sector, key=lambda x: x.get("market_cap", 0), reverse=True
        )
        for stock in sorted_stocks:
            stock_sectors_id_mapping[stock["symbol"]] = settings.SECTOR_ID_MAP.get(sector, -1)

    return stock_sectors_id_mapping
