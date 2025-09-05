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


def get_available_stocks() -> list[str]:
    """Fetch the top market capitalization of USA companies"""
    logger.info(f"Reading from {settings.AVAILABLE_STOCK_SOURCE}...")
    df = pd.read_csv(settings.AVAILABLE_STOCK_SOURCE)
    logger.info(f"Read {len(df)} stocks from the source")

    logger.info(f"Filtering stocks with market cap > {settings.MIN_MARKET_CAP}")
    available_stocks_list = df[df["marketcap"] > settings.MIN_MARKET_CAP]["Symbol"].tolist()
    logger.info(f"Filtered down to {len(available_stocks_list)} stocks")

    logger.info(
        f"Add selective ETF symbols to the available stocks list: {settings.SELECTIVE_ETF}"
    )
    available_stocks_list.extend(settings.SELECTIVE_ETF)

    return available_stocks_list


def process_raw_data(
    cur_raw_df: pd.DataFrame, stock: str, interval_mins: int
) -> tuple[pd.DataFrame | None, list[str]]:
    """Process raw data for a single stock"""
    msg = []
    cur_processed_df = cur_raw_df.copy()
    market_open = datetime.time(9, 30)
    market_close = datetime.time(16, 0)
    cur_processed_df = cur_processed_df.between_time(market_open, market_close)

    if cur_processed_df.empty:
        msg.append(f"Stock {stock}: No data within market hours. Skip it")
        return None, msg

    if not isinstance(cur_processed_df.index, pd.DatetimeIndex):
        msg.append(f"Stock {stock}: Index is not DatetimeIndex after filtering. Skip it")
        return None, msg

    closing_auction_mask = cur_processed_df.index.time == market_close
    current_indices = cur_processed_df.index[closing_auction_mask]
    new_indices = current_indices - datetime.timedelta(minutes=1)
    index_updates = pd.Series(new_indices, index=current_indices).to_dict()
    cur_processed_df.rename(index=index_updates, inplace=True)

    ohlcv_rules = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    try:
        cur_processed_df = cur_processed_df.resample(f"{interval_mins}min").agg(
            ohlcv_rules  # type: ignore
        )
    except (TypeError, ValueError) as e:
        msg.append(f"Stock {stock}: Resample failed: {str(e)}. Skip it")
        return None, msg
    cur_processed_df.dropna(inplace=True)

    cur_processed_df.columns = (
        cur_processed_df.columns.droplevel(1)
        if isinstance(cur_processed_df.columns, pd.MultiIndex)
        else cur_processed_df.columns
    )

    nyse = mcal.get_calendar("NYSE")

    """ Filter non-trading dates """
    assert isinstance(cur_processed_df.index, pd.DatetimeIndex)
    unique_dates = pd.to_datetime(cur_processed_df.index.date).unique()
    schedule_in_range = nyse.schedule(start_date=unique_dates.min(), end_date=unique_dates.max())
    assert isinstance(schedule_in_range.index, pd.DatetimeIndex)
    trading_days_set = set(pd.to_datetime(schedule_in_range.index.date))

    non_trading_dates_mask = ~pd.Series(unique_dates).isin(list(trading_days_set))
    non_trading_dates_found = unique_dates[non_trading_dates_mask]

    if not non_trading_dates_found.empty:
        last_non_trading_date = non_trading_dates_found.max().date()
        msg.append(
            f"Stock {stock}: Found non-trading date {last_non_trading_date}. "
            f"Removing data on or before this date."
        )
        cur_processed_df = cur_processed_df[cur_processed_df.index.date > last_non_trading_date]

    if cur_processed_df.empty:
        msg.append(f"Stock {stock}: No data remains after removing non-trading dates. Skip it")
        return None, msg

    """ Filter missing trading dates """
    assert isinstance(cur_processed_df.index, pd.DatetimeIndex)
    unique_dates = pd.to_datetime(cur_processed_df.index.date).unique()
    unique_dates_set = set(unique_dates)
    schedule_in_range = nyse.schedule(start_date=unique_dates.min(), end_date=unique_dates.max())
    assert isinstance(schedule_in_range.index, pd.DatetimeIndex)
    all_trading_days_in_range = set(pd.to_datetime(schedule_in_range.index.date))

    missing_dates = all_trading_days_in_range - unique_dates_set
    if missing_dates:
        last_missing_date = max(missing_dates).date()
        msg.append(
            f"Stock {stock}: Found missing trading date up to {last_missing_date}. "
            f"Removing data on or before this date."
        )
        cur_processed_df = cur_processed_df[cur_processed_df.index.date > last_missing_date]

    if cur_processed_df.empty:
        msg.append(
            f"Stock {stock}: No data remains after removing days preceding data gaps. Skip it"
        )
        return None, msg

    """ Filter for valid trading days """
    assert isinstance(cur_processed_df.index, pd.DatetimeIndex)
    daily_first_timestamps = cur_processed_df.groupby(cur_processed_df.index.date).apply(
        lambda group: group.index.min()
    )
    invalid_start_mask = daily_first_timestamps.dt.time != market_open
    invalid_start_days = daily_first_timestamps[invalid_start_mask]

    if not invalid_start_days.empty:
        last_invalid_date = invalid_start_days.index.max()
        msg.append(
            f"Stock {stock}: Found day {last_invalid_date} with invalid start time. "
            f"Removing data on or before this date."
        )
        cur_processed_df = cur_processed_df[cur_processed_df.index.date > last_invalid_date]

    if cur_processed_df.empty:
        msg.append(
            f"Stock {stock}: No valid data remains after removing days "
            f"with incorrect start times. Skip it"
        )
        return None, msg

    """ Pad missing data within each day """
    assert isinstance(cur_processed_df.index, pd.DatetimeIndex)
    daily_counts = cur_processed_df.groupby(cur_processed_df.index.date).size()
    short_days = daily_counts[daily_counts < settings.DATA_PER_DAY]

    if not short_days.empty:
        padding_rows = []
        for day, actual_count in short_days.items():
            msg.append(
                f"Stock {stock}: Found day {day} with {actual_count} data points. "
                f"Padding to {settings.DATA_PER_DAY} points."
            )
            num_to_pad = settings.DATA_PER_DAY - actual_count

            last_row_of_day = cur_processed_df[cur_processed_df.index.date == day].iloc[-1]
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
            cur_processed_df = pd.concat([cur_processed_df, padding_df])
            cur_processed_df.sort_index(inplace=True)

    assert isinstance(cur_processed_df.index, pd.DatetimeIndex)
    daily_counts = cur_processed_df.groupby(cur_processed_df.index.date).size()
    incorrect_counts = daily_counts[daily_counts != settings.DATA_PER_DAY]
    if not incorrect_counts.empty:
        msg.append(
            f"Found days with incorrect data point counts after padding for stock {stock}. Skip it"
        )
        return None, msg

    return cur_processed_df[["Open", "High", "Low", "Close", "Volume"]], msg


def fetch_one_stock_from_schwab(
    task: tuple[str, schwabdev.Client, datetime.datetime, datetime.datetime, int],
) -> tuple[str, pd.DataFrame | None, list[str]]:
    """Fetch trading day data for one stock from Schwab"""
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
    except Exception as e:
        return (
            stock,
            None,
            [f"Failed to fetch data for stock {stock} after retries: {str(e)}. Skip it"],
        )

    assert cur_raw_data is not None
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
) -> dict[str, pd.DataFrame]:
    """Fetch trading day data from Schwab"""
    end_date = datetime.datetime.now(datetime.timezone.utc)
    start_date = end_date - datetime.timedelta(days=period_days)
    fetched_data: dict[str, pd.DataFrame] = {}

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
            for m in msg:
                logger.warning(m)
            if daily_df is None:
                continue
            fetched_data[stock] = daily_df

    return fetched_data


def load_local_data(
    stocks: list[str], base_dir: str, interval_mins: int = settings.DATA_INTERVAL_MINS
) -> dict[str, pd.DataFrame]:
    """Load the local data for the given stocks"""
    logger.info("Loading local data")
    fetched_data: dict[str, pd.DataFrame] = {}

    base_dir_path = Path(base_dir)
    if not base_dir_path.is_dir():
        logger.error(f"Base directory {base_dir_path} does not exist")
        raise ValueError(f"Base directory {base_dir_path} does not exist")

    year_folders = []
    for item in base_dir_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            year_folders.append(int(item.name))

    start_date = datetime.date.fromisoformat(settings.TRAINING_DATA_START_DATE)
    earliest_year = start_date.year
    if settings.TESTING_DATA_END_DATE is not None:
        end_date = datetime.date.fromisoformat(settings.TESTING_DATA_END_DATE)
    else:
        end_date = datetime.date.today()
    latest_year = end_date.year
    start_date = pd.Timestamp(start_date, tz="America/New_York")
    end_date = pd.Timestamp(end_date, tz="America/New_York")
    logger.info(f"Filtering data between {start_date} and {end_date}")

    with tqdm(stocks, desc="Fetch stocks data") as pbar:
        for stock in pbar:
            data_list = []
            try:
                file_name = f"{stock}.csv"
                file_path = base_dir_path / file_name
                if not file_path.exists():
                    raise FileNotFoundError
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                data_list.append(df)
            except FileNotFoundError:
                if earliest_year > 0:
                    missing_year = None
                    for year in range(earliest_year, latest_year + 1):
                        pbar.set_postfix({"stock": stock, "year": year}, refresh=True)
                        year_str = str(year)
                        file_name = f"{stock}.csv"
                        file_path = base_dir_path / year_str / file_name

                        try:
                            df_year = pd.read_csv(file_path, index_col=0, parse_dates=True)
                            data_list.append(df_year)
                            missing_year = False
                        except FileNotFoundError:
                            if missing_year is not None:
                                logger.warning(
                                    f"Missing year data for {stock} in year {year}. Skip it"
                                )
                                missing_year = True
                                break

                    if missing_year is not None and missing_year:
                        continue

            if not data_list:
                logger.warning(f"No data found for {stock} in local files. Skip it")
                continue

            cur_raw_df = pd.concat(data_list)
            cur_raw_df.sort_index(inplace=True)
            cur_raw_df = cur_raw_df.tz_localize("America/New_York", ambiguous="infer")
            cur_raw_df = cur_raw_df.loc[start_date:end_date]
            if cur_raw_df.empty:
                logger.warning(
                    f"No data found for {stock} within the specified date range. Skip it"
                )
                continue

            cur_processed_df, msg = process_raw_data(cur_raw_df, stock, interval_mins)

            for m in msg:
                logger.warning(m)

            if cur_processed_df is None:
                continue

            fetched_data[stock] = cur_processed_df

    return fetched_data


def merge_fetched_data(existing_data: dict, new_data: dict) -> tuple[dict, dict]:
    """Merge newly fetched data with previously saved data."""
    merged = existing_data.copy()
    train_data = {}
    logger.info(
        f"Starting merge: {len(new_data)} stocks to merge into existing {len(existing_data)}"
    )

    for stock in new_data:
        if stock in existing_data:
            # Concatenate and remove duplicates by index (timestamp)
            original_len = len(existing_data[stock])
            merged_df = pd.concat([existing_data[stock], new_data[stock]])
            merged_df = merged_df[~merged_df.index.duplicated(keep="last")].sort_index()
            merged_len = len(merged_df)
            logger.info(f"Merging existing stock: {stock} with {merged_len - original_len}")
        else:
            merged_df = new_data[stock]
            logger.info(f"Adding new stock: {stock} with {len(new_data[stock])} rows")

        merged[stock] = merged_df
        train_data[stock] = merged_df

    logger.info(f"Merging complete. Total stocks after merge: {len(merged)}")
    return merged, train_data


def sample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_proportion: dict[str, float],
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sample training data based on the given sampling proportion."""
    sampled_X_train_list: list[pd.DataFrame] = []
    sampled_y_train_list: list[pd.Series] = []

    y_train.name = "Target"

    combined_df = pd.concat([X_train, y_train], axis=1)

    for stock, proportion in sampling_proportion.items():
        stock_mask = X_train["stock_id"] == stock_code_to_id(stock)

        cur_stock_combined_orig = combined_df[stock_mask]

        if cur_stock_combined_orig.empty:
            logger.warning(
                f"Stock '{stock}' specified in sampling_proportion has no data in X_train. "
                f"Skip it."
            )
            continue

        cur_stock_sample_number = math.floor(len(cur_stock_combined_orig) * proportion)

        if cur_stock_sample_number == 0:
            logger.info(
                f"Stock '{stock}' target sample number is 0. Skipping sampling for this stock."
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

    logger.info(f"Successfully sampled training data. New training set size: {len(shuffled_X)}")
    return shuffled_X, shuffled_y


def get_stock_sector_id_mapping(available_stocks: list) -> dict[str, int]:
    """Get train stocks across different sectors with minimum market cap"""
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
