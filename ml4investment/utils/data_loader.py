import datetime
import logging
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import schwabdev
import yfinance as yf
from sklearn.utils import shuffle
from tqdm import tqdm

from ml4investment.config.global_settings import settings
from ml4investment.utils.utils import retry_api_call

logger = logging.getLogger(__name__)


def get_available_stocks() -> list[str]:
    """Fetch the top market capitalization of USA companies"""
    logger.info(f"Reading from {settings.AVAILABLE_STOCK_SOURCE}...")
    df = pd.read_csv(settings.AVAILABLE_STOCK_SOURCE)
    logger.info(f"Read {len(df)} stocks from the source")

    logger.info(f"Filtering stocks with market cap > {settings.MIN_MARKET_CAP}")
    available_stocks_list = df[df["marketcap"] > settings.MIN_MARKET_CAP][
        "Symbol"
    ].tolist()
    logger.info(f"Filtered down to {len(available_stocks_list)} stocks")

    logger.info(
        f"Add selective ETF symbols to the available stocks list: {settings.SELECTIVE_ETF}"
    )
    available_stocks_list.extend(settings.SELECTIVE_ETF)

    return available_stocks_list


def fetch_one_stock_from_schwab(
    task: tuple[str, schwabdev.Client, datetime.datetime, datetime.datetime, int, bool],
) -> tuple[str, pd.DataFrame | None]:
    """Fetch trading day data for one stock from Schwab"""
    stock, client, start_date, end_date, interval_mins, check_valid = task

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
        logger.warning(
            f"Failed to fetch data for stock {stock} after retries: {str(e)}. Skip it"
        )
        return stock, None

    assert cur_raw_data is not None
    cur_raw_df = pd.DataFrame(cur_raw_data["candles"])
    if cur_raw_df is None or cur_raw_df.empty:
        logger.warning(f"Stock: {stock}. contains no data. Skip it.")
        return stock, None

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

    market_open = datetime.time(9, 30)
    market_close = datetime.time(16, 0)
    cur_raw_df = cur_raw_df.between_time(market_open, market_close)

    if cur_raw_df.empty:
        logger.warning(f"Stock {stock}: No data within market hours. Skip it")
        return stock, None

    if not isinstance(cur_raw_df.index, pd.DatetimeIndex):
        logger.warning(
            f"Stock {stock}: Index is not DatetimeIndex after filtering. Skip it"
        )
        return stock, None

    closing_auction_mask = cur_raw_df.index.time == market_close
    current_indices = cur_raw_df.index[closing_auction_mask]
    new_indices = current_indices - datetime.timedelta(minutes=1)
    index_updates = pd.Series(new_indices, index=current_indices).to_dict()
    cur_raw_df.rename(index=index_updates, inplace=True)

    ohlcv_rules = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    try:
        cur_raw_df = cur_raw_df.resample(f"{interval_mins}min").agg(ohlcv_rules)  # type: ignore
    except (TypeError, ValueError) as e:
        logger.warning(f"Stock {stock}: Resample failed: {str(e)}. Skip it")
        return stock, None
    cur_raw_df.dropna(inplace=True)

    cur_raw_df.columns = (
        cur_raw_df.columns.droplevel(1)
        if isinstance(cur_raw_df.columns, pd.MultiIndex)
        else cur_raw_df.columns
    )

    assert isinstance(cur_raw_df.index, pd.DatetimeIndex)
    unique_dates = pd.Series(cur_raw_df.index.date).unique()

    if len(unique_dates) > 0 and check_valid:
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(
            start_date=unique_dates.min(), end_date=unique_dates.max()
        )

        assert isinstance(schedule.index, pd.DatetimeIndex)
        trading_days = schedule.index.date
        non_trading_dates = [d for d in unique_dates if d not in trading_days]
        if non_trading_dates:
            logger.warning(
                f"Non-trading dates found for stock {stock}: {non_trading_dates}. Skip it"
            )
            return stock, None

        unique_dates_set = set(unique_dates)
        missing_trading_dates = [d for d in trading_days if d not in unique_dates_set]
        if missing_trading_dates:
            logger.warning(
                f"Missing data for trading dates for stock {stock}: {missing_trading_dates}. Skip it"
            )
            return stock, None

        daily_counts = cur_raw_df.groupby(cur_raw_df.index.date).size()
        short_days = daily_counts[daily_counts < settings.DATA_PER_DAY]

        if not short_days.empty:
            logger.info(
                f"For stock {stock}, found {len(short_days)} day(s) with insufficient data. Padding them to {settings.DATA_PER_DAY} points."
            )

            padding_rows = []
            for day, actual_count in short_days.items():
                day_df = cur_raw_df[cur_raw_df.index.date == day]
                actual_first_timestamp = day_df.index.min()
                if market_open != actual_first_timestamp.time():
                    logger.warning(
                        f"First data point for stock {stock} on {day} is not at market open. Skip it"
                    )
                    return stock, None

                num_to_pad = settings.DATA_PER_DAY - actual_count

                last_row_of_day = cur_raw_df[cur_raw_df.index.date == day].iloc[-1]
                last_close = last_row_of_day["Close"]
                last_timestamp = last_row_of_day.name

                for i in range(num_to_pad):
                    new_timestamp = last_timestamp + pd.Timedelta(
                        minutes=(i + 1) * interval_mins
                    )
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
                cur_raw_df = pd.concat([cur_raw_df, padding_df])
                cur_raw_df.sort_index(inplace=True)

        assert isinstance(cur_raw_df.index, pd.DatetimeIndex)
        daily_counts = cur_raw_df.groupby(cur_raw_df.index.date).size()
        incorrect_counts = daily_counts[daily_counts != settings.DATA_PER_DAY]
        if not incorrect_counts.empty:
            logger.warning(
                f"Found days with incorrect data point counts after padding for stock {stock}. Skip it"
            )
            return stock, None

    return stock, cur_raw_df


def fetch_data_from_schwab(
    client: schwabdev.Client,
    stocks: list[str],
    period_days: int = settings.FETCH_PERIOD_DAYS,
    interval_mins: int = settings.DATA_INTERVAL_MINS,
    check_valid: bool = True,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Fetch trading day data from Schwab"""
    end_date = datetime.datetime.now(datetime.timezone.utc)
    start_date = end_date - datetime.timedelta(days=period_days)
    fetched_data: dict[str, pd.DataFrame] = {}
    cleaned_stocks_set = stocks.copy()

    tasks = [
        (stock, client, start_date, end_date, interval_mins, check_valid)
        for stock in stocks
    ]
    num_processes = min(max(1, cpu_count() - 1), settings.MAX_NUM_PROCESSES)

    with Pool(processes=num_processes) as pool:
        results_iterator = pool.imap(fetch_one_stock_from_schwab, tasks)
        pbar = tqdm(
            results_iterator,
            total=len(tasks),
            desc="Fetching and validating stocks data",
        )

        for stock, daily_df in pbar:
            pbar.set_postfix({"stock": stock}, refresh=True)
            if daily_df is None:
                cleaned_stocks_set.remove(stock)
                continue
            fetched_data[stock] = daily_df

    logger.info(f"Final cleaned available stocks fetched: {len(cleaned_stocks_set)}")

    return fetched_data, cleaned_stocks_set


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
            logger.info(
                f"Merging existing stock: {stock} with {merged_len - original_len}"
            )
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
    target_sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sample training data based on the given sampling proportion."""
    assert np.isclose(sum(sampling_proportion.values()), 1.0, atol=1e-6), (
        "Sampling proportions must sum to 1 across all stocks."
    )

    sampled_X_train_list: list[pd.DataFrame] = []
    sampled_y_train_list: list[pd.Series] = []

    logger.info(f"Target total training sample size: {target_sample_size}")

    assert isinstance(X_train.index, pd.DatetimeIndex)
    X_train_months = X_train.index.strftime("%Y-%m")

    y_train.name = "Target"

    combined_df = pd.concat([X_train, y_train], axis=1)

    for month, proportion in sampling_proportion.items():
        month_mask = X_train_months == month

        cur_month_combined_orig = combined_df[month_mask]

        if cur_month_combined_orig.empty:
            logger.warning(
                f"Month '{month}' specified in sampling_proportion has no data in X_train. Skipping."
            )
            continue

        cur_month_sample_number = round(target_sample_size * proportion)

        if cur_month_sample_number == 0:
            logger.info(
                f"Month '{month}' target sample number is 0. Skipping sampling for this month."
            )
            continue

        sampled_month_combined = cur_month_combined_orig.sample(
            n=cur_month_sample_number, random_state=seed, replace=True
        )

        sampled_month_X_train = sampled_month_combined.drop(columns=[y_train.name])
        sampled_month_y_train = sampled_month_combined[y_train.name]

        sampled_X_train_list.append(sampled_month_X_train)
        sampled_y_train_list.append(sampled_month_y_train)

    if not sampled_X_train_list:
        return pd.DataFrame(), pd.Series(dtype="object")

    final_X_train_sampled = pd.concat(sampled_X_train_list)
    final_y_train_sampled = pd.concat(sampled_y_train_list)

    shuffled = shuffle(final_X_train_sampled, final_y_train_sampled, random_state=seed)

    assert shuffled is not None, "Shuffle returned None unexpectedly"
    shuffled_X, shuffled_y = shuffled
    assert isinstance(shuffled_X, pd.DataFrame)
    assert isinstance(shuffled_y, pd.Series)

    logger.info(
        f"Successfully sampled training data. New training set size: {len(shuffled_X)}"
    )
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
            info_list.append(
                {"symbol": stock, "sector": sector, "market_cap": market_cap}
            )

    grouped_by_sector = defaultdict(list)
    for stock_data in info_list:
        grouped_by_sector[stock_data["sector"]].append(stock_data)

    stock_sectors_id_mapping = {}
    for sector, stocks_in_sector in grouped_by_sector.items():
        sorted_stocks = sorted(
            stocks_in_sector, key=lambda x: x.get("market_cap", 0), reverse=True
        )
        for stock in sorted_stocks:
            stock_sectors_id_mapping[stock["symbol"]] = settings.SECTOR_ID_MAP.get(
                sector, -1
            )

    return stock_sectors_id_mapping
