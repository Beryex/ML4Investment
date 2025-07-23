import pandas as pd
import yfinance as yf
import logging
import pandas_market_calendars as mcal
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.utils import shuffle

from ml4investment.config import settings

logger = logging.getLogger(__name__)


def fetch_data_from_yfinance(stocks: list, period: str = '2y', interval: str = settings.DATA_INTERVAL, check_valid: bool = True) -> pd.DataFrame:
    """ Fetch trading day data for a given stock for the last given days with given interval from yfinance """
    logger.info(f"Fetching data from yfinance")
    fetched_data = {}

    data = yf.download(stocks, period=period, interval=interval, auto_adjust=True, progress=False, timeout=300).tz_convert('America/New_York')
    
    with tqdm(stocks, desc="Processing fetched stocks data") as pbar:
        for stock in pbar:
            pbar.set_postfix({'stock': stock,}, refresh=True)
            df = data.xs(stock, level='Ticker', axis=1)

            assert not df.empty, f"No data fetched for {stock}"
            df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
            unique_dates = pd.Series(df.index.date).unique()

            if len(unique_dates) > 0 and check_valid:
                nyse = mcal.get_calendar('NYSE')
                schedule = nyse.schedule(
                    start_date=unique_dates.min(),
                    end_date=unique_dates.max()
                )

                trading_days = schedule.index.date
                non_trading_dates = [d for d in unique_dates if d not in trading_days]
                assert not non_trading_dates, f"Non-trading dates found: {non_trading_dates}"
                
                unique_dates_set = set(unique_dates)
                missing_trading_dates = [d for d in trading_days if d not in unique_dates_set]
                assert not missing_trading_dates, f"Missing data for trading dates: {missing_trading_dates}"
                
                valid_time_mask = df.index.map(lambda x: nyse.open_at_time(schedule, x))
                assert valid_time_mask.all(), "Found timestamps outside trading hours"

            fetched_data[stock] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    return fetched_data


def load_local_data(stocks: list, base_dir: str, check_valid: bool = True) -> pd.DataFrame:
    """ Load the local data for the given srocks """
    logger.info(f"Loading local data")
    fetched_data = {}

    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        logger.error(f"Base directory {base_dir} does not exist")
        raise ValueError(f"Base directory {base_dir} does not exist")
    
    year_folders = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            year_folders.append(int(item.name))
    earliest_year = min(year_folders)
    latest_year = max(year_folders)

    with tqdm(stocks, desc="Fetch stocks data") as pbar:
        for stock in pbar:
            
            data_list = []
            try:
                file_name = f"{stock}.csv"
                file_path = base_dir / file_name
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                data_list.append(df)
            except FileNotFoundError:
                tqdm.write(f"Data not found for {stock} directly in .csv format, search in year folders")
                for year in range(earliest_year, latest_year + 1):
                    pbar.set_postfix({'stock': stock, 'year': year}, refresh=True)
                    year_str = str(year)
                    file_name = f"{stock}.csv"
                    file_path = base_dir / year_str / file_name
                    
                    try:
                        df_year = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    except FileNotFoundError:
                        continue
                    data_list.append(df_year)
            
            if len(data_list) == 0:
                logger.warning(f"No data found for {stock} in local files")
                continue

            data = pd.concat(data_list)
            data = data.between_time('09:30', '15:30')
            data.sort_index(inplace=True)
            data = data.tz_localize('America/New_York', ambiguous='infer')
            unique_dates = pd.Series(data.index.date).unique()

            if len(unique_dates) > 0 and check_valid:
                nyse = mcal.get_calendar('NYSE')
                schedule = nyse.schedule(
                    start_date=unique_dates.min(),
                    end_date=unique_dates.max()
                )

                trading_days = schedule.index.date
                non_trading_dates = [d for d in unique_dates if d not in trading_days]
                if non_trading_dates:
                    logger.warning(f"Removing non-trading dates found for {stock}: {non_trading_dates}")
                    data = data[data.index.normalize().isin(pd.to_datetime(trading_days))]

                unique_dates_set = set(pd.Series(data.index.date).unique())
                missing_trading_dates = [d for d in trading_days if d not in unique_dates_set]
                if missing_trading_dates:
                    logger.warning(f"Filling missing trading dates for {stock} using ffill: {missing_trading_dates}")

                    all_expected_timestamps = pd.DatetimeIndex([])
                    for dt_day in pd.to_datetime(trading_days):
                        start_ts = pd.Timestamp(f"{dt_day.date()} 09:30:00", tz='America/New_York')
                        end_ts = pd.Timestamp(f"{dt_day.date()} 15:30:00", tz='America/New_York')
                        
                        daily_range = pd.date_range(start=start_ts, end=end_ts, freq='30min')
                        all_expected_timestamps = all_expected_timestamps.union(daily_range)

                    all_expected_timestamps = pd.to_datetime(all_expected_timestamps)
                    data = data.reindex(all_expected_timestamps)
                    data = data.ffill()

                final_unique_dates = pd.Series(data.index.date).unique()
                final_unique_dates_set = set(final_unique_dates)

                # No non-trading dates should exist
                final_non_trading_dates = [d for d in final_unique_dates if d not in trading_days]
                assert not final_non_trading_dates, \
                    f"Post-processing: Non-trading dates still found: {final_non_trading_dates} for {stock}"

                #No missing trading dates should exist
                final_missing_trading_dates = [d for d in trading_days if d not in final_unique_dates_set]
                assert not final_missing_trading_dates, \
                    f"Post-processing: Missing data for trading dates still exist: {final_missing_trading_dates} for {stock}"

                start_time = pd.Timestamp('09:30').time()
                end_time = pd.Timestamp('15:30').time()
                
                time_series = data.index.time
                valid_time_mask = (time_series >= start_time) & (time_series <= end_time)

                assert valid_time_mask.all(), f"Found timestamps outside manual trading hours (09:30-15:30) for {stock}"

            fetched_data[stock] = data[['Open', 'High', 'Low', 'Close', 'Volume']]

    return fetched_data


def merge_fetched_data(existing_data: dict, new_data: dict) -> tuple[dict, dict]:
    """Merge newly fetched data with previously saved data."""
    merged = existing_data.copy()
    train_data = {}
    logger.info(f"Starting merge: {len(new_data)} stocks to merge into existing {len(existing_data)}")

    for stock in new_data:
        if stock in existing_data:
            # Concatenate and remove duplicates by index (timestamp)
            original_len = len(existing_data[stock])
            merged_df = pd.concat([existing_data[stock], new_data[stock]])
            merged_df = merged_df[~merged_df.index.duplicated(keep='last')].sort_index()
            merged_len = len(merged_df)
            logger.info(f"Merging existing stock: {stock} with {merged_len - original_len}")
        else:
            merged_df = new_data[stock]
            logger.info(f"Adding new stock: {stock} with {len(new_data[stock])} rows")

        merged[stock] = merged_df
        train_data[stock] = merged_df

    logger.info(f"Merging complete. Total stocks after merge: {len(merged)}")
    return merged, train_data


def sample_training_data(X_train: pd.DataFrame, 
                         y_train: pd.Series, 
                         sampling_proportion: dict, 
                         target_sample_size: int,
                         seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """ Sample training data based on the given sampling proportion. """
    assert np.isclose(sum(sampling_proportion.values()), 1.0, atol=1e-6), \
        "Sampling proportions must sum to 1 across all stocks."

    sampled_X_train_list = []
    sampled_y_train_list = []

    logger.info(f"Target total training sample size: {target_sample_size}")

    X_train_months = X_train.index.strftime('%Y-%m')

    y_train.name = 'Target'

    combined_df = pd.concat([X_train, y_train], axis=1)


    for month, proportion in sampling_proportion.items():
        month_mask = (X_train_months == month)

        cur_month_combined_orig = combined_df[month_mask]

        if cur_month_combined_orig.empty:
            logger.warning(f"Month '{month}' specified in sampling_proportion has no data in X_train. Skipping.")
            continue
        
        cur_month_sample_number = round(target_sample_size * proportion)
        
        if cur_month_sample_number == 0:
            logger.info(f"Month '{month}' target sample number is 0. Skipping sampling for this month.")
            continue
        
        sampled_month_combined = cur_month_combined_orig.sample(
            n=cur_month_sample_number,
            random_state=seed,
            replace=True
        )

        sampled_month_X_train = sampled_month_combined.drop(columns=[y_train.name])
        sampled_month_y_train = sampled_month_combined[y_train.name]

        sampled_X_train_list.append(sampled_month_X_train)
        sampled_y_train_list.append(sampled_month_y_train)

    final_X_train_sampled = pd.concat(sampled_X_train_list)
    final_y_train_sampled = pd.concat(sampled_y_train_list)

    final_X_train_sampled, final_y_train_sampled = shuffle(
        final_X_train_sampled, final_y_train_sampled, random_state=seed
    )
    
    logger.info(f"Successfully sampled training data. New training set size: {len(final_X_train_sampled)}")
    return final_X_train_sampled, final_y_train_sampled


def generate_stock_sectors_id_mapping(train_stock_list: list) -> dict:
    """ Get target stocks across different sectors with minimum market cap """
    info_list = []
    with tqdm(train_stock_list, desc="Fetch stocks data") as pbar:
        for stock in pbar:
            pbar.set_postfix({'stock': stock,}, refresh=True)

            stock_info = yf.Ticker(stock).info
            sector = stock_info.get("sector", "Others")
            market_cap = stock_info.get("marketCap", 0)
            info_list.append({"symbol": stock, "sector": sector, "market_cap": market_cap})
    
    grouped_by_sector = defaultdict(list)
    for stock_data in info_list:
        grouped_by_sector[stock_data['sector']].append(stock_data)

    stock_sectors_id_mapping = {}
    for sector, stocks_in_sector in grouped_by_sector.items():
        logger.info(f"\n--- Sector: {sector} ---")
        sorted_stocks = sorted(stocks_in_sector, key=lambda x: x.get('market_cap', 0), reverse=True)
        for stock in sorted_stocks:
            stock_sectors_id_mapping[stock['symbol']] = settings.SECTOR_ID_MAP[sector]
    
    return stock_sectors_id_mapping
