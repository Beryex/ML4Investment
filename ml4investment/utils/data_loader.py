import pandas as pd
import yfinance as yf
import logging
import pandas_market_calendars as mcal
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from ml4investment.config import settings

logger = logging.getLogger(__name__)


def fetch_data_from_yfinance(stocks: list, period: str = '2y', interval: str = settings.DATA_INTERVAL, check_valid: bool = False) -> pd.DataFrame:
    """ Fetch trading day data for a given stock for the last given days with given interval from yfinance """
    logger.info(f"Fetching data from yfinance for {stocks}")
    fetched_data = {}

    with tqdm(stocks, desc="Fetch stocks data") as pbar:
        for stock in pbar:
            pbar.set_postfix({'stock': stock,}, refresh=True)
            
            data = yf.download(stock, period=period, interval=interval, auto_adjust=True, progress=False).tz_convert('America/New_York')
            
            assert not data.empty, f"No data fetched for {stock}"
            data.columns = data.columns.droplevel(1) if isinstance(data.columns, pd.MultiIndex) else data.columns
            unique_dates = pd.Series(data.index.date).unique()

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
                
                valid_time_mask = data.index.map(lambda x: nyse.open_at_time(schedule, x))
                assert valid_time_mask.all(), "Found timestamps outside trading hours"
            
                logger.info(f"Data validation passed for {stock}")

            fetched_data[stock] = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    return fetched_data


def load_local_data(stocks: list, base_dir: str, check_valid: bool = False) -> pd.DataFrame:
    """ Load the local data for the given srocks """
    logger.info(f"Loading local data for {stocks}")
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
                        tqdm.write(f"Data not found for {stock} in year {year}, skip it")
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
                assert not non_trading_dates, f"Non-trading dates found: {non_trading_dates} for {stock}"

                unique_dates_set = set(unique_dates)
                missing_trading_dates = [d for d in trading_days if d not in unique_dates_set]
                assert not missing_trading_dates, f"Missing data for trading dates: {missing_trading_dates} for {stock}"

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
