import pandas as pd
import yfinance as yf
import logging
import pandas_market_calendars as mcal
from tqdm import tqdm

from ml4investment.config import settings

logger = logging.getLogger(__name__)


def fetch_trading_day_data(stocks: list, period: str = '2y', interval: str = settings.DATA_INTERVAL, check_valid: bool = False) -> pd.DataFrame:
    """ Fetch trading day data for a given stock for the last given days with given interval """
    logger.info(f"Fetching data for {stocks}")
    fetched_data = {}

    with tqdm(stocks, desc="Fetch stocks data") as pbar:
        for stock in pbar:
            pbar.set_postfix({'stock': stock,}, refresh=True)
            
            data = yf.download(stock, period=period, interval=interval, auto_adjust=True, progress=False).tz_convert('America/New_York')
            
            assert not data.empty, f"No data fetched for {stock}"
            data.columns = data.columns.droplevel(1) if isinstance(data.columns, pd.MultiIndex) else data.columns
            
            nyse = mcal.get_calendar('NYSE')
            unique_dates = pd.Series(data.index.date).unique()

            if len(unique_dates) > 0 and check_valid:
                schedule = nyse.schedule(
                    start_date=unique_dates.min(),
                    end_date=unique_dates.max()
                )
                trading_days = schedule.index.date
                
                non_trading_dates = [d for d in unique_dates if d not in trading_days]
                assert not non_trading_dates, f"Non-trading dates found: {non_trading_dates}"
                
                valid_time_mask = data.index.map(lambda x: nyse.open_at_time(schedule, x))
                assert valid_time_mask.all(), "Found timestamps outside trading hours"
            
                logger.info(f"Data validation passed for {stock}")

            fetched_data[stock] = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    return fetched_data


def merge_fetched_data(existing_data: dict, new_data: dict) -> dict:
    """Merge newly fetched data with previously saved data."""
    merged = existing_data.copy()
    logger.info(f"Starting merge: {len(new_data)} stocks to merge into existing {len(existing_data)}")

    for stock in new_data:
        if stock in existing_data:
            # Concatenate and remove duplicates by index (timestamp)
            merged_df = pd.concat([existing_data[stock], new_data[stock]])
            merged_df = merged_df[~merged_df.index.duplicated(keep='last')].sort_index()
        else:
            merged_df = new_data[stock]
            logger.info(f"Adding new stock: {stock} with {len(new_data[stock])} rows")

        merged[stock] = merged_df

    logger.info(f"Merging complete. Total stocks after merge: {len(merged)}")
    return merged


def get_target_stocks(train_stock_list: list) -> list:
    """ Get target stocks across different sectors with minimum market cap """
    info_list = []
    with tqdm(train_stock_list, desc="Fetch stocks data") as pbar:
        for stock in pbar:
            pbar.set_postfix({'stock': stock,}, refresh=True)

            stock_info = yf.Ticker(stock).info
            sector = stock_info.get("sector", "Others")
            market_cap = stock_info.get("marketCap", 0)
            info_list.append({"symbol": stock, "sector": sector, "market_cap": market_cap})
            
    df = pd.DataFrame(info_list)
    logger.info(f"Get target stocks with minimum market cap: {settings.MIN_CAP}")
    df = df[df["market_cap"] >= settings.MIN_CAP].copy()
    df["sector"] = df["sector"].fillna("Others")

    target_stock_list = []
    for sector, target_count in settings.TARGET_STOCK_DISTRIBUTION.items():
        logger.info(f"Get target stocks in sector: {sector} with count: {target_count}")
        sector_df = df[df["sector"] == sector].sort_values("market_cap", ascending=False)
        target_stock_list += sector_df.head(target_count)["symbol"].tolist()
    
    return target_stock_list
