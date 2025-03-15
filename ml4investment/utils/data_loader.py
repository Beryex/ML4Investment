import pandas as pd
import yfinance as yf
import logging
import pandas_market_calendars as mcal

logger = logging.getLogger(__name__)


def fetch_trading_day_data(stock: str, period: str = '2y', interval: str = '1h') -> pd.DataFrame:
    """ Fetch trading day data for a given stock for the last given days with given interval """
    logger.info(f"Fetching data for {stock}") 
    data = yf.download(stock, period=period, interval=interval).tz_convert('America/New_York')
    
    assert not data.empty, f"No data fetched for {stock}"
    data.columns = data.columns.droplevel(1) if isinstance(data.columns, pd.MultiIndex) else data.columns
    
    nyse = mcal.get_calendar('NYSE')
    unique_dates = pd.Series(data.index.date).unique()
    if len(unique_dates) > 0:
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
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]
