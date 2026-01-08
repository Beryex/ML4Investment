import pandas as pd

from ml4investment.config.global_settings import settings
from ml4investment.utils import feature_calculating
from ml4investment.utils.feature_calculating import (
    calculate_features,
    calculate_one_stock_features,
)


class _DummyPool:
    def __init__(self, processes=None):
        self.processes = processes

    def imap(self, func, iterable):
        return map(func, iterable)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_calculate_one_stock_features_columns(make_intraday_df):
    df = make_intraday_df("AAPL", "2024-06-03")
    daily_df = calculate_one_stock_features(df)

    assert isinstance(daily_df.index, pd.DatetimeIndex)
    assert not daily_df.empty
    assert "stock_code" in daily_df.columns
    assert "Target" in daily_df.columns
    assert daily_df["stock_code"].iloc[0] == "AAPL"
    assert str(daily_df["stock_id"].dtype) == "Int64"
    assert str(daily_df["sector_id"].dtype) == "Int64"


def test_calculate_features_embeds_etf(make_intraday_df, monkeypatch):
    df_a = make_intraday_df("AAPL", "2024-06-03")
    df_etf = make_intraday_df("QQQ", "2024-06-03")
    combined = pd.concat([df_a, df_etf]).sort_index()

    monkeypatch.setattr(settings, "SELECTIVE_ETF", ["QQQ"])
    monkeypatch.setattr(settings, "MAX_NUM_PROCESSES", 1)
    monkeypatch.setattr(feature_calculating, "Pool", _DummyPool)

    result = calculate_features(combined)

    assert "stock_code" in result.columns
    assert "QQQ_Return_1d" in result.columns
    assert "QQQ_Volatility_5d" in result.columns
    assert "QQQ_MACD_hist_daily" in result.columns
    assert "QQQ" not in result["stock_code"].unique().tolist()
