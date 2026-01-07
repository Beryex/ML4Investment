import numpy as np
import pandas as pd
import pytest


def _build_ohlcv(index: pd.DatetimeIndex) -> pd.DataFrame:
    base = np.linspace(100.0, 100.0 + len(index) - 1, len(index))
    return pd.DataFrame(
        {
            "Open": base,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base + 0.5,
            "Volume": np.arange(1, len(index) + 1),
        },
        index=index,
    )


@pytest.fixture
def make_intraday_df():
    def _make(
        stock_code: str,
        date: str,
        periods: int = 13,
        freq: str = "30min",
        tz: str | None = "America/New_York",
    ) -> pd.DataFrame:
        start = pd.Timestamp(f"{date} 09:30")
        index = pd.date_range(start=start, periods=periods, freq=freq)
        if tz:
            index = index.tz_localize(tz)
        df = _build_ohlcv(index)
        df["stock_code"] = stock_code
        return df

    return _make


@pytest.fixture
def make_daily_features_df():
    def _make(
        stock_codes: list[str],
        start_date: str = "2024-06-03",
        periods: int = 6,
        tz: str = "America/New_York",
    ) -> pd.DataFrame:
        index = pd.date_range(start=start_date, periods=periods, freq="D", tz=tz)
        rows = []
        for stock_code in stock_codes:
            stock_id = int(sum(ord(c) * 256**i for i, c in enumerate(reversed(stock_code))))
            sector_id = 1
            for ts in index:
                rows.append(
                    {
                        "datetime": ts,
                        "stock_code": stock_code,
                        "stock_id": stock_id,
                        "sector_id": sector_id,
                        "feature_a": float(ts.day),
                        "Target": float(ts.day % 3) / 100.0,
                    }
                )
        df = pd.DataFrame(rows).set_index("datetime")
        return df

    return _make
