import math

import pandas as pd
import pytest

from ml4investment.config.global_settings import settings
from ml4investment.utils.data_loader import (
    load_local_data,
    merge_fetched_data,
    process_raw_data,
    sample_training_data,
)
from ml4investment.utils.utils import stock_code_to_id


def test_process_raw_data_valid(make_intraday_df):
    df = make_intraday_df("AAPL", "2024-06-03").drop(columns=["stock_code"])
    processed, msg = process_raw_data(df, "AAPL", interval_mins=30)

    assert processed is not None
    assert isinstance(processed.index, pd.DatetimeIndex)
    assert len(processed) == settings.DATA_PER_DAY
    assert msg == []


def test_process_raw_data_pads_short_day(make_intraday_df):
    df = make_intraday_df(
        "AAPL", "2024-06-04", periods=settings.DATA_PER_DAY - 1
    ).drop(columns=["stock_code"])
    processed, msg = process_raw_data(df, "AAPL", interval_mins=30)

    assert processed is not None
    assert len(processed) == settings.DATA_PER_DAY
    assert any("Padding" in entry for entry in msg)
    assert (processed["Volume"] == 1).any()


def test_load_local_data_reads_files(tmp_path, make_intraday_df, monkeypatch):
    df = make_intraday_df("AAPL", "2024-06-03", tz=None).drop(columns=["stock_code"])
    file_path = tmp_path / "AAPL.csv"
    df.to_csv(file_path)

    monkeypatch.setattr(settings, "TRAINING_DATA_START_DATE", "2024-06-03")
    monkeypatch.setattr(settings, "TESTING_DATA_END_DATE", "2024-06-03")

    result = load_local_data(["AAPL"], base_dir=str(tmp_path), interval_mins=30)

    assert not result.empty
    assert "stock_code" in result.columns
    assert result["stock_code"].unique().tolist() == ["AAPL"]


def test_merge_fetched_data_overwrites_duplicates():
    ts = pd.Timestamp("2024-06-03 09:30", tz="America/New_York")
    index = pd.DatetimeIndex([ts], name="datetime")

    existing = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Volume": [100],
            "stock_code": ["AAPL"],
        },
        index=index,
    )
    new = pd.DataFrame(
        {
            "Open": [1.2],
            "High": [2.2],
            "Low": [0.7],
            "Close": [1.8],
            "Volume": [120],
            "stock_code": ["AAPL"],
        },
        index=index,
    )

    merged = merge_fetched_data(existing, new)

    assert len(merged) == 1
    assert merged.loc[ts, "Close"] == pytest.approx(1.8)


def test_sample_training_data_respects_proportions():
    stock_a = "AAA"
    stock_b = "BBB"
    stock_a_id = stock_code_to_id(stock_a)
    stock_b_id = stock_code_to_id(stock_b)

    X_train = pd.DataFrame(
        {
            "stock_id": [stock_a_id] * 5 + [stock_b_id] * 5,
            "feature": list(range(10)),
        }
    )
    y_train = pd.Series(range(10))

    sampled_X, sampled_y = sample_training_data(
        X_train,
        y_train,
        sampling_proportion={stock_a: 0.4, stock_b: 0.0},
        seed=1,
    )

    assert not sampled_X.empty
    assert sampled_X["stock_id"].unique().tolist() == [stock_a_id]
    assert len(sampled_X) == math.floor(5 * 0.4)
    assert len(sampled_X) == len(sampled_y)
