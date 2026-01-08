import numpy as np
import pandas as pd
import pytest

from ml4investment.config.global_settings import settings
from ml4investment.utils.model_predicting import get_prev_actual_ranking, get_stocks_portfolio


def test_get_prev_actual_ranking_filters_by_time():
    dates = pd.date_range("2024-06-03", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "stock_code": ["AAA", "AAA", "BBB"],
            "y_actual": [0.1, 0.2, -0.3],
        },
        index=[dates[0], dates[1], dates[1]],
    )

    result = get_prev_actual_ranking(
        stock_codes=["AAA", "BBB"],
        historical_df=df,
        current_ts=dates[2],
        actual_col="y_actual",
    )

    assert result == {"AAA": 0.2, "BBB": -0.3}


def test_get_prev_actual_ranking_requires_columns():
    df = pd.DataFrame({"y_actual": [0.1]})
    with pytest.raises(ValueError):
        get_prev_actual_ranking(["AAA"], df)


def test_get_stocks_portfolio_buy_long(monkeypatch):
    monkeypatch.setattr(settings, "STOCK_SELECTION_STRATEGY", "BUY_LONG")
    monkeypatch.setattr(settings, "NUMBER_OF_STOCKS_TO_BUY", 2)

    candidates = pd.DataFrame(
        {
            "stock_code": ["AAA", "BBB", "CCC"],
            "prediction": [0.3, 0.1, -0.2],
        }
    )

    selected = get_stocks_portfolio(candidates)

    assert set(selected["stock_code"]) == {"AAA", "BBB"}
    assert (selected["action"] == "BUY_LONG").all()
    assert np.isclose(selected["weight"].sum(), 1.0)


def test_get_stocks_portfolio_adapt(monkeypatch):
    monkeypatch.setattr(settings, "STOCK_SELECTION_STRATEGY", "ADAPT")
    monkeypatch.setattr(settings, "NUMBER_OF_STOCKS_TO_BUY", 1)

    candidates = pd.DataFrame(
        {
            "stock_code": ["AAA", "BBB"],
            "prediction": [-0.3, -0.1],
        }
    )

    selected = get_stocks_portfolio(candidates)

    assert len(selected) == 1
    assert selected.iloc[0]["action"] == "SELL_SHORT"


def test_get_stocks_portfolio_combines_momentum(monkeypatch):
    monkeypatch.setattr(settings, "STOCK_SELECTION_STRATEGY", "BUY_LONG")
    monkeypatch.setattr(settings, "NUMBER_OF_STOCKS_TO_BUY", 1)
    monkeypatch.setattr(settings, "STOCK_SELECTION_MOMENTUM", 0.5)

    candidates = pd.DataFrame(
        {
            "stock_code": ["AAA", "BBB"],
            "prediction": [0.3, 0.1],
        }
    )

    prev_actuals = {"AAA": 0.05, "BBB": 0.4}
    selected = get_stocks_portfolio(candidates, prev_actuals=prev_actuals)

    assert set(selected["stock_code"]) == {"AAA", "BBB"}
    assert np.isclose(selected["weight"].sum(), 1.0)


def test_get_stocks_portfolio_both_strategy(monkeypatch):
    monkeypatch.setattr(settings, "STOCK_SELECTION_STRATEGY", "BOTH")
    monkeypatch.setattr(settings, "NUMBER_OF_STOCKS_TO_BUY", 2)

    candidates = pd.DataFrame(
        {
            "stock_code": ["AAA", "BBB", "CCC"],
            "prediction": [0.2, -0.5, 0.1],
        }
    )

    selected = get_stocks_portfolio(candidates)

    assert len(selected) == 2
    assert set(selected["action"]) == {"BUY_LONG", "SELL_SHORT"}
    assert np.isclose(selected["weight"].sum(), 1.0)


def test_get_stocks_portfolio_buy_long_first(monkeypatch):
    monkeypatch.setattr(settings, "STOCK_SELECTION_STRATEGY", "BUY_LONG_FIRST")
    monkeypatch.setattr(settings, "NUMBER_OF_STOCKS_TO_BUY", 2)

    candidates = pd.DataFrame(
        {
            "stock_code": ["AAA", "BBB", "CCC"],
            "prediction": [0.3, 0.2, -0.4],
        }
    )

    selected = get_stocks_portfolio(candidates)

    assert len(selected) == 2
    assert "BUY_LONG" in set(selected["action"])
    assert np.isclose(selected["weight"].sum(), 1.0)


def test_get_prev_actual_ranking_returns_empty_on_empty_stock_list():
    df = pd.DataFrame({"stock_code": ["AAA"], "y_actual": [0.1]})
    result = get_prev_actual_ranking(stock_codes=[], historical_df=df)
    assert result == {}
