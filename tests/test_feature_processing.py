from ml4investment.config.global_settings import settings
from ml4investment.utils.feature_processing import (
    process_features_for_backtest,
    process_features_for_predict,
    process_features_for_train_and_validate,
)


def _patch_dates(monkeypatch):
    monkeypatch.setattr(settings, "TRAINING_DATA_START_DATE", "2024-06-03")
    monkeypatch.setattr(settings, "TRAINING_DATA_END_DATE", "2024-06-05")
    monkeypatch.setattr(settings, "VALIDATION_DATA_START_DATE", "2024-06-06")
    monkeypatch.setattr(settings, "VALIDATION_DATA_END_DATE", "2024-06-07")
    monkeypatch.setattr(settings, "TESTING_DATA_START_DATE", "2024-06-03")
    monkeypatch.setattr(settings, "TESTING_DATA_END_DATE", "2024-06-07")


def test_process_features_for_train_and_validate(make_daily_features_df, monkeypatch):
    _patch_dates(monkeypatch)
    df = make_daily_features_df(["AAPL", "MSFT"], periods=5)

    X_train, y_train, X_validate, y_validate, config = process_features_for_train_and_validate(
        df,
        apply_clip="global-level",
        apply_scale="global-level",
        seed=1,
    )

    assert "stock_code" not in X_train.columns
    assert "stock_code" not in X_validate.columns
    assert len(X_train) > 0
    assert len(X_validate) > 0
    assert len(y_train) == len(X_train)
    assert len(y_validate) == len(X_validate)
    assert config["apply_clip"] == "global-level"
    assert "bounds" in config
    assert "scaler" in config


def test_process_features_for_backtest(make_daily_features_df, monkeypatch):
    _patch_dates(monkeypatch)
    df = make_daily_features_df(["AAPL", "MSFT"], periods=5)

    _, _, _, _, config = process_features_for_train_and_validate(
        df,
        apply_clip="global-level",
        apply_scale="global-level",
        seed=1,
    )

    X_backtest, y_backtest = process_features_for_backtest(df, config, ["AAPL"])

    assert "stock_code" not in X_backtest.columns
    assert len(X_backtest) == len(y_backtest)
    assert len(X_backtest) > 0


def test_process_features_for_predict_handles_stale(make_daily_features_df, monkeypatch):
    _patch_dates(monkeypatch)
    df = make_daily_features_df(["AAPL", "MSFT"], periods=5)
    df = df.sort_index()

    stale_mask = (df["stock_code"] == "MSFT") & (df.index == df.index.max())
    df = df[~stale_mask]

    _, _, _, _, config = process_features_for_train_and_validate(
        df,
        apply_clip="global-level",
        apply_scale="global-level",
        seed=1,
    )

    X_predict = process_features_for_predict(df, config, ["AAPL", "MSFT"])

    assert "stock_code" not in X_predict.columns
    assert len(X_predict) == 1
