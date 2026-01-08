import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from ml4investment.config.global_settings import settings

pd.set_option("future.no_silent_downcasting", True)

logger = logging.getLogger(__name__)


def _apply_clipping_and_scaling(
    X: pd.DataFrame,
    numerical_cols: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, int]:
    """Apply clipping and scaling based on the provided configuration.

    Args:
        X: Feature DataFrame.
        numerical_cols: Numerical column names to transform.
        config: Configuration dict containing clip/scale settings.

    Returns:
        Tuple of (processed DataFrame, number of rows clipped).
    """
    X_processed = X.copy()
    X_processed[numerical_cols] = X_processed[numerical_cols].astype("float64")
    apply_clip = config.get("apply_clip")
    apply_scale = config.get("apply_scale")

    data_before_clip = X_processed[numerical_cols].copy()
    if apply_clip == "global-level":
        bounds = config.get("bounds", {}).get("global")
        if bounds:
            X_processed[numerical_cols] = X_processed[numerical_cols].clip(
                bounds["lower"], bounds["upper"], axis=1
            )
    elif apply_clip == "stock-level":
        for stock_code, group in X_processed.groupby("stock_code"):
            stock_bounds = config.get("bounds", {}).get(stock_code)
            if stock_bounds:
                group_indices = group.index
                clipped_values = group[numerical_cols].clip(
                    stock_bounds["lower"], stock_bounds["upper"], axis=1
                )
                X_processed.loc[group_indices, numerical_cols] = clipped_values

    data_after_clip = X_processed[numerical_cols]
    clipped_mask = (data_before_clip != data_after_clip).any(axis=1)
    clipped_rows_count = clipped_mask.sum()

    if apply_scale == "global-level":
        scaler = config.get("scaler")
        if scaler:
            X_processed[numerical_cols] = scaler.transform(X_processed[numerical_cols])
    elif apply_scale == "stock-level":
        for stock_code, group in X_processed.groupby("stock_code"):
            scaler = config.get("scaler", {}).get(stock_code)
            if scaler:
                group_indices = group.index
                scaled_values = scaler.transform(group[numerical_cols])

                scaled_df = pd.DataFrame(
                    scaled_values, index=group_indices, columns=numerical_cols
                )

                X_processed.loc[group_indices, numerical_cols] = scaled_df

    assert isinstance(X_processed, pd.DataFrame)

    return X_processed, clipped_rows_count


def _sanitize_features(df: pd.DataFrame, drop_na: bool) -> pd.DataFrame:
    """Replace inf values and optionally drop rows with NaNs.

    Args:
        df: Input DataFrame.
        drop_na: Whether to drop rows containing NaN values.

    Returns:
        Cleaned DataFrame.
    """
    cleaned = df.copy()
    cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    if drop_na:
        cleaned.dropna(inplace=True)
    return cleaned


def _build_categorical_types(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build and apply categorical dtypes for stock and sector identifiers.

    Args:
        df: Cleaned DataFrame with stock_id and sector_id columns.
        config: Config dict to store the categorical dtypes.

    Returns:
        DataFrame with categorical dtypes applied.
    """
    all_stock_ids = sorted(set(df["stock_id"]))
    cat_stock_id_type = pd.CategoricalDtype(categories=all_stock_ids)
    config["cat_stock_id_type"] = cat_stock_id_type

    all_sector_ids = sorted(set(df["sector_id"]))
    cat_sector_id_type = pd.CategoricalDtype(categories=all_sector_ids)
    config["cat_sector_id_type"] = cat_sector_id_type

    df["stock_id"] = df["stock_id"].astype(cat_stock_id_type)
    df["sector_id"] = df["sector_id"].astype(cat_sector_id_type)
    return df


def _apply_categorical_types(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Apply categorical dtypes from the config to the DataFrame.

    Args:
        df: DataFrame with stock_id and sector_id columns.
        config: Config dict containing categorical dtypes.

    Returns:
        DataFrame with categorical dtypes applied.
    """
    df["stock_id"] = df["stock_id"].astype(config["cat_stock_id_type"])
    df["sector_id"] = df["sector_id"].astype(config["cat_sector_id_type"])
    return df


def _resolve_train_validate_windows() -> tuple[
    pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp
]:
    """Resolve training and validation time windows in NY time.

    Returns:
        Tuple of (train_start, train_end, val_start, val_end).
    """
    train_start = pd.to_datetime(settings.TRAINING_DATA_START_DATE).tz_localize("America/New_York")
    train_end = pd.to_datetime(settings.TRAINING_DATA_END_DATE).tz_localize("America/New_York")
    val_start = pd.to_datetime(settings.VALIDATION_DATA_START_DATE).tz_localize("America/New_York")
    val_end = pd.to_datetime(settings.VALIDATION_DATA_END_DATE).tz_localize("America/New_York")
    return train_start, train_end, val_start, val_end


def _split_train_validate(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into training and validation windows.

    Args:
        df: Input DataFrame.
        train_start: Training window start.
        train_end: Training window end.
        val_start: Validation window start.
        val_end: Validation window end.

    Returns:
        Tuple of (train_df, validate_df).
    """
    train_df = df[(df.index >= train_start) & (df.index <= train_end)]
    validate_df = df[(df.index >= val_start) & (df.index <= val_end)]
    return train_df, validate_df


def _split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split features and target from a labeled DataFrame.

    Args:
        df: DataFrame containing Target column.

    Returns:
        Tuple of (features, target).
    """
    feature_cols = [col for col in df.columns if col != "Target"]
    return df[feature_cols], df["Target"]


def _select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select feature columns from a DataFrame.

    Args:
        df: DataFrame possibly containing Target column.

    Returns:
        Feature DataFrame without Target.
    """
    feature_cols = [col for col in df.columns if col != "Target"]
    return df[feature_cols]


def _get_column_groups(X: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Determine boolean, categorical, and numerical column groups.

    Args:
        X: Feature DataFrame.

    Returns:
        Tuple of (boolean_cols, categorical_cols, numerical_cols).
    """
    boolean_cols = X.select_dtypes(include="bool").columns.tolist()
    categorical_cols = settings.CATEGORICAL_FEATURES + ["stock_code"]
    numerical_cols = [col for col in X.columns if col not in boolean_cols + categorical_cols]
    return boolean_cols, categorical_cols, numerical_cols


def _configure_clip_bounds(
    X_train: pd.DataFrame, numerical_cols: list[str], config: dict[str, Any]
) -> None:
    """Configure clipping bounds in the processing config.

    Args:
        X_train: Training feature DataFrame.
        numerical_cols: Numerical column names.
        config: Config dict to update.
    """
    apply_clip = config.get("apply_clip")
    if apply_clip == "global-level":
        lower_bound = X_train[numerical_cols].quantile(settings.CLIP_LOWER_QUANTILE_RATIO)
        upper_bound = X_train[numerical_cols].quantile(settings.CLIP_UPPER_QUANTILE_RATIO)
        config["bounds"] = {"global": {"lower": lower_bound, "upper": upper_bound}}
    elif apply_clip == "stock-level":
        bounds = X_train.groupby("stock_code")[numerical_cols].quantile(
            [settings.CLIP_LOWER_QUANTILE_RATIO, settings.CLIP_UPPER_QUANTILE_RATIO]  # type: ignore
        )
        config["bounds"] = {
            stock: {
                "lower": bounds.loc[(stock, settings.CLIP_LOWER_QUANTILE_RATIO)],  # type: ignore
                "upper": bounds.loc[(stock, settings.CLIP_UPPER_QUANTILE_RATIO)],  # type: ignore
            }
            for stock, _ in X_train.groupby("stock_code")
        }


def _configure_scalers(
    X_train: pd.DataFrame, numerical_cols: list[str], config: dict[str, Any]
) -> None:
    """Configure scalers in the processing config.

    Args:
        X_train: Training feature DataFrame.
        numerical_cols: Numerical column names.
        config: Config dict to update.
    """
    apply_scale = config.get("apply_scale")
    if apply_scale == "global-level":
        scaler = RobustScaler().fit(X_train[numerical_cols])
        config["scaler"] = scaler
    elif apply_scale == "stock-level":
        scalers = {
            stock_code: RobustScaler().fit(group[numerical_cols])
            for stock_code, group in X_train.groupby("stock_code")
        }
        config["scaler"] = scalers


def _shuffle_training_data(
    X_train: pd.DataFrame, y_train: pd.Series, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Shuffle training data while keeping features and target aligned.

    Args:
        X_train: Training features.
        y_train: Training target.
        seed: Random seed.

    Returns:
        Tuple of (X_train_shuffled, y_train_shuffled).
    """
    combined_train_df = pd.concat([X_train, y_train], axis=1)
    combined_train_df.sort_index(inplace=True)
    shuffled_df = combined_train_df.sample(frac=1, random_state=seed)
    y_train = shuffled_df["Target"]
    X_train = shuffled_df.drop(columns=["Target"])
    return X_train, y_train


def _drop_stock_code(X: pd.DataFrame) -> pd.DataFrame:
    """Drop the stock_code column from a feature DataFrame.

    Args:
        X: Feature DataFrame.

    Returns:
        DataFrame without the stock_code column.
    """
    X.pop("stock_code")
    return X


def _log_processing_summary(
    original_data_points: int, final_data_points: int, clipped_rows_count: int
) -> None:
    """Log a standard feature processing summary.

    Args:
        original_data_points: Number of rows before processing.
        final_data_points: Number of rows after processing.
        clipped_rows_count: Number of rows affected by clipping.
    """
    points_dropped = original_data_points - final_data_points

    percentage_dropped = (points_dropped / original_data_points) * 100
    percentage_clipped = (clipped_rows_count / original_data_points) * 100

    logger.info("Feature processing complete")
    logger.info(f"  Original data points: {original_data_points}")
    logger.info(f"  Final data points: {final_data_points}")
    logger.info(f"  Dropped {points_dropped} points: ({percentage_dropped:.2f}%)")
    logger.info(f"  Clipped {clipped_rows_count} points: ({percentage_clipped:.2f}%)")


def _resolve_backtest_window(daily_features_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve backtesting window, respecting the index timezone.

    Args:
        daily_features_df: Daily feature DataFrame used for timezone alignment.

    Returns:
        Tuple of (test_start, test_end).
    """
    test_start = pd.to_datetime(settings.TESTING_DATA_START_DATE).tz_localize("America/New_York")
    test_end = pd.to_datetime(
        settings.TESTING_DATA_END_DATE or pd.Timestamp.now(tz="America/New_York")
    )
    index_tz = daily_features_df.index.tz  # type: ignore
    if index_tz is not None:
        test_start = test_start.tz_convert(index_tz)
        if test_end.tzinfo is None:
            test_end = test_end.tz_localize(index_tz)
        else:
            test_end = test_end.tz_convert(index_tz)
    else:
        if test_start.tzinfo is not None:
            test_start = test_start.tz_convert(None)
        if test_end.tzinfo is not None:
            test_end = test_end.tz_convert(None)
    return test_start, test_end


def _select_latest_snapshot(daily_features_df: pd.DataFrame) -> pd.DataFrame:
    """Select the latest available snapshot per stock.

    Args:
        daily_features_df: Daily feature DataFrame.

    Returns:
        DataFrame containing the latest row per stock.
    """
    latest_df = daily_features_df.groupby("stock_code").tail(1).copy()

    max_ts = latest_df.index.max()
    lagging = latest_df[latest_df.index != max_ts]
    if not lagging.empty:
        logger.warning(
            "Dropping %d stocks with stale data (latest snapshot: %s). Examples: %s",
            len(lagging),
            max_ts,
            ", ".join(
                f"{code} @ {idx}" for idx, code in zip(lagging.index, lagging["stock_code"])
            ),
        )
        latest_df = latest_df[latest_df.index == max_ts]

    assert latest_df.index.min() == latest_df.index.max()
    return latest_df


def process_features_for_train_and_validate(
    daily_features_df: pd.DataFrame,
    apply_clip: str = "skip",
    apply_scale: str = "skip",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict]:
    """Process features for training and validation.

    Args:
        daily_features_df: Daily feature DataFrame.
        apply_clip: Clipping strategy.
        apply_scale: Scaling strategy.
        seed: Random seed for shuffling.

    Returns:
        Tuple of (X_train, y_train, X_validate, y_validate, config).
    """
    logger.info("Begin feature processing for training and validation...")
    original_data_points = len(daily_features_df)

    process_feature_config: dict[str, Any] = {
        "apply_clip": apply_clip,
        "apply_scale": apply_scale,
    }

    df = _sanitize_features(daily_features_df, drop_na=True)
    df = _build_categorical_types(df, process_feature_config)

    train_start, train_end, val_start, val_end = _resolve_train_validate_windows()

    logger.info(f"Using data from {train_start} to {train_end} for training.")
    logger.info(f"Using data from {val_start} to {val_end} for validation.")

    train_df, validate_df = _split_train_validate(df, train_start, train_end, val_start, val_end)

    X_train, y_train = _split_features_target(train_df)
    X_validate, y_validate = _split_features_target(validate_df)

    _, _, numerical_cols = _get_column_groups(X_train)

    logger.info(f"Clipping method: {apply_clip}")
    logger.info(f"Scaling method: {apply_scale}")

    _configure_clip_bounds(X_train, numerical_cols, process_feature_config)
    _configure_scalers(X_train, numerical_cols, process_feature_config)

    X_train, train_clipped_rows_count = _apply_clipping_and_scaling(
        X_train, numerical_cols, process_feature_config
    )
    X_validate, valid_clipped_rows_count = _apply_clipping_and_scaling(
        X_validate, numerical_cols, process_feature_config
    )
    clipped_rows_count = train_clipped_rows_count + valid_clipped_rows_count

    X_train = _drop_stock_code(X_train)
    X_validate = _drop_stock_code(X_validate)

    X_train, y_train = _shuffle_training_data(X_train, y_train, seed)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_validate, pd.DataFrame)
    assert isinstance(y_train, pd.Series)

    assert X_train.isnull().sum().sum() == 0, "NaN values found in the final X_train."
    assert np.isinf(X_train.select_dtypes(include=np.number)).sum().sum() == 0, (
        "Infinity values found in the final X_train."
    )

    assert X_validate.isnull().sum().sum() == 0, "NaN values found in the final X_validate."
    assert np.isinf(X_validate.select_dtypes(include=np.number)).sum().sum() == 0, (
        "Infinity values found in the final X_validate."
    )

    final_data_points = len(X_train) + len(X_validate)
    _log_processing_summary(original_data_points, final_data_points, clipped_rows_count)

    return (
        X_train,
        y_train,
        X_validate,
        y_validate,
        process_feature_config,
    )


def process_features_for_backtest(
    daily_features_df: pd.DataFrame, config_data: dict, predict_stock_list: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Process features for backtesting.

    Args:
        daily_features_df: Daily feature DataFrame.
        config_data: Processing config from training.
        predict_stock_list: Stock codes to include.

    Returns:
        Tuple of (X_backtest, y_backtest).
    """
    logger.info("Begin feature processing for backtesting...")

    test_start, test_end = _resolve_backtest_window(daily_features_df)
    logger.info(f"Using data from {test_start} to {test_end} for backtesting.")

    mask = (daily_features_df.index >= test_start) & (daily_features_df.index <= test_end)
    mask &= daily_features_df["stock_code"].isin(predict_stock_list)

    backtest_df = daily_features_df[mask].copy()
    original_data_points = len(backtest_df)

    backtest_df = _apply_categorical_types(backtest_df, config_data)
    backtest_df = _sanitize_features(backtest_df, drop_na=True)

    X_backtest, y_backtest = _split_features_target(backtest_df)

    _, _, numerical_cols = _get_column_groups(X_backtest)

    X_backtest, clipped_rows_count = _apply_clipping_and_scaling(
        X_backtest, numerical_cols, config_data
    )

    X_backtest = _drop_stock_code(X_backtest)

    assert X_backtest.isnull().sum().sum() == 0, "NaN values found in the final X_backtest."
    assert np.isinf(X_backtest.select_dtypes(include=np.number)).sum().sum() == 0, (
        "Infinity values found in the final X_backtest."
    )

    final_data_points = len(X_backtest)
    _log_processing_summary(original_data_points, final_data_points, clipped_rows_count)

    return X_backtest, y_backtest


def process_features_for_predict(
    daily_features_df: pd.DataFrame, config_data: dict, predict_stock_list: list[str]
) -> pd.DataFrame:
    """Process features for prediction.

    Args:
        daily_features_df: Daily feature DataFrame.
        config_data: Processing config from training.
        predict_stock_list: Stock codes to include.

    Returns:
        Feature DataFrame ready for prediction.
    """
    logger.info("Begin feature processing for backtesting...")

    latest_df = _select_latest_snapshot(daily_features_df)

    predict_df = latest_df.loc[latest_df["stock_code"].isin(predict_stock_list)].copy()
    original_data_points = len(predict_df)

    predict_df = _apply_categorical_types(predict_df, config_data)
    predict_df = _sanitize_features(predict_df, drop_na=False)

    X_predict = _select_feature_columns(predict_df)
    X_predict = X_predict.dropna()

    _, _, numerical_cols = _get_column_groups(X_predict)

    X_predict, clipped_rows_count = _apply_clipping_and_scaling(
        X_predict, numerical_cols, config_data
    )

    X_predict = _drop_stock_code(X_predict)

    assert X_predict.isnull().sum().sum() == 0, "NaN values found in the final X_predict."
    assert np.isinf(X_predict.select_dtypes(include=np.number)).sum().sum() == 0, (
        "Infinity values found in the final X_predict."
    )

    final_data_points = len(X_predict)
    _log_processing_summary(original_data_points, final_data_points, clipped_rows_count)

    return X_predict
