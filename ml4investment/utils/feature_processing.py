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
    """Apply clipping and scaling based on the provided configuration"""
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


def process_features_for_train_and_validate(
    daily_features_df: pd.DataFrame,
    apply_clip: str = "skip",
    apply_scale: str = "skip",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict]:
    """Process Data, including washing, removing Nan, scaling and spliting for training"""
    logger.info("Begin feature processing for training and validation...")
    original_data_points = len(daily_features_df)

    process_feature_config: dict[str, Any] = {
        "apply_clip": apply_clip,
        "apply_scale": apply_scale,
    }

    """ Global preprocessing """
    df = daily_features_df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    all_stock_ids = sorted(set(df["stock_id"]))
    cat_stock_id_type = pd.CategoricalDtype(categories=all_stock_ids)
    process_feature_config["cat_stock_id_type"] = cat_stock_id_type

    all_sector_ids = sorted(set(df["sector_id"]))
    cat_sector_id_type = pd.CategoricalDtype(categories=all_sector_ids)
    process_feature_config["cat_sector_id_type"] = cat_sector_id_type

    df["stock_id"] = df["stock_id"].astype(cat_stock_id_type)
    df["sector_id"] = df["sector_id"].astype(cat_sector_id_type)

    """ Split Train and Validation """
    train_start = pd.to_datetime(settings.TRAINING_DATA_START_DATE).tz_localize("America/New_York")
    train_end = pd.to_datetime(settings.TRAINING_DATA_END_DATE).tz_localize("America/New_York")
    val_start = pd.to_datetime(settings.VALIDATION_DATA_START_DATE).tz_localize("America/New_York")
    val_end = pd.to_datetime(settings.VALIDATION_DATA_END_DATE).tz_localize("America/New_York")

    logger.info(f"Using data from {train_start} to {train_end} for training.")
    logger.info(f"Using data from {val_start} to {val_end} for validation.")

    train_df = df[(df.index >= train_start) & (df.index <= train_end)]
    validate_df = df[(df.index >= val_start) & (df.index <= val_end)]

    feature_cols = [col for col in df.columns if col != "Target"]
    target_col = "Target"

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_validate, y_validate = validate_df[feature_cols], validate_df[target_col]

    """ Post-processing features """
    boolean_cols = X_train.select_dtypes(include="bool").columns.tolist()
    categorical_cols = settings.CATEGORICAL_FEATURES + ["stock_code"]
    numerical_cols = [col for col in X_train.columns if col not in boolean_cols + categorical_cols]

    logger.info(f"Clipping method: {apply_clip}")
    logger.info(f"Scaling method: {apply_scale}")

    if apply_clip == "global-level":
        lower_bound = X_train[numerical_cols].quantile(settings.CLIP_LOWER_QUANTILE_RATIO)
        upper_bound = X_train[numerical_cols].quantile(settings.CLIP_UPPER_QUANTILE_RATIO)
        process_feature_config["bounds"] = {"global": {"lower": lower_bound, "upper": upper_bound}}
    elif apply_clip == "stock-level":
        bounds = X_train.groupby("stock_code")[numerical_cols].quantile(
            [settings.CLIP_LOWER_QUANTILE_RATIO, settings.CLIP_UPPER_QUANTILE_RATIO]  # type: ignore
        )
        process_feature_config["bounds"] = {
            stock: {
                "lower": bounds.loc[(stock, settings.CLIP_LOWER_QUANTILE_RATIO)],  # type: ignore
                "upper": bounds.loc[(stock, settings.CLIP_UPPER_QUANTILE_RATIO)],  # type: ignore
            }
            for stock, _ in X_train.groupby("stock_code")
        }

    if apply_scale == "global-level":
        scaler = RobustScaler().fit(X_train[numerical_cols])
        process_feature_config["scaler"] = scaler
    elif apply_scale == "stock-level":
        scalers = {
            stock_code: RobustScaler().fit(group[numerical_cols])
            for stock_code, group in X_train.groupby("stock_code")
        }
        process_feature_config["scaler"] = scalers

    X_train, train_clipped_rows_count = _apply_clipping_and_scaling(
        X_train, numerical_cols, process_feature_config
    )
    X_validate, valid_clipped_rows_count = _apply_clipping_and_scaling(
        X_validate, numerical_cols, process_feature_config
    )
    clipped_rows_count = train_clipped_rows_count + valid_clipped_rows_count

    X_train.pop("stock_code")
    X_validate.pop("stock_code")

    combined_train_df = pd.concat([X_train, y_train], axis=1)
    combined_train_df.sort_index(inplace=True)
    shuffled_df = combined_train_df.sample(frac=1, random_state=seed)
    y_train = shuffled_df["Target"]
    X_train = shuffled_df.drop(columns=["Target"])
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
    points_dropped = original_data_points - final_data_points

    percentage_dropped = (points_dropped / original_data_points) * 100
    percentage_clipped = (clipped_rows_count / original_data_points) * 100

    logger.info("Feature processing complete")
    logger.info(f"  Original data points: {original_data_points}")
    logger.info(f"  Final data points: {final_data_points}")
    logger.info(f"  Dropped {points_dropped} points: ({percentage_dropped:.2f}%)")
    logger.info(f"  Clipped {clipped_rows_count} points: ({percentage_clipped:.2f}%)")

    return (
        X_train,
        y_train,
        X_validate,
        y_validate,
        process_feature_config,
    )


def process_features_for_backtest(
    daily_features_df: pd.DataFrame, config_data: dict, predict_stock_list: list
) -> tuple[pd.DataFrame, pd.Series]:
    """Process Data, including washing, removing Nan, scaling and spliting for backtest"""
    logger.info("Begin feature processing for backtesting...")

    """ Preprocessing Features"""
    test_start = pd.to_datetime(settings.TESTING_DATA_START_DATE).tz_localize("America/New_York")
    test_end = pd.to_datetime(
        settings.TESTING_DATA_END_DATE or pd.Timestamp.now(tz="America/New_York")
    )
    index_tz = daily_features_df.index.tz   # type: ignore
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
    logger.info(f"Using data from {test_start} to {test_end} for backtesting.")

    mask = (daily_features_df.index >= test_start) & (daily_features_df.index <= test_end)
    mask &= daily_features_df["stock_code"].isin(predict_stock_list)

    backtest_df = daily_features_df[mask].copy()
    original_data_points = len(backtest_df)

    backtest_df["stock_id"] = backtest_df["stock_id"].astype(config_data["cat_stock_id_type"])
    backtest_df["sector_id"] = backtest_df["sector_id"].astype(config_data["cat_sector_id_type"])

    backtest_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    backtest_df.dropna(inplace=True)

    feature_cols = [col for col in backtest_df.columns if col != "Target"]
    X_backtest = backtest_df[feature_cols]
    y_backtest = backtest_df["Target"]

    """ Post-processing features """
    boolean_cols = X_backtest.select_dtypes(include="bool").columns.tolist()
    categorical_cols = settings.CATEGORICAL_FEATURES + ["stock_code"]
    numerical_cols = [
        col for col in X_backtest.columns if col not in boolean_cols + categorical_cols
    ]

    X_backtest, clipped_rows_count = _apply_clipping_and_scaling(
        X_backtest, numerical_cols, config_data
    )

    X_backtest.pop("stock_code")

    assert X_backtest.isnull().sum().sum() == 0, "NaN values found in the final X_backtest."
    assert np.isinf(X_backtest.select_dtypes(include=np.number)).sum().sum() == 0, (
        "Infinity values found in the final X_backtest."
    )

    final_data_points = len(X_backtest)
    points_dropped = original_data_points - final_data_points

    percentage_dropped = (points_dropped / original_data_points) * 100
    percentage_clipped = (clipped_rows_count / original_data_points) * 100

    logger.info("Feature processing complete")
    logger.info(f"  Original data points: {original_data_points}")
    logger.info(f"  Final data points: {final_data_points}")
    logger.info(f"  Dropped {points_dropped} points: ({percentage_dropped:.2f}%)")
    logger.info(f"  Clipped {clipped_rows_count} points: ({percentage_clipped:.2f}%)")

    return X_backtest, y_backtest


def process_features_for_predict(
    daily_features_df: pd.DataFrame, config_data: dict, predict_stock_list: list[str]
) -> pd.DataFrame:
    """Process Data, including washing, removing Nan, scaling and spliting for prediction"""
    logger.info("Begin feature processing for backtesting...")

    """ Preprocessing Features"""
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
    predict_df = latest_df.loc[
        latest_df["stock_code"].isin(predict_stock_list)
    ].copy()
    original_data_points = len(predict_df)

    predict_df["stock_id"] = predict_df["stock_id"].astype(config_data["cat_stock_id_type"])
    predict_df["sector_id"] = predict_df["sector_id"].astype(config_data["cat_sector_id_type"])

    predict_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    feature_cols = [col for col in predict_df.columns if col != "Target"]
    X_predict = predict_df[feature_cols]

    X_predict = X_predict.dropna()

    """ Post-processing features """
    boolean_cols = X_predict.select_dtypes(include="bool").columns.tolist()
    categorical_cols = settings.CATEGORICAL_FEATURES + ["stock_code"]
    numerical_cols = [
        col for col in X_predict.columns if col not in boolean_cols + categorical_cols
    ]

    X_predict, clipped_rows_count = _apply_clipping_and_scaling(
        X_predict, numerical_cols, config_data
    )

    X_predict.pop("stock_code")

    assert X_predict.isnull().sum().sum() == 0, "NaN values found in the final X_predict."
    assert np.isinf(X_predict.select_dtypes(include=np.number)).sum().sum() == 0, (
        "Infinity values found in the final X_predict."
    )

    final_data_points = len(X_predict)
    points_dropped = original_data_points - final_data_points

    percentage_dropped = (points_dropped / original_data_points) * 100
    percentage_clipped = (clipped_rows_count / original_data_points) * 100

    logger.info("Feature processing complete")
    logger.info(f"  Original data points: {original_data_points}")
    logger.info(f"  Final data points: {final_data_points}")
    logger.info(f"  Dropped {points_dropped} points: ({percentage_dropped:.2f}%)")
    logger.info(f"  Clipped {clipped_rows_count} points: ({percentage_clipped:.2f}%)")

    return X_predict
