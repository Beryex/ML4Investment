import argparse
import json
import logging
import os
import pickle
from multiprocessing import cpu_count
from typing import Any

import pandas as pd
import wandb
from wandb.sdk.wandb_run import Run

from ml4investment.config.global_settings import settings
from ml4investment.utils.data_loader import sample_training_data
from ml4investment.utils.feature_calculating import calculate_features
from ml4investment.utils.feature_processing import process_features_for_train_and_validate
from ml4investment.utils.logging import configure_logging, setup_wandb
from ml4investment.utils.model_training import (
    model_training,
    optimize_data_sampling_proportion,
    optimize_features,
    optimize_model_hyperparameters,
    validate_model,
)
from ml4investment.utils.utils import set_random_seed

logger = logging.getLogger(__name__)


def _load_json_file(path: str) -> dict[str, Any]:
    """Load a JSON file from disk.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON as a dictionary.
    """
    with open(path, "r") as file_handle:
        return json.load(file_handle)


def _save_json_file(path: str, payload: dict[str, Any]) -> None:
    """Save a dictionary payload to JSON.

    Args:
        path: JSON file path.
        payload: Dictionary to write.
    """
    with open(path, "w") as file_handle:
        json.dump(payload, file_handle, indent=4)


def _build_default_model_hyperparams(seed: int) -> dict[str, Any]:
    """Build default model hyperparameters with runtime settings.

    Args:
        seed: Random seed.

    Returns:
        Model hyperparameter dictionary.
    """
    model_hyperparams = settings.FIXED_TRAINING_CONFIG.copy()
    model_hyperparams.update({"seed": seed})
    model_hyperparams.update({"num_threads": min(max(1, cpu_count()), settings.MAX_NUM_PROCESSES)})
    return model_hyperparams


def _load_training_data(
    fetched_data_df: pd.DataFrame, train_stock_list: list[str]
) -> pd.DataFrame:
    """Filter fetched data for the training stock list and date window.

    Args:
        fetched_data_df: Raw fetched intraday data.
        train_stock_list: Stock codes used for training.

    Returns:
        Filtered intraday DataFrame.
    """
    train_data_start_date = settings.TRAINING_DATA_START_DATE
    validation_data_end_date = settings.VALIDATION_DATA_END_DATE
    logger.info(
        "Load input fetched data, starting from %s to %s",
        train_data_start_date,
        validation_data_end_date,
    )
    train_data_df = fetched_data_df[fetched_data_df["stock_code"].isin(train_stock_list)]
    return train_data_df.loc[train_data_start_date:validation_data_end_date]


def _prepare_feature_data(
    train_data_df: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict[str, Any]]:
    """Calculate and process features for training/validation.

    Args:
        train_data_df: Filtered intraday data for training.
        seed: Random seed.

    Returns:
        Tuple of (X_train, y_train, X_validate, y_validate, config).
    """
    daily_features_df = calculate_features(train_data_df)
    return process_features_for_train_and_validate(
        daily_features_df,
        apply_clip=settings.APPLY_CLIP,
        apply_scale=settings.APPLY_SCALE,
        seed=seed,
    )


def _apply_feature_selection(
    X_train: pd.DataFrame,
    X_validate: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load or initialize feature selection for model training.

    Args:
        X_train: Training feature DataFrame.
        X_validate: Validation feature DataFrame.
        args: CLI arguments.

    Returns:
        Tuple of (X_train, X_validate, selected_features).
    """
    if args.optimize_features:
        logger.info("Optimize model features from the scratch, using all features initially")
        optimal_features = X_train.columns.tolist()
        return X_train, X_validate, optimal_features

    try:
        logger.info("Attempting to load input model features from %s", args.features_pth)
        optimal_features = _load_json_file(args.features_pth)["features"]
        logger.info("Successfully loaded features.")
        return X_train[optimal_features], X_validate[optimal_features], optimal_features
    except Exception as exc:
        logger.warning(
            "Failed to load features from %s. Error: %s. Falling back to using all features.",
            args.features_pth,
            exc,
        )
        optimal_features = X_train.columns.tolist()
        return X_train, X_validate, optimal_features


def _resolve_model_hyperparams(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    """Load or initialize model hyperparameters.

    Args:
        args: CLI arguments.
        seed: Random seed.

    Returns:
        Model hyperparameter dictionary.
    """
    if args.optimize_model_hyperparameters:
        logger.info("Optimize model hyperparameters from the scratch")
        return _build_default_model_hyperparams(seed)

    try:
        logger.info(
            "Attempting to load input model hyperparameters from %s",
            args.model_hyperparams_pth,
        )
        optimal_model_hyperparams = _load_json_file(args.model_hyperparams_pth)
        logger.info("Successfully loaded model hyperparameters.")
        return optimal_model_hyperparams
    except Exception as exc:
        logger.warning(
            "Failed to load model hyperparameters from %s. Error: %s. "
            "Falling back to default hyperparameters.",
            args.model_hyperparams_pth,
            exc,
        )
        return _build_default_model_hyperparams(seed)


def _resolve_sampling_proportion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    train_stock_list: list[str],
    model_hyperparams: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, float]:
    """Load or optimize data sampling proportions.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        X_validate: Validation feature DataFrame.
        y_validate: Validation target Series.
        train_stock_list: Stock codes used for training.
        model_hyperparams: Model hyperparameters.
        args: CLI arguments.
        seed: Random seed.

    Returns:
        Sampling proportion mapping by stock code.
    """
    if args.optimize_data_sampling_proportion:
        logger.info("Optimize data sampling proportion from the scratch")
        return optimize_data_sampling_proportion(
            X_train,
            y_train,
            X_validate,
            y_validate,
            train_stock_list=train_stock_list,
            categorical_features=settings.CATEGORICAL_FEATURES,
            model_hyperparams=model_hyperparams,
            given_data_sampling_proportion_pth=args.data_sampling_proportion_pth,
            seed=seed,
            verbose=args.verbose,
        )

    try:
        logger.info(
            "Attempting to load input data sampling proportion from %s",
            args.data_sampling_proportion_pth,
        )
        sampling_proportion = _load_json_file(args.data_sampling_proportion_pth)
        logger.info("Successfully loaded data sampling proportion.")
        return sampling_proportion
    except Exception as exc:
        logger.warning(
            "Failed to load data sampling proportion from %s. Error: %s. "
            "Falling back to use all data.",
            args.data_sampling_proportion_pth,
            exc,
        )
        return {stock: 1.0 for stock in train_stock_list if stock not in settings.SELECTIVE_ETF}


def _maybe_optimize_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    model_hyperparams: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Optionally optimize the feature subset using Optuna.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        X_validate: Validation feature DataFrame.
        y_validate: Validation target Series.
        model_hyperparams: Model hyperparameters.
        args: CLI arguments.
        seed: Random seed.

    Returns:
        Tuple of (X_train, X_validate, selected_features).
    """
    if not args.optimize_features:
        return X_train, X_validate, X_train.columns.tolist()

    optimal_features = optimize_features(
        X_train,
        y_train,
        X_validate,
        y_validate,
        all_features=X_train.columns.tolist(),
        categorical_features=settings.CATEGORICAL_FEATURES,
        given_features_pth=args.features_pth,
        model_hyperparams=model_hyperparams,
        seed=seed,
        verbose=args.verbose,
    )
    return X_train[optimal_features], X_validate[optimal_features], optimal_features


def _maybe_optimize_model_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validate: pd.DataFrame,
    y_validate: pd.Series,
    model_hyperparams: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    """Optionally optimize model hyperparameters.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        X_validate: Validation feature DataFrame.
        y_validate: Validation target Series.
        model_hyperparams: Existing model hyperparameters.
        args: CLI arguments.
        seed: Random seed.

    Returns:
        Updated model hyperparameters.
    """
    if not args.optimize_model_hyperparameters:
        return model_hyperparams

    return optimize_model_hyperparameters(
        X_train,
        y_train,
        X_validate,
        y_validate,
        categorical_features=settings.CATEGORICAL_FEATURES,
        given_model_hyperparams_pth=args.model_hyperparams_pth,
        seed=seed,
        verbose=args.verbose,
    )


def _log_dataset_summary(X_train: pd.DataFrame, X_validate: pd.DataFrame) -> None:
    """Log dataset summaries for training and validation.

    Args:
        X_train: Training feature DataFrame.
        X_validate: Validation feature DataFrame.
    """
    logger.info("Oldest date in training data: %s", X_train.index.min())
    logger.info("Newest date in training data: %s", X_train.index.max())
    logger.info("Total processed samples in training data: %s", X_train.shape[0])
    logger.info("Number of features in training data: %s", X_train.shape[1])

    logger.info("Oldest date in validation data: %s", X_validate.index.min())
    logger.info("Newest date in validation data: %s", X_validate.index.max())
    logger.info("Total processed samples in validation data: %s", X_validate.shape[0])
    logger.info("Number of features in validation data: %s", X_validate.shape[1])


def _select_predict_stocks(
    sorted_stocks: list[str], target_stock_list: list[str], args: argparse.Namespace
) -> list[str]:
    """Select prediction stock list based on validation ranking.

    Args:
        sorted_stocks: Stock list ranked by validation metrics.
        target_stock_list: Default target stocks.
        args: CLI arguments.

    Returns:
        List of selected stock codes.
    """
    if args.optimize_predict_stocks:
        logger.info("Begin predict stocks optimization")
        logger.info(
            "Using %s as the predict stocks optimization metric with max drawdown threshold %s "
            "with target number %s",
            settings.PREDICT_STOCK_OPTIMIZE_METRIC,
            settings.PREDICT_STOCK_OPTIMIZE_MAX_DRAWDOWN_THRESHOLD,
            settings.PREDICT_STOCK_NUMBER,
        )

        predict_stock_list = sorted_stocks[: settings.PREDICT_STOCK_NUMBER]
        logger.info(
            "Selected %d stocks for prediction: %s",
            len(predict_stock_list),
            ", ".join(predict_stock_list),
        )
        return predict_stock_list

    logger.info("No predict stocks optimization. Using all target stocks as predict stocks")
    return target_stock_list


def _save_training_artifacts(
    process_feature_config: dict[str, Any],
    optimal_data_sampling_proportion: dict[str, float],
    optimal_features: list[str],
    optimal_model_hyperparams: dict[str, Any],
    predict_stock_list: list[str],
    args: argparse.Namespace,
) -> None:
    """Persist artifacts created during training.

    Args:
        process_feature_config: Feature processing configuration.
        optimal_data_sampling_proportion: Sampling proportion mapping.
        optimal_features: Selected feature names.
        optimal_model_hyperparams: Model hyperparameters.
        predict_stock_list: Selected prediction stock list.
        args: CLI arguments.
    """
    os.makedirs(os.path.dirname(args.save_process_feature_config_pth), exist_ok=True)
    with open(args.save_process_feature_config_pth, "wb") as file_handle:
        pickle.dump(process_feature_config, file_handle)
    logger.info(
        "Processing features configuration saved to %s", args.save_process_feature_config_pth
    )

    os.makedirs(os.path.dirname(args.save_model_pth), exist_ok=True)
    optimal_model_hyperparams_path = args.save_model_hyperparams_pth

    os.makedirs(os.path.dirname(args.save_data_sampling_proportion_pth), exist_ok=True)
    _save_json_file(args.save_data_sampling_proportion_pth, optimal_data_sampling_proportion)
    logger.info(
        "Optimized data sampling proportion saved to %s",
        args.save_data_sampling_proportion_pth,
    )

    os.makedirs(os.path.dirname(args.save_features_pth), exist_ok=True)
    _save_json_file(args.save_features_pth, {"features": optimal_features})
    logger.info("Optimized features saved to %s", args.save_features_pth)

    os.makedirs(os.path.dirname(optimal_model_hyperparams_path), exist_ok=True)
    _save_json_file(optimal_model_hyperparams_path, optimal_model_hyperparams)
    logger.info("Optimized model hyperparameters saved to %s", optimal_model_hyperparams_path)

    os.makedirs(os.path.dirname(args.save_predict_stocks_pth), exist_ok=True)
    _save_json_file(args.save_predict_stocks_pth, {"predict_stocks": predict_stock_list})
    logger.info("Predict stocks saved to %s", args.save_predict_stocks_pth)

    return


def train(
    run: Run,
    train_stock_list: list[str],
    target_stock_list: list[str],
    fetched_data_df: pd.DataFrame,
    seed: int,
    args: argparse.Namespace,
) -> None:
    """Train model based on the given stocks.

    Args:
        run: Active W&B run.
        train_stock_list: Stock codes used for training.
        target_stock_list: Stock codes eligible for prediction.
        fetched_data_df: Intraday fetched data.
        seed: Random seed.
        args: CLI arguments.
    """
    logger.info("Start training model given stocks: %s", train_stock_list)
    logger.info("Current trading time: %s", pd.Timestamp.now(tz="America/New_York"))
    set_random_seed(seed)

    train_data_df = _load_training_data(fetched_data_df, train_stock_list)

    (
        X_train,
        y_train,
        X_validate,
        y_validate,
        process_feature_config,
    ) = _prepare_feature_data(train_data_df, seed)

    X_train, X_validate, optimal_features = _apply_feature_selection(X_train, X_validate, args)

    optimal_model_hyperparams = _resolve_model_hyperparams(args, seed)
    logger.info(
        "Training model with objective: %s",
        optimal_model_hyperparams.get("objective"),
    )
    logger.info(
        "Training model with optimize metric: %s",
        optimal_model_hyperparams.get("metric"),
    )

    optimal_data_sampling_proportion = _resolve_sampling_proportion(
        X_train,
        y_train,
        X_validate,
        y_validate,
        train_stock_list=train_stock_list,
        model_hyperparams=optimal_model_hyperparams,
        args=args,
        seed=seed,
    )

    X_train, y_train = sample_training_data(
        X_train,
        y_train,
        sampling_proportion=optimal_data_sampling_proportion,
        seed=seed,
    )

    X_train, X_validate, optimal_features = _maybe_optimize_features(
        X_train,
        y_train,
        X_validate,
        y_validate,
        model_hyperparams=optimal_model_hyperparams,
        args=args,
        seed=seed,
    )

    optimal_model_hyperparams = _maybe_optimize_model_hyperparams(
        X_train,
        y_train,
        X_validate,
        y_validate,
        optimal_model_hyperparams,
        args,
        seed,
    )

    _log_dataset_summary(X_train, X_validate)

    final_model, _ = model_training(
        X_train,
        y_train,
        X_validate,
        y_validate,
        categorical_features=settings.CATEGORICAL_FEATURES,
        model_hyperparams=optimal_model_hyperparams,
        show_training_log=True,
    )

    (
        valid_day_number,
        valid_mae,
        valid_mse,
        valid_sign_acc,
        valid_precision,
        valid_recall,
        valid_f1,
        vaild_average_daily_gain,
        vaild_overall_gain,
        valid_annualized_sharpe_ratio,
        validate_max_drawdown,
        sorted_stocks,
    ) = validate_model(
        final_model,
        X_validate,
        y_validate,
        target_stock_list=target_stock_list,
        verbose=args.verbose,
    )

    predict_stock_list = _select_predict_stocks(sorted_stocks, target_stock_list, args)

    os.makedirs(os.path.dirname(args.save_model_pth), exist_ok=True)
    final_model.save_model(args.save_model_pth)
    logger.info("Model saved to %s", args.save_model_pth)

    _save_training_artifacts(
        process_feature_config,
        optimal_data_sampling_proportion,
        optimal_features,
        optimal_model_hyperparams,
        predict_stock_list,
        args,
    )

    wandb.log(
        {
            "valid_day_number": valid_day_number,
            "valid_mae": valid_mae,
            "valid_mse": valid_mse,
            "valid_sign_acc": valid_sign_acc,
            "valid_precision": valid_precision,
            "valid_recall": valid_recall,
            "valid_f1": valid_f1,
            "valid_average_daily_gain": vaild_average_daily_gain,
            "valid_overall_gain": vaild_overall_gain,
            "valid_annualized_sharpe_ratio": valid_annualized_sharpe_ratio,
            "valid_max_drawdown": validate_max_drawdown,
        }
    )

    run.finish()

    logger.info("Training process completed.")


def _validate_optimization_flags(args: argparse.Namespace) -> None:
    """Ensure only one optimization flag is enabled.

    Args:
        args: CLI arguments.
    """
    activated_optimizations = [
        args.optimize_data_sampling_proportion,
        args.optimize_features,
        args.optimize_model_hyperparameters,
    ]
    num_activated = sum(1 for flag in activated_optimizations if flag)

    if num_activated <= 1:
        return

    error_message = (
        "Error: Only one optimization option can be active at a time. "
        f"Detected {num_activated} active optimizations: "
    )
    active_options_names = []
    if args.optimize_data_sampling_proportion:
        active_options_names.append("optimize_data_sampling_proportion")
    if args.optimize_features:
        active_options_names.append("optimize_features")
    if args.optimize_model_hyperparameters:
        active_options_names.append("optimize_model_hyperparameters")

    error_message += ", ".join(active_options_names) + ". Please enable only one."

    logging.error(error_message)
    raise ValueError(error_message)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for training CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_stocks", type=str, default="config/train_stocks.json")
    parser.add_argument("--target_stocks", type=str, default="config/target_stocks.json")
    parser.add_argument(
        "--fetched_data_pth", "-fdp", type=str, default="data/fetched_data.parquet"
    )
    parser.add_argument(
        "--optimize_data_sampling_proportion",
        "-odsp",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--data_sampling_proportion_pth",
        "-dspp",
        type=str,
        default="data/prod_data_sampling_proportion.json",
    )
    parser.add_argument("--optimize_features", "-of", action="store_true", default=False)
    parser.add_argument("--features_pth", "-fp", type=str, default="data/prod_features.json")
    parser.add_argument(
        "--optimize_model_hyperparameters", "-omhp", action="store_true", default=False
    )
    parser.add_argument(
        "--model_hyperparams_pth",
        "-mhpp",
        type=str,
        default="data/prod_model_hyperparams.json",
    )
    parser.add_argument("--optimize_predict_stocks", "-ops", action="store_true", default=False)

    parser.add_argument(
        "--save_process_feature_config_pth",
        "-spfcp",
        type=str,
        default="data/prod_process_feature_config.pkl",
    )
    parser.add_argument(
        "--save_data_sampling_proportion_pth",
        "-sdspp",
        type=str,
        default="data/prod_data_sampling_proportion.json",
    )
    parser.add_argument("--save_model_pth", "-smp", type=str, default="data/prod_model.model")
    parser.add_argument("--save_features_pth", "-sfp", type=str, default="data/prod_features.json")
    parser.add_argument(
        "--save_model_hyperparams_pth",
        "-smhpp",
        type=str,
        default="data/prod_model_hyperparams.json",
    )
    parser.add_argument("--save_predict_stocks_pth", type=str, default="data/predict_stocks.json")

    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--seed", "-s", type=int, default=settings.SEED)

    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    configure_logging(env="train", file_name="train.log")
    logger = logging.getLogger("ml4investment.train")

    _validate_optimization_flags(args)

    train_stock_list = _load_json_file(args.train_stocks)["train_stocks"]
    target_stock_list = _load_json_file(args.target_stocks)["target_stocks"]
    fetched_data_df = pd.read_parquet(args.fetched_data_pth)
    seed = args.seed

    run = setup_wandb(config=vars(args))

    train(run, train_stock_list, target_stock_list, fetched_data_df, seed, args)
