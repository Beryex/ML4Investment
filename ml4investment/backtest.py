import argparse
import json
import logging
import pickle
from typing import Any

import lightgbm as lgb
import pandas as pd
import wandb
from wandb.sdk.wandb_run import Run

from ml4investment.config.global_settings import settings
from ml4investment.utils.feature_calculating import calculate_features
from ml4investment.utils.feature_processing import process_features_for_backtest
from ml4investment.utils.logging import configure_logging, setup_wandb
from ml4investment.utils.model_backtesting import get_detailed_static_result
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


def _load_backtest_data(
    fetched_data_df: pd.DataFrame, train_stock_list: list[str]
) -> pd.DataFrame:
    """Filter fetched data for backtesting.

    Args:
        fetched_data_df: Raw intraday data.
        train_stock_list: Stock codes used for training.

    Returns:
        Filtered intraday DataFrame for backtesting.
    """
    train_data_start_date = settings.TRAINING_DATA_START_DATE
    logger.info("Load input fetched data, starting from %s", train_data_start_date)
    train_data_df = fetched_data_df[fetched_data_df["stock_code"].isin(train_stock_list)]
    return train_data_df.loc[train_data_start_date:]


def _prepare_backtest_features(
    backtest_data_df: pd.DataFrame,
    process_feature_config: dict[str, Any],
    predict_stock_list: list[str],
    selected_features: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Calculate, process, and filter features for backtesting.

    Args:
        backtest_data_df: Intraday data for backtesting.
        process_feature_config: Feature processing configuration.
        predict_stock_list: Stock codes to include.
        selected_features: Feature list selected during training.

    Returns:
        Tuple of (X_backtest, y_backtest).
    """
    daily_features_df = calculate_features(backtest_data_df)

    X_backtest, y_backtest = process_features_for_backtest(
        daily_features_df, process_feature_config, predict_stock_list
    )

    X_backtest = X_backtest[selected_features]

    return X_backtest, y_backtest


def _log_backtest_summary(X_backtest: pd.DataFrame) -> None:
    """Log summary information for backtest dataset.

    Args:
        X_backtest: Backtest feature DataFrame.
    """
    start_date = X_backtest.index.min()
    end_date = X_backtest.index.max()
    logger.info("Oldest date in backtest data: %s", start_date)
    logger.info("Newest date in backtest data: %s", end_date)
    logger.info("Total processed samples in backtest data: %s", X_backtest.shape[0])
    logger.info("Number of features in backtest data: %s", X_backtest.shape[1])


def _run_backtest_metrics(
    model: lgb.Booster,
    X_backtest: pd.DataFrame,
    y_backtest: pd.Series,
    predict_stock_list: list[str],
    args: argparse.Namespace,
) -> tuple[int, float, float, float, float, float, float, float, float, float, float, list[str]]:
    """Run backtest evaluation metrics.

    Args:
        model: Trained LightGBM model.
        X_backtest: Backtest features.
        y_backtest: Backtest targets.
        predict_stock_list: Stock codes evaluated.
        args: CLI arguments.

    Returns:
        Backtest metrics tuple.
    """
    return get_detailed_static_result(
        model=model,
        X=X_backtest,
        y=y_backtest,
        predict_stock_list=predict_stock_list,
        name="Backtest",
        verbose=args.verbose,
    )


def _log_backtest_metrics(metrics: dict[str, float]) -> None:
    """Log backtest metrics to W&B.

    Args:
        metrics: Metric dictionary to log.
    """
    wandb.log(metrics)


def backtest(
    run: Run,
    train_stock_list: list[str],
    predict_stock_list: list[str],
    fetched_data_df: pd.DataFrame,
    process_feature_config: dict[str, Any],
    selected_features: list[str],
    model: lgb.Booster,
    seed: int,
    args: argparse.Namespace,
) -> None:
    """Backtest the model performance for the given stocks for the last week.

    Args:
        run: Active W&B run.
        train_stock_list: Stock codes used for training.
        predict_stock_list: Stock codes to evaluate.
        fetched_data_df: Intraday fetched data.
        process_feature_config: Feature processing configuration.
        selected_features: Feature list selected during training.
        model: Trained LightGBM model.
        seed: Random seed.
        args: CLI arguments.
    """
    logger.info("Start backtesting based on the given stocks: %s", predict_stock_list)
    logger.info("Current trading time: %s", pd.Timestamp.now(tz="America/New_York"))
    set_random_seed(seed)

    backtest_data_df = _load_backtest_data(fetched_data_df, train_stock_list)

    X_backtest, y_backtest = _prepare_backtest_features(
        backtest_data_df,
        process_feature_config,
        predict_stock_list,
        selected_features,
    )

    _log_backtest_summary(X_backtest)

    (
        backtest_day_number,
        backtest_mae,
        backtest_mse,
        backtest_sign_acc,
        backtest_precision,
        backtest_recall,
        backtest_f1,
        backtest_average_daily_gain,
        backtest_overall_gain,
        backtest_annualized_sharpe_ratio,
        backtest_max_drawdown,
        _,
    ) = _run_backtest_metrics(model, X_backtest, y_backtest, predict_stock_list, args)

    _log_backtest_metrics(
        {
            "backtest_day_number": backtest_day_number,
            "backtest_mae": backtest_mae,
            "backtest_mse": backtest_mse,
            "backtest_sign_acc": backtest_sign_acc,
            "backtest_precision": backtest_precision,
            "backtest_recall": backtest_recall,
            "backtest_f1": backtest_f1,
            "backtest_average_daily_gain": backtest_average_daily_gain,
            "backtest_overall_gain": backtest_overall_gain,
            "backtest_annualized_sharpe_ratio": backtest_annualized_sharpe_ratio,
            "backtest_max_drawdown": backtest_max_drawdown,
        }
    )

    run.finish()

    logger.info("Backtesting process completed.")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for backtesting CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_stocks", "-ts", type=str, default="config/train_stocks.json")
    parser.add_argument("--predict_stocks", "-ps", type=str, default="data/predict_stocks.json")
    parser.add_argument(
        "--fetched_data_pth", "-fdp", type=str, default="data/fetched_data.parquet"
    )

    parser.add_argument(
        "--process_feature_config_pth",
        "-pfcp",
        type=str,
        default="data/prod_process_feature_config.pkl",
    )
    parser.add_argument("--features_pth", "-fp", type=str, default="data/prod_features.json")
    parser.add_argument("--model_pth", "-mp", type=str, default="data/prod_model.model")

    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--seed", "-s", type=int, default=settings.SEED)

    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    configure_logging(env="backtest", file_name="backtest.log")
    logger = logging.getLogger("ml4investment.backtest")

    train_stock_list = _load_json_file(args.train_stocks)["train_stocks"]
    predict_stock_list = _load_json_file(args.predict_stocks)["predict_stocks"]
    fetched_data_df = pd.read_parquet(args.fetched_data_pth)
    process_feature_config = pickle.load(open(args.process_feature_config_pth, "rb"))
    selected_features = _load_json_file(args.features_pth)["features"]
    model = lgb.Booster(model_file=args.model_pth)
    seed = args.seed

    run = setup_wandb(config=vars(args))

    backtest(
        run,
        train_stock_list,
        predict_stock_list,
        fetched_data_df,
        process_feature_config,
        selected_features,
        model,
        seed,
        args,
    )
