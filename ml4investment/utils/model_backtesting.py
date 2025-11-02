import logging
from collections import defaultdict
from typing import cast

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from prettytable import PrettyTable
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from ml4investment.config.global_settings import settings
from ml4investment.utils.model_predicting import get_stocks_portfolio
from ml4investment.utils.utils import _coerce_stock_id, id_to_stock_code

logger = logging.getLogger(__name__)


def get_shap_analysis(
    model: lgb.Booster,
    X: pd.DataFrame,
    y: pd.Series,
    preds: np.ndarray,
    name: str = "",
):
    """Perform SHAP analysis and generate summary plots"""
    logger.info("Calculating SHAP values for feature contribution analysis...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    rng = np.random.default_rng(settings.SEED)
    plot_name_suffix = name if name else "default"

    # 1. Global Summary Plot
    logger.info("Generating global SHAP summary plot...")
    shap.summary_plot(
        shap_values,
        X,
        show=False,
        rng=rng,
        max_display=settings.SHAP_PLOT_MAX_DISPLAY_FEATURES,
    )
    global_plot_path = settings.SHAP_SUMMARY_GLOBAL_IMG_PTH_TPL.format(plot_name_suffix)
    plt.savefig(global_plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Global SHAP summary plot saved to: {global_plot_path}")

    # 2. Error Analysis Plot
    logger.info("Generating SHAP summary plot for incorrect predictions...")
    error_mask = np.sign(preds) != np.sign(y)
    if np.any(error_mask):
        shap_errors = shap_values[error_mask]
        X_errors = X[error_mask]

        shap.summary_plot(
            shap_errors,
            X_errors,
            show=False,
            rng=rng,
            max_display=settings.SHAP_PLOT_MAX_DISPLAY_FEATURES,
        )
        plt.title(f"Feature Contributions on Errors ({name})")  # Set title after plotting
        error_plot_path = settings.SHAP_SUMMARY_ERROR_IMG_PTH_TPL.format(plot_name_suffix)
        plt.savefig(error_plot_path, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP error analysis plot saved to: {error_plot_path}")
    else:
        logger.info("No incorrect predictions found, skipping error analysis plot.")

    # 3. Correct Prediction Analysis Plot
    logger.info("Generating SHAP summary plot for correct predictions...")
    correct_mask = np.sign(preds) == np.sign(y)
    if np.any(correct_mask):
        shap_correct = shap_values[correct_mask]
        X_correct = X[correct_mask]

        shap.summary_plot(
            shap_correct,
            X_correct,
            show=False,
            rng=rng,
            max_display=settings.SHAP_PLOT_MAX_DISPLAY_FEATURES,
        )
        plt.title(f"Feature Contributions on Correct Predictions ({name})")
        correct_plot_path = settings.SHAP_SUMMARY_CORRECT_IMG_PATH_TPL.format(plot_name_suffix)
        plt.savefig(correct_plot_path, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP correct analysis plot saved to: {correct_plot_path}")
    else:
        logger.info("No correct predictions found, skipping correct analysis plot.")


def get_detailed_static_result(
    model: lgb.Booster,
    X: pd.DataFrame,
    y: pd.Series,
    predict_stock_list: list,
    name: str = "",
    verbose: bool = True,
) -> tuple[int, float, float, float, float, float, float, float, float, float, float, list[str]]:
    """Display detailed static result of the model predictions"""
    preds = model.predict(X, num_iteration=model.best_iteration)
    assert isinstance(preds, np.ndarray)

    get_shap_analysis(model, X, y, preds, name)

    results_df = X[["stock_id"]].copy()
    results_df["y_actual"] = y
    results_df["prediction"] = preds

    """Compute overall metrics"""
    mae_overall = mean_absolute_error(results_df["y_actual"], results_df["prediction"])
    mse_overall = mean_squared_error(results_df["y_actual"], results_df["prediction"])
    sign_acc_overall = float(
        (np.sign(results_df["y_actual"]) == np.sign(results_df["prediction"])).mean()
    )

    binary_y_true = (results_df["y_actual"] > 0).astype(int)
    binary_y_pred = (results_df["prediction"] > 0).astype(int)
    precision_overall = float(precision_score(binary_y_true, binary_y_pred, zero_division=0))
    recall_overall = float(recall_score(binary_y_true, binary_y_pred, zero_division=0))
    f1_overall = float(f1_score(binary_y_true, binary_y_pred, zero_division=0))

    """Compute stock-level metrics"""
    stock_metrics = defaultdict(dict)

    for stock_id, stock_df in results_df.groupby("stock_id", observed=True):
        stock_code = id_to_stock_code(int(stock_id))  # type: ignore
        if stock_code not in predict_stock_list:
            continue

        y_true_stock = stock_df["y_actual"]
        y_pred_stock = stock_df["prediction"]

        stock_metrics[stock_code]["mae"] = mean_absolute_error(y_true_stock, y_pred_stock)
        stock_metrics[stock_code]["mse"] = mean_squared_error(y_true_stock, y_pred_stock)
        stock_metrics[stock_code]["sign_acc"] = (
            np.sign(y_true_stock) == np.sign(y_pred_stock)
        ).mean()

        b_y_true = (y_true_stock > 0).astype(int)
        b_y_pred = (y_pred_stock > 0).astype(int)
        stock_metrics[stock_code]["precision"] = precision_score(
            b_y_true, b_y_pred, zero_division=0
        )
        stock_metrics[stock_code]["recall"] = recall_score(b_y_true, b_y_pred, zero_division=0)
        stock_metrics[stock_code]["f1"] = f1_score(b_y_true, b_y_pred, zero_division=0)

        positive_mask = y_pred_stock > 0
        if positive_mask.any():
            gain_factors = 1 + y_true_stock[positive_mask]
            overall_gain = gain_factors.prod()
        else:
            overall_gain = 1.0
        stock_metrics[stock_code]["overall_gain"] = overall_gain
        stock_metrics[stock_code]["avg_daily_gain"] = cast(float, overall_gain) ** (
            1 / len(stock_df)
        )

        # Sharpe ratio and max drawdown
        returns_series = y_true_stock.copy()
        daily_returns_np_stock = returns_series.to_numpy()
        if np.std(daily_returns_np_stock) > 0:
            sharpe_ratio = np.mean(daily_returns_np_stock) / np.std(daily_returns_np_stock)
            sharpe_ratio *= np.sqrt(settings.TRADING_DAYS_PER_YEAR)
        else:
            sharpe_ratio = 0.0

        cumulative_returns_stock = np.cumprod(1 + daily_returns_np_stock)
        peak_stock = np.maximum.accumulate(cumulative_returns_stock)
        drawdown_stock = (cumulative_returns_stock - peak_stock) / peak_stock
        max_drawdown_stock = np.min(drawdown_stock) if len(drawdown_stock) > 0 else 0.0

        stock_metrics[stock_code]["sharpe_ratio"] = sharpe_ratio
        stock_metrics[stock_code]["max_drawdown"] = max_drawdown_stock

    """Compute daily-level metrics"""
    gain_actual = 1.0
    daily_results_table_data = []
    daily_returns_list = []

    unique_days = results_df.index.unique()
    day_number = len(unique_days)

    logger.info(f"Selecting stocks using strategy: {settings.STOCK_SELECTION_STRATEGY}")
    for date, daily_df in results_df.groupby(results_df.index):
        daily_df_filtered = daily_df[
            daily_df["stock_id"].map(id_to_stock_code).isin(predict_stock_list)
        ]

        positive_preds_df = daily_df_filtered[daily_df_filtered["prediction"] > 0]

        total_candidates = len(daily_df_filtered)
        predicted_positive_ratio = (
            len(positive_preds_df) / total_candidates if total_candidates else 0.0
        )
        actual_positive_ratio = (
            (daily_df_filtered["y_actual"] > 0).mean() if total_candidates else 0.0
        )

        selected_portfolio = get_stocks_portfolio(daily_df_filtered)

        if selected_portfolio.empty:
            daily_gain_predict = 1.0
            daily_gain_actual = 1.0
        else:
            weight_sum = float(selected_portfolio["weight"].sum())
            assert np.isclose(weight_sum, 1.0)

            long_mask = selected_portfolio["action"] == "BUY_LONG"
            predicted_factor = np.where(
                long_mask,
                1 + selected_portfolio["prediction"],
                1 - selected_portfolio["prediction"],
            )
            actual_factor = np.where(
                long_mask,
                1 + selected_portfolio["y_actual"],
                1 - selected_portfolio["y_actual"],
            )

            daily_gain_predict = float((predicted_factor * selected_portfolio["weight"]).sum())
            daily_gain_actual = float((actual_factor * selected_portfolio["weight"]).sum())

        daily_returns_list.append(daily_gain_actual - 1)

        predicted_long_stocks = [
            id_to_stock_code(_coerce_stock_id(row.stock_id))
            for row in selected_portfolio.itertuples()
            if row.action == "BUY_LONG"
        ]
        predicted_short_stocks = [
            id_to_stock_code(_coerce_stock_id(row.stock_id))
            for row in selected_portfolio.itertuples()
            if row.action == "SELL_SHORT"
        ]

        optimal_buy_long_df = daily_df_filtered.sort_values("y_actual", ascending=False).head(1)
        if not optimal_buy_long_df.empty:
            optimal_buy_long_stocks = [
                id_to_stock_code(_coerce_stock_id(optimal_buy_long_df.iloc[0]["stock_id"]))
            ]
            optimal_buy_long_gain = float(1 + optimal_buy_long_df.iloc[0]["y_actual"])
        else:
            optimal_buy_long_stocks = []
            optimal_buy_long_gain = 1.0

        optimal_sell_short_df = daily_df_filtered.sort_values("y_actual", ascending=True).head(1)
        if not optimal_sell_short_df.empty:
            optimal_sell_short_stocks = [
                id_to_stock_code(_coerce_stock_id(optimal_sell_short_df.iloc[0]["stock_id"]))
            ]
            optimal_sell_short_gain = float(1 - optimal_sell_short_df.iloc[0]["y_actual"])
        else:
            optimal_sell_short_stocks = []
            optimal_sell_short_gain = 1.0

        gain_actual *= daily_gain_actual

        daily_results_table_data.append(
            {
                "day": date.strftime("%Y-%m-%d"),
                "daily_gain_predict": daily_gain_predict,
                "daily_gain_actual": daily_gain_actual,
                "predicted_buy_long_stocks": predicted_long_stocks,
                "predicted_sell_short_stocks": predicted_short_stocks,
                "optimal_buy_long_stocks": optimal_buy_long_stocks,
                "optimal_sell_short_stocks": optimal_sell_short_stocks,
                "optimal_buy_long_gain": optimal_buy_long_gain,
                "optimal_sell_short_gain": optimal_sell_short_gain,
                "cumulative_gain": gain_actual,
                "predicted_positive_ratio": predicted_positive_ratio,
                "actual_positive_ratio": actual_positive_ratio,
            }
        )

    average_daily_gain = gain_actual ** (1 / day_number) if day_number > 0 else 1.0

    daily_returns_np = np.array(daily_returns_list)
    if np.std(daily_returns_np) > 0:
        daily_sharpe_ratio = np.mean(daily_returns_np) / np.std(daily_returns_np)
        annualized_sharpe_ratio = daily_sharpe_ratio * np.sqrt(settings.TRADING_DAYS_PER_YEAR)
    else:
        annualized_sharpe_ratio = 0.0

    cumulative_returns = np.cumprod(1 + daily_returns_np)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

    optimize_metric = settings.PREDICT_STOCK_OPTIMIZE_METRIC

    sorted_stocks = sorted(
        stock_metrics.keys(),
        key=lambda s: stock_metrics[s].get(optimize_metric, 0.0),
        reverse=True,
    )

    start_date = X.index.min()
    end_date = X.index.max()

    if verbose:
        stock_static_table = PrettyTable()
        stock_static_table.field_names = [
            "Stock",
            "MAE",
            "MSE",
            "Sign Acc",
            "Precision",
            "Recall",
            "F1",
            "Avg Daily Gain",
            "Overall Gain",
            "Sharpe Ratio",
            "Max Drawdown",
        ]
        for stock in sorted_stocks:
            metrics = stock_metrics[stock]
            stock_static_table.add_row(
                [
                    stock,
                    f"{metrics['mae']:.7f}",
                    f"{metrics['mse']:.7f}",
                    f"{metrics['sign_acc'] * 100:.2f}%",
                    f"{metrics['precision'] * 100:.2f}%",
                    f"{metrics['recall'] * 100:.2f}%",
                    f"{metrics['f1'] * 100:.2f}%",
                    f"{metrics['avg_daily_gain']:+.4%}",
                    f"{metrics['overall_gain']:+.2%}",
                    f"{metrics['sharpe_ratio']:.3f}",
                    f"{metrics['max_drawdown']:.2%}",
                ],
                divider=True,
            )
        title_str = f"{name} Stock-level Static Result from {start_date} to {end_date}"
        logger.info(f"\n{stock_static_table.get_string(title=title_str)}")

        daily_static_table = PrettyTable()
        daily_static_table.field_names = [
            "Day",
            "Predict Daily Gain",
            "Actual Daily Gain",
            "Cumulative Gain",
            "Predicted BUY LONG Stocks",
            "Optimal BUY LONG Stocks",
            "Optimal BUY LONG Daily Gain",
            "Predicted SELL SHORT Stocks",
            "Optimal SELL SHORT Stocks",
            "Optimal SELL SHORT Daily Gain",
            "Predicted Upside Ratio",
            "Actual Upside Ratio",
        ]
        for res in daily_results_table_data:
            daily_static_table.add_row(
                [
                    res["day"],
                    f"{res['daily_gain_predict']:+.2%}",
                    f"{res['daily_gain_actual']:+.2%}",
                    f"{res['cumulative_gain']:+.2%}",
                    res["predicted_buy_long_stocks"],
                    res["optimal_buy_long_stocks"],
                    f"{res['optimal_buy_long_gain']:+.2%}",
                    res["predicted_sell_short_stocks"],
                    res["optimal_sell_short_stocks"],
                    f"{res['optimal_sell_short_gain']:+.2%}",
                    f"{res['predicted_positive_ratio']:.2%}",
                    f"{res['actual_positive_ratio']:.2%}",
                ],
                divider=True,
            )
        title_str = f"{name} Daily-level Static Result from {start_date} to {end_date}"
        logger.info(f"\n{daily_static_table.get_string(title=title_str)}")

    overall_static_table = PrettyTable()
    overall_static_table.field_names = [
        "Trading Days",
        "MAE",
        "MSE",
        "Sign Acc",
        "Precision",
        "Recall",
        "F1",
        "Avg Daily Gain",
        "Overall Gain",
        "Annualized Sharpe Ratio",
        "Max Drawdown",
    ]
    overall_static_table.add_row(
        [
            f"{day_number}",
            f"{mae_overall:.7f}",
            f"{mse_overall:.7f}",
            f"{sign_acc_overall * 100:.2f}%",
            f"{precision_overall * 100:.2f}%",
            f"{recall_overall * 100:.2f}%",
            f"{f1_overall * 100:.2f}%",
            f"{average_daily_gain:+.4%}",
            f"{gain_actual:+.2%}",
            f"{annualized_sharpe_ratio:.3f}",
            f"{max_drawdown:.2%}",
        ],
        divider=True,
    )
    title_str = f"{name} Overall Static Result from {start_date} to {end_date}"
    logger.info(f"\n{overall_static_table.get_string(title=title_str)}")

    return (
        day_number,
        mae_overall,
        mse_overall,
        sign_acc_overall,
        precision_overall,
        recall_overall,
        f1_overall,
        average_daily_gain,
        gain_actual,
        annualized_sharpe_ratio,
        max_drawdown,
        sorted_stocks,
    )
