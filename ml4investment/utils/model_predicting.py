import logging
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from ml4investment.config.global_settings import settings

logger = logging.getLogger(__name__)


def _validate_historical_columns(historical_df: pd.DataFrame, actual_col: str) -> None:
    """Validate historical data has required columns.

    Args:
        historical_df: Historical DataFrame to validate.
        actual_col: Column containing actual values.
    """
    if "stock_code" not in historical_df.columns:
        raise ValueError("Column 'stock_code' not found in historical data.")
    if actual_col not in historical_df.columns:
        raise ValueError(f"Column '{actual_col}' not found in historical data.")


def _normalize_stock_codes(stock_codes: Iterable[str]) -> pd.Index:
    """Normalize stock codes into a unique Index.

    Args:
        stock_codes: Iterable of stock codes.

    Returns:
        Unique Index of stock codes as strings.
    """
    return pd.Index(stock_codes).astype(str).unique()


def _filter_historical_data(
    historical_df: pd.DataFrame, stock_codes: pd.Index, current_ts: pd.Timestamp | None
) -> pd.DataFrame:
    """Filter historical data by stock codes and optional timestamp.

    Args:
        historical_df: Historical DataFrame.
        stock_codes: Stock code Index.
        current_ts: Optional cutoff timestamp.

    Returns:
        Filtered historical DataFrame.
    """
    history = historical_df[historical_df["stock_code"].isin(stock_codes)].copy()
    if history.empty:
        return history

    if current_ts is not None:
        history = history[history.index < current_ts]
    return history


def get_prev_actual_ranking(
    stock_codes: Iterable[str],
    historical_df: pd.DataFrame,
    current_ts: pd.Timestamp | None = None,
    actual_col: str = "y_actual",
) -> dict[str, float]:
    """Return previous-day actual values for the given stock universe."""
    _validate_historical_columns(historical_df, actual_col)

    unique_codes = _normalize_stock_codes(stock_codes)
    if unique_codes.empty:
        return {}

    history = _filter_historical_data(historical_df, unique_codes, current_ts)
    if history.empty:
        return {}

    history.sort_index(inplace=True)
    prev_actual = history.groupby("stock_code", observed=True)[actual_col].last()
    prev_actual = prev_actual.dropna()

    return prev_actual.to_dict()


def _empty_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Create an empty portfolio DataFrame with weight/action columns.

    Args:
        df: Source DataFrame to mirror.

    Returns:
        Empty DataFrame with weight/action columns.
    """
    result = df.head(0).copy()
    result["weight"] = pd.Series(dtype="float64")
    result["action"] = pd.Series(dtype="object")
    return result


def _select_direction(direction: str, metric_col: str, data: pd.DataFrame) -> pd.DataFrame:
    """Select stocks by direction and metric.

    Args:
        direction: BUY_LONG or SELL_SHORT.
        metric_col: Column used for ranking.
        data: Candidate DataFrame.

    Returns:
        Selected portfolio DataFrame.
    """
    if direction == "BUY_LONG":
        eligible = data[data[metric_col] > 0.0]
        ascending = False
    elif direction == "SELL_SHORT":
        eligible = data[data[metric_col] < 0.0]
        ascending = True
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    if eligible.empty:
        return _empty_portfolio(data)

    selected = (
        eligible.sort_values(by=metric_col, ascending=ascending)
        .head(settings.NUMBER_OF_STOCKS_TO_BUY)
        .copy()
    )
    if selected.empty:
        return _empty_portfolio(data)

    total_abs_metric = float(selected[metric_col].abs().sum())
    if total_abs_metric <= 0:
        selected["weight"] = 1.0 / len(selected)
    else:
        selected["weight"] = selected[metric_col].abs() / total_abs_metric

    selected["action"] = direction
    return selected


def _select_both_directions(metric_col: str, data: pd.DataFrame) -> pd.DataFrame:
    """Select candidates across both directions by absolute metric.

    Args:
        metric_col: Column used for ranking.
        data: Candidate DataFrame.

    Returns:
        Selected portfolio DataFrame.
    """
    non_zero = data[data[metric_col] != 0]
    if non_zero.empty:
        return _empty_portfolio(data)

    abs_sorted = non_zero.assign(_abs_metric=non_zero[metric_col].abs()).sort_values(
        by="_abs_metric", ascending=False
    )
    selected = abs_sorted.head(settings.NUMBER_OF_STOCKS_TO_BUY).drop(columns="_abs_metric").copy()
    if selected.empty:
        return _empty_portfolio(data)

    total_abs_metric = float(selected[metric_col].abs().sum())
    if total_abs_metric <= 0:
        selected["weight"] = 1.0 / len(selected)
    else:
        selected["weight"] = selected[metric_col].abs() / total_abs_metric

    selected["action"] = np.where(selected[metric_col] > 0, "BUY_LONG", "SELL_SHORT")
    return selected


def _select_buy_long_first(metric_col: str, data: pd.DataFrame) -> pd.DataFrame:
    """Select BUY_LONG first, then SELL_SHORT if slots remain.

    Args:
        metric_col: Column used for ranking.
        data: Candidate DataFrame.

    Returns:
        Selected portfolio DataFrame.
    """
    long_candidates = data[data[metric_col] > 0.0].copy()
    long_selected = long_candidates.sort_values(by=metric_col, ascending=False).head(
        settings.NUMBER_OF_STOCKS_TO_BUY
    )

    remaining_slots = settings.NUMBER_OF_STOCKS_TO_BUY - len(long_selected)

    short_selected = pd.DataFrame()
    if remaining_slots > 0:
        short_candidates = data[data[metric_col] < 0.0].copy()
        if not short_candidates.empty:
            short_selected = short_candidates.sort_values(by=metric_col, ascending=True).head(
                remaining_slots
            )

    combined = pd.concat([long_selected, short_selected])
    if combined.empty:
        return _empty_portfolio(data)

    combined["action"] = np.where(combined[metric_col] > 0, "BUY_LONG", "SELL_SHORT")
    total_abs_metric = float(combined[metric_col].abs().sum())
    if total_abs_metric <= 0:
        combined["weight"] = 1.0 / len(combined)
    else:
        combined["weight"] = combined[metric_col].abs() / total_abs_metric

    return combined


def _build_portfolio(metric_col: str, data: pd.DataFrame) -> pd.DataFrame:
    """Build portfolio based on strategy.

    Args:
        metric_col: Column used for ranking.
        data: Candidate DataFrame.

    Returns:
        Portfolio DataFrame.
    """
    working_df = data.dropna(subset=[metric_col])
    if working_df.empty:
        return _empty_portfolio(data)

    strategy = settings.STOCK_SELECTION_STRATEGY

    if strategy == "BUY_LONG":
        return _select_direction("BUY_LONG", metric_col, working_df)

    if strategy == "SELL_SHORT":
        return _select_direction("SELL_SHORT", metric_col, working_df)

    if strategy == "ADAPT":
        positive_ratio = (working_df[metric_col] > 0).mean()
        primary = "BUY_LONG" if positive_ratio >= 0.5 else "SELL_SHORT"
        portfolio = _select_direction(primary, metric_col, working_df)
        if portfolio.empty:
            alternate = "SELL_SHORT" if primary == "BUY_LONG" else "BUY_LONG"
            portfolio = _select_direction(alternate, metric_col, working_df)
        return portfolio

    if strategy == "BOTH":
        return _select_both_directions(metric_col, working_df)

    if strategy == "BUY_LONG_FIRST":
        return _select_buy_long_first(metric_col, working_df)

    logger.error("Unknown STOCK_SELECTION_STRATEGY: %s", strategy)
    raise ValueError(f"Unknown STOCK_SELECTION_STRATEGY: {strategy}")


def _add_momentum_column(
    candidates: pd.DataFrame, prev_actuals: Mapping[str, float] | pd.Series
) -> pd.DataFrame:
    """Add momentum values to candidates.

    Args:
        candidates: Candidate DataFrame.
        prev_actuals: Previous actuals mapping or Series.

    Returns:
        Updated candidates DataFrame.
    """
    momentum_series = (
        pd.Series(prev_actuals, dtype="float64")
        if not isinstance(prev_actuals, pd.Series)
        else prev_actuals.astype("float64")
    )
    momentum_series.index = momentum_series.index.astype(str)
    candidates["momentum"] = candidates["stock_code"].astype(str).map(momentum_series)
    return candidates


def _combine_with_momentum(
    base: pd.DataFrame, momentum: pd.DataFrame, base_df: pd.DataFrame, weight: float
) -> pd.DataFrame:
    """Combine base and momentum portfolios by weight.

    Args:
        base: Base portfolio.
        momentum: Momentum portfolio.
        base_df: Original candidate DataFrame.
        weight: Momentum weight.

    Returns:
        Combined portfolio DataFrame.
    """
    if base.empty and momentum.empty:
        return _empty_portfolio(base_df)

    primary_weight = 1.0 - weight

    combined_frames = []
    if not base.empty:
        tmp = base.copy()
        tmp["weight"] *= primary_weight
        combined_frames.append(tmp)
    if not momentum.empty:
        tmp = momentum.copy()
        tmp["weight"] *= weight
        combined_frames.append(tmp)

    if not combined_frames:
        return _empty_portfolio(base_df)

    combined = pd.concat(combined_frames, ignore_index=True)
    if combined.empty:
        return _empty_portfolio(base_df)

    combined["signed_weight"] = np.where(
        combined["action"] == "BUY_LONG", combined["weight"], -combined["weight"]
    )
    agg = combined.groupby("stock_code", as_index=False, observed=True)["signed_weight"].sum()
    agg["weight"] = agg["signed_weight"].abs()
    agg = agg[agg["weight"] > 0]
    if agg.empty:
        return _empty_portfolio(base_df)

    agg["action"] = np.where(agg["signed_weight"] >= 0, "BUY_LONG", "SELL_SHORT")
    total = agg["weight"].sum()
    if total <= 0:
        return _empty_portfolio(base_df)

    agg["weight"] /= total
    agg = agg.drop(columns=["signed_weight"])

    enriched = agg.merge(
        base_df.drop_duplicates("stock_code"),
        on="stock_code",
        how="left",
        suffixes=("", "_cand"),
    )
    return enriched


def _validate_candidate_columns(candidates: pd.DataFrame) -> None:
    """Validate candidate DataFrame has required columns.

    Args:
        candidates: Candidate DataFrame.
    """
    for col in ["stock_code", "prediction"]:
        if col not in candidates.columns:
            raise ValueError(f"Column '{col}' not found in candidates DataFrame.")


def get_stocks_portfolio(
    candidates: pd.DataFrame, prev_actuals: Mapping[str, float] | pd.Series | None = None
) -> pd.DataFrame:
    """Select candidates under the configured strategy, supporting momentum blending."""
    _validate_candidate_columns(candidates)

    candidates_local = candidates.copy()

    momentum_weight = settings.STOCK_SELECTION_MOMENTUM
    use_momentum = momentum_weight > 0
    if use_momentum and prev_actuals is not None:
        candidates_local = _add_momentum_column(candidates_local, prev_actuals)

    base_portfolio = _build_portfolio("prediction", candidates_local)
    if not use_momentum:
        return base_portfolio

    momentum_portfolio = _build_portfolio("momentum", candidates_local)
    if momentum_portfolio.empty:
        return base_portfolio

    if base_portfolio.empty:
        momentum_portfolio = momentum_portfolio.copy()
        total = momentum_portfolio["weight"].sum()
        if total > 0:
            momentum_portfolio["weight"] /= total
        return momentum_portfolio

    return _combine_with_momentum(
        base_portfolio, momentum_portfolio, candidates_local, momentum_weight
    )
