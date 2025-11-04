import logging
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from ml4investment.config.global_settings import settings

logger = logging.getLogger(__name__)


def get_prev_actual_ranking(
    stock_codes: Iterable[str],
    historical_df: pd.DataFrame,
    current_ts: pd.Timestamp | None = None,
    actual_col: str = "y_actual",
) -> dict[str, float]:
    """Return previous-day actual values for the given stock universe."""
    if "stock_code" not in historical_df.columns:
        raise ValueError("Column 'stock_code' not found in historical data.")
    if actual_col not in historical_df.columns:
        raise ValueError(f"Column '{actual_col}' not found in historical data.")

    unique_codes = pd.Index(stock_codes).astype(str).unique()
    if unique_codes.empty:
        return {}

    history = historical_df[historical_df["stock_code"].isin(unique_codes)].copy()
    if history.empty:
        return {}

    if current_ts is not None:
        history = history[history.index < current_ts]
        if history.empty:
            return {}

    history.sort_index(inplace=True)
    prev_actual = history.groupby("stock_code", observed=True)[actual_col].last()
    prev_actual = prev_actual.dropna()

    return prev_actual.to_dict()


def get_stocks_portfolio(
    candidates: pd.DataFrame, prev_actuals: Mapping[str, float] | pd.Series | None = None
) -> pd.DataFrame:
    """Select candidates under the configured strategy, supporting momentum blending."""
    for col in ["stock_code", "prediction"]:
        if col not in candidates.columns:
            raise ValueError(f"Column '{col}' not found in candidates DataFrame.")

    def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
        result = df.head(0).copy()
        result["weight"] = pd.Series(dtype="float64")
        result["action"] = pd.Series(dtype="object")
        return result

    def _select_direction(direction: str, metric_col: str, data: pd.DataFrame) -> pd.DataFrame:
        if direction == "BUY_LONG":
            eligible = data[data[metric_col] > 0.0]
            ascending = False
        elif direction == "SELL_SHORT":
            eligible = data[data[metric_col] < 0.0]
            ascending = True
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        if eligible.empty:
            return _empty_like(data)

        selected = (
            eligible.sort_values(by=metric_col, ascending=ascending)
            .head(settings.NUMBER_OF_STOCKS_TO_BUY)
            .copy()
        )
        if selected.empty:
            return _empty_like(data)

        total_abs_metric = float(selected[metric_col].abs().sum())
        if total_abs_metric <= 0:
            selected["weight"] = 1.0 / len(selected)
        else:
            selected["weight"] = selected[metric_col].abs() / total_abs_metric

        selected["action"] = direction
        return selected

    def _build_portfolio(metric_col: str, data: pd.DataFrame) -> pd.DataFrame:
        working_df = data.dropna(subset=[metric_col])
        if working_df.empty:
            return _empty_like(data)

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
            non_zero = working_df[working_df[metric_col] != 0]
            if non_zero.empty:
                return _empty_like(working_df)

            abs_sorted = non_zero.assign(_abs_metric=non_zero[metric_col].abs()).sort_values(
                by="_abs_metric", ascending=False
            )
            selected = (
                abs_sorted.head(settings.NUMBER_OF_STOCKS_TO_BUY)
                .drop(columns="_abs_metric")
                .copy()
            )
            if selected.empty:
                return _empty_like(working_df)

            total_abs_metric = float(selected[metric_col].abs().sum())
            if total_abs_metric <= 0:
                selected["weight"] = 1.0 / len(selected)
            else:
                selected["weight"] = selected[metric_col].abs() / total_abs_metric

            selected["action"] = np.where(selected[metric_col] > 0, "BUY_LONG", "SELL_SHORT")
            return selected

        if strategy == "BUY_LONG_FIRST":
            long_candidates = working_df[working_df[metric_col] > 0.0].copy()
            long_selected = long_candidates.sort_values(by=metric_col, ascending=False).head(
                settings.NUMBER_OF_STOCKS_TO_BUY
            )

            remaining_slots = settings.NUMBER_OF_STOCKS_TO_BUY - len(long_selected)

            short_selected = pd.DataFrame()
            if remaining_slots > 0:
                short_candidates = working_df[working_df[metric_col] < 0.0].copy()
                if not short_candidates.empty:
                    short_selected = short_candidates.sort_values(
                        by=metric_col, ascending=True
                    ).head(remaining_slots)

            combined = pd.concat([long_selected, short_selected])
            if combined.empty:
                return _empty_like(working_df)

            combined["action"] = np.where(combined[metric_col] > 0, "BUY_LONG", "SELL_SHORT")
            total_abs_metric = float(combined[metric_col].abs().sum())
            if total_abs_metric <= 0:
                combined["weight"] = 1.0 / len(combined)
            else:
                combined["weight"] = combined[metric_col].abs() / total_abs_metric

            return combined

        logger.error("Unknown STOCK_SELECTION_STRATEGY: %s", strategy)
        raise ValueError(f"Unknown STOCK_SELECTION_STRATEGY: {strategy}")

    def _combine_with_momentum(
        base: pd.DataFrame, momentum: pd.DataFrame, base_df: pd.DataFrame, weight: float
    ) -> pd.DataFrame:
        if base.empty and momentum.empty:
            return _empty_like(base_df)

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
            return _empty_like(base_df)

        combined = pd.concat(combined_frames, ignore_index=True)
        if combined.empty:
            return _empty_like(base_df)

        combined["signed_weight"] = np.where(
            combined["action"] == "BUY_LONG", combined["weight"], -combined["weight"]
        )
        agg = combined.groupby("stock_code", as_index=False, observed=True)["signed_weight"].sum()
        agg["weight"] = agg["signed_weight"].abs()
        agg = agg[agg["weight"] > 0]
        if agg.empty:
            return _empty_like(base_df)

        agg["action"] = np.where(agg["signed_weight"] >= 0, "BUY_LONG", "SELL_SHORT")
        total = agg["weight"].sum()
        if total <= 0:
            return _empty_like(base_df)

        agg["weight"] /= total
        agg = agg.drop(columns=["signed_weight"])

        enriched = agg.merge(
            base_df.drop_duplicates("stock_code"),
            on="stock_code",
            how="left",
            suffixes=("", "_cand"),
        )
        return enriched

    candidates_local = candidates.copy()

    momentum_weight = settings.STOCK_SELECTION_MOMENTUM
    use_momentum = momentum_weight > 0 and prev_actuals is not None
    if use_momentum:
        momentum_series = (
            pd.Series(prev_actuals, dtype="float64")
            if not isinstance(prev_actuals, pd.Series)
            else prev_actuals.astype("float64")
        )
        momentum_series.index = momentum_series.index.astype(str)
        candidates_local["momentum"] = (
            candidates_local["stock_code"].astype(str).map(momentum_series)
        )

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
