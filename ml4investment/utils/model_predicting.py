import datetime
import logging
import os
import random
import time
from collections import defaultdict
from typing import cast

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests.exceptions
import schwabdev
import shap
from dotenv import load_dotenv
from prettytable import PrettyTable
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from ml4investment.config.global_settings import settings

logger = logging.getLogger(__name__)


def get_stocks_portfolio(candidates: pd.DataFrame) -> pd.DataFrame:
    """Select candidates under the configured strategy and allocate weights."""
    if "prediction" not in candidates.columns:
        raise ValueError("Column 'prediction' not found in candidates DataFrame.")

    def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
        result = df.head(0).copy()
        result["weight"] = pd.Series(dtype="float64")
        result["action"] = pd.Series(dtype="object")
        return result

    def _select_direction(direction: str) -> pd.DataFrame:
        if direction == "BUY_LONG":
            eligible = candidates[candidates["prediction"] > 0.0]
            ascending = False
        elif direction == "SELL_SHORT":
            eligible = candidates[candidates["prediction"] < 0.0]
            ascending = True
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        if eligible.empty:
            return _empty_like(candidates)

        selected = eligible.sort_values(by="prediction", ascending=ascending).head(
            settings.NUMBER_OF_STOCKS_TO_BUY
        ).copy()
        if selected.empty:
            return _empty_like(candidates)

        total_abs_prediction = float(selected["prediction"].abs().sum())
        if total_abs_prediction <= 0:
            selected["weight"] = 1.0 / len(selected)
        else:
            selected["weight"] = selected["prediction"].abs() / total_abs_prediction

        selected["action"] = direction
        return selected

    strategy = settings.STOCK_SELECTION_STRATEGY

    if strategy == "BUY_LONG":
        return _select_direction("BUY_LONG")

    elif strategy == "SELL_SHORT":
        return _select_direction("SELL_SHORT")

    elif strategy == "ADAPT":
        if candidates.empty:
            return _empty_like(candidates)

        positive_ratio = (candidates["prediction"] > 0).mean()
        primary = "BUY_LONG" if positive_ratio >= 0.5 else "SELL_SHORT"
        portfolio = _select_direction(primary)
        if portfolio.empty:
            alternate = "SELL_SHORT" if primary == "BUY_LONG" else "BUY_LONG"
            portfolio = _select_direction(alternate)
        return portfolio

    elif strategy == "BOTH":
        non_zero = candidates[candidates["prediction"] != 0]
        if non_zero.empty:
            return _empty_like(candidates)

        abs_sorted = non_zero.assign(_abs_pred=non_zero["prediction"].abs()).sort_values(
            by="_abs_pred", ascending=False
        )
        selected = abs_sorted.head(settings.NUMBER_OF_STOCKS_TO_BUY).drop(columns="_abs_pred").copy()
        if selected.empty:
            return _empty_like(candidates)

        total_abs_prediction = float(selected["prediction"].abs().sum())
        if total_abs_prediction <= 0:
            selected["weight"] = 1.0 / len(selected)
        else:
            selected["weight"] = selected["prediction"].abs() / total_abs_prediction

        selected["action"] = np.where(selected["prediction"] > 0, "BUY_LONG", "SELL_SHORT")
        return selected

    elif strategy == "BUY_LONG_FIRST":
        long_candidates = candidates[candidates["prediction"] > 0.0].copy()

        long_selected = long_candidates.sort_values(by="prediction", ascending=False).head(
            settings.NUMBER_OF_STOCKS_TO_BUY
        )

        remaining_slots = settings.NUMBER_OF_STOCKS_TO_BUY - len(long_selected)

        short_selected = pd.DataFrame()
        if remaining_slots > 0:
            short_candidates = candidates[candidates["prediction"] < 0.0].copy()
            if not short_candidates.empty:
                short_selected = short_candidates.sort_values(by="prediction", ascending=True).head(
                    remaining_slots
                )

        combined = pd.concat([long_selected, short_selected])
        if combined.empty:
            return _empty_like(candidates)

        combined["action"] = np.where(combined["prediction"] > 0, "BUY_LONG", "SELL_SHORT")
        total_abs_prediction = float(combined["prediction"].abs().sum())
        if total_abs_prediction <= 0:
            combined["weight"] = 1.0 / len(combined)
        else:
            combined["weight"] = combined["prediction"].abs() / total_abs_prediction

        return combined

    else:
        logger.error(f"Unknown STOCK_SELECTION_STRATEGY: {strategy}")
        raise ValueError(f"Unknown STOCK_SELECTION_STRATEGY: {strategy}")
