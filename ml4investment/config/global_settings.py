import json
import os
from pathlib import Path
from typing import Dict, List, Optional

CONFIG_DIR = Path(__file__).resolve().parent


class Settings:
    PROJECT_NAME: str = "ml4investment"
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")
    MAX_NUM_PROCESSES: int = 10
    SEED: int = 42

    # Data Fetching
    AVAILABLE_STOCK_SOURCE: str = "https://companiesmarketcap.com/usa/largest-companies-in-the-usa-by-market-cap/?download=csv"
    MIN_MARKET_CAP: float = 1e9
    SELECTIVE_ETF: list[str] = ["QQQ", "SPY", "DIA", "VTV", "VUG", "GLD", "EFA", "KWEB"]
    FETCH_PERIOD_DAYS: int = 30
    DATA_INTERVAL_MINS: int = 30
    DATA_PER_DAY: int = 13
    TRADING_DAYS_PER_YEAR: int = 252

    # Feature Engineering
    CATEGORICAL_FEATURES: List[str] = ["stock_id", "sector_id"]
    SECTOR_ID_MAP: Dict[str, int] = {
        "Technology": 1,
        "Healthcare": 2,
        "Financial Services": 3,
        "Consumer Defensive": 4,
        "Consumer Cyclical": 5,
        "Industrials": 6,
        "Energy": 7,
        "Communication Services": 8,
        "Utilities": 9,
        "Real Estate": 10,
        "Basic Materials": 11,
        "Others": 12,
    }
    STOCK_SECTOR_ID_MAP_PTH = os.getenv(
        "STOCK_SECTOR_ID_MAP_PTH",
        str(CONFIG_DIR / "stock_sector_id_mapping.json"),
    )
    STOCK_SECTOR_ID_MAP: Dict[str, int] = json.load(open(STOCK_SECTOR_ID_MAP_PTH, "r"))
    APPLY_CLIP: str = os.getenv("APPLY_CLIP", "skip")
    APPLY_SCALE: str = os.getenv("APPLY_SCALE", "skip")
    CLIP_LOWER_QUANTILE_RATIO: float = 0.005
    CLIP_UPPER_QUANTILE_RATIO: float = 0.995

    # Model Training
    TRAINING_DATA_START_DATE: str = os.getenv("TRAIN_START_DATE", "2023-11-30")
    TRAINING_DATA_END_DATE: str = "2024-11-30"
    VALIDATION_DATA_START_DATE: str = "2024-12-01"
    VALIDATION_DATA_END_DATE: str = "2025-05-31"
    N_SPLIT: int = 5
    NUM_ROUNDS: int = int(os.getenv("NUM_ROUNDS", 1000))
    TRAIN_OBJECTIVE: str = os.getenv("TRAIN_OBJECTIVE", "regression_l1")
    OPTIMIZE_METRIC: str = os.getenv("OPTIMIZE_METRIC", "l1")
    FIXED_TRAINING_CONFIG: dict = {
        "objective": TRAIN_OBJECTIVE,
        "metric": OPTIMIZE_METRIC,
        "verbosity": -1,
        "boosting_type": "dart",
        "num_rounds": NUM_ROUNDS,
        "feature_fraction": 1.0,
        "bagging_freq": 0,
        "bagging_fraction": 1.0,
        "force_row_wise": True,
        "deterministic": True,
    }

    ITERATIVE_OPTIMIZATION_STEPS: int = int(os.getenv("ITERATIVE_OPTIMIZATION_STEPS", 1))
    PRUNING_WARMUP_STEPS: int = int(os.getenv("PRUNING_WARMUP_STEPS", 1001))
    DATA_OPTIMIZATION_SAMPLING_MULTIVARIATE: bool = (
        os.getenv("DATA_OPTIMIZATION_SAMPLING_MULTIVARIATE", "true").lower() == "true"
    )
    DATA_SAMPLING_PROPORTION_SEARCH_LIMIT: int = int(
        os.getenv("DATA_SAMPLING_PROPORTION_SEARCH_LIMIT", 100)
    )
    FEATURE_OPTIMIZATION_SAMPLING_MULTIVARIATE: bool = (
        os.getenv("FEATURE_OPTIMIZATION_SAMPLING_MULTIVARIATE", "true").lower() == "true"
    )
    FEATURE_SEARCH_LIMIT: int = int(os.getenv("FEATURE_SEARCH_LIMIT", 100))
    MODEL_OPTIMIZATION_SAMPLING_MULTIVARIATE: bool = (
        os.getenv("MODEL_OPTIMIZATION_SAMPLING_MULTIVARIATE", "true").lower() == "true"
    )
    HYPERPARAMETER_SEARCH_LIMIT: int = int(os.getenv("HYPERPARAMETER_SEARCH_LIMIT", 100))
    PREDICT_STOCK_OPTIMIZE_METRIC: str = "sharpe_ratio"
    PREDICT_STOCK_OPTIMIZE_MAX_DRAWDOWN_THRESHOLD: float = 0.10
    PREDICT_STOCK_NUMBER: int = 6

    # Prediction
    # strategy chosen from BUY_LONG, SELL_SHORT, ADAPT, BOTH, BUY_LONG_FIRST
    STOCK_SELECTION_STRATEGY: str = os.getenv("STOCK_SELECTION_STRATEGY", "BUY_LONG_FIRST")
    STOCK_SELECTION_MOMENTUM: float = min(
        max(float(os.getenv("STOCK_SELECTION_MOMENTUM", 0.0)), 0.0), 1.0
    )
    NUMBER_OF_STOCKS_TO_BUY: int = 1
    OPENING_STATUS: set[str] = {"PENDING_ACTIVATION", "WORKING", "OPEN", "QUEUED"}

    # Evaluation
    TESTING_DATA_START_DATE: str = "2025-06-01"
    TESTING_DATA_END_DATE: Optional[str] = None
    SHAP_PLOT_MAX_DISPLAY_FEATURES: int = 20
    SHAP_SUMMARY_GLOBAL_IMG_PTH_TPL: str = "data/shap_summary_global_{}.png"
    SHAP_SUMMARY_ERROR_IMG_PTH_TPL: str = "data/shap_summary_errors_{}.png"
    SHAP_SUMMARY_CORRECT_IMG_PATH_TPL: str = "data/shap_summary_correct_{}.png"


settings = Settings()
