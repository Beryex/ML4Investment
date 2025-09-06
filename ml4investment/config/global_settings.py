import json
import os
from typing import Dict, List, Optional


class Settings:
    PROJECT_NAME: str = "ml4investment"
    MAX_NUM_PROCESSES: int = 10
    SEED: int = 42

    # Data Fetching
    AVAILABLE_STOCK_SOURCE: str = "https://companiesmarketcap.com/usa/largest-companies-in-the-usa-by-market-cap/?download=csv"
    MIN_MARKET_CAP: float = 1e9
    SELECTIVE_ETF: list[str] = ["QQQ", "SPY", "DIA", "VTV", "VUG", "GLD", "EFA", "KWEB"]
    FETCH_PERIOD_DAYS: int = 10
    DATA_INTERVAL_MINS: int = 30
    DATA_PER_DAY: int = 13

    # Feature Engineering
    CATEGORICAL_FEATURES: List[str] = ["stock_id", "stock_sector"]
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
    STOCK_SECTOR_ID_MAP_PTH: str = "config/stock_sector_id_mapping.json"
    STOCK_SECTOR_ID_MAP: Dict[str, int] = json.load(open(STOCK_SECTOR_ID_MAP_PTH, "r"))
    APPLY_CLIP: bool = False
    APPLY_SCALE: bool = False
    CLIP_LOWER_QUANTILE_RATIO: float = 0.005
    CLIP_UPPER_QUANTILE_RATIO: float = 0.995

    # Model Training
    TRAINING_DATA_START_DATE: str = os.getenv("TRAIN_START_DATE", "2023-11-30")
    TRAINING_DATA_END_DATE: str = "2024-11-30"
    VALIDATION_DATA_START_DATE: str = "2024-12-01"
    VALIDATION_DATA_END_DATE: str = "2025-05-31"
    N_SPLIT: int = 5
    NUM_ROUNDS: int = 1000
    WARMUP_ROUNDS: int = 100
    DATA_SAMPLING_PROPORTION_SEARCH_LIMIT: int = 100
    HYPERPARAMETER_SEARCH_LIMIT: int = 100
    FEATURE_SEARCH_LIMIT: int = 100
    PREDICT_STOCK_SEARCH_LIMIT: int = 20
    FIXED_TRAINING_CONFIG: dict = {
        "objective": "regression_l1",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": "dart",
        "num_rounds": NUM_ROUNDS,
        "feature_fraction": 1.0,
        "bagging_freq": 0,
        "bagging_fraction": 1.0,
        "force_row_wise": True,
        "deterministic": True,
    }

    # Prediction
    PREDICT_STOCK_NUMBER: int = 6
    NUMBER_OF_STOCKS_TO_BUY: int = 1
    OPENING_STATUS: set[str] = {"PENDING_ACTIVATION", "WORKING", "OPEN", "QUEUED"}

    # Evaluation
    TESTING_DATA_START_DATE: str = "2025-06-01"
    TESTING_DATA_END_DATE: Optional[str] = None


settings = Settings()
