# Data Fetching
TRAIN_DAYS = '2y'
MIN_CAP = 1e9
TARGET_STOCK_DISTRIBUTION = {
    "Technology": 20,
    "Healthcare": 5,
    "Financial Services": 5,
    "Consumer Defensive": 5,
    "Consumer Cyclical": 5,
    "Industrials": 5,
    "Communication Services": 5
}

# Feature Engineering
DATA_INTERVAL = '1h'
DATA_PER_DAY = 7

# Model Training
THREAD_NUM = 12
N_TRIALS = 100
MAE_THRESHOLD = 0.015
SIGN_ACCURACY_THRESHOLD = 0.50

# Evaluation
TEST_FETCH_DAYS = '3mo'
TEST_DAY_NUMBER = 21
