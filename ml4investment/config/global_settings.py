# Data Fetching
TRAIN_DAYS = '2y'
MIN_CAP = 1e9
TARGET_STOCK_DISTRIBUTION = {
    "Technology": 20,
    "Healthcare": 3,
    "Financial Services": 3,
    "Consumer Defensive": 3,
    "Consumer Cyclical": 3,
    "Industrials": 3,
    "Energy": 3,
    "Communication Services": 3,
    "Utilities": 3,
    "Real Estate": 3,
    "Basic Materials": 3
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
NUMBER_OF_STOCKS_TO_BUY = 3
