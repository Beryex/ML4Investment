# Data Fetching
TRAIN_DAYS = '15d'
DATA_INTERVAL = '30m'
DATA_PER_DAY = 13

# Feature Engineering
SECTOR_ID_MAP = {
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
    "Others": 12
}
STOCK_SECTOR_ID_MAP = {'MSFT': 1, 'NVDA': 1, 'AAPL': 1, 'AVGO': 1, 'TSM': 1, 'ORCL': 1, 'PLTR': 1, 'ASML': 1, 'CRM': 1, 'NOW': 1, 'AMD': 1, 'UBER': 1, 'ADBE': 1, 'TXN': 1, 'SHOP': 1, 'AMAT': 1, 'PANW': 1, 'ANET': 1, 'LRCX': 1, 'CRWD': 1, 'KLAC': 1, 'INTC': 1, 'CDNS': 1, 'SNPS': 1, 'FTNT': 1, 'SNOW': 1, 'TEAM': 1, 'MRVL': 1, 'NET': 1, 'DDOG': 1, 'FIS': 1, 'TTD': 1, 'ZS': 1, 'SMCI': 1, 'ZM': 1, 'ASX': 1, 'OKTA': 1, 'DOCU': 1, 'TWLO': 1, 'AFRM': 1, 'TER': 1, 'YMM': 1, 'IONQ': 1, 'DBX': 1, 'PATH': 1, 'ENPH': 1, 'BILL': 1, 'KC': 1, 'RGTI': 1, 
                       'LLY': 2, 'JNJ': 2, 'ABBV': 2, 'NVO': 2, 'UNH': 2, 'ABT': 2, 'ISRG': 2, 'TMO': 2, 'BSX': 2, 'SYK': 2, 'DHR': 2, 'PFE': 2, 'MDT': 2, 'HCA': 2, 'ELV': 2, 'CVS': 2, 'ZTS': 2, 'BDX': 2, 'ALC': 2, 'IDXX': 2, 'VEEV': 2, 'CNC': 2, 'HUM': 2, 'IQV': 2, 'NTRA': 2, 'LH': 2, 'DGX': 2, 'BAX': 2, 'WST': 2, 'ILMN': 2, 'HOLX': 2, 'ICLR': 2, 'EXAS': 2, 'MRNA': 2, 'GMED': 2, 'NVCR': 2, 'BEAM': 2, 'TDOC': 2, 
                       'JPM': 3, 'V': 3, 'MA': 3, 'BAC': 3, 'WFC': 3, 'AXP': 3, 'MS': 3, 'GS': 3, 'PGR': 3, 'SCHW': 3, 'SPGI': 3, 'BLK': 3, 'C': 3, 'ICE': 3, 'CME': 3, 'MCO': 3, 'PYPL': 3, 'USB': 3, 'COIN': 3, 'TRV': 3, 'HOOD': 3, 'MET': 3, 'ALL': 3, 'DFS': 3, 'AIG': 3, 'NDAQ': 3, 'MSCI': 3, 'PRU': 3, 'OWL': 3, 'TW': 3, 'RJF': 3, 'STT': 3, 'RKT': 3, 'MKL': 3, 'SYF': 3, 'FUTU': 3, 'SOFI': 3, 'SF': 3, 'MKTX': 3, 'UPST': 3, 
                       'WMT': 4, 'COST': 4, 'KO': 4, 'PEP': 4, 'CL': 4, 'MNST': 4, 'KDP': 4, 'TGT': 4, 'KHC': 4, 'EL': 4, 'TSN': 4, 'BJ': 4, 'EDU': 4, 'TAL': 4, 
                       'AMZN': 5, 'TSLA': 5, 'HD': 5, 'BABA': 5, 'MCD': 5, 'PDD': 5, 'TJX': 5, 'LOW': 5, 'SBUX': 5, 'SE': 5, 'NKE': 5, 'MAR': 5, 'CMG': 5, 'HLT': 5, 'ROST': 5, 'JD': 5, 'TCOM': 5, 'YUM': 5, 'LULU': 5, 'CCL': 5, 'LI': 5, 'DRI': 5, 'XPEV': 5, 'ULTA': 5, 'YUMC': 5, 'RIVN': 5, 'BURL': 5, 'DPZ': 5, 'CHWY': 5, 'TXRH': 5, 'HTHT': 5, 'NIO': 5, 'NCLH': 5, 'FND': 5, 'LCID': 5, 'VIPS': 5, 'ETSY': 5, 'RH': 5, 'PTON': 5, 'QS': 5, 
                       'GE': 6, 'RTX': 6, 'CAT': 6, 'BA': 6, 'HON': 6, 'UNP': 6, 'DE': 6, 'ETN': 6, 'LMT': 6, 'WM': 6, 'UPS': 6, 'MMM': 6, 'RSG': 6, 'EMR': 6, 'NOC': 6, 'CSX': 6, 'FDX': 6, 'NSC': 6, 'PCAR': 6, 'DAL': 6, 'UAL': 6, 'LUV': 6, 'PLUG': 6, 
                       'XOM': 7, 'CVX': 7, 
                       'GOOGL': 8, 'META': 8, 'NFLX': 8, 'DIS': 8, 'SPOT': 8, 'DASH': 8, 'NTES': 8, 'RBLX': 8, 'BIDU': 8, 'WBD': 8, 'SNAP': 8, 'BILI': 8, 'WB': 8, 'IQ': 8, 'MOMO': 8, 
                       'NEE': 9, 
                       'LIN': 11, 'APD': 11, 'FCX': 11, 'DOW': 11, 'CLF': 11,
                       'SPY': 12, 'QQQ': 12, 'VUG': 12, 'EFA': 12, 'GLD': 12, 'VTV': 12, 'DIA': 12, 'KWEB': 12}
CLIP_LOWER_QUANTILE_RATIO = 0.005
CLIP_UPPER_QUANTILE_RATIO = 0.995

# Model Training
TRAINING_DATA_START_DATE = '2016-01-01'
TRAINING_DATA_END_DATE = '2025-01-01'
N_SPLIT = 5
HYPERPARAMETER_SEARCH_LIMIT = 100
FEATURE_SEARCH_LIMIT = 100
SIGN_ACC_THRESHOLD = 0.50
PREDICT_STOCK_NUMBER = 10

# Prediction
NUMBER_OF_STOCKS_TO_BUY = 2
TOTAL_BALANCE = 6500

# Evaluation
TEST_DAY_NUMBER = 33
