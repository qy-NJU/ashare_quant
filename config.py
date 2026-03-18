import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
DB_DATA_DIR = os.path.join(DATA_DIR, 'db')

# Logging
LOG_LEVEL = 'INFO'

# Stock Selection Parameters (Example)
MIN_MARKET_CAP = 50e8  # 50亿
MAX_PE = 30
MIN_ROE = 10  # 10%

# Trading parameters
COMMISSION_RATE = 0.0003  # 3/10000 commission
STAMP_DUTY = 0.001        # 1/1000 stamp duty (sell only)
SLIPPAGE = 0.001          # 0.1% slippage
