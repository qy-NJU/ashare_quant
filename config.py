"""
Global configuration file for the A-Share Quant project.
Store your sensitive tokens and global settings here.
"""

import os

# Tushare API Token
# You can get it from https://tushare.pro/register
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "058107e25020ebc9be4568849327db2c2295d7e6606df6e25536a004")
