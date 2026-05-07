"""
Global configuration file for the A-Share Quant project.
Store your sensitive tokens and global settings here.
"""

import os

# Tushare API Token
# You can get it from https://tushare.pro/register
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "879e1294dee35f3103429bc6cf4afc4b50057eec6e83e24ade3940d4")
