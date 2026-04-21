import pandas as pd
from scripts.sync_data import sync_stock_list, sync_fund_flow_for_symbol
from data.source.baostock_source import BaostockSource
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

source = BaostockSource()
stock_df = sync_stock_list(source)
symbols = stock_df['symbol'].tolist()
print(f"Syncing fund flow data for {len(symbols)} symbols...")
completed_fund = 0
with ThreadPoolExecutor(max_workers=20) as executor:
    fund_futures = {executor.submit(sync_fund_flow_for_symbol, sym): sym for sym in symbols}
    for future in as_completed(fund_futures):
        completed_fund += 1
        if completed_fund % 100 == 0:
            print(f"Fund Flow Progress: {completed_fund}/{len(symbols)}...")
print("All fund sync tasks completed.")
