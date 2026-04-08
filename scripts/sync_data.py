import os
import sys
import pandas as pd
import datetime
import time

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.source.baostock_source import BaostockSource

LOCAL_LAKE_DIR = 'data/local_lake'
DAILY_K_DIR = os.path.join(LOCAL_LAKE_DIR, 'daily_k')
BASICS_DIR = os.path.join(LOCAL_LAKE_DIR, 'basics')

os.makedirs(DAILY_K_DIR, exist_ok=True)
os.makedirs(BASICS_DIR, exist_ok=True)

def sync_stock_list(source):
    print("Syncing A-share stock list...")
    df = source.get_stock_list()
    if df.empty:
        print("Failed to get stock list.")
        return pd.DataFrame()
        
    save_path = os.path.join(BASICS_DIR, 'stock_list.parquet')
    df.to_parquet(save_path, engine='pyarrow', index=False)
    print(f"Stock list saved to {save_path} ({len(df)} stocks)")
    return df

def sync_daily_data_for_symbol(source, symbol, start_date='2015-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
    save_path = os.path.join(DAILY_K_DIR, f"{symbol}.parquet")
    
    # Check if we have local data for incremental update
    if os.path.exists(save_path):
        try:
            local_df = pd.read_parquet(save_path)
            if not local_df.empty:
                # Find the last date we have data for
                last_date = local_df.index[-1] if isinstance(local_df.index, pd.DatetimeIndex) else pd.to_datetime(local_df['date']).max()
                
                # If the last date is already today or later than end_date, skip
                if pd.to_datetime(last_date).strftime('%Y-%m-%d') >= end_date:
                    # print(f"[{symbol}] Data is up to date (Latest: {last_date.strftime('%Y-%m-%d')})")
                    return
                    
                # We need to fetch from the next day
                start_date = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                # print(f"[{symbol}] Incremental sync from {start_date} to {end_date}")
        except Exception as e:
            print(f"[{symbol}] Failed to read local parquet, will do full sync. Error: {e}")
            local_df = pd.DataFrame()
    else:
        # print(f"[{symbol}] Full sync from {start_date} to {end_date}")
        local_df = pd.DataFrame()
        
    # Fetch new data
    try:
        # Baostock format expects YYYY-MM-DD
        new_df = source.get_daily_data(symbol, start_date=start_date, end_date=end_date, adjust='qfq')
        
        if not new_df.empty:
            # Combine local and new
            if not local_df.empty:
                # Reset index to merge
                if local_df.index.name == 'date':
                    local_df = local_df.reset_index()
                if new_df.index.name == 'date':
                    new_df = new_df.reset_index()
                    
                combined_df = pd.concat([local_df, new_df])
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
                combined_df = combined_df.sort_values('date')
            else:
                combined_df = new_df.reset_index() if new_df.index.name == 'date' else new_df
                
            # Save back
            combined_df.to_parquet(save_path, engine='pyarrow', index=False)
            print(f"[{symbol}] Sync complete. Saved {len(new_df)} new rows. Total: {len(combined_df)}")
        else:
            # print(f"[{symbol}] No new data found for the given period.")
            pass
            
    except Exception as e:
        print(f"[{symbol}] Failed to fetch data: {e}")

def sync_all_daily_data(symbols_limit=None):
    source = BaostockSource()
    
    # 1. Sync stock list
    stock_df = sync_stock_list(source)
    if stock_df.empty:
        return
        
    symbols = stock_df['symbol'].tolist()
    
    # For index
    symbols.append("sh.000300")
    
    if symbols_limit:
        symbols = symbols[:symbols_limit]
        print(f"LIMITING sync to first {symbols_limit} symbols for testing.")
        
    print(f"Starting daily data sync for {len(symbols)} symbols...")
    
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    count = 0
    for sym in symbols:
        count += 1
        if count % 100 == 0:
            print(f"Progress: {count}/{len(symbols)}...")
            
        sync_daily_data_for_symbol(source, sym, start_date='2020-01-01', end_date=end_date)
        
        # Simple rate limiting for Baostock
        if count % 50 == 0:
            time.sleep(1)
            
    print("All sync tasks completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sync A-Share data to Local Data Lake")
    parser.add_argument('--limit', type=int, default=None, help='Limit number of symbols to sync (for testing)')
    args = parser.parse_args()
    
    sync_all_daily_data(symbols_limit=args.limit)
