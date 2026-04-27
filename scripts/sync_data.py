import os
import sys
import pandas as pd
import datetime
import time
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.source.baostock_source import BaostockSource

import akshare as ak

LOCAL_LAKE_DIR = 'data/local_lake'
DAILY_K_DIR = os.path.join(LOCAL_LAKE_DIR, 'daily_k')
BASICS_DIR = os.path.join(LOCAL_LAKE_DIR, 'basics')
EVENTS_DIR = os.path.join(LOCAL_LAKE_DIR, 'events')
FINANCIALS_DIR = os.path.join(LOCAL_LAKE_DIR, 'financials')
FUNDS_DIR = os.path.join(LOCAL_LAKE_DIR, 'funds')

os.makedirs(DAILY_K_DIR, exist_ok=True)
os.makedirs(BASICS_DIR, exist_ok=True)
os.makedirs(EVENTS_DIR, exist_ok=True)
os.makedirs(FINANCIALS_DIR, exist_ok=True)
os.makedirs(FUNDS_DIR, exist_ok=True)

def sync_fund_flow_for_symbol(symbol):
    cache_file = os.path.join(FUNDS_DIR, f"fund_{symbol}.parquet")
    if os.path.exists(cache_file):
        return
    try:
        df = ak.stock_individual_fund_flow(symbol="sh600000", market="sh") # This is just an example, akshare is complicated
        # Actually AkShareSource has a get_fund_flow method
        from data.source.akshare_source import AkShareSource
        source = AkShareSource()
        df = source.get_fund_flow(symbol)
        if not df.empty:
            df.to_parquet(cache_file, engine='pyarrow', index=True)
    except Exception as e:
        pass

def sync_stock_list(source, force_update=False):
    save_path = os.path.join(BASICS_DIR, 'stock_list.parquet')
    
    if not force_update and os.path.exists(save_path):
        # Check if the file is less than 7 days old
        file_mtime = os.path.getmtime(save_path)
        if (time.time() - file_mtime) < 7 * 24 * 3600:
            print(f"Loading stock list from local cache: {save_path} (Skipping API call)")
            try:
                return pd.read_parquet(save_path)
            except Exception as e:
                print(f"Failed to read local stock list: {e}. Will fetch from API.")
                
    print("Syncing A-share stock list from API...")
    df = source.get_stock_list()
    if df.empty:
        print("Failed to get stock list.")
        return pd.DataFrame()
        
    df.to_parquet(save_path, engine='pyarrow', index=False)
    print(f"Stock list saved to {save_path} ({len(df)} stocks)")
    return df

def sync_industry_mapping(force_update=False):
    save_path = os.path.join(BASICS_DIR, 'industry_map.parquet')
    
    if not force_update and os.path.exists(save_path):
        file_mtime = os.path.getmtime(save_path)
        if (time.time() - file_mtime) < 7 * 24 * 3600:
            print(f"Loading industry mapping from local cache: {save_path} (Skipping API call)")
            try:
                return pd.read_parquet(save_path)
            except Exception as e:
                print(f"Failed to read local industry mapping: {e}. Will fetch from API.")

    print("Syncing Industry Classification from Baostock API...")
    import baostock as bs
    
    # 注意：不要在这里单独 login/logout，因为 baostock 使用全局 socket 连接
    # 多次 login/logout 会导致 socket 被关闭，影响其他数据源的正常使用
    # 直接查询即可，baostock 会自动复用已有的连接
    rs = bs.query_stock_industry()
    industry_data = []
    while (rs.error_code == '0') & rs.next():
        industry_data.append(rs.get_row_data())
    
    if rs.error_code != '0':
        print(f"Industry query failed: error_code={rs.error_code}, error_msg={rs.error_msg}")
        return pd.DataFrame()
        
    if industry_data:
        df = pd.DataFrame(industry_data, columns=rs.fields)
        df['symbol'] = df['code'].apply(lambda x: x.split('.')[1] if '.' in x else x)
        df[['symbol', 'industry']].to_parquet(save_path, engine='pyarrow', index=False)
        print(f"Industry mapping saved to {save_path} ({len(df)} records)")
        return df[['symbol', 'industry']]
    else:
        print("No industry data found.")
        return pd.DataFrame()

def sync_financial_data_for_symbol(source, symbol, start_year=2015, end_year=None):
    if end_year is None:
        end_year = datetime.datetime.now().year

    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            # Check if this quarter is in the future
            if year == datetime.datetime.now().year and quarter > (datetime.datetime.now().month - 1) // 3 + 1:
                continue
                
            cache_file = os.path.join(FINANCIALS_DIR, f"fin_{symbol}_{year}_{quarter}.parquet")
            if os.path.exists(cache_file):
                continue
                
            df_profit = source.get_profit_data(symbol, year, quarter)
            df_growth = source.get_growth_data(symbol, year, quarter)
            
            if df_profit.empty and df_growth.empty:
                continue
                
            if df_profit.empty:
                df_merged = df_growth
            elif df_growth.empty:
                df_merged = df_profit
            else:
                common_cols = list(set(df_profit.columns) & set(df_growth.columns))
                df_merged = pd.merge(df_profit, df_growth, on=common_cols, how='outer')
                
            if not df_merged.empty:
                df_merged.to_parquet(cache_file, engine='pyarrow', index=False)

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

def sync_lhb_data(start_date='20200101', end_date=None):
    """
    Sync Longhu Bang (Dragon Tiger List / Institutional & Hot Money trading) data from AkShare.
    """
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y%m%d')
        
    print(f"Syncing Longhu Bang (LHB) data from {start_date} to {end_date}...")
    save_path = os.path.join(EVENTS_DIR, 'lhb_data.parquet')
    
    local_df = pd.DataFrame()
    if os.path.exists(save_path):
        try:
            local_df = pd.read_parquet(save_path)
            if not local_df.empty:
                date_col = '上榜日' if '上榜日' in local_df.columns else '上榜日期'
                last_date = pd.to_datetime(local_df[date_col]).max().strftime('%Y%m%d')
                if last_date >= end_date:
                    print("LHB data is already up to date.")
                    return
                # Start fetching from the day after the last date
                start_date = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y%m%d')
                print(f"Incremental LHB sync from {start_date} to {end_date}")
        except Exception as e:
            print(f"Failed to read local LHB parquet, doing full sync. Error: {e}")
            
    # Fetch from AkShare
    try:
        # stock_lhb_detail_em returns data for a specific date range
        new_df = ak.stock_lhb_detail_em(start_date=start_date, end_date=end_date)
        
        if not new_df.empty:
            if not local_df.empty:
                combined_df = pd.concat([local_df, new_df])
                date_col_combined = '上榜日' if '上榜日' in combined_df.columns else '上榜日期'
                combined_df = combined_df.drop_duplicates(subset=[date_col_combined, '代码'], keep='last')
                combined_df = combined_df.sort_values(date_col_combined)
            else:
                combined_df = new_df
                
            combined_df.to_parquet(save_path, engine='pyarrow', index=False)
            print(f"LHB Sync complete. Saved {len(new_df)} new rows. Total: {len(combined_df)}")
        else:
            print("No new LHB data found.")
    except Exception as e:
        print(f"Failed to fetch LHB data: {e}")


def sync_all_daily_data(symbols_limit=None, start_date='2020-01-01', source_name='baostock', tushare_token=None, force_update=False):
    if source_name.lower() == 'tushare':
        from data.source.tushare_source import TushareSource
        source = TushareSource(token=tushare_token)
        print("Using Tushare as the primary data source.")
    else:
        from data.source.baostock_source import BaostockSource
        source = BaostockSource()
        print("Using Baostock as the primary data source.")
    
    # 1. Sync stock list
    stock_df = sync_stock_list(source, force_update=force_update)
    if stock_df.empty:
        return
        
    # 2. Sync Industry Mapping
    save_path = os.path.join(BASICS_DIR, 'industry_map.parquet')
    if not force_update and os.path.exists(save_path):
        file_mtime = os.path.getmtime(save_path)
        if (time.time() - file_mtime) < 7 * 24 * 3600:
            print(f"Loading industry mapping from local cache: {save_path} (Skipping API call)")
        else:
            if source_name.lower() == 'tushare' and hasattr(source, 'get_industry_mapping'):
                print("Syncing Industry Classification from Tushare API...")
                ind_df = source.get_industry_mapping()
                if not ind_df.empty:
                    ind_df.to_parquet(save_path, engine='pyarrow', index=False)
                    print(f"Industry mapping saved to {save_path} ({len(ind_df)} records)")
            else:
                sync_industry_mapping(force_update=True)
    else:
        if source_name.lower() == 'tushare' and hasattr(source, 'get_industry_mapping'):
            print("Syncing Industry Classification from Tushare API...")
            ind_df = source.get_industry_mapping()
            if not ind_df.empty:
                ind_df.to_parquet(save_path, engine='pyarrow', index=False)
                print(f"Industry mapping saved to {save_path} ({len(ind_df)} records)")
        else:
            sync_industry_mapping(force_update=True)
        
    symbols = stock_df['symbol'].tolist()
    
    # For index
    symbols.append("sh.000300")
    
    if symbols_limit:
        symbols = symbols[:symbols_limit]
        print(f"LIMITING sync to first {symbols_limit} symbols for testing.")
        
    print(f"Starting daily data sync for {len(symbols)} symbols...")
    
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_year = int(start_date.split('-')[0])
    end_year = int(end_date.split('-')[0])
    
    if source_name.lower() == 'tushare' and hasattr(source, 'get_daily_data_batch'):
        print(f"Starting BATCH sync for {len(symbols)} symbols using Tushare...")
        success_count = 0
        failed_symbols = []
        
        # 1. Determine needed start_date for each symbol
        from collections import defaultdict
        date_groups = defaultdict(list)
        
        print("Checking local data status...")
        for sym in tqdm(symbols, desc="Checking status"):
            save_path = os.path.join(DAILY_K_DIR, f"{sym}.parquet")
            needed_start = start_date
            if os.path.exists(save_path):
                try:
                    local_df = pd.read_parquet(save_path)
                    if not local_df.empty:
                        last_date = local_df.index[-1] if isinstance(local_df.index, pd.DatetimeIndex) else pd.to_datetime(local_df['date']).max()
                        last_date_str = pd.to_datetime(last_date).strftime('%Y-%m-%d')
                        if last_date_str < end_date:
                            needed_start = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                        else:
                            needed_start = None # up to date
                except:
                    pass
            date_groups[needed_start].append(sym)
            
        # 2. Process each date group
        for needed_start, syms in date_groups.items():
            if needed_start is None:
                # Already up to date, just sync financials/funds
                for sym in syms:
                    try:
                        sync_financial_data_for_symbol(source, sym, start_year=start_year, end_year=end_year)
                        sync_fund_flow_for_symbol(sym)
                        success_count += 1
                    except Exception as e:
                        failed_symbols.append((sym, str(e)))
                continue
                
            # Calculate batch size for this needed_start
            start_dt = pd.to_datetime(needed_start)
            end_dt = pd.to_datetime(end_date)
            days = max(1, (end_dt - start_dt).days)
            trading_days = max(1, int(days * 250 / 365))
            
            # Max rows per request is 6000. Use 4000 as safety margin.
            batch_size = max(1, int(4000 / trading_days))
            batch_size = min(batch_size, 200) # Cap at 200 codes per request
            
            for i in tqdm(range(0, len(syms), batch_size), desc=f"Batch sync from {needed_start}"):
                batch_symbols = syms[i:i+batch_size]
                try:
                    batch_data = source.get_daily_data_batch(batch_symbols, start_date=needed_start, end_date=end_date)
                    
                    for sym in batch_symbols:
                        save_path = os.path.join(DAILY_K_DIR, f"{sym}.parquet")
                        new_df = batch_data.get(sym, pd.DataFrame())
                        
                        local_df = pd.DataFrame()
                        if os.path.exists(save_path):
                            try:
                                local_df = pd.read_parquet(save_path)
                            except:
                                pass
                                
                        if not new_df.empty:
                            if not local_df.empty:
                                if local_df.index.name == 'date':
                                    local_df = local_df.reset_index()
                                if new_df.index.name == 'date':
                                    new_df = new_df.reset_index()
                                combined_df = pd.concat([local_df, new_df])
                                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
                                combined_df = combined_df.sort_values('date')
                            else:
                                combined_df = new_df.reset_index() if new_df.index.name == 'date' else new_df
                            combined_df.to_parquet(save_path, engine='pyarrow', index=False)
                            
                        # Financials and fund flow
                        sync_financial_data_for_symbol(source, sym, start_year=start_year, end_year=end_year)
                        sync_fund_flow_for_symbol(sym)
                        success_count += 1
                        
                    time.sleep(0.15) # Rate limit for Tushare
                except Exception as e:
                    for sym in batch_symbols:
                        failed_symbols.append((sym, str(e)))
                        
        print(f"All sync tasks completed. Success: {success_count}, Failed: {len(failed_symbols)}")
        if failed_symbols:
            print(f"Sample failures: {failed_symbols[:5]}")
        return

    # Sequential sync (Fallback for baostock or if no batch method)
    print(f"Starting sequential sync for {len(symbols)} symbols...")

    success_count = 0
    failed_symbols = []

    for sym in tqdm(symbols, desc="Syncing", unit="stock"):
        try:
            sync_daily_data_for_symbol(source, sym, start_date=start_date, end_date=end_date)
            sync_financial_data_for_symbol(source, sym, start_year=start_year, end_year=end_year)
            sync_fund_flow_for_symbol(sym)
            success_count += 1
            
            # Rate limiting for Tushare to avoid ban (approx 5 requests per second)
            if source_name.lower() == 'tushare':
                time.sleep(0.2)
                
        except Exception as e:
            failed_symbols.append((sym, str(e)))

    print(f"All sync tasks completed. Success: {success_count}, Failed: {len(failed_symbols)}")
    if failed_symbols:
        print(f"Sample failures: {failed_symbols[:5]}")

if __name__ == "__main__":
    import argparse
    import yaml
    
    # Try to load config
    tushare_token = None
    default_source = 'baostock'
    try:
        import config
        tushare_token = getattr(config, 'TUSHARE_TOKEN', None)
        if tushare_token == "your_tushare_token_here":
            tushare_token = None
    except ImportError:
        pass
        
    try:
        with open('configs/pipeline_config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
            default_source = cfg.get('data', {}).get('sync_source', 'baostock')
    except:
        pass
        
    parser = argparse.ArgumentParser(description="Sync A-Share data to Local Data Lake")
    parser.add_argument('--limit', type=int, default=None, help='Limit number of symbols to sync (for testing)')
    parser.add_argument('--events_only', action='store_true', help='Only sync event data like LHB')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date for syncing data (YYYY-MM-DD)')
    parser.add_argument('--source', type=str, default=default_source, choices=['baostock', 'tushare'], help='Data source to use')
    parser.add_argument('--token', type=str, default=tushare_token, help='Tushare token (if source is tushare)')
    parser.add_argument('--force', action='store_true', help='Force update of basic info like stock list and industry map')
    args = parser.parse_args()
    
    # LHB data expects YYYYMMDD format
    lhb_start_date = args.start_date.replace('-', '')
    sync_lhb_data(start_date=lhb_start_date)
    
    if not args.events_only:
        sync_all_daily_data(symbols_limit=args.limit, start_date=args.start_date, source_name=args.source, tushare_token=args.token, force_update=args.force)
