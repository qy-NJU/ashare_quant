import pandas as pd
import numpy as np
import os
from data.repository import DataRepository
from data.source.baostock_source import BaostockSource
from features.pipeline import FeaturePipeline
from features.factors.pandas_ta_factor import PandasTAFactor
from features.factors.technical import LabelGenerator
from models.xgboost_model import XGBoostWrapper
from strategies.ml_strategy import MLStrategy
from backtest.engine import BacktestEngine

def run_v3_pipeline():
    print("=== V3.0 Pipeline: Baostock + Parquet + PandasTA + XGBoost Incremental ===")
    
    # 1. Initialize Data Repository with Baostock
    repo = DataRepository([BaostockSource()], cache_dir='data/cache')
    
    # Get a few stocks for testing (Baostock returns HS300)
    stocks_df = repo.get_stock_list()
    if stocks_df.empty:
        print("Failed to get stock list. Check network or login.")
        return
        
    test_symbols = stocks_df['symbol'].head(3).tolist()
    print(f"Testing with symbols: {test_symbols}")
    
    # 2. Setup Feature Pipeline
    pipeline = FeaturePipeline([
        PandasTAFactor(), # SMA, RSI, MACD, BBands
        LabelGenerator(horizon=3) # Target: 3-day future return
    ])
    
    # 3. Data Processing & Incremental Training Loop
    model = XGBoostWrapper()
    
    # Define time periods to simulate incremental flow
    periods = [
        ('2023-01-01', '2023-03-31'), # Base training
        ('2023-04-01', '2023-06-30')  # Incremental update
    ]
    
    for i, (start, end) in enumerate(periods):
        print(f"\n--- Processing Period {i+1}: {start} to {end} ---")
        X_batch, y_batch = [], []
        
        for symbol in test_symbols:
            # Data will be cached in Parquet automatically
            df = repo.get_daily_data(symbol, start, end)
            if df.empty or len(df) < 30:
                continue
                
            features_df = pipeline.transform(df)
            if features_df.empty:
                continue
                
            target_col = 'target_3d'
            features_df = features_df.dropna(subset=[target_col])
            
            X = features_df.drop(columns=[target_col, 'open', 'high', 'low', 'close', 'volume', 'date'], errors='ignore')
            y = features_df[target_col]
            
            X_batch.append(X)
            y_batch.append(y)
            
        if not X_batch:
            print("No valid data in this period.")
            continue
            
        X_full = pd.concat(X_batch)
        y_full = pd.concat(y_batch)
        
        # Train / Incremental Train
        if i == 0:
            print(f"Base Training on {len(X_full)} samples...")
            model.train(X_full, y_full)
        else:
            print(f"Incremental Training on new {len(X_full)} samples...")
            model.partial_train(X_full, y_full)
            
    # Save Model
    os.makedirs('models/saved', exist_ok=True)
    model.save('models/saved/xgb_incremental.json')
    
    # 4. Strategy & Backtest (Out of sample)
    print("\n--- Running Backtest (Out of Sample) ---")
    prediction_pipeline = FeaturePipeline([PandasTAFactor()])
    
    strategy = MLStrategy(
        name="XGBoost_TA",
        model=model,
        feature_pipeline=prediction_pipeline,
        top_k=1
    )
    
    # Use repo for backtest data
    engine = BacktestEngine(strategy, repo, initial_capital=100000.0)
    # We must pass the exact symbols to the strategy if we don't want it to query all
    # For demo, we patch the mock list logic inside MLStrategy implicitly by overriding repo
    
    # Backtest Period
    results = engine.run('20230701', '20230715')
    print(results.head())

if __name__ == "__main__":
    run_v3_pipeline()
