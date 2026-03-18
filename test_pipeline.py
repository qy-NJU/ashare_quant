import pandas as pd
from strategies.momentum_strategy import MomentumStrategy
from backtest.engine import BacktestEngine
from utils import mock_data_loader

def run_test():
    print("Initializing Strategy...")
    strategy = MomentumStrategy(name="TestMomentum", period=5, top_n=2)
    
    print("Initializing Backtest Engine...")
    engine = BacktestEngine(strategy, mock_data_loader, initial_capital=100000.0)
    
    print("Running Backtest...")
    # Run for a short period
    start_date = "20230101"
    end_date = "20230110"
    
    results = engine.run(start_date, end_date)
    
    print("\nBacktest Results Head:")
    print(results.head())

if __name__ == "__main__":
    run_test()
