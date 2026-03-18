import argparse
import pandas as pd
from strategies.momentum_strategy import MomentumStrategy
from backtest.engine import BacktestEngine
from utils import data_loader, mock_data_loader
import config

def main():
    parser = argparse.ArgumentParser(description='A-Share Quant Strategy Runner')
    parser.add_argument('--mock', action='store_true', help='Use mock data for testing')
    parser.add_argument('--start', type=str, default='20230101', help='Start date YYYYMMDD')
    parser.add_argument('--end', type=str, default='20230131', help='End date YYYYMMDD')
    parser.add_argument('--strategy', type=str, default='momentum', help='Strategy name')
    
    args = parser.parse_args()
    
    # Select data source
    if args.mock:
        print("Using MOCK data source.")
        loader = mock_data_loader
    else:
        print("Using REAL data source (AkShare).")
        loader = data_loader
        
    # Select strategy
    if args.strategy == 'momentum':
        strategy = MomentumStrategy(name="Momentum20", period=20, top_n=5)
    else:
        print(f"Unknown strategy: {args.strategy}")
        return
        
    # Run backtest
    print(f"Running backtest from {args.start} to {args.end}...")
    engine = BacktestEngine(strategy, loader, initial_capital=100000.0)
    results = engine.run(args.start, args.end)
    
    # Save results
    results.to_csv('backtest_results.csv')
    print("Results saved to backtest_results.csv")
    
    # Simple Plot (if possible)
    try:
        import matplotlib.pyplot as plt
        results['value'].plot(title='Portfolio Value')
        plt.show()
    except Exception as e:
        print(f"Could not plot results: {e}")

if __name__ == "__main__":
    main()
