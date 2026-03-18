import pandas as pd
from features.pipeline import FeaturePipeline
from features.factors.technical import TechnicalFactors, LabelGenerator
from data.repository import DataRepository
from data.source.mock_source import MockDataSource
from models.machine_learning import SklearnWrapper
from strategies.ml_strategy import MLStrategy
from backtest.engine import BacktestEngine
import datetime

def integration_test():
    print("Initializing Data Repository...")
    # Add sources (Priority: Mock)
    sources = [MockDataSource()]
    repo = DataRepository(sources)
    
    print("Initializing Feature Pipeline...")
    # Create feature definitions
    pipeline = FeaturePipeline([
        TechnicalFactors(), # MA5, MA20, ROC, Vol
        LabelGenerator(horizon=5) # Target return
    ])
    
    print("Training Model...")
    # 1. Prepare Training Data
    # Fetch historical data for a few stocks to train
    stocks = repo.get_stock_list()
    if stocks.empty:
        print("No stocks found.")
        return
        
    train_start = '20220101'
    train_end = '20221231'
    
    X_train = []
    y_train = []
    
    for symbol in stocks['symbol']:
        df = repo.get_daily_data(symbol, train_start, train_end)
        if df.empty:
            continue
            
        # Transform data to features + label
        # The pipeline calculates both features and label
        # But we need to separate X and y
        # However, our pipeline implementation just concat everything
        # We need to manually separate later or modify pipeline
        
        # Let's assume pipeline returns everything including target
        # And we know target column name is 'target_5d'
        
        # Quick hack: modify pipeline or handle here
        # Actually pipeline returns all columns
        # We need to instantiate factors separately? No, pipeline is fine.
        
        # The LabelGenerator adds 'target_5d'
        # TechnicalFactors adds 'ma5', 'ma20', 'roc5', 'vol20'
        
        features_df = pipeline.transform(df)
        if features_df.empty:
            continue
            
        target_col = 'target_5d'
        if target_col not in features_df.columns:
            continue
            
        # Drop rows where target is NaN (last 5 days)
        features_df = features_df.dropna(subset=[target_col])
        
        X = features_df.drop(columns=[target_col, 'open', 'high', 'low', 'close', 'volume', 'date'], errors='ignore')
        y = features_df[target_col]
        
        X_train.append(X)
        y_train.append(y)
        
    if not X_train:
        print("No training data generated.")
        return
        
    X_train_full = pd.concat(X_train)
    y_train_full = pd.concat(y_train)
    
    print(f"Training on {len(X_train_full)} samples...")
    
    # 2. Train Model
    from sklearn.linear_model import Ridge
    model = SklearnWrapper(Ridge(alpha=1.0))
    model.train(X_train_full, y_train_full)
    
    print("Initializing Strategy...")
    # Create ML Strategy using trained model
    # Note: Strategy needs pipeline to transform live data
    # But for live data we don't need LabelGenerator
    # So we should create a separate pipeline for prediction (without label)
    # Or just ignore label column if present
    
    prediction_pipeline = FeaturePipeline([
        TechnicalFactors() # Only technicals for prediction
    ])
    
    strategy = MLStrategy(
        name="RidgeRegressionTop5",
        model=model,
        feature_pipeline=prediction_pipeline,
        top_k=2
    )
    
    print("Running Backtest...")
    # Backtest on out-of-sample data
    test_start = '20230101'
    test_end = '20230115'
    
    engine = BacktestEngine(strategy, repo, initial_capital=100000.0)
    results = engine.run(test_start, test_end)
    
    print("\nBacktest Results Head:")
    print(results.head())

if __name__ == "__main__":
    integration_test()
