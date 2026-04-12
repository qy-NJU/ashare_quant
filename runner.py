import yaml
import os
import pandas as pd
import importlib

# Import all possible components to make them available for eval/getattr
from data.repository import DataRepository
from data.source.baostock_source import BaostockSource
from data.source.mock_source import MockDataSource
from data.source.akshare_source import AkShareSource
from data.pool_manager import StockPoolManager
from features.pipeline import FeaturePipeline
from features.factors.pandas_ta_factor import PandasTAFactor
from features.factors.fundamental import BoardFactor
from features.factors.event_driven import EventFactor
from features.factors.financial import FinancialFactor
from features.factors.fund_flow import FundFlowFactor
from features.factors.market import MarketFactor
from features.factors.subjective import SubjectiveFactor
from features.factors.pattern import PatternFactor
from features.factors.technical import LabelGenerator
from features.processor import CrossSectionalProcessor, DynamicFilter
from models.xgboost_model import XGBoostWrapper
from models.machine_learning import SklearnWrapper
from strategies.ml_strategy import MLStrategy
from backtest.engine import BacktestEngine

class PipelineRunner:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)['pipeline']
        print(f"Loaded configuration for pipeline: {self.config['name']}")

    def _instantiate_class(self, class_name, params=None):
        """Dynamically instantiate a class by its name from the current global scope."""
        if params is None:
            params = {}
        cls = globals().get(class_name)
        if not cls:
            raise ValueError(f"Class {class_name} not found or not imported in runner.py")
        return cls(**params)

    def build_data_layer(self):
        print("Building Data Layer...")
        data_cfg = self.config['data']
        # For Local Data Lake, sources are deprecated but kept in config for compatibility
        return DataRepository(cache_dir=data_cfg.get('cache_dir', 'data/local_lake'))

    def build_feature_pipeline(self, include_label=True):
        print(f"Building Feature Pipeline (include_label={include_label})...")
        factors = []
        for feat_cfg in self.config['features']:
            name = feat_cfg['name']
            if not include_label and name == "LabelGenerator":
                continue 
            
            params = feat_cfg.get('params', {})
            # Special handling for MarketFactor (needs data preparation)
            if name == "MarketFactor":
                factor = self._instantiate_class(name, params)
                # Pre-fetch index data for the entire possible range
                # Ideally this should be dynamic, but for now we fetch a wide range
                factor.prepare_index_data("2020-01-01", "2025-12-31")
                factors.append(factor)
            else:
                factors.append(self._instantiate_class(name, params))
            
        return FeaturePipeline(factors)

    def build_model(self):
        print("Building Model...")
        model_cfg = self.config['model']
        params = model_cfg.get('params', {})
        if model_cfg['name'] == 'XGBoostWrapper':
            return XGBoostWrapper(params=params)
        else:
            return self._instantiate_class(model_cfg['name'], params)

    def get_target_symbols(self):
        """Get the target symbols based on configuration."""
        data_cfg = self.config['data']
        
        # If specific symbols are provided, use them directly
        if data_cfg.get('symbols'):
            print(f"Using explicitly configured symbols: {data_cfg['symbols']}")
            return data_cfg['symbols']
            
        # Otherwise, use pool manager to filter
        if 'pool' in data_cfg:
            pool_cfg = data_cfg['pool']
            print(f"Using StockPoolManager with config: {pool_cfg}")
            pool_manager = StockPoolManager(data_repo=self.build_data_layer())
            return pool_manager.get_filtered_symbols(
                board=pool_cfg.get('board'),
                exchange=pool_cfg.get('exchange'),
                max_count=pool_cfg.get('max_count'),
                exclude_st=pool_cfg.get('exclude_st', True)
            )
            
        print("No symbols or pool config found. Returning empty list.")
        return []

    def run(self):
        mode = self.config.get('mode', 'train')
        
        # 1. Init Data
        repo = self.build_data_layer()
        symbols = self.get_target_symbols()
        
        if not symbols:
            print("No symbols to process. Exiting.")
            return
            
        model = self.build_model()
        
        # Build processor and dynamic filter
        prep_cfg = self.config.get('preprocessing', {})
        use_mad = prep_cfg.get('mad_clip', True)
        use_zscore = prep_cfg.get('z_score', True)
        processor = CrossSectionalProcessor(use_mad_clip=use_mad, use_zscore=use_zscore)
        
        filter_cfg = prep_cfg.get('dynamic_filter', {})
        enable_filter = filter_cfg.get('enable', True)
        min_turnover = filter_cfg.get('min_avg_turnover', 10000000) # Default 10 million RMB
        min_listed_days = filter_cfg.get('min_listed_days', 120)    # Default half year
        dynamic_filter = DynamicFilter(min_avg_turnover=min_turnover, min_listed_days=min_listed_days)
        
        # --- INFERENCE MODE ---
        if mode == 'inference':
            print("\n=== RUNNING IN INFERENCE MODE ===")
            # Load model
            load_path = self.config['model'].get('load_path')
            if not load_path or not os.path.exists(load_path):
                print(f"Error: load_path {load_path} not found for inference.")
                return
            model.load(load_path)
            
            # Prepare data for TODAY (or latest available)
            # We need enough history to calculate features (e.g. 30-60 days)
            # Note: Since Baostock free data might not have 2026 data, we mock 'today' as 2023-06-01 for demo
            import datetime
            # today = datetime.datetime.now()
            today = datetime.datetime(2023, 6, 1)
            end_date_str = today.strftime('%Y%m%d')
            start_date_str = (today - datetime.timedelta(days=150)).strftime('%Y%m%d')
            
            print(f"Fetching recent data for prediction ({start_date_str} - {end_date_str})...")
            
            # Build inference pipeline
            infer_pipeline = self.build_feature_pipeline(include_label=False)
            
            # For excess return or ranking, we might need a benchmark
            # Let's fetch benchmark data once per window (sh.000300 is HS300 in Baostock)
            benchmark_df = repo.get_daily_data("sh.000300", start_date_str, end_date_str)
            
            # --- Apply Cross-Sectional Processing on Predictions (if needed) ---
            # Wait, prediction features should be cross-sectionally processed!
            # Since we predict stock by stock above, cross-sectional features for the latest day are NOT calculated correctly.
            # We must gather all latest features, process them together, then predict.
            print("Gathering features for all symbols to apply cross-sectional processing...")
            
            latest_features_list = []
            valid_symbols = []
            
            count = 0
            for sym in symbols:
                count += 1
                if count % 100 == 0:
                    print(f"Processing {count}/{len(symbols)}...")
                    
                # Fetch data
                df = repo.get_daily_data(sym, start_date_str, end_date_str)
                if df.empty or len(df) < 30: continue
                
                # Apply Dynamic Filter (Zombie stocks, New stocks)
                if enable_filter:
                    df = dynamic_filter.filter(df)
                    if df.empty: continue
                
                # Merge benchmark
                if not benchmark_df.empty:
                    try:
                        df = df.join(benchmark_df['close'].rename('benchmark_close'), how='left')
                    except:
                        pass
                        
                # Add symbol column
                df['symbol'] = sym
                
                # Transform
                try:
                    features_df = infer_pipeline.transform(df)
                    if features_df.empty: continue
                    
                    # Take the LATEST row for prediction
                    latest_features = features_df.iloc[[-1]]
                    latest_features_list.append(latest_features)
                    valid_symbols.append(sym)
                except Exception as e:
                    pass
            
            if not latest_features_list:
                print("No features generated.")
                return
                
            # Combine all latest features
            X_infer_full = pd.concat(latest_features_list)
            
            # Apply Cross-Sectional Processing
            feature_cols = X_infer_full.drop(columns=['open', 'high', 'low', 'close', 'volume', 'date', 'symbol'], errors='ignore').columns.tolist()
            X_infer_full = processor.process(X_infer_full, feature_cols)
            
            # Predict
            X_pred = X_infer_full[feature_cols]
            scores = model.predict(X_pred)
            
            predictions = []
            for i, sym in enumerate(valid_symbols):
                latest_date = X_infer_full.index[i].strftime('%Y-%m-%d')
                predictions.append({'symbol': sym, 'score': scores[i], 'data_date': latest_date})
                
            pred_df = pd.DataFrame(predictions)
            pred_df = pred_df.sort_values('score', ascending=False)
            
            top_k = self.config['strategy']['params'].get('top_k', 10)
            print(f"\n=== Top {top_k} Predictions based on latest available data ===")
            print(pred_df[['symbol', 'score', 'data_date']].head(top_k))
            
            # Save predictions
            os.makedirs('data/predictions', exist_ok=True)
            out_file = f"data/predictions/pred_{end_date_str}.csv"
            pred_df.to_csv(out_file, index=False)
            print(f"\nFull predictions saved to {out_file}")
            return

        # --- TRAIN/BACKTEST MODE ---
        
        # 2. Build Training Pipeline
        train_pipeline = self.build_feature_pipeline(include_label=True)
        
        # 3. Training Loop (Incremental)
        train_windows = self.config['windows'].get('train', [])
        for i, window in enumerate(train_windows):
            start = window['start'].replace('-', '')
            end = window['end'].replace('-', '')
            print(f"\n--- Training Phase {i+1}: {start} to {end} ---")
            
            X_batch, y_batch = [], []
            
            # For excess return or ranking, we might need a benchmark
            # Let's fetch benchmark data once per window (sh.000300 is HS300 in Baostock)
            benchmark_df = repo.get_daily_data("sh.000300", start, end)
            
            for sym in symbols:
                df = repo.get_daily_data(sym, start, end)
                if df.empty: continue
                
                # Apply Dynamic Filter (Zombie stocks, New stocks)
                if enable_filter:
                    df = dynamic_filter.filter(df)
                    if df.empty: continue
                
                # Merge benchmark close if needed for excess return
                if not benchmark_df.empty:
                    # Align by date index
                    # Ensure both have date index
                    try:
                        # Convert benchmark_df to same timezone/format if needed, but usually YYYY-MM-DD matches
                        df = df.join(benchmark_df['close'].rename('benchmark_close'), how='left')
                    except Exception as e:
                        pass
                
                # Add symbol column for fundamental/board/subjective features to map correctly
                df['symbol'] = sym
                
                features_df = train_pipeline.transform(df)
                if features_df.empty: continue
                
                target_col = [c for c in features_df.columns if c.startswith('target_')]
                if not target_col: continue
                target_col = target_col[0]
                
                features_df = features_df.dropna(subset=[target_col])
                X = features_df.drop(columns=[target_col, 'open', 'high', 'low', 'close', 'volume', 'date', 'symbol'], errors='ignore')
                y = features_df[target_col]
                
                X_batch.append(X)
                y_batch.append(y)
                
            if X_batch:
                X_full = pd.concat(X_batch)
                y_full = pd.concat(y_batch)
                
                # Ensure X_full has unique columns
                X_full = X_full.loc[:, ~X_full.columns.duplicated()]
                
                # Apply Cross-Sectional Processing (MAD + Z-Score) on Features
                feature_cols = X_full.columns.tolist()
                X_full = processor.process(X_full, feature_cols)
                
                # Check if we need to do cross-sectional percentage ranking for labels
                label_gen_cfg = next((f for f in self.config['features'] if f['name'] == 'LabelGenerator'), None)
                if label_gen_cfg and label_gen_cfg.get('params', {}).get('target_type') == 'rank_pct':
                    print("Applying cross-sectional percentage ranking to labels...")
                    # Group by date index and calculate percentage rank
                    y_full = y_full.groupby(level=0).rank(pct=True)
                
                # If using ranking objective, we need groups (query structure)
                groups = None
                if 'rank:' in model.params.get('objective', ''):
                    # For ranking, data must be grouped by date (query)
                    # We need to sort by date first to ensure contiguous groups
                    combined = pd.concat([X_full, y_full], axis=1)
                    # Sort by index (date)
                    combined = combined.sort_index()
                    y_full = combined[y_full.name]
                    X_full = combined.drop(columns=[y_full.name])
                    
                    # Calculate group sizes (number of stocks per date)
                    groups = X_full.groupby(X_full.index).size().values
                
                if i == 0:
                    model.train(X_full, y_full, groups=groups)
                else:
                    model.partial_train(X_full, y_full, groups=groups)
            else:
                print("No data available for this window.")

        # Save Model
        save_path = self.config['model'].get('save_path')
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save(save_path)

        # 4. Backtesting
        backtest_win = self.config['windows']['backtest']
        b_start = backtest_win['start'].replace('-', '')
        b_end = backtest_win['end'].replace('-', '')
        print(f"\n--- Backtesting Phase: {b_start} to {b_end} ---")
        
        # Build inference pipeline
        infer_pipeline = self.build_feature_pipeline(include_label=False)
        
        # Build Strategy
        strat_cfg = self.config['strategy']
        strat_params = strat_cfg.get('params', {})
        
        if strat_cfg['name'] == 'MLStrategy':
            strategy = MLStrategy(name="YAML_Strategy", model=model, feature_pipeline=infer_pipeline, universe=symbols, 
                                  processor=processor, dynamic_filter=dynamic_filter, **strat_params)
        else:
            strategy = self._instantiate_class(strat_cfg['name'], strat_params)

        # Run Engine
        engine = BacktestEngine(strategy, repo, initial_capital=strat_cfg.get('initial_capital', 100000.0))
        results = engine.run(b_start, b_end)
        
        print("\nPipeline Execution Completed. Final Results:")
        print(results.tail())

import sys

if __name__ == "__main__":
    config_file = "configs/pipeline_config.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    runner = PipelineRunner(config_file)
    runner.run()
