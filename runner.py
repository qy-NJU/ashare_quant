import yaml
import os
import pandas as pd
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Import all possible components to make them available for eval/getattr
from data.repository import DataRepository
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
from features.factors.reversal import ReversalFactor
from features.factors.technical import LabelGenerator
from features.processor import CrossSectionalProcessor, DynamicFilter
from models.xgboost_model import XGBoostWrapper
from models.machine_learning import SklearnWrapper
from strategies.ml_strategy import MLStrategy
from backtest.engine import BacktestEngine


def _compute_stock_features_batch(args):
    """
    Worker function for parallel training feature computation.
    Processes a BATCH of stocks to amortize FeaturePipeline initialization cost.
    Must be at module level for multiprocessing pickle to work.
    """
    stock_batch, start, end, include_label, dynamic_filter_config, feature_pipeline_config, benchmark_df_dict, benchmark_index, enable_filter, cache_dir = args

    results = {}  # sym -> features_df or None

    try:
        # Import inside worker to ensure fresh instances
        from data.repository import DataRepository
        from features.processor import DynamicFilter
        from features.pipeline import FeaturePipeline, FACTOR_MAP

        repo = DataRepository(cache_dir=cache_dir)
        dynamic_filter = DynamicFilter(**dynamic_filter_config) if dynamic_filter_config and enable_filter else None

        # Reconstruct benchmark DataFrame for MarketFactor
        benchmark_df_worker = None
        if benchmark_df_dict is not None and benchmark_index is not None:
            benchmark_df_worker = pd.DataFrame(benchmark_df_dict, index=pd.DatetimeIndex(benchmark_index))
            # Pre-compute index features for MarketFactor
            benchmark_df_worker['idx_ret'] = benchmark_df_worker['close'].pct_change()
            benchmark_df_worker['idx_ma20'] = benchmark_df_worker['close'].rolling(20).mean()
            benchmark_df_worker['idx_trend'] = (benchmark_df_worker['close'] / benchmark_df_worker['idx_ma20']) - 1
            benchmark_df_worker['idx_vol20'] = benchmark_df_worker['idx_ret'].rolling(20).std()

        # Reconstruct feature pipeline ONCE per batch (expensive, especially EventFactor loads all LHB data)
        factors = []
        for factor_info in feature_pipeline_config:
            factor_class = FACTOR_MAP[factor_info['class']]
            factor_instance = factor_class(**factor_info.get('params', {}))

            # Set up MarketFactor with pre-computed index data (same as _process_single_stock)
            if factor_info['class'] == 'MarketFactor' and benchmark_df_worker is not None:
                factor_instance.index_df = benchmark_df_worker

            factors.append(factor_instance)
        feature_pipeline = FeaturePipeline(factors)

        for sym in stock_batch:
            try:
                # Load raw data
                df = repo.get_daily_data(sym, start, end)
                if df.empty:
                    results[sym] = None
                    continue

                # Apply Dynamic Filter
                if dynamic_filter:
                    df = dynamic_filter.filter(df)
                    if df.empty:
                        results[sym] = None
                        continue

                # Merge benchmark_close as time-aligned series (not scalar)
                if benchmark_df_worker is not None and 'close' in benchmark_df_worker.columns:
                    # Align benchmark close to the stock's date index
                    aligned_benchmark = benchmark_df_worker['close'].reindex(df.index, method='ffill')
                    df['benchmark_close'] = aligned_benchmark

                # Add symbol column
                df['symbol'] = sym

                # Compute features
                features_df = feature_pipeline.transform(df)
                if features_df.empty:
                    results[sym] = None
                    continue

                results[sym] = features_df
            except Exception as e:
                results[sym] = None

        return results
    except Exception as e:
        # Return empty results on batch-level error
        print(f"Batch error: {e}")
        return {sym: None for sym in stock_batch}

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

    def _get_feature_cache_path(self, symbol, start, end, include_label):
        """
        Generate a unique cache path for the features based on the pipeline config.
        This prevents re-calculating 100+ indicators if the config hasn't changed.
        """
        # Create a deterministic string representation of the features config
        config_str = json.dumps(self.config['features'], sort_keys=True)
        config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]

        cache_dir = os.path.join(self.config['data'].get('cache_dir', 'data/local_lake'), 'features', config_hash)
        os.makedirs(cache_dir, exist_ok=True)

        label_str = "with_label" if include_label else "no_label"
        return os.path.join(cache_dir, f"{symbol}_{start}_{end}_{label_str}.parquet")

    def _serialize_feature_pipeline_config(self, feature_pipeline):
        """
        Serialize feature pipeline config for multiprocessing workers.
        """
        config = []
        for factor in feature_pipeline.factors:
            class_name = factor.__class__.__name__
            params = {}

            # Known init param names for each factor class
            # Only pass params if the factor's __init__ actually accepts them
            if class_name == 'PandasTAFactor':
                params = {'strategy': getattr(factor, 'strategy_mode', 'default'),
                         'features': getattr(factor, 'features', None)}
            elif class_name == 'BoardFactor':
                params = {'encode_method': getattr(factor, 'encode_method', 'category')}
            elif class_name == 'FinancialFactor':
                params = {'cache_dir': getattr(factor.repo, 'cache_dir', 'data/local_lake')}
            elif class_name == 'FundFlowFactor':
                params = {'cache_dir': getattr(factor.repo, 'cache_dir', 'data/local_lake')}
            elif class_name == 'MarketFactor':
                params = {'index_symbol': getattr(factor, 'index_symbol', 'sh.000300')}
            elif class_name == 'LabelGenerator':
                params = {'horizon': getattr(factor, 'horizon', 5),
                         'target_type': getattr(factor, 'target_type', 'regression'),
                         'decay_weights': getattr(factor, 'decay_weights', None)}
            else:
                # EventFactor, SubjectiveFactor, PatternFactor, TechnicalFactors: no params
                params = {}

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            config.append({'class': class_name, 'params': params})

        return config

    def build_model(self):
        print("Building Model...")
        model_cfg = self.config['model']
        params = model_cfg.get('params', {})
        num_boost_round = model_cfg.get('num_boost_round', 50)  # 从配置读取，默认50
        if model_cfg['name'] == 'XGBoostWrapper':
            model = XGBoostWrapper(**params)
            model.num_boost_round = num_boost_round  # 存储以供后续使用
            return model
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

    def _run_analysis(self, results, engine, model_metrics=None):
        if results is None or results.empty:
            print("No results to analyze.")
            return
            
        print("\nRunning Strategy Analysis...")
        try:
            from analysis import StrategyEvaluator, ReportGenerator
        except ImportError as e:
            print(f"Could not import analysis module: {e}")
            return
            
        evaluator = StrategyEvaluator()
        
        # Calculate daily returns from portfolio value
        results['daily_return'] = results['value'].pct_change().fillna(0)
        strategy_metrics = evaluator.evaluate_returns(results['daily_return'])
        
        trades_df = pd.DataFrame(engine.trade_log) if engine.trade_log else pd.DataFrame()
        trade_stats = evaluator.analyze_trades(trades_df)
        
        reporter = ReportGenerator()
        report_path = reporter.generate_markdown_report(
            model_metrics=model_metrics,
            strategy_metrics=strategy_metrics,
            trade_stats=trade_stats
        )
        print(f"Analysis Report generated at: {report_path}")

    def run(self):
        """
        Execute the full pipeline based on config
        """
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
        
        # --- 1. PREDICT ONLY MODE ---
        if mode == 'predict_only':
            print("\n=== STARTING PREDICTION MODE ===")

            # Load the model
            model_cfg = self.config['model']
            model_path = model_cfg.get('load_path', 'models/saved/config_xgb.json')

            if not os.path.exists(model_path):
                print(f"Error: Model file not found at {model_path}")
                return
                
            model = self._instantiate_class(model_cfg['name'], model_cfg.get('params', {}))
            model.load(model_path)
            print(f"Model loaded successfully from {model_path}")
            
            # Target prediction date
            pred_cfg = self.config.get('prediction', {})
            target_date = pred_cfg.get('target_date')
            if not target_date:
                print("Error: 'prediction.target_date' not specified in config.")
                return
            
            # Clean date formatting
            target_date_clean = target_date.replace('-', '')
            print(f"Generating Top N predictions based on data up to {target_date_clean}...")
            
            # Build feature pipeline without labels
            infer_pipeline = self.build_feature_pipeline(include_label=False)
            
            # We use MLStrategy's select_stocks logic directly to generate signals
            strat_cfg = self.config['strategy']
            strat_params = strat_cfg.get('params', {})
            strategy = MLStrategy(name="YAML_Strategy", model=model, feature_pipeline=infer_pipeline, universe=symbols, 
                                  processor=processor, dynamic_filter=dynamic_filter, **strat_params)
                                  
            # Load real portfolio data
            portfolio_cfg = self.config.get('portfolio', {})
            current_positions = portfolio_cfg.get('current_positions', {}) or {}
            position_costs = portfolio_cfg.get('position_costs', {}) or {}
            current_prices = {}
            
            # Fetch latest prices for current positions to trigger stop-loss/take-profit
            for sym in current_positions.keys():
                try:
                    df_latest = repo.get_daily_data(sym, target_date_clean, target_date_clean)
                    if not df_latest.empty:
                        current_prices[sym] = df_latest['close'].iloc[-1]
                except Exception as e:
                    print(f"Warning: Could not fetch latest price for {sym}: {e}")
            
            # Force days_since_rebalance so it actually predicts
            strategy.days_since_rebalance = strategy.rebalance_period + 1 
            
            print("\n--- 正在分析老仓健康度 (止盈/止损检查) ---")
            target_weights = strategy.select_stocks(
                target_date_clean, 
                repo, 
                current_positions=current_positions,
                current_prices=current_prices,
                position_costs=position_costs
            )
            
            if target_weights is None:
                target_weights = {}

            print(f"\n--- TOP STOCKS TO BUY/HOLD FOR NEXT TRADING DAY ---")
            print(f"Based on data at: {target_date}")
            
            available_capital = strat_cfg.get('available_capital', 100000.0)
            print(f"Account Total Capital: ¥{available_capital:,.2f}")
            
            # Fetch stock names
            try:
                stock_list = repo.get_stock_list()
                name_map = dict(zip(stock_list['symbol'], stock_list['name']))
            except:
                name_map = {}
                
            # Determine what to SELL
            sell_symbols = [sym for sym in current_positions.keys() if sym not in target_weights]
            if sell_symbols:
                print(f"\n🔴 [SELL INSTRUCTIONS]")
                for sym in sell_symbols:
                    name = name_map.get(sym, "Unknown")
                    qty = current_positions[sym]
                    print(f"   Sell All: {sym:<8} | Name: {name:<10} | Qty: {qty}")
            
            # Determine what to BUY / HOLD
            if not target_weights:
                print("\n⚪ No stocks selected. Holding cash. (Market filter or no valid predictions)")
            else:
                print(f"\n🟢 [BUY/HOLD INSTRUCTIONS]")
                for sym, weight in sorted(target_weights.items(), key=lambda x: x[1], reverse=True):
                    name = name_map.get(sym, "Unknown")
                    # Fetch latest price to calculate shares
                    try:
                        latest_data = repo.get_daily_data(sym, target_date_clean, target_date_clean)
                        if not latest_data.empty:
                            # Double check suspension status before recommending buy
                            vol = latest_data['volume'].iloc[-1]
                            if pd.isna(vol) or vol == 0:
                                print(f"   [SUSPENDED] SKIP: {sym:<8} | Name: {name:<10} | Target Weight: {weight*100:5.1f}% | Reason: Stock is currently suspended (Volume is 0)")
                                continue
                                
                            latest_price = latest_data['close'].iloc[-1]
                            allocated_funds = available_capital * weight
                            target_shares = int(allocated_funds / latest_price / 100) * 100
                            
                            # Check if it's already held
                            current_shares = current_positions.get(sym, 0)
                            shares_to_buy = target_shares - current_shares
                            
                            if shares_to_buy > 0:
                                action = "BUY" if current_shares == 0 else "ADD"
                                print(f"   {action}: {sym:<8} | Name: {name:<10} | Target Weight: {weight*100:5.1f}% | Est. Price: {latest_price:6.2f} | Need to Buy: {shares_to_buy:6d} shares")
                            elif shares_to_buy < 0:
                                print(f"   REDUCE: {sym:<8} | Name: {name:<10} | Target Weight: {weight*100:5.1f}% | Est. Price: {latest_price:6.2f} | Need to Sell: {-shares_to_buy:6d} shares")
                            else:
                                if current_shares == 0:
                                    print(f"   SKIP: {sym:<8} | Name: {name:<10} | Target Weight: {weight*100:5.1f}% | Est. Price: {latest_price:6.2f} | Reason: Allocated funds not enough to buy 100 shares")
                                else:
                                    print(f"   HOLD: {sym:<8} | Name: {name:<10} | Target Weight: {weight*100:5.1f}% | Est. Price: {latest_price:6.2f} | Shares Match Target")
                        else:
                            print(f"   [SUSPENDED] SKIP: {sym:<8} | Name: {name:<10} | Target Weight: {weight*100:5.1f}% | Reason: Stock is currently suspended (No data for target date)")
                            # Since this was skipped, we should conceptually reallocate this weight to other stocks or cash
                    except Exception as e:
                        print(f"   Error calculating shares for {sym}: {e}")
                    
            print("================================================\n")
            return

        # --- BACKTEST ONLY MODE (skip training, use saved model) ---
        elif mode == 'backtest_only':
            print("\n=== RUNNING BACKTEST ONLY MODE (skipping training) ===")
            # Load saved model
            load_path = self.config['model'].get('load_path')
            if not load_path or not os.path.exists(load_path):
                print(f"Error: saved model {load_path} not found. Please train first.")
                return
            model.load(load_path)
            print(f"Loaded saved model from {load_path}")

            # Skip training, go directly to backtest
            # Build inference pipeline for backtest
            infer_pipeline = self.build_feature_pipeline(include_label=False)

            # Get backtest period
            b_start = self.config['windows'].get('backtest', {}).get('start', '').replace('-', '')
            b_end = self.config['windows'].get('backtest', {}).get('end', '').replace('-', '')
            if not b_start or not b_end:
                print("Error: backtest start/end not configured.")
                return

            print(f"\n--- Backtest Phase: {b_start} to {b_end} ---")
            print("Starting backtest from {} to {} with {} capital.".format(
                b_start, b_end, self.config['strategy'].get('initial_capital', 100000.0)))

            # Build Strategy with saved model
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
            
            # Export Trades
            os.makedirs('data/backtest', exist_ok=True)
            if engine.trade_log:
                trades_df = pd.DataFrame(engine.trade_log)
                trades_csv = f'data/backtest/trades_{b_start}_{b_end}.csv'
                trades_df.to_csv(trades_csv, index=False)
                print(f"Exported {len(engine.trade_log)} trades to {trades_csv}")
            else:
                print("No trades executed during backtest.")

            print("\nPipeline Execution Completed. Final Results:")
            print(results.tail())
            
            # Run Analysis
            self._run_analysis(results, engine)
            
            return

        # --- TRAIN/BACKTEST/INCREMENTAL MODE ---
        
        if mode == 'incremental_train':
            print("\n=== RUNNING INCREMENTAL TRAIN MODE ===")
            load_path = self.config['model'].get('load_path')
            if not load_path or not os.path.exists(load_path):
                print(f"Error: saved model {load_path} not found. Cannot do incremental training.")
                return
            model.load(load_path)
            print(f"Loaded base model from {load_path}")
        
        # 2. Build Training Pipeline
        train_pipeline = self.build_feature_pipeline(include_label=True)
        
        # 3. Training Loop (Incremental)
        train_windows = self.config['windows'].get('train', [])
        
        last_model_metrics = None
        for i, window in enumerate(train_windows):
            start = window['start'].replace('-', '')
            end = window['end'].replace('-', '')
            print(f"\n--- Training Phase {i+1}: {start} to {end} ---")
            
            X_batch, y_batch = [], []
            
            # For excess return or ranking, we might need a benchmark
            # Let's fetch benchmark data once per window (sh.000300 is HS300 in Baostock)
            benchmark_df = repo.get_daily_data("sh.000300", start, end)
            
            opt_cfg = self.config.get('training_optimization', {})
            sample_rate = opt_cfg.get('sample_rate', 1.0)
            drop_middle = opt_cfg.get('drop_middle', False)
            drop_middle_threshold = opt_cfg.get('drop_middle_threshold', 0.3)
            
            # Serialize configs for parallel workers
            dynamic_filter_config = None
            if enable_filter and dynamic_filter:
                dynamic_filter_config = {
                    'min_avg_turnover': dynamic_filter.min_avg_turnover,
                    'min_listed_days': dynamic_filter.min_listed_days
                }
            feature_pipeline_config = self._serialize_feature_pipeline_config(train_pipeline)
            # Pass full benchmark time series (as dict+index for pickling) instead of scalar
            benchmark_df_dict = benchmark_df[['close']].to_dict('list') if not benchmark_df.empty else None
            benchmark_index = benchmark_df.index.tolist() if not benchmark_df.empty else None

            # Get absolute path for cache_dir to pass to workers
            data_cache_dir = self.config['data'].get('cache_dir', 'data/local_lake')
            if not os.path.isabs(data_cache_dir):
                data_cache_dir = os.path.abspath(data_cache_dir)

            # Prepare batches: first check cache, only compute if not cached
            uncached_symbols = []
            cache_paths = {}
            for sym in symbols:
                cache_path = self._get_feature_cache_path(sym, start, end, include_label=True)
                cache_paths[sym] = cache_path
                if not os.path.exists(cache_path):
                    uncached_symbols.append(sym)

            # Load cached features first (fast, no multiprocessing needed)
            cached_count = 0
            for sym in symbols:
                if sym in uncached_symbols:
                    continue
                cache_path = cache_paths[sym]
                try:
                    features_df = pd.read_parquet(cache_path)
                    if features_df.empty:
                        continue

                    target_col = [c for c in features_df.columns if c.startswith('target_')]
                    if not target_col:
                        continue
                    target_col = target_col[0]

                    features_df = features_df.dropna(subset=[target_col])

                    if sample_rate < 1.0:
                        features_df = features_df.sample(frac=sample_rate, random_state=42)

                    if features_df.empty:
                        continue

                    X = features_df.drop(columns=[target_col, 'open', 'high', 'low', 'close', 'volume', 'date', 'symbol'], errors='ignore')
                    X = X.select_dtypes(include=['number', 'category'])
                    y = features_df[target_col]

                    X_batch.append(X)
                    y_batch.append(y)
                    cached_count += 1
                except:
                    # Cache corrupted, treat as uncached
                    uncached_symbols.append(sym)

            # Parallel feature computation for uncached stocks in batches
            if uncached_symbols:
                # Split into batches to amortize FeaturePipeline init cost
                # Each batch processes multiple stocks with one FeaturePipeline instance
                num_workers = min(mp.cpu_count(), 8)
                batch_size = 20  # Small batch size for faster progress feedback
                batches = [uncached_symbols[i:i+batch_size] for i in range(0, len(uncached_symbols), batch_size)]

                print(f"Computing features for {len(uncached_symbols)} uncached stocks in {len(batches)} batches ({num_workers} workers, ~{batch_size} stocks/batch)...")

                tasks = [(batch, start, end, True, dynamic_filter_config, feature_pipeline_config, benchmark_df_dict, benchmark_index, enable_filter, data_cache_dir)
                         for batch in batches]

                completed_batches = 0
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {executor.submit(_compute_stock_features_batch, task): task[0] for task in tasks}
                    for future in as_completed(futures):
                        batch_results = future.result()  # dict: sym -> features_df
                        completed_batches += 1
                        print(f"  Batch {completed_batches}/{len(batches)} completed, {len([v for v in batch_results.values() if v is not None])} stocks succeeded")

                        for sym, features_df in batch_results.items():
                            if features_df is None or features_df.empty:
                                continue

                            # Save to cache
                            try:
                                features_df.to_parquet(cache_paths[sym], engine='pyarrow')
                            except:
                                pass

                            target_col = [c for c in features_df.columns if c.startswith('target_')]
                            if not target_col:
                                continue
                            target_col = target_col[0]

                            features_df = features_df.dropna(subset=[target_col])

                            if sample_rate < 1.0:
                                features_df = features_df.sample(frac=sample_rate, random_state=42)

                            if features_df.empty:
                                continue

                            X = features_df.drop(columns=[target_col, 'open', 'high', 'low', 'close', 'volume', 'date', 'symbol'], errors='ignore')
                            X = X.select_dtypes(include=['number', 'category'])
                            y = features_df[target_col]

                            X_batch.append(X)
                            y_batch.append(y)

            if X_batch:
                # Keep original index (dates) for proper groupby in ranking
                X_full = pd.concat(X_batch)
                y_full = pd.concat(y_batch)
                
                # Ensure X_full has unique columns
                X_full = X_full.loc[:, ~X_full.columns.duplicated()]
                
                # Apply Cross-Sectional Processing (MAD + Z-Score) on Features
                feature_cols = X_full.columns.tolist()
                X_full = processor.process(X_full, feature_cols)
                
                # Check if we need to do cross-sectional percentage ranking for labels
                label_gen_cfg = next((f for f in self.config['features'] if f['name'] == 'LabelGenerator'), None)
                if label_gen_cfg and label_gen_cfg.get('params', {}).get('target_type') in ('rank_pct', 'decay_weighted'):
                    print("Applying cross-sectional percentage ranking to labels...")
                    y_full = y_full.groupby(level=0).rank(pct=True)
                    
                # Apply Data Sampling (Drop Middle)
                if drop_middle:
                    print(f"Applying Drop Middle sampling (threshold: {drop_middle_threshold})...")
                    # We drop samples where the target value is around the median (e.g. between 0.35 and 0.65)
                    # For rank_pct, the median is 0.5. 
                    lower_bound = 0.5 - drop_middle_threshold / 2
                    upper_bound = 0.5 + drop_middle_threshold / 2
                    
                    # Create mask to keep extremes
                    keep_mask = (y_full <= lower_bound) | (y_full >= upper_bound)
                    
                    X_full = X_full[keep_mask]
                    y_full = y_full[keep_mask]
                    print(f"Kept {len(y_full)} extreme samples after dropping middle.")
                
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
                
                if i == 0 and mode != 'incremental_train':
                    model.train(X_full, y_full, groups=groups,
                                num_boost_round=getattr(model, 'num_boost_round', 50))
                else:
                    model.partial_train(X_full, y_full, groups=groups)
                    
                # Evaluate on training set
                train_preds = model.predict(X_full)
                
                try:
                    from analysis import ModelEvaluator
                    df_eval = pd.DataFrame({
                        'date': y_full.index,
                        'true': y_full.values,
                        'pred': train_preds
                    })
                    ic_series = ModelEvaluator.calculate_ic_series(df_eval, 'date', 'true', 'pred')
                    mean_ic = ic_series['IC'].mean()
                    mean_rank_ic = ic_series['Rank_IC'].mean()
                    
                    reg_metrics = ModelEvaluator.evaluate_regression(df_eval['true'], df_eval['pred'])
                    reg_metrics['Mean_IC_Series'] = float(mean_ic)
                    reg_metrics['Mean_Rank_IC_Series'] = float(mean_rank_ic)
                    
                    last_model_metrics = reg_metrics
                    
                    print(f"  --> Training Set Rank IC: {mean_rank_ic:.4f} (over {len(ic_series.dropna())} days)")
                    print(f"  --> Training Set MSE: {reg_metrics['MSE']:.4f}")
                except Exception as e:
                    print(f"  --> Could not calculate Model Metrics: {e}")
                    
            else:
                print("No data available for this window.")

        # Save Model
        save_path = self.config['model'].get('save_path')
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save(save_path)

        # --- EVALUATE ON OOS (Backtest) DATA ---
        backtest_win = self.config['windows']['backtest']
        b_start = backtest_win['start'].replace('-', '')
        b_end = backtest_win['end'].replace('-', '')
        
        print(f"\n=== Evaluating Predictive Power on OOS Data ({b_start} to {b_end}) ===")
        
        # Prepare tasks for OOS feature computation
        oos_pipeline = self.build_feature_pipeline(include_label=True)
        dynamic_filter_config = None
        if enable_filter and dynamic_filter:
            dynamic_filter_config = {
                'min_avg_turnover': dynamic_filter.min_avg_turnover,
                'min_listed_days': dynamic_filter.min_listed_days
            }
        oos_pipeline_config = self._serialize_feature_pipeline_config(oos_pipeline)
        
        # We need a lookback period to calculate features for b_start
        # pandas-ta needs up to 255 trading days for some indicators (e.g. PVIe_255). 
        # 400 calendar days gives us roughly 270 trading days.
        lookback_days = max(400, dynamic_filter.min_listed_days + 150) if dynamic_filter else 400
        b_start_fetch = (pd.to_datetime(backtest_win['start']) - pd.Timedelta(days=lookback_days)).strftime('%Y%m%d')
        
        benchmark_df = repo.get_daily_data("sh.000300", b_start_fetch, b_end)
        benchmark_df_dict = benchmark_df.to_dict('list') if not benchmark_df.empty else None
        benchmark_index = benchmark_df.index.tolist() if not benchmark_df.empty else None
        
        data_cache_dir = getattr(repo, 'cache_dir', 'data/local_lake')
        if not os.path.isabs(data_cache_dir):
            data_cache_dir = os.path.abspath(data_cache_dir)
            
        import hashlib, json
        config_str = json.dumps(oos_pipeline_config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]
        
        tasks = []
        for symbol in symbols:
            # We construct task parameter tuple matching what _compute_stock_features_batch expects
            # format: (batch_symbols, start_date, end_date, is_train, dynamic_filter_config, feature_pipeline_config, benchmark_df_dict, benchmark_index, cache_dir)
            tasks.append(([symbol], b_start_fetch, b_end, True, dynamic_filter_config, oos_pipeline_config, benchmark_df_dict, benchmark_index, enable_filter, data_cache_dir))
            
        # Parallel execution
        X_oos_list = []
        y_oos_list = []
        
        print(f"Computing/Loading OOS features for {len(symbols)} stocks...")
        with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), 8)) as executor:
            futures = {executor.submit(_compute_stock_features_batch, task): task[0][0] for task in tasks}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    res_dict = future.result()
                    if res_dict and symbol in res_dict:
                        df = res_dict[symbol]
                        if df is not None and not df.empty:
                            # Filter to actual backtest window
                            df = df.loc[b_start:b_end]
                            if df.empty: continue
                            
                            target_cols = [c for c in df.columns if c.startswith('target_')]
                            if not target_cols: continue
                            target_col = target_cols[0]
                            
                            df = df.dropna(subset=[target_col])
                            if df.empty: continue
                            
                            X = df.drop(columns=[target_col, 'open', 'high', 'low', 'close', 'volume', 'date', 'symbol'], errors='ignore')
                            X = X.select_dtypes(include=['number', 'category'])
                            y = df[target_col]
                            
                            X_oos_list.append(X)
                            y_oos_list.append(y)
                except Exception as e:
                    pass

        if X_oos_list:
            X_oos_full = pd.concat(X_oos_list)
            y_oos_full = pd.concat(y_oos_list)
            
            # Apply Cross-Sectional Processing
            feature_cols = X_oos_full.columns.tolist()
            X_oos_full = processor.process(X_oos_full, feature_cols)
            
            # Predict
            oos_preds = model.predict(X_oos_full)
            
            # Calculate metrics
            from scipy.stats import spearmanr
            import numpy as np
            
            rank_ics = []
            df_eval = pd.DataFrame({'pred': oos_preds, 'true': y_oos_full.values}, index=y_oos_full.index)
            
            for date, group in df_eval.groupby(level=0):
                if len(group) > 1 and group['true'].std() > 0:
                    corr, _ = spearmanr(group['pred'], group['true'])
                    if not pd.isna(corr):
                        rank_ics.append(corr)
                        
            mean_ic = sum(rank_ics) / len(rank_ics) if rank_ics else 0
            icir = (mean_ic / np.std(rank_ics)) if rank_ics and np.std(rank_ics) > 0 else 0
            
            print(f"OOS Rank IC: {mean_ic:.4f} (ICIR: {icir:.4f} over {len(rank_ics)} days)")
            
            # Accuracy metric (Directional accuracy for top decile vs bottom decile)
            # This shows if our high-score predictions actually go up more than low-score ones
            if len(df_eval) > 10:
                top_10_pct = df_eval['pred'].quantile(0.9)
                bottom_10_pct = df_eval['pred'].quantile(0.1)
                
                top_true_mean = df_eval[df_eval['pred'] >= top_10_pct]['true'].mean()
                bottom_true_mean = df_eval[df_eval['pred'] <= bottom_10_pct]['true'].mean()
                
                print(f"OOS Top 10% Score Avg True Return/Rank: {top_true_mean:.4f}")
                print(f"OOS Bottom 10% Score Avg True Return/Rank: {bottom_true_mean:.4f}")
        else:
            print("No OOS data available for evaluation.")

        # 4. Backtesting
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
        
        # Export Trades
        os.makedirs('data/backtest', exist_ok=True)
        trades_df = pd.DataFrame()
        if engine.trade_log:
            trades_df = pd.DataFrame(engine.trade_log)
            trades_csv = f'data/backtest/trades_{b_start}_{b_end}.csv'
            trades_df.to_csv(trades_csv, index=False)
            print(f"Exported {len(engine.trade_log)} trades to {trades_csv}")
        else:
            print("No trades executed during backtest.")
        
        print("\nPipeline Execution Completed. Final Results:")
        print(results.tail())
        
        # Run Analysis
        self._run_analysis(results, engine, model_metrics=last_model_metrics)

import sys

if __name__ == "__main__":
    config_file = "configs/pipeline_config.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    runner = PipelineRunner(config_file)
    runner.run()
