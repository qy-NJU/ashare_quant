from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os


def _process_single_stock(args):
    """
    Worker function for parallel stock processing.
    Must be at module level for multiprocessing pickle to work.
    """
    symbol, date, lookback_days, benchmark_close, dynamic_filter_config, feature_pipeline_config = args

    try:
        # Import here to avoid circular imports and ensure fresh instances in each process
        from data.repository import DataRepository
        from features.processor import DynamicFilter
        from features.pipeline import FeaturePipeline, FACTOR_MAP

        # Create fresh instances in this process
        data_loader = DataRepository()

        # Reconstruct dynamic filter
        dynamic_filter = None
        if dynamic_filter_config:
            dynamic_filter = DynamicFilter(**dynamic_filter_config)

        # Reconstruct feature pipeline
        factors = []
        for factor_info in feature_pipeline_config:
            factor_class = FACTOR_MAP[factor_info['class']]
            factors.append(factor_class(**factor_info.get('params', {})))
        feature_pipeline = FeaturePipeline(factors)

        # Fetch data
        start_dt = pd.to_datetime(date) - pd.Timedelta(days=lookback_days)
        df = data_loader.get_daily_data(symbol, start_date=start_dt.strftime('%Y%m%d'), end_date=date)

        if df.empty or len(df) < 30:
            return None

        # Apply Dynamic Filter
        if dynamic_filter:
            df = dynamic_filter.filter(df)
            if df.empty:
                return None

        # Merge benchmark
        if benchmark_close is not None and not pd.isna(benchmark_close):
            df['benchmark_close'] = benchmark_close

        # Inject symbol
        df['symbol'] = symbol

        # Generate features
        features_df = feature_pipeline.transform(df)
        if features_df.empty:
            return None

        # Return the latest row features and symbol
        return {
            'symbol': symbol,
            'features': features_df.iloc[-1]
        }
    except Exception as e:
        return None

class MLStrategy(BaseStrategy):
    """
    Strategy based on machine learning predictions.
    """
    def __init__(self, name, model, feature_pipeline, top_k=5, rebalance_period=20, universe=None, 
                 use_market_filter=False, max_turnover=1.0, weight_method='equal', processor=None, dynamic_filter=None):
        """
        Args:
            name (str): Strategy name.
            model (BaseModel): Trained model.
            feature_pipeline (FeaturePipeline): Pipeline to generate features for prediction.
            top_k (int): Number of stocks to select.
            rebalance_period (int): Days between rebalancing.
            universe (list): Optional list of stock symbols to restrict trading to.
            use_market_filter (bool): If True, stop buying when market is in downtrend.
            max_turnover (float): Maximum allowed portfolio turnover per rebalance (0.0 to 1.0).
            weight_method (str): 'equal' or 'score' (weight by model prediction score).
            processor (CrossSectionalProcessor): Processor for cross-sectional MAD and Z-Score.
            dynamic_filter (DynamicFilter): Filter for zombie/newly listed stocks.
        """
        super().__init__(name)
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.top_k = top_k
        self.rebalance_period = rebalance_period
        self.universe = universe
        self.processor = processor
        self.dynamic_filter = dynamic_filter
        
        # Advanced configurations
        self.use_market_filter = use_market_filter
        self.max_turnover = max_turnover
        self.weight_method = weight_method
        
        self.days_since_rebalance = 0

    def _check_market_filter(self, date, data_loader):
        """Simple market filter: check if SH000300 is above 20-day SMA."""
        if not self.use_market_filter:
            return True
            
        try:
            # Using a mock broad index symbol or fetch from repo
            # In a real system, you'd fetch "000300" (HS300)
            df = data_loader.get_daily_data("sh.000300", 
                                            start_date=(pd.to_datetime(date) - pd.Timedelta(days=40)).strftime('%Y%m%d'), 
                                            end_date=date)
            if len(df) < 20:
                return True # Not enough data, pass filter
            
            sma20 = df['close'].rolling(20).mean().iloc[-1]
            current_close = df['close'].iloc[-1]
            
            is_uptrend = current_close > sma20
            if not is_uptrend:
                print(f"[{date}] Market Filter Triggered: Market is in downtrend. Halting buys.")
            return is_uptrend
        except:
            return True

    def select_stocks(self, date, data_loader, current_positions=None):
        """
        Select stocks based on model prediction and advanced rules.
        """
        if current_positions is None:
            current_positions = {}

        # 1. Rebalance frequency check
        if self.days_since_rebalance < self.rebalance_period and self.days_since_rebalance != 0:
            self.days_since_rebalance += 1
            # Return current positions as target weights (hold)
            # Assuming equal weight for currently held if we just return keys
            return list(current_positions.keys())

        self.days_since_rebalance = 1 # Reset counter

        # 2. Market Filter
        if not self._check_market_filter(date, data_loader):
            return {} # Empty dict means sell all and hold cash

        # Determine candidates
        if self.universe:
            candidates = self.universe
        else:
            try:
                stock_list_df = data_loader.get_stock_list()
                if stock_list_df.empty:
                    return []
                candidates = stock_list_df['symbol'].tolist()
            except:
                return []

        # If candidates list is too large and no universe specified, limit it for performance in demo
        if not self.universe and len(candidates) > 20:
            candidates = candidates[:20]

        # Collect latest features for all candidates to form a cross-section
        latest_features_list = []
        valid_symbols = []

        # We need benchmark close for some factors
        # Fetch enough history: min_listed_days (120) + buffer for rolling calculations (~80 for safety)
        # This ensures stocks listed 120+ days ago have valid rows after dynamic_filter
        lookback_days = max(220, self.dynamic_filter.min_listed_days + 100) if self.dynamic_filter else 220
        benchmark_df = data_loader.get_daily_data("sh.000300",
                                                start_date=(pd.to_datetime(date) - pd.Timedelta(days=lookback_days)).strftime('%Y%m%d'),
                                                end_date=date)

        # Extract benchmark close series for parallel workers
        benchmark_close = benchmark_df['close'] if not benchmark_df.empty else None

        # Prepare configs for parallel workers
        dynamic_filter_config = None
        if self.dynamic_filter:
            dynamic_filter_config = {
                'min_avg_turnover': self.dynamic_filter.min_avg_turnover,
                'min_listed_days': self.dynamic_filter.min_listed_days
            }

        # Serialize feature pipeline config
        # Each factor class can be reconstructed from its init signature
        # We capture known init params based on factor class
        feature_pipeline_config = []
        for factor in self.feature_pipeline.factors:
            class_name = factor.__class__.__name__
            params = {}

            # Known init param names for each factor class
            if class_name == 'PandasTAFactor':
                params = {'name': factor.name, 'strategy': getattr(factor, 'strategy_mode', 'default'),
                         'features': getattr(factor, 'features', None)}
            elif class_name == 'BoardFactor':
                params = {'name': factor.name, 'encode_method': getattr(factor, 'encode_method', 'category')}
            elif class_name == 'FinancialFactor':
                params = {'name': factor.name, 'cache_dir': getattr(factor, 'cache_dir', 'data/cache')}
            elif class_name == 'FundFlowFactor':
                params = {'name': factor.name, 'cache_dir': getattr(factor, 'cache_dir', 'data/cache')}
            elif class_name == 'MarketFactor':
                params = {'name': factor.name, 'index_symbol': getattr(factor, 'index_symbol', 'sh.000300')}
            elif class_name == 'EventFactor':
                params = {'name': factor.name}
            elif class_name == 'SubjectiveFactor':
                params = {'name': factor.name}
            elif class_name == 'PatternFactor':
                params = {'name': factor.name}
            elif class_name == 'LabelGenerator':
                params = {'horizon': getattr(factor, 'horizon', 5),
                         'target_type': getattr(factor, 'target_type', 'regression')}
            else:
                # Fallback: use factor's own params attribute if available
                if hasattr(factor, 'params') and isinstance(factor.params, dict):
                    params = factor.params
                elif hasattr(factor, '_params') and isinstance(factor._params, dict):
                    params = factor._params

            # Remove None values to use defaults
            params = {k: v for k, v in params.items() if v is not None}

            feature_pipeline_config.append({'class': class_name, 'params': params})

        # Use multiprocessing for parallel stock processing
        num_workers = min(mp.cpu_count(), len(candidates), 8)  # Cap at 8 workers
        filtered_count = {'empty_df': 0, 'len_small': 0, 'dynamic': 0, 'transform': 0, 'exception': 0}

        # Prepare tasks: (symbol, date, lookback_days, benchmark_close_value, dynamic_filter_config, feature_pipeline_config)
        tasks = []
        for symbol in candidates:
            # Pass a single benchmark close value (latest) for each stock
            bench_close = benchmark_close.iloc[-1] if benchmark_close is not None and len(benchmark_close) > 0 else None
            tasks.append((symbol, date, lookback_days, bench_close, dynamic_filter_config, feature_pipeline_config))

        # Parallel execution
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_single_stock, task): task[0] for task in tasks}

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result is None:
                        # Categorize the failure
                        filtered_count['exception'] += 1
                    else:
                        latest_features_list.append(result['features'])
                        valid_symbols.append(result['symbol'])
                except Exception as e:
                    filtered_count['exception'] += 1

        # Debug: count how far we got
        print(f"[DEBUG {date}] valid={len(valid_symbols)}, filtered={filtered_count}")

        # Debug: check feature names from first successful symbol
        if latest_features_list:
            sample_features = latest_features_list[0]
            print(f"[DEBUG {date}] Sample features ({len(sample_features.columns)} cols): {list(sample_features.columns)[:15]}...")

        # Debug: count how far we got
        print(f"[DEBUG {date}] After parallel processing: valid_symbols={len(valid_symbols)} out of {len(candidates)}")

        if not latest_features_list:
            # Debug: check why no features were generated
            print(f"[DEBUG {date}] No features generated. Tried {len(candidates)} candidates.")
            if self.dynamic_filter:
                print(f"[DEBUG] DynamicFilter: min_listed_days={self.dynamic_filter.min_listed_days}, min_turnover={self.dynamic_filter.min_avg_turnover}")
            return []

        # Form Cross-Section
        X_full = pd.concat(latest_features_list)

        # Apply Cross-Sectional Processor (MAD Clip & Z-Score)
        if self.processor:
            feature_cols = X_full.drop(columns=['open', 'high', 'low', 'close', 'volume', 'date', 'symbol'], errors='ignore').columns.tolist()
            X_full = self.processor.process(X_full, feature_cols)

        # Drop non-feature columns before prediction
        # Also drop string/object columns that XGBoost cannot handle
        X_pred = X_full.drop(columns=['open', 'high', 'low', 'close', 'volume', 'date', 'symbol'], errors='ignore')
        # Keep only numeric and category columns (XGBoost supports these with enable_categorical=True)
        X_pred = X_pred.select_dtypes(include=['number', 'category'])

        # Batch Predict
        try:
            scores = self.model.predict(X_pred)
        except Exception as e:
            print(f"[{date}] Prediction failed: {e}")
            return []

        predictions = [(sym, score) for sym, score in zip(valid_symbols, scores)]

        # Sort by score descending
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Select Top K
        top_predictions = predictions[:self.top_k]
        selected = [x[0] for x in top_predictions]

        # 3. Turnover Control (Optional)
        if self.max_turnover < 1.0 and current_positions:
            # Simple logic: Keep existing top performers, only replace worst ones
            current_symbols = set(current_positions.keys())
            target_symbols = set(selected)
            
            to_sell = current_symbols - target_symbols
            to_buy = target_symbols - current_symbols
            
            # Limit number of trades based on max_turnover
            max_trades = max(1, int(len(current_positions) * self.max_turnover))
            
            if len(to_sell) > max_trades:
                # Need to reduce turnover. Keep some of the original positions
                # Ideally keep the ones with the highest scores among those we planned to sell
                pass # Simplified for demo, full logic requires sorting current holdings by new score
                
        # 4. Weight Allocation
        if self.weight_method == 'score':
            # Assign weights proportional to prediction score (assuming score > 0)
            # Clip negative scores to 0
            scores = [max(0.01, x[1]) for x in top_predictions]
            total_score = sum(scores)
            if total_score > 0:
                target_weights = {top_predictions[i][0]: scores[i]/total_score for i in range(len(top_predictions))}
                print(f"[{date}] ML Selected with Score Weights: {target_weights}")
                return target_weights
                
        # Default: Equal weight (just return list)
        print(f"[{date}] ML Selected: {selected}")
        return selected
