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
    symbol, date, lookback_days, benchmark_df_dict, benchmark_index, dynamic_filter_config, feature_pipeline_config, cache_dir, cache_enabled, config_hash = args

    try:
        # Import here to avoid circular imports and ensure fresh instances in each process
        from data.repository import DataRepository
        from features.processor import DynamicFilter
        from features.pipeline import FeaturePipeline, FACTOR_MAP

        # Create fresh instances in this process
        data_loader = DataRepository(cache_dir=cache_dir)

        # Check feature cache first
        if cache_enabled and config_hash:
            import hashlib, json
            cache_subdir = os.path.join(cache_dir, 'features', config_hash)
            os.makedirs(cache_subdir, exist_ok=True)
            cache_file = os.path.join(cache_subdir, f"{symbol}_{date}_inference.parquet")
            if os.path.exists(cache_file):
                try:
                    features_df = pd.read_parquet(cache_file)
                    return {
                        'symbol': symbol,
                        'features': features_df.iloc[-1] if len(features_df) > 0 else None,
                        'from_cache': True
                    }
                except:
                    pass

        # Reconstruct benchmark DataFrame for MarketFactor
        benchmark_df_worker = None
        if benchmark_df_dict is not None and benchmark_index is not None:
            benchmark_df_worker = pd.DataFrame(benchmark_df_dict, index=pd.DatetimeIndex(benchmark_index))
            # Pre-compute index features
            benchmark_df_worker['idx_ret'] = benchmark_df_worker['close'].pct_change()
            benchmark_df_worker['idx_ma20'] = benchmark_df_worker['close'].rolling(20).mean()
            benchmark_df_worker['idx_trend'] = (benchmark_df_worker['close'] / benchmark_df_worker['idx_ma20']) - 1
            benchmark_df_worker['idx_vol20'] = benchmark_df_worker['idx_ret'].rolling(20).std()
            benchmark_df_worker = benchmark_df_worker[['close', 'idx_ret', 'idx_trend', 'idx_vol20']]

        # Reconstruct dynamic filter
        dynamic_filter = None
        if dynamic_filter_config:
            dynamic_filter = DynamicFilter(**dynamic_filter_config)

        # Reconstruct feature pipeline
        factors = []
        for factor_info in feature_pipeline_config:
            factor_class = FACTOR_MAP[factor_info['class']]
            factor_instance = factor_class(**factor_info.get('params', {}))

            # Special handling for MarketFactor: set up pre-computed index data
            if factor_info['class'] == 'MarketFactor' and benchmark_df_worker is not None:
                factor_instance.index_df = benchmark_df_worker

            factors.append(factor_instance)
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

        # Merge benchmark: compute latest close from benchmark_df_dict for PandasTAFactor
        if benchmark_df_dict is not None:
            benchmark_df_worker = pd.DataFrame(benchmark_df_dict)
            if not benchmark_df_worker.empty and 'close' in benchmark_df_worker.columns:
                bench_close_value = benchmark_df_worker['close'].iloc[-1]
                df['benchmark_close'] = bench_close_value

        # Inject symbol
        df['symbol'] = symbol

        # Generate features
        features_df = feature_pipeline.transform(df)
        if features_df.empty:
            return None

        # Save to cache
        if cache_enabled and config_hash:
            try:
                features_df.to_parquet(cache_file, engine='pyarrow')
            except:
                pass

        # Return the latest row features and symbol
        return {
            'symbol': symbol,
            'features': features_df.iloc[-1]
        }
    except Exception as e:
        # Return error info for debugging
        return {
            'symbol': symbol,
            'error': str(e)
        }

class MLStrategy(BaseStrategy):
    """
    Strategy based on machine learning predictions.
    """
    def __init__(self, name, model, feature_pipeline, top_k=5, rebalance_period=20, universe=None, 
                 use_market_filter=False, max_turnover=1.0, weight_method='equal', processor=None, dynamic_filter=None, target_position_ratio=1.0, **kwargs):
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
            target_position_ratio (float): Target total position ratio (0.0 to 1.0).
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
        self.target_position_ratio = target_position_ratio
        
        self.days_since_rebalance = 0
        
        # State to track highest prices for trailing stop
        self.highest_prices = {}

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

    def select_stocks(self, date, data_loader, current_positions=None, current_prices=None, position_costs=None):
        """
        Select stocks based on model prediction and advanced rules.
        """
        if current_positions is None:
            current_positions = {}
        if current_prices is None:
            current_prices = {}
        if position_costs is None:
            position_costs = {}

        # Clean up highest_prices for symbols no longer held
        for sym in list(self.highest_prices.keys()):
            if sym not in current_positions:
                del self.highest_prices[sym]

        # --- Dynamic Stop Loss & Take Profit (Always Check) ---
        symbols_to_sell = []
        stop_loss_threshold = -0.08  # -8% Accumulated Stop Loss
        daily_drop_threshold = -0.07 # -7% Single day drop (Intraday flush)
        trailing_stop_threshold = -0.10 # -10% from highest peak (Let profits run but cut if trend reverses)
        
        # 1. Evaluate Old Positions Health (Longfeihu Strategy)
        losing_positions_count = 0
        for symbol, qty in current_positions.items():
            if symbol in position_costs and symbol in current_prices:
                avg_cost = position_costs[symbol]
                curr_price = current_prices[symbol]
                if avg_cost > 0:
                    pnl_ratio = (curr_price - avg_cost) / avg_cost
                    if pnl_ratio < -0.03: # Floating loss > 3% is considered weak
                        losing_positions_count += 1
                        
        # 2. Dynamic Position Sizing based on Health
        allow_new_buys = True
        dynamic_target_ratio = self.target_position_ratio
        
        if len(current_positions) > 0 and losing_positions_count >= len(current_positions) / 2.0:
            # Headwind period: More than half of old positions are weak. Stop new buys and reduce target ratio.
            allow_new_buys = False
            dynamic_target_ratio = max(0.3, self.target_position_ratio / 2.0)
            print(f"[{date}] 【逆风期风控】老仓表现恶劣 (弱势比例: {losing_positions_count}/{len(current_positions)})，停止开新仓，总仓位目标压降至 {dynamic_target_ratio*100:.0f}%")
        
        for symbol, qty in current_positions.items():
            if symbol in position_costs and symbol in current_prices:
                avg_cost = position_costs[symbol]
                curr_price = current_prices[symbol]
                
                # Update highest price for trailing stop
                if symbol not in self.highest_prices:
                    self.highest_prices[symbol] = curr_price
                else:
                    self.highest_prices[symbol] = max(self.highest_prices[symbol], curr_price)
                
                highest_price = self.highest_prices[symbol]
                
                if avg_cost > 0:
                    pnl_ratio = (curr_price - avg_cost) / avg_cost
                    
                    # Trailing Stop Loss (for winning trades)
                    if highest_price > avg_cost * 1.05:
                        drawdown_from_peak = (curr_price - highest_price) / highest_price
                        if drawdown_from_peak <= trailing_stop_threshold:
                            print(f"[{date}] Trailing Stop Triggered for {symbol}: Peak {highest_price:.2f}, Curr {curr_price:.2f} (Drawdown {drawdown_from_peak*100:.2f}%)")
                            symbols_to_sell.append(symbol)
                            continue
                            
                    # Accumulated Stop Loss
                    if pnl_ratio <= stop_loss_threshold:
                        print(f"[{date}] Accumulated Stop Loss Triggered for {symbol}: PnL {pnl_ratio*100:.2f}% (Cost: {avg_cost:.2f}, Curr: {curr_price:.2f})")
                        symbols_to_sell.append(symbol)
                        continue
                        
                    # In headwind period, accelerate closing of weak positions (accelerated stop loss)
                    if not allow_new_buys and pnl_ratio < -0.05:
                        print(f"[{date}] Headwind Accelerated Stop Loss Triggered for {symbol}: PnL {pnl_ratio*100:.2f}%")
                        symbols_to_sell.append(symbol)
                        continue
                        
                    # Single Day Drop
                    try:
                        df_recent = data_loader.get_daily_data(symbol, start_date=(pd.to_datetime(date)-pd.Timedelta(days=5)).strftime('%Y%m%d'), end_date=date)
                        if len(df_recent) >= 2:
                            prev_close = df_recent.iloc[-2]['close']
                            daily_drop = (curr_price - prev_close) / prev_close
                            if daily_drop <= daily_drop_threshold:
                                print(f"[{date}] Daily Flush Stop Loss Triggered for {symbol}: Daily Drop {daily_drop*100:.2f}%")
                                symbols_to_sell.append(symbol)
                                continue
                    except Exception:
                        pass
                        
        # Remove stopped out symbols from active positions
        active_positions = {s: q for s, q in current_positions.items() if s not in symbols_to_sell}

        # 3. Conditional Rebalance Check
        # If we are in headwind and stopped buying, we should return scaled down active positions
        if not allow_new_buys:
            if active_positions:
                w = dynamic_target_ratio / len(active_positions)
                return {s: w for s in active_positions.keys()}
            return {}

        if self.days_since_rebalance < self.rebalance_period and self.days_since_rebalance != 0:
            self.days_since_rebalance += 1
            if active_positions:
                w = dynamic_target_ratio / len(active_positions)
                return {s: w for s in active_positions.keys()}
            return {}

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
                         'target_type': getattr(factor, 'target_type', 'regression')}
            else:
                # EventFactor, SubjectiveFactor, PatternFactor, TechnicalFactors: no params
                params = {}

            # Remove None values to use defaults
            params = {k: v for k, v in params.items() if v is not None}

            feature_pipeline_config.append({'class': class_name, 'params': params})

        # Get absolute path for cache_dir to pass to workers
        cache_dir = getattr(data_loader, 'cache_dir', 'data/local_lake')
        if not os.path.isabs(cache_dir):
            cache_dir = os.path.abspath(cache_dir)

        # Compute config hash for feature cache
        import hashlib, json
        config_str = json.dumps(feature_pipeline_config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]

        # Use multiprocessing for parallel stock processing
        num_workers = min(mp.cpu_count(), len(candidates), 8)  # Cap at 8 workers
        filtered_count = {'empty_df': 0, 'len_small': 0, 'dynamic': 0, 'transform': 0, 'exception': 0, 'from_cache': 0}

        # Convert benchmark_df to dict for pickling (DataFrame can't be pickled directly)
        # Use 'list' orientation to get {col: [values]}, then reconstruct with index
        benchmark_df_dict = benchmark_df.to_dict('list') if not benchmark_df.empty else None
        benchmark_index = benchmark_df.index.tolist() if not benchmark_df.empty else None

        # Prepare tasks: (symbol, date, lookback_days, benchmark_df_dict, benchmark_index, dynamic_filter_config, feature_pipeline_config, cache_dir, cache_enabled, config_hash)
        tasks = []
        for symbol in candidates:
            tasks.append((symbol, date, lookback_days, benchmark_df_dict, benchmark_index, dynamic_filter_config, feature_pipeline_config, cache_dir, True, config_hash))

        # Parallel execution
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_single_stock, task): task[0] for task in tasks}

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result is None:
                        filtered_count['exception'] += 1
                    elif 'error' in result:
                        # Debug: log the error
                        if len(filtered_count.get('errors', [])) < 5:
                            filtered_count.setdefault('errors', []).append(f"{symbol}: {result['error']}")
                        filtered_count['exception'] += 1
                    else:
                        if result.get('from_cache'):
                            filtered_count['from_cache'] = filtered_count.get('from_cache', 0) + 1
                        latest_features_list.append(result['features'])
                        valid_symbols.append(result['symbol'])
                except Exception as e:
                    filtered_count['exception'] += 1

        # Debug: count how far we got
        print(f"[DEBUG {date}] valid={len(valid_symbols)}, filtered={filtered_count}")

        # Debug: check feature names from first successful symbol
        if latest_features_list:
            sample_features = latest_features_list[0]
            print(f"[DEBUG {date}] Sample features ({len(sample_features)} cols): {list(sample_features.index)[:15]}...")

        # Debug: count how far we got
        print(f"[DEBUG {date}] After parallel processing: valid_symbols={len(valid_symbols)} out of {len(candidates)}")

        if not latest_features_list:
            # Debug: check why no features were generated
            print(f"[DEBUG {date}] No features generated. Tried {len(candidates)} candidates.")
            if self.dynamic_filter:
                print(f"[DEBUG] DynamicFilter: min_listed_days={self.dynamic_filter.min_listed_days}, min_turnover={self.dynamic_filter.min_avg_turnover}")
            return []

        # Form Cross-Section (each item is a Series, need to convert to single-row DataFrame)
        X_full = pd.concat([s.to_frame().T for s in latest_features_list])

        # Fix: When Series has mixed types (including string columns like board_industry),
        # the entire Series becomes object dtype. After concat, ALL columns become object.
        # Convert numeric columns back to proper numeric types.
        for col in X_full.columns:
            if X_full[col].dtype == 'object':
                # Try to convert to numeric
                converted = pd.to_numeric(X_full[col], errors='coerce')
                # Only keep if conversion worked (no NaN from coercion)
                if not converted.isna().all():
                    X_full[col] = converted

        # Apply Cross-Sectional Processor (MAD Clip & Z-Score)
        if self.processor:
            feature_cols = X_full.drop(columns=['open', 'high', 'low', 'close', 'volume', 'date', 'symbol'], errors='ignore').columns.tolist()
            X_full = self.processor.process(X_full, feature_cols)

        # Drop non-feature columns before prediction
        # Also drop string/object columns that XGBoost cannot handle
        X_pred = X_full.drop(columns=['open', 'high', 'low', 'close', 'volume', 'date', 'symbol'], errors='ignore')

        # DEBUG: check columns before select_dtypes
        print(f"[DEBUG {date}] Before select_dtypes - columns: {list(X_pred.columns)}, dtypes: {dict(X_pred.dtypes.value_counts())}")

        # Keep only numeric and category columns (XGBoost supports these with enable_categorical=True)
        X_pred = X_pred.select_dtypes(include=['number', 'category'])

        # DEBUG: trace X_pred columns
        print(f"[DEBUG {date}] X_pred shape: {X_pred.shape}, X_pred columns: {list(X_pred.columns)[:10]}...")

        # Batch Predict
        try:
            scores = self.model.predict(X_pred)
        except Exception as e:
            print(f"[{date}] Prediction failed: {e}")
            return []

        predictions = [(sym, score) for sym, score in zip(valid_symbols, scores)]

        # Sort by score descending
        predictions.sort(key=lambda x: x[1], reverse=True)

        # --- Filter Illiquid / Limit Up Stocks ---
        valid_predictions = []
        for sym, score in predictions:
            idx = valid_symbols.index(sym)
            row = X_full.iloc[idx]
            orig_row = latest_features_list[idx]
            
            vol = orig_row.get('volume', 1)
            is_limit_up = row.get('sub_is_limit_up', 0)
            
            if vol == 0 or is_limit_up == 1:
                # print(f"[{date}] Skipping {sym} due to liquidity/limit-up risk (Score: {score:.4f})")
                continue
                
            valid_predictions.append((sym, score))

        # --- Rebalance: Keep strong old positions even if not in Top K ---
        # If an active position is very strong (e.g., it was limit up today) OR it still ranks in the top 30%, we keep it
        forced_keeps = []
        keep_threshold = int(len(valid_predictions) * 0.3) # Top 30%
        top_30_symbols = [x[0] for x in valid_predictions[:keep_threshold]]
        
        for sym in active_positions.keys():
            if sym in valid_symbols:
                idx = valid_symbols.index(sym)
                is_limit_up = X_full.iloc[idx].get('sub_is_limit_up', 0) == 1
                is_in_top_30 = sym in top_30_symbols
                
                if is_limit_up:
                    forced_keeps.append(sym)
                    print(f"[{date}] Letting profits run for {sym} (Limit Up today)")
                elif is_in_top_30:
                    forced_keeps.append(sym)
                    # print(f"[{date}] Holding {sym} (Still in Top 30%)")
                    
        # Select Top K from valid predictions, avoiding duplicates with forced_keeps
        # First, count how many new slots we have
        new_slots = max(0, self.top_k - len(forced_keeps))
        
        selected = list(forced_keeps)
        for sym, score in valid_predictions:
            if new_slots <= 0:
                break
            if sym not in selected:
                selected.append(sym)
                new_slots -= 1
                
        # We need to reconstruct top_predictions for score weighting
        final_top_predictions = []
        for sym in selected:
            # Find its score
            score = 0.01
            for p_sym, p_score in predictions:
                if p_sym == sym:
                    score = p_score
                    break
            final_top_predictions.append((sym, score))

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
            # Assign weights proportional to prediction score
            # Since scores from rank:pairwise can be anything (including negative or very small differences),
            # we use softmax to convert them into meaningful probabilities/weights.
            scores = np.array([x[1] for x in final_top_predictions])
            # Subtract max for numerical stability
            exp_scores = np.exp(scores - np.max(scores))
            softmax_weights = exp_scores / np.sum(exp_scores)
            
            target_weights = {final_top_predictions[i][0]: float(softmax_weights[i]) * dynamic_target_ratio for i in range(len(final_top_predictions))}
            print(f"[{date}] ML Selected with Score Weights: {target_weights}")
            return target_weights
                
        # Default: Equal weight (enforce dynamic_target_ratio)
        if selected:
            w = dynamic_target_ratio / len(selected)
            target_weights = {sym: w for sym in selected}
            print(f"[{date}] ML Selected with Equal Weights: {target_weights}")
            return target_weights
            
        return {}
