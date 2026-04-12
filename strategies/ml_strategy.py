from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np

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
        benchmark_df = data_loader.get_daily_data("sh.000300", 
                                                start_date=(pd.to_datetime(date) - pd.Timedelta(days=100)).strftime('%Y%m%d'), 
                                                end_date=date)
        
        for symbol in candidates:
            start_dt = pd.to_datetime(date) - pd.Timedelta(days=100)
            df = data_loader.get_daily_data(symbol, start_date=start_dt.strftime('%Y%m%d'), end_date=date)
            
            if df.empty or len(df) < 30: # Minimum required
                continue
                
            # Apply Dynamic Filter (Zombie stocks, New stocks)
            if self.dynamic_filter:
                df = self.dynamic_filter.filter(df)
                if df.empty: continue
            
            # Merge benchmark
            if not benchmark_df.empty:
                try:
                    df = df.join(benchmark_df['close'].rename('benchmark_close'), how='left')
                except:
                    pass
            
            # Inject symbol for BoardFactor and EventFactor
            df['symbol'] = symbol
                
            try:
                features_df = self.feature_pipeline.transform(df)
                if features_df.empty:
                    continue
                
                # Take the LATEST row to form the cross-section
                latest_features = features_df.iloc[[-1]] 
                latest_features_list.append(latest_features)
                valid_symbols.append(symbol)
            except Exception as e:
                continue
                
        if not latest_features_list:
            return []
            
        # Form Cross-Section
        X_full = pd.concat(latest_features_list)
        
        # Apply Cross-Sectional Processor (MAD Clip & Z-Score)
        if self.processor:
            feature_cols = X_full.drop(columns=['open', 'high', 'low', 'close', 'volume', 'date', 'symbol'], errors='ignore').columns.tolist()
            X_full = self.processor.process(X_full, feature_cols)
            
        # Drop non-feature columns before prediction
        X_pred = X_full.drop(columns=['open', 'high', 'low', 'close', 'volume', 'date', 'symbol'], errors='ignore')
        
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
