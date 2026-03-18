import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, strategy, data_loader, initial_capital=100000.0, commission=0.0003):
        self.strategy = strategy
        self.data_loader = data_loader
        self.initial_capital = initial_capital
        self.commission = commission
        
        self.cash = initial_capital
        self.positions = {} # Symbol -> Quantity
        self.portfolio_history = []
        
    def run(self, start_date, end_date):
        """
        Run the backtest.
        """
        # Get trading dates
        # Assuming we can get market calendar from a benchmark like 000300
        # For simplicity, use business days
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        print(f"Starting backtest from {start_date} to {end_date} with {self.initial_capital} capital.")
        
        for current_date in dates:
            date_str = current_date.strftime('%Y%m%d')
            
            # 1. Update Portfolio Value (Mark to Market)
            portfolio_value = self.cash
            current_prices = {}
            
            for symbol, qty in self.positions.items():
                try:
                    # Get price for today
                    df = self.data_loader.get_stock_daily(symbol, start_date=date_str, end_date=date_str)
                    if not df.empty:
                        price = df.iloc[0]['收盘']
                        current_prices[symbol] = price
                        portfolio_value += price * qty
                    else:
                        # Use last known price or 0 if data missing (simplified)
                        # In real system, use fillna or previous close
                        pass 
                except Exception:
                    pass
            
            self.portfolio_history.append({
                'date': current_date,
                'value': portfolio_value
            })
            
            # 2. Generate Signals
            # Strategy can return a list of symbols (equal weight) or a dict of {symbol: target_weight}
            signals = self.strategy.select_stocks(date_str, self.data_loader, current_positions=self.positions)
            
            # 3. Execute Trades
            if signals is not None and len(signals) > 0:
                target_weights = {}
                if isinstance(signals, list):
                    # Equal weight distribution
                    n = len(signals)
                    target_weights = {s: 1.0/n for s in signals}
                elif isinstance(signals, dict):
                    target_weights = signals
                    
                # Normalize weights just in case
                total_weight = sum(target_weights.values())
                if total_weight > 1.0:
                    target_weights = {k: v/total_weight for k, v in target_weights.items()}
                
                # Sell phase
                for symbol in list(self.positions.keys()):
                    if symbol not in target_weights or target_weights[symbol] == 0:
                        # Sell all
                        price = current_prices.get(symbol)
                        if price:
                            qty = self.positions.pop(symbol)
                            revenue = price * qty * (1 - self.commission - 0.001) # + stamp duty
                            self.cash += revenue
                            print(f"[{date_str}] Sell {symbol}: {qty} shares @ {price:.2f}")
                    else:
                        # Rebalance: Sell partial if target value < current value
                        price = current_prices.get(symbol)
                        if price:
                            target_val = portfolio_value * target_weights[symbol]
                            current_val = self.positions[symbol] * price
                            if current_val > target_val:
                                diff_val = current_val - target_val
                                qty_to_sell = int(diff_val / price / 100) * 100
                                if qty_to_sell > 0:
                                    revenue = price * qty_to_sell * (1 - self.commission - 0.001)
                                    self.cash += revenue
                                    self.positions[symbol] -= qty_to_sell
                                    print(f"[{date_str}] Sell {symbol}: {qty_to_sell} shares @ {price:.2f}")
                            
                # Buy phase
                for symbol, weight in target_weights.items():
                    target_val = portfolio_value * weight
                    price = current_prices.get(symbol)
                    
                    if not price:
                        try:
                            df = self.data_loader.get_stock_daily(symbol, start_date=date_str, end_date=date_str)
                            if not df.empty:
                                price = df.iloc[0]['收盘']
                                current_prices[symbol] = price
                        except:
                            pass
                            
                    if price:
                        current_qty = self.positions.get(symbol, 0)
                        current_val = current_qty * price
                        
                        if current_val < target_val:
                            # Buy more
                            diff_val = target_val - current_val
                            qty_to_buy = int(diff_val / price / 100) * 100 # Round to 100 shares
                            if qty_to_buy > 0:
                                cost = qty_to_buy * price * (1 + self.commission)
                                if self.cash >= cost:
                                    self.cash -= cost
                                    self.positions[symbol] = current_qty + qty_to_buy
                                    print(f"[{date_str}] Buy {symbol}: {qty_to_buy} shares @ {price:.2f}")
                                    
        # Calculate performance
        df_result = pd.DataFrame(self.portfolio_history)
        df_result.set_index('date', inplace=True)
        
        final_value = df_result.iloc[-1]['value']
        total_return = (final_value / self.initial_capital) - 1
        
        print(f"\nBacktest Finished.")
        print(f"Final Value: {final_value:.2f}")
        print(f"Total Return: {total_return:.2%}")
        
        return df_result
