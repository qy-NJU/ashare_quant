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
        Run the backtest with T+1 execution logic, slippage, stamp duty, and limit up/down checks.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        print(f"Starting backtest from {start_date} to {end_date} with {self.initial_capital} capital.")
        
        pending_signals = {}  # Signals generated on T-1 to be executed on T
        
        for current_date in dates:
            date_str = current_date.strftime('%Y%m%d')
            
            # --- 1. Execute Trades from T-1 Signals at T's Open Price ---
            if pending_signals:
                target_weights = pending_signals
                
                # Fetch today's data to get open prices and pre_close for limit up/down check
                current_opens = {}
                pre_closes = {}
                for symbol in set(list(self.positions.keys()) + list(target_weights.keys())):
                    try:
                        # get_daily_data might return multiple days if we query a range, but we query 1 day
                        # To get pre_close safely, we can query T-5 to T and take the last 2 rows
                        start_dt = (current_date - pd.Timedelta(days=10)).strftime('%Y%m%d')
                        df = self.data_loader.get_daily_data(symbol, start_date=start_dt, end_date=date_str)
                        if not df.empty and date_str in df.index.strftime('%Y%m%d').values:
                            today_idx = df.index.get_loc(df[df.index.strftime('%Y%m%d') == date_str].index[0])
                            current_opens[symbol] = df.iloc[today_idx]['open']
                            if today_idx > 0:
                                pre_closes[symbol] = df.iloc[today_idx-1]['close']
                            else:
                                pre_closes[symbol] = current_opens[symbol] # Fallback
                    except Exception as e:
                        # print(f"Failed to fetch open price for {symbol}: {e}")
                        pass
                
                # Calculate estimated portfolio value at open for target allocation
                portfolio_value_at_open = self.cash
                for symbol, qty in self.positions.items():
                    price = current_opens.get(symbol, 0) # If suspended, price might be 0, we can't trade it anyway
                    if price > 0:
                        portfolio_value_at_open += price * qty
                    else:
                        # Try to get last close if suspended
                        try:
                            start_dt = (current_date - pd.Timedelta(days=20)).strftime('%Y%m%d')
                            df = self.data_loader.get_daily_data(symbol, start_date=start_dt, end_date=date_str)
                            if not df.empty:
                                portfolio_value_at_open += df.iloc[-1]['close'] * qty
                        except:
                            pass

                # Execute Sells
                for symbol in list(self.positions.keys()):
                    if symbol not in target_weights or target_weights[symbol] == 0:
                        # Sell all
                        price = current_opens.get(symbol)
                        pre_close = pre_closes.get(symbol)
                        if price and pre_close:
                            # Check Limit Down (cannot sell)
                            if price <= pre_close * 0.905:
                                print(f"[{date_str}] Limit Down! Cannot sell {symbol} @ {price:.2f}")
                                continue
                                
                            qty = self.positions.pop(symbol)
                            # Apply slippage (e.g. 0.002) and stamp duty (0.0005) and commission
                            slippage = 0.002
                            stamp_duty = 0.0005
                            actual_price = price * (1 - slippage)
                            revenue = actual_price * qty * (1 - self.commission - stamp_duty)
                            self.cash += revenue
                            print(f"[{date_str}] Sell {symbol}: {qty} shares @ {actual_price:.2f} (Open: {price:.2f})")
                    else:
                        # Rebalance: Sell partial
                        price = current_opens.get(symbol)
                        pre_close = pre_closes.get(symbol)
                        if price and pre_close:
                            target_val = portfolio_value_at_open * target_weights[symbol]
                            current_val = self.positions[symbol] * price
                            if current_val > target_val:
                                # Check Limit Down
                                if price <= pre_close * 0.905:
                                    print(f"[{date_str}] Limit Down! Cannot sell {symbol}")
                                    continue
                                    
                                diff_val = current_val - target_val
                                qty_to_sell = int(diff_val / price / 100) * 100
                                if qty_to_sell > 0:
                                    slippage = 0.002
                                    stamp_duty = 0.0005
                                    actual_price = price * (1 - slippage)
                                    revenue = actual_price * qty_to_sell * (1 - self.commission - stamp_duty)
                                    self.cash += revenue
                                    self.positions[symbol] -= qty_to_sell
                                    print(f"[{date_str}] Sell {symbol}: {qty_to_sell} shares @ {actual_price:.2f} (Open: {price:.2f})")

                # Execute Buys
                for symbol, weight in target_weights.items():
                    target_val = portfolio_value_at_open * weight
                    price = current_opens.get(symbol)
                    pre_close = pre_closes.get(symbol)
                    
                    if price and pre_close:
                        current_qty = self.positions.get(symbol, 0)
                        current_val = current_qty * price
                        
                        if current_val < target_val:
                            # Check Limit Up (cannot buy)
                            if price >= pre_close * 1.095:
                                print(f"[{date_str}] Limit Up! Cannot buy {symbol} @ {price:.2f}")
                                continue
                                
                            diff_val = target_val - current_val
                            qty_to_buy = int(diff_val / price / 100) * 100
                            if qty_to_buy > 0:
                                slippage = 0.002
                                actual_price = price * (1 + slippage)
                                cost = qty_to_buy * actual_price * (1 + self.commission)
                                if self.cash >= cost:
                                    self.cash -= cost
                                    self.positions[symbol] = current_qty + qty_to_buy
                                    print(f"[{date_str}] Buy {symbol}: {qty_to_buy} shares @ {actual_price:.2f} (Open: {price:.2f})")
            
            # --- 2. Mark to Market at T's Close Price ---
            portfolio_value = self.cash
            current_closes = {}
            for symbol, qty in self.positions.items():
                try:
                    df = self.data_loader.get_daily_data(symbol, start_date=date_str, end_date=date_str)
                    if not df.empty:
                        price = df.iloc[-1]['close']
                        current_closes[symbol] = price
                        portfolio_value += price * qty
                    else:
                        # Fallback to previous known close
                        start_dt = (current_date - pd.Timedelta(days=20)).strftime('%Y%m%d')
                        df_prev = self.data_loader.get_daily_data(symbol, start_date=start_dt, end_date=date_str)
                        if not df_prev.empty:
                            portfolio_value += df_prev.iloc[-1]['close'] * qty
                except Exception as e:
                    pass
            
            self.portfolio_history.append({
                'date': current_date,
                'value': portfolio_value
            })
            
            # --- 3. Generate Signals for T+1 ---
            signals = self.strategy.select_stocks(date_str, self.data_loader, current_positions=self.positions)
            
            pending_signals = {}
            if signals is not None and len(signals) > 0:
                if isinstance(signals, list):
                    n = len(signals)
                    pending_signals = {s: 1.0/n for s in signals}
                elif isinstance(signals, dict):
                    pending_signals = signals
                    
                total_weight = sum(pending_signals.values())
                if total_weight > 1.0:
                    pending_signals = {k: v/total_weight for k, v in pending_signals.items()}
                                    
        # Calculate performance
        df_result = pd.DataFrame(self.portfolio_history)
        if df_result.empty:
            print("No portfolio history generated.")
            return df_result
            
        df_result.set_index('date', inplace=True)
        
        final_value = df_result.iloc[-1]['value']
        total_return = (final_value / self.initial_capital) - 1
        
        days = len(df_result)
        if days > 0:
            annualized_return = (final_value / self.initial_capital) ** (252 / days) - 1
        else:
            annualized_return = 0.0

        df_result['cummax'] = df_result['value'].cummax()
        df_result['drawdown'] = (df_result['value'] - df_result['cummax']) / df_result['cummax']
        max_drawdown = abs(df_result['drawdown'].min())
        
        print(f"\nBacktest Finished.")
        print(f"Final Value: {final_value:.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        
        return df_result
