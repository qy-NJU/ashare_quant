import pandas as pd
import numpy as np
from datetime import datetime

class StrategyEvaluator:
    """
    Evaluates the financial and trading performance of a quantitative strategy.
    """
    
    def __init__(self, risk_free_rate: float = 0.03, trading_days: int = 252):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def evaluate_returns(self, daily_returns: pd.Series) -> dict:
        """
        Evaluate daily returns series of a strategy.
        
        Args:
            daily_returns: Series of daily portfolio returns
            
        Returns:
            dict containing return and risk metrics
        """
        if len(daily_returns) == 0:
            return {}
            
        # Cumulative return
        cum_returns = (1 + daily_returns).cumprod()
        total_return = cum_returns.iloc[-1] - 1
        
        # Annualized return
        years = len(daily_returns) / self.trading_days
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Annualized volatility
        ann_volatility = daily_returns.std() * np.sqrt(self.trading_days)
        
        # Sharpe Ratio
        excess_return = ann_return - self.risk_free_rate
        sharpe = excess_return / ann_volatility if ann_volatility > 0 else 0
        
        # Maximum Drawdown
        roll_max = cum_returns.cummax()
        drawdown = cum_returns / roll_max - 1.0
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        calmar = ann_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0
        
        return {
            'Total_Return': float(total_return),
            'Annualized_Return': float(ann_return),
            'Annualized_Volatility': float(ann_volatility),
            'Sharpe_Ratio': float(sharpe),
            'Max_Drawdown': float(max_drawdown),
            'Calmar_Ratio': float(calmar)
        }

    def analyze_trades(self, trades_df: pd.DataFrame) -> dict:
        """
        Analyze a dataframe of raw trade records.
        Assumes columns: date, symbol, action, qty, price, amount, fee, reason
        
        Args:
            trades_df: DataFrame of raw trades from backtest
            
        Returns:
            dict containing trade statistics
        """
        if len(trades_df) == 0:
            return {}
            
        # Identify completed trades (buy to sell)
        completed_trades = []
        positions = {}
        
        for _, row in trades_df.iterrows():
            sym = row['symbol']
            if sym not in positions:
                positions[sym] = {
                    'qty': 0, 'cash_flow': 0.0, 'start_date': row['date'], 
                    'reasons': set(), 'buy_qty': 0, 'buy_amount': 0.0, 
                    'sell_qty': 0, 'sell_amount': 0.0, 'total_fee': 0.0
                }
            
            pos = positions[sym]
            if abs(pos['qty']) < 1e-6:
                pos['start_date'] = row['date']
                pos['cash_flow'] = 0.0
                pos['reasons'] = set()
                pos['buy_qty'] = 0
                pos['buy_amount'] = 0.0
                pos['sell_qty'] = 0
                pos['sell_amount'] = 0.0
                pos['total_fee'] = 0.0
            
            pos['reasons'].add(str(row['reason']))
            pos['total_fee'] += row['fee']
            
            if row['action'] == 'BUY':
                pos['qty'] += row['qty']
                pos['cash_flow'] -= row['amount']
                pos['buy_qty'] += row['qty']
                pos['buy_amount'] += row['amount']
            elif row['action'] == 'SELL':
                pos['qty'] -= row['qty']
                pos['cash_flow'] += row['amount']
                pos['sell_qty'] += row['qty']
                pos['sell_amount'] += row['amount']
                
            if abs(pos['qty']) < 1e-6:
                avg_buy = pos['buy_amount'] / pos['buy_qty'] if pos['buy_qty'] > 0 else 0
                avg_sell = pos['sell_amount'] / pos['sell_qty'] if pos['sell_qty'] > 0 else 0
                
                # Convert dates to datetime to calculate holding period
                try:
                    start_dt = pd.to_datetime(str(pos['start_date']), format='%Y%m%d')
                    end_dt = pd.to_datetime(str(row['date']), format='%Y%m%d')
                    holding_days = (end_dt - start_dt).days
                except:
                    holding_days = 0
                    
                completed_trades.append({
                    'symbol': sym,
                    'start_date': pos['start_date'],
                    'end_date': row['date'],
                    'holding_days': holding_days,
                    'avg_buy_price': avg_buy,
                    'avg_sell_price': avg_sell,
                    'pnl': pos['cash_flow'],
                    'return_pct': pos['cash_flow'] / pos['buy_amount'] if pos['buy_amount'] > 0 else 0,
                    'total_fee': pos['total_fee'],
                    'reasons': ', '.join(pos['reasons'])
                })
                
        df = pd.DataFrame(completed_trades)
        if len(df) == 0:
            return {'Total_Trades': 0}
            
        # Trade statistics
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(df)
        
        avg_profit = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
        
        avg_holding_days = df['holding_days'].mean()
        total_fees = df['total_fee'].sum()
        total_pnl = df['pnl'].sum()
        
        # Get top 5 profits and losses
        df_sorted = df.sort_values('pnl')
        top_5_losses = df_sorted.head(5).to_dict(orient='records')
        top_5_profits = df_sorted.tail(5).sort_values('pnl', ascending=False).to_dict(orient='records')
        
        return {
            'Total_Trades': len(df),
            'Win_Rate': float(win_rate),
            'Profit_Loss_Ratio': float(profit_loss_ratio),
            'Average_Profit': float(avg_profit),
            'Average_Loss': float(avg_loss),
            'Average_Holding_Days': float(avg_holding_days),
            'Total_Fees': float(total_fees),
            'Gross_Profit': float(total_pnl),
            'Net_Profit': float(total_pnl - total_fees),
            'Top_5_Profits': top_5_profits,
            'Top_5_Losses': top_5_losses
        }
