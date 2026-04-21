import json
import os
from datetime import datetime

class ReportGenerator:
    """
    Generates analysis reports combining model evaluation and strategy backtest results.
    """
    
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_markdown_report(self, model_metrics: dict, strategy_metrics: dict, 
                                trade_stats: dict, system_metrics: dict = None) -> str:
        """
        Generate a markdown report from the given metrics dictionaries.
        
        Args:
            model_metrics: dict from ModelEvaluator
            strategy_metrics: dict from StrategyEvaluator
            trade_stats: dict from StrategyEvaluator
            system_metrics: dict of system performance metrics
            
        Returns:
            Path to the generated report file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.output_dir, f'analysis_report_{timestamp}.md')
        
        with open(report_path, 'w') as f:
            f.write(f"# Quant Model Analysis Report\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 1. Model Predictive Performance
            f.write("## 1. Model Predictive Performance\n")
            if model_metrics:
                for k, v in model_metrics.items():
                    f.write(f"- **{k}**: {v:.4f}\n")
            else:
                f.write("No model metrics available.\n")
            f.write("\n")
                
            # 2. Financial & Strategy Performance
            f.write("## 2. Strategy Financial Performance\n")
            if strategy_metrics:
                f.write(f"- **Total Return**: {strategy_metrics.get('Total_Return', 0)*100:.2f}%\n")
                f.write(f"- **Annualized Return**: {strategy_metrics.get('Annualized_Return', 0)*100:.2f}%\n")
                f.write(f"- **Annualized Volatility**: {strategy_metrics.get('Annualized_Volatility', 0)*100:.2f}%\n")
                f.write(f"- **Sharpe Ratio**: {strategy_metrics.get('Sharpe_Ratio', 0):.4f}\n")
                f.write(f"- **Max Drawdown**: {strategy_metrics.get('Max_Drawdown', 0)*100:.2f}%\n")
                f.write(f"- **Calmar Ratio**: {strategy_metrics.get('Calmar_Ratio', 0):.4f}\n")
            else:
                f.write("No strategy metrics available.\n")
            f.write("\n")
                
            # 3. Trading Behavior & Statistics
            f.write("## 3. Trading Behavior & Statistics\n")
            if trade_stats:
                f.write(f"- **Total Trades**: {trade_stats.get('Total_Trades', 0)}\n")
                f.write(f"- **Win Rate**: {trade_stats.get('Win_Rate', 0)*100:.2f}%\n")
                f.write(f"- **Profit/Loss Ratio**: {trade_stats.get('Profit_Loss_Ratio', 0):.2f}\n")
                f.write(f"- **Average Holding Days**: {trade_stats.get('Average_Holding_Days', 0):.1f}\n")
                f.write(f"- **Total PnL**: {trade_stats.get('Gross_Profit', 0):.2f}\n")
                f.write(f"- **Total Fees**: {trade_stats.get('Total_Fees', 0):.2f}\n")
                
                # Top 5 Profits
                f.write("\n### Top 5 Profits\n")
                f.write("| Symbol | Start Date | End Date | Holding Days | PnL | Return % | Reasons |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                for trade in trade_stats.get('Top_5_Profits', []):
                    f.write(f"| {trade['symbol']} | {trade['start_date']} | {trade['end_date']} | "
                            f"{trade['holding_days']} | {trade['pnl']:.2f} | "
                            f"{trade['return_pct']*100:.2f}% | {trade['reasons']} |\n")
                            
                # Top 5 Losses
                f.write("\n### Top 5 Losses\n")
                f.write("| Symbol | Start Date | End Date | Holding Days | PnL | Return % | Reasons |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                for trade in trade_stats.get('Top_5_Losses', []):
                    f.write(f"| {trade['symbol']} | {trade['start_date']} | {trade['end_date']} | "
                            f"{trade['holding_days']} | {trade['pnl']:.2f} | "
                            f"{trade['return_pct']*100:.2f}% | {trade['reasons']} |\n")
            else:
                f.write("No trade statistics available.\n")
            f.write("\n")
                
            # 4. System & Engineering Performance
            f.write("## 4. System Engineering Performance\n")
            if system_metrics:
                for k, v in system_metrics.items():
                    if isinstance(v, float):
                        f.write(f"- **{k}**: {v:.4f}\n")
                    else:
                        f.write(f"- **{k}**: {v}\n")
            else:
                f.write("No system metrics provided.\n")
                
        return report_path

    def save_json_report(self, data: dict) -> str:
        """
        Save the raw metrics as a JSON file for programmatic access later.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.output_dir, f'analysis_data_{timestamp}.json')
        
        with open(report_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        return report_path
