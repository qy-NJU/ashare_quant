import numpy as np
import pandas as pd


class RiskController:

    def __init__(self,
                 max_single_stock_weight=0.15,
                 max_industry_weight=0.30,
                 max_total_position=0.80,
                 max_stocks_same_industry=2,
                 blacklist=None,
                 liquidity_min_amount=20_000_000,
                 market_state_recognizer=None):
        self.max_single_stock_weight = max_single_stock_weight
        self.max_industry_weight = max_industry_weight
        self.max_total_position = max_total_position
        self.max_stocks_same_industry = max_stocks_same_industry
        self.blacklist = set(blacklist) if blacklist else set()
        self.liquidity_min_amount = liquidity_min_amount
        self.market_state_recognizer = market_state_recognizer

        self.violations_log = []
        self.position_caps = {}

    # ── pre-trade checks ──────────────────────────────────────

    def pre_trade_check(self, symbol, date, data_repo, stock_list_df=None):
        violations = []

        if symbol in self.blacklist:
            violations.append(f"{symbol}: blacklisted")
            return False, violations

        if stock_list_df is not None and not stock_list_df.empty:
            row = stock_list_df[stock_list_df['symbol'] == symbol]
            if not row.empty:
                name = str(row.iloc[0].get('name', ''))
                if 'ST' in name.upper() or '*ST' in name.upper():
                    violations.append(f"{symbol}: ST stock")
                    return False, violations

        df = data_repo.get_daily_data(symbol,
                                      start_date=(pd.to_datetime(date) - pd.Timedelta(days=25)).strftime('%Y%m%d'),
                                      end_date=date)
        if df.empty:
            violations.append(f"{symbol}: no data")
            return False, violations

        volume = df['volume']
        close = df['close']
        if 'amount' in df.columns:
            avg_amount = df['amount'].rolling(20).mean().iloc[-1]
        else:
            avg_amount = (volume * close).rolling(20).mean().iloc[-1]

        if pd.isna(avg_amount) or avg_amount < self.liquidity_min_amount:
            violations.append(f"{symbol}: insufficient liquidity ({avg_amount:.0f})")
            return False, violations

        if len(df) >= 1 and volume.iloc[-1] == 0:
            violations.append(f"{symbol}: suspended")
            return False, violations

        if self._is_limit_up(df):
            violations.append(f"{symbol}: limit-up")
            return False, violations

        return True, violations

    def _is_limit_up(self, df):
        if len(df) < 2:
            return False
        prev_close = df['close'].iloc[-2]
        current_close = df['close'].iloc[-1]
        if prev_close <= 0:
            return False
        change_pct = (current_close - prev_close) / prev_close

        symbol = str(df.index[-1])
        if 'sz.30' in symbol:
            return change_pct >= 0.199
        elif 'sh.688' in symbol:
            return change_pct >= 0.199
        elif 'bj.' in symbol:
            return change_pct >= 0.299
        else:
            return change_pct >= 0.099

    # ── position checks ───────────────────────────────────────

    def validate_position(self, target_positions, current_positions, date,
                          data_repo=None, board_manager=None, dynamic_cap=None):
        result = dict(target_positions)
        violations = []

        effective_cap = self.max_total_position
        if dynamic_cap is not None:
            effective_cap = min(effective_cap, dynamic_cap)

        total_weight = sum(result.values())
        if total_weight > effective_cap:
            scale = effective_cap / total_weight
            result = {s: w * scale for s, w in result.items()}
            violations.append(f"position_capped: {total_weight:.2%} → {effective_cap:.2%}")

        for sym, weight in list(result.items()):
            if weight > self.max_single_stock_weight:
                result[sym] = self.max_single_stock_weight
                violations.append(f"{sym}: max_weight_cap to {self.max_single_stock_weight:.2%}")

        if board_manager is not None:
            industry_exposure = {}
            for sym in result:
                industry = self._get_industry(sym, board_manager, data_repo)
                if industry:
                    industry_exposure.setdefault(industry, 0)
                    industry_exposure[industry] += result[sym]

            for industry, exposure in industry_exposure.items():
                if exposure > self.max_industry_weight:
                    scale = self.max_industry_weight / exposure
                    for sym in result:
                        if self._get_industry(sym, board_manager, data_repo) == industry:
                            result[sym] *= scale
                    violations.append(f"industry_cap: {industry} → {self.max_industry_weight:.2%}")

        return result, violations

    def validate_total_position(self, target_positions, market_state, data_repo, date):
        dynamic_cap = self.max_total_position

        if market_state == 'crash':
            dynamic_cap = 0.10
        elif market_state == 'trending_down':
            if self.market_state_recognizer:
                reduction, reason = self.market_state_recognizer.should_reduce_position(date, data_repo)
                if reduction > 0:
                    dynamic_cap = self.max_total_position * (1 - reduction)

        total = sum(target_positions.values())
        if total > dynamic_cap:
            scale = dynamic_cap / total
            return {s: w * scale for s, w in target_positions.items()}, dynamic_cap
        return target_positions, dynamic_cap

    def _get_industry(self, symbol, board_manager, data_repo):
        try:
            if board_manager:
                mapping = board_manager.get_industry_mapping()
                if not mapping.empty:
                    sym_clean = symbol.replace('sh.', '').replace('sz.', '').replace('bj.', '')
                    row = mapping[mapping['symbol'] == sym_clean]
                    if not row.empty:
                        return row.iloc[0].get('industry')
        except Exception:
            pass
        return None

    # ── post-trade analysis ───────────────────────────────────

    def post_trade_analysis(self, portfolio_value_history, trade_log):
        analysis = {}

        if portfolio_value_history and len(portfolio_value_history) > 1:
            if isinstance(portfolio_value_history[0], dict):
                values = np.array([d.get('value', d.get('total', 0)) for d in portfolio_value_history])
            else:
                values = np.array(portfolio_value_history)

            if values.min() <= 0:
                return analysis

            returns = np.diff(values) / values[:-1]

            analysis['daily_return_mean'] = float(np.mean(returns))
            analysis['daily_return_std'] = float(np.std(returns))
            analysis['sharpe'] = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0

            cumulative = values / values[0]
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = cumulative / running_max - 1
            analysis['max_drawdown'] = float(drawdowns.min())

            analysis['total_return'] = float(values[-1] / values[0] - 1)

        if trade_log:
            trades_df = pd.DataFrame(trade_log)
            analysis['total_trades'] = len(trades_df)
            if 'pnl' in trades_df.columns:
                analysis['win_rate'] = float((trades_df['pnl'] > 0).mean())
                analysis['avg_pnl'] = float(trades_df['pnl'].mean())

        return analysis

    def add_to_blacklist(self, symbol, reason=""):
        self.blacklist.add(symbol)
        self.violations_log.append({'symbol': symbol, 'action': 'blacklist_add', 'reason': reason})

    def remove_from_blacklist(self, symbol):
        self.blacklist.discard(symbol)
