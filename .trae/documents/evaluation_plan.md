# 量化模型与系统分析评估计划 (Analysis & Evaluation Plan)

为了全面评估量化交易模型和系统在每次训练和回测后的表现，我们将重新构建 `analysis` 模块。评估体系将分为以下四个核心维度：

## 1. 模型预测能力评估 (Model Predictive Performance)
主要用于评估机器学习/深度学习模型本身的预测准确度，脱离具体交易规则：
- **IC / Rank IC (信息系数)**: 评估模型预测收益率与实际收益率之间的相关性（截面和时序）。
- **回归指标 (Regression Metrics)**: MSE (均方误差), MAE (平均绝对误差) - 适用于预测具体收益率数值的模型。
- **分类指标 (Classification Metrics)**: Accuracy (准确率), Precision (精确率), Recall (召回率), F1-Score, AUC - 适用于预测涨跌方向的模型。

## 2. 策略回测财务指标 (Financial & Strategy Performance)
用于评估模型预测转化为实际交易信号后的资金曲线表现：
- **收益指标**:
  - 累计收益率 (Cumulative Return)
  - 年化收益率 (Annualized Return)
  - 超额收益率 (Excess Return / Alpha) - 相对于基准（如沪深300）的超额表现。
- **风险指标**:
  - 最大回撤 (Maximum Drawdown, MDD) - 衡量极端风险。
  - 年化波动率 (Annualized Volatility)。
- **风险调整后收益**:
  - 夏普比率 (Sharpe Ratio) - 承受单位风险带来的超额收益。
  - 卡玛比率 (Calmar Ratio) - 年化收益率与最大回撤的比值。

## 3. 交易行为与统计分析 (Trading Behavior & Statistics)
深入分析每一笔交易的具体表现，寻找策略优化空间（类似我们刚刚做的 csv 分析）：
- **胜率与盈亏比**: 交易胜率 (Win Rate)、平均盈利/平均亏损 (Profit/Loss Ratio)。
- **极端交易分析**: 最大单笔盈利/亏损 (Top 5 Profits/Losses)，分析尾部风险。
- **持仓特征**: 平均持仓周期、换手率 (Turnover Rate)、单日最大开仓数量。
- **成本分析**: 手续费、滑点占总利润的比重。

## 4. 系统工程与性能评估 (System & Engineering Performance)
用于评估系统运行的效率与稳定性：
- **耗时评估**: 特征工程耗时、模型训练耗时、单步推理延迟 (Inference Latency)。
- **资源占用**: 内存消耗峰值、GPU显存使用率（如有）。

---

## 实施步骤 (Implementation Steps)

1. **重构分析模块目录**:
   - 创建 `analysis/model_evaluator.py` (计算IC, MSE, 准确率等)
   - 创建 `analysis/strategy_evaluator.py` (计算夏普, 回撤, 胜率等)
   - 创建 `analysis/report_generator.py` (整合数据，生成可视化图表或Markdown报告)
2. **集成到训练管道**:
   - 在 `train.py` 或 `main.py` 结束后自动调用分析模块，并输出 Summary。
3. **数据持久化**:
   - 将每次的评估结果记录到 `logs/` 或 `reports/` 目录下，方便对比不同版本模型的好坏。
