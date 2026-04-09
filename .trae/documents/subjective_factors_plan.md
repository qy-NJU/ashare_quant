# 主观交易逻辑量化因子提取与实现计划

## 一、 现状分析与目标 (Current State Analysis & Goal)

在目前的 `ashare_quant` 框架中，特征工程主要依赖于 `pandas-ta` 提供的传统技术指标（如 MACD、RSI、均线等）。这些指标在 A 股市场中往往滞后且同质化严重。
为了提升模型的 Alpha 捕捉能力，我们需要将 A 股短线交易中最核心的**主观博弈逻辑**（如打板情绪、换手突破、量价背离、弱转强等）翻译为机器可以理解的**量化因子**。

**目标**：在 `features/factors/` 目录下新增自定义因子模块，将“海安炒家”提供的实战短线逻辑转化为具体的 Pandas 特征列，并无缝接入现有的 `FeaturePipeline` 中。

## 二、 提议的因子与实现逻辑 (Proposed Factors)

我们将新增一个文件 `features/factors/subjective.py`，包含一个 `SubjectiveFactor` 类，计算以下四大核心逻辑因子：

### 1. 极限情绪因子：涨停与溢价 (Limit-up & Premium)
A 股的涨跌停制度是情绪的放大器。我们需要刻画股票最近是否处于极端情绪中。
- **is_limit_up**: 当日是否涨停。逻辑：`close >= round(pre_close * 1.095, 2)`（简化版，暂不考虑 20% 的科创/创业板，统一按 10% 近似）。
- **is_limit_down**: 当日是否跌停。逻辑：`close <= round(pre_close * 0.905, 2)`。
- **high_premium_tags**: 昨日最高溢价。逻辑：`high / pre_close - 1`。如果昨天涨停，今天的最高溢价能反映承接力度。

### 2. 资金共识因子：换手突破 (Turnover Breakout)
真正的龙头需要换手支持，缩量往往走不远。
- **turnover_rate_est**: 估算换手率（如果数据源没有直接提供）。逻辑：`volume * close` 作为成交额代理，再除以某个滚动均值来衡量“相对换手热度”。
- **vol_breakout_ratio**: 成交量爆发率。逻辑：`volume / volume.rolling(5).mean()`。反映今日资金介入的急剧程度。
- **price_new_high_20**: 20日新高标志。逻辑：`close >= close.rolling(20).max()`。上方无套牢盘的标志。

### 3. 风险预警因子：高位量价背离 (Price-Volume Divergence)
放量滞涨或长上影线是短线见顶的危险信号。
- **upper_shadow_ratio**: 上影线比例。逻辑：`(high - max(open, close)) / pre_close`。上影线越长，抛压越重。
- **high_vol_stagnation**: 高位放量滞涨。逻辑：近10日涨幅 > 20% 且 今日成交量 > 2倍的5日均量，但今日收盘涨幅 < 2% 甚至收跌。

### 4. 预期差因子：弱转强 (Weak to Strong)
这是短线超额收益的核心。昨天烂板或长上影（弱），今天却大幅高开（强）。
- **weak_to_strong_signal**: 弱转强特征。逻辑：昨日留下长上影线（`upper_shadow_ratio > 0.03`）且放量，但今日开盘超预期强势（`open > pre_close * 1.02`）。

## 三、 架构接入 (Architecture Integration)

1. **创建 `SubjectiveFactor` 类**：
   - 继承 `BaseFactor` 接口。
   - 在 `calculate(self, df)` 方法中，利用 Pandas 的向量化操作（shift, rolling, np.where 等）高效计算上述所有因子。
2. **修改 `configs/pipeline_config.yaml`**：
   - 在 `features` 列表中注册 `SubjectiveFactor`，使其随流水线自动执行。
3. **兼容性处理**：
   - 这些因子产生的空值（如 rolling 产生的开端 NaN）将由现有的 pipeline 自动处理。
   - 产生的绝对数值因子（如 `vol_breakout_ratio`）将完美受益于上一阶段刚刚完成的 **MAD 去极值和 Z-Score 标准化**，消除极端离群值的干扰。

## 四、 验证步骤 (Verification Steps)
1. 编写代码后，运行 `runner.py configs/pipeline_config.yaml`。
2. 检查日志，确认 `SubjectiveFactor` 成功被 Pipeline 加载并执行。
3. 运行 `analyze_importance.py`。
4. **核心验证**：观察 XGBoost 输出的特征重要性排名（Feature Importance），检查我们新加入的短线逻辑因子（如 `vol_breakout_ratio`、`is_limit_up` 等）是否排在了传统技术指标（如单纯的 MA 或 RSI）的前面。如果排名靠前，说明模型确实从这些主观逻辑中榨取到了 Alpha！