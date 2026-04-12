# A股形态选股量化实现计划

## 一、 现状分析与目标 (Current State Analysis & Goal)

在A股交易中，“形态选股”是右侧交易者和趋势跟踪者最常用的方法。经典的形态（如均线多头排列、平台突破、龙头首阴反包、MACD底背离等）能够直观地反映出多空力量的对比和主力资金的控盘意图。
目前我们的 `ashare_quant` 框架已经有了传统技术指标（`pandas-ta`）和主观游资逻辑（`SubjectiveFactor`），但在**经典K线与均线组合形态**的量化识别上仍有空白。

**目标**：在 `features/factors/` 目录下新增 `pattern.py` 模块，将高胜率的A股经典形态转化为 `0/1` 的布尔值特征或连续强度的特征，让 XGBoost 模型能够识别图形。

## 二、 提议的形态因子 (Proposed Pattern Factors)

我们将实现以下四类实战中最常用、最有效的形态因子：

### 1. 趋势形态：均线多头排列 (Bullish MA Alignment)
- **逻辑**：短期均线 > 中期均线 > 长期均线，且均线呈向上发散状态。这是最稳健的主升浪形态。
- **量化公式**：`MA5 > MA10 > MA20 > MA60`，且 `MA20` 的斜率为正（`MA20 > MA20.shift(3)`）。

### 2. 爆发形态：箱体/平台突破 (Box/Consolidation Breakout)
- **逻辑**：股价在较窄的区间内横盘震荡洗盘（波动率极小），随后某天放量大涨（最好是涨停）突破区间最高点，开启主升浪。
- **量化公式**：
  - 过去20天的最高价与最低价之差较小：`(Max_20 - Min_20) / Min_20 < 0.15`。
  - 今日收盘价突破前期高点：`Close > Max_20.shift(1)`。
  - 放量：`Volume > 1.5 * MA(Volume, 5)`。

### 3. 接力形态：龙头首阴反包 (First Drop Engulfing)
- **逻辑**：强势股（连续大涨或涨停）首次出现回调收阴线（缩量），次日立刻高开高走收大阳线（或涨停），吞没昨日阴线，确立二波行情。
- **量化公式**：
  - T-2日大涨（涨幅 > 7%）。
  - T-1日收阴线（`Close < Open`）且缩量（`Volume < Volume.shift(1)`）。
  - T日收大阳线（`Close > Open`），且收盘价突破T-1日最高价（`Close > High.shift(1)`）。

### 4. 抄底形态：MACD 底背离 (MACD Bullish Divergence)
- **逻辑**：股价创出新低，但 MACD 的绿柱子变短，或者 DIF 线上穿 DEA 线的位置比上一次金叉的位置要高。代表杀跌动能衰竭。
- **量化公式**：
  - 股价近 5 日最低点 < 过去 20 日的最低点（创新低）。
  - MACD Histogram 的最小值 > 上一个波谷的最小值（动能背离）。
  - （由于严格的波峰波谷识别较复杂，我们采用简化版：价格创新低，但MACD指标值大于前低时的MACD值）。

## 三、 架构接入 (Architecture Integration)

1. **创建 `PatternFactor` 类**：
   - 继承 `BaseFactor` 接口，新建 `features/factors/pattern.py`。
   - 在 `calculate(self, df)` 中，利用 Pandas 计算出上述形态的布尔值，并转化为整数（1/0）或浮点数特征。
2. **修改流水线与配置**：
   - 在 `features/pipeline.py` 中注册 `PatternFactor`。
   - 在 `configs/pipeline_config.yaml` 的 `features` 列表中添加 `- name: "PatternFactor"`。
3. **兼容性**：
   - 这些 0/1 标志特征经过 `CrossSectionalProcessor` 时，由于使用了 Z-Score，会自动转变为相对强度的截面打分，完全兼容现有框架。

## 四、 验证步骤 (Verification Steps)
1. 编写代码后，运行 `runner.py configs/pipeline_config.yaml`，确认 Pipeline 能正常执行，无报错。
2. 观察输出日志，确保特征正常生成。
3. 运行 `analyze_importance.py` 观察形态特征（如 `pat_box_breakout`）是否对模型提供了增量 Alpha。