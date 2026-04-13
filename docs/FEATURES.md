# 因子库详解 (Feature Engineering Guide)

## 一、因子概览

本框架内置 9 大类因子，涵盖技术面、基本面、资金面、主观交易逻辑、形态识别和事件驱动：

| 因子类 | 数量 | 说明 |
|--------|------|------|
| PandasTAFactor | 130+ | pandas-ta 库提供的技术指标 |
| FinancialFactor | 10+ | ROE、净利润增长率等财务指标 |
| FundFlowFactor | 5+ | 主力资金净流入等 |
| MarketFactor | 3+ | 大盘基准对比特征 |
| BoardFactor | 1 | 行业/概念编码 |
| SubjectiveFactor | 8+ | 涨停溢价、弱转强、量价背离等 |
| PatternFactor | 5+ | 均线多头、箱体突破、红三兵等 |
| EventFactor | 3+ | 龙虎榜资金对齐 |
| LabelGenerator | 5种目标 | 收益率/排名标签生成 |

---

## 二、因子详解

### 2.1 PandasTAFactor (技术指标因子)

基于 [pandas-ta](https://github.com/twopirllc/pandas-ta) 库，提供 130+ 技术指标。

**配置示例**:
```yaml
- name: "PandasTAFactor"
  params:
    strategy: "custom"  # custom 表示使用自定义指标列表
    features:
      - 'MACD_12_26_9'      # MACD 主线
      - 'MACDs_12_26_9'     # MACD 信号线
      - 'MACDh_12_26_9'     # MACD 柱状图
      - 'RSI_14'            # 相对强弱指数
      - 'BBU_5_2.0_2.0'     # 布林带上轨
      - 'BBL_5_2.0_2.0'     # 布林带下轨
      - 'BBM_5_2.0_2.0'     # 布林带中轨
      - 'BBP_5_2.0_2.0'     # 布林带位置
      - 'BBB_5_2.0_2.0'     # 布林带带宽
      - 'ATRr_14'           # 真实波幅
      - 'SMA_10'            # 简单移动平均
      - 'D_9_3'             # D 值 (Stochastic)
      - 'K_9_3'             # K 值 (Stochastic)
      - 'J_9_3'             # J 值 (Stochastic)
      - 'OBV'               # 能量潮
      - 'fin_roe'           # 财务 ROE
      - 'fin_net_profit'    # 净利润
      - 'fin_yoy_ni'        # 净利润同比增长
      - 'main_net_inflow'   # 主力净流入
      - 'main_net_ratio'    # 主力净流入占比
      - 'super_net_ratio'   # 超大单净流入占比
```

**常用指标速查**:

| 指标 | 命名格式 | 说明 |
|------|----------|------|
| MACD | `MACD_12_26_9` | DIF/DEA/柱状图 |
| RSI | `RSI_14` | 相对强弱指数 |
| Bollinger | `BB(U/L/M/P)_5_2.0_2.0` | 布林带上下轨/中轨/位置/带宽 |
| KD | `K_9_3`, `D_9_3`, `J_9_3` | 随机指标 |
| ATR | `ATRr_14` | 真实波幅 |
| SMA/EMA | `SMA_10`, `EMA_12` | 移动平均 |
| OBV | `OBV` | 能量潮 |

### 2.2 FinancialFactor (财务因子)

从财务报表中提取的基本面指标。

**计算指标**:
- `roe`: 净资产收益率 (净利润 / 净资产)
- `roa`: 资产收益率 (净利润 / 总资产)
- `gross_margin`: 毛利率
- `net_margin`: 净利率
- `debt_to_equity`: 资产负债率
- `current_ratio`: 流动比率
- `quick_ratio`: 速动比率
- `yoy_growth`: 营收同比增长率
- `qoq_growth`: 营收环比增长率
- `profit_growth`: 利润同比增长率

**配置示例**:
```yaml
- name: "FinancialFactor"
  params: {}
```

### 2.3 FundFlowFactor (资金流因子)

追踪资金在个股中的流入流出情况。

**计算指标**:
- `main_net_inflow`: 主力资金净流入 (万元)
- `main_net_ratio`: 主力资金净流入占比
- `super_net_inflow`: 超大单资金净流入
- `super_net_ratio`: 超大单净流入占比
- `retail_net_inflow`: 散户资金净流入

**计算逻辑**:
```python
# 主力通常定义为:
# - 成交量 > 50 万元的买卖单
# 或者使用其他定义 (如 100 万以上)
main_net_inflow = 主动买入额 - 主动卖出额
main_net_ratio = main_net_inflow / 总成交额
```

### 2.4 MarketFactor (大盘基准因子)

计算个股与大盘基准的对比特征。

**计算指标**:
- `benchmark_close`: 基准指数收盘价 (用于计算个股与大盘的相关性/超额收益)
- `excess_return`: 个股收益率 - 基准收益率
- `correlation_20d`: 过去20日与基准的相关系数

**配置示例**:
```yaml
- name: "MarketFactor"
  params:
    index_symbol: "sh.000300"  # 沪深300作为基准
```

### 2.5 BoardFactor (行业/概念因子)

对个股所属的行业板块进行类别编码。

**计算指标**:
- `board_code`: 行业/概念板块代码
- `board_name`: 板块名称

**编码方式**:
```yaml
- name: "BoardFactor"
  params:
    encode_method: "category"  # 使用 pandas category 类型
```

### 2.6 SubjectiveFactor (主观交易逻辑因子)

将A股市场中一些**主观交易经验**量化为因子。这些因子捕捉的是"市场共识"和"资金行为"。

#### 2.6.1 涨停溢价 (LimitUpPremium)

**定义**: 涨停次日继续上涨的概率高于普通股票。

**计算逻辑**:
```python
# 判断昨日是否涨停
is_limit_up_yesterday = (close[-1] / close[-2] - 1) >= 0.099  # 近似10%

# 涨停溢价因子 = 昨日涨停 AND 今日高开
limit_up_premium = is_limit_up_yesterday and open[-1] > close[-2]
```

#### 2.6.2 弱转强 (WeakToStrong)

**定义**: 昨日走势弱（收盘接近日内低点），今日转强（高开或放量上涨）。

**计算逻辑**:
```python
# 昨日弱势: 收盘价低于开盘价，且接近日内低点
weak_yesterday = close[-1] < open[-1] and close[-1] < (high[-1] + low[-1]) / 2

# 今日转强: 跳空高开或放量大涨
strong_today = open[-1] > close[-2] * 1.01 or volume[-1] > volume[-2] * 1.5

weak_to_strong = weak_yesterday and strong_today
```

#### 2.6.3 量价背离 (PriceVolumeDivergence)

**定义**: 价格上涨但成交量萎缩（可能见顶），或价格下跌但成交量放大（可能见底）。

**计算逻辑**:
```python
# 计算过去5日价格变化和成交量变化的相关性
price_change = (close[-1] - close[-5]) / close[-5]
volume_change = (volume[-1] - volume[-5]) / volume[-5]

# 量价背离: 价格涨但量跌，或价格跌但量涨
divergence = price_change * volume_change < 0
```

#### 2.6.4 换手率突破 (TurnoverBreakout)

**定义**: 换手率突然放大，通常伴随突破。

```python
# 换手率突破: 今日换手率 > 过去20日平均换手率的2倍
avg_turnover_20d = turnover[-20:].mean()
turnover_breakout = turnover[-1] > avg_turnover_20d * 2
```

### 2.7 PatternFactor (形态识别因子)

通过向量化代码识别经典技术形态。

#### 2.7.1 均线多头排列 (BullishMAAlignment)

**定义**: 短期均线在长期均线上方依次排列。

**识别条件**:
```python
ma5 = close.rolling(5).mean()
ma10 = close.rolling(10).mean()
ma20 = close.rolling(20).mean()
ma60 = close.rolling(60).mean()

bullish_ma = ma5 > ma10 > ma20 > ma60
```

#### 2.7.2 箱体平台突破 (BoxBreakout)

**定义**: 股价在箱体震荡后突破上沿。

**识别条件**:
```python
# 计算过去20日最高点和最低点
highest_20d = high[-20:].max()
lowest_20d = low[-20:].min()

# 突破: 今日收盘价突破箱体上沿
breakout = close[-1] > highest_20d

# 伴随放量
volume_increase = volume[-1] > volume[-20:].mean() * 1.5
```

#### 2.7.3 龙头首阴反包 (FirstDropEngulfing)

**定义**: 涨停次日出现阴线，第三日反包收回。

**识别条件**:
```python
# 昨日涨停
limit_up_1 = close[-2] / close[-3] >= 0.099

# 今日收阴
drop_today = close[-1] < open[-1]

# 明日 (当前) 反包: 收盘价 > 昨日收盘价
engulfing = close[-1] > close[-2]

first_drop_engulfing = limit_up_1 and drop_today and engulfing
```

#### 2.7.4 红三兵 (ThreeWhiteSoldiers)

**定义**: 三根连续上涨的小阳线，收盘价逐步抬升。

**识别条件**:
```python
# 三根阳线，收盘价依次上升
c1 = close[-3] > open[-3]  # 第一天
c2 = close[-2] > open[-2] and close[-2] > close[-3]  # 第二天
c3 = close[-1] > open[-1] and close[-1] > close[-2]  # 第三天

# 实体不能太大 (不是涨停)
body_size = close - open
not_limit_up = body_size < close * 0.07

three_soldiers = c1 and c2 and c3 and not_limit_up
```

#### 2.7.5 MACD底背离 (MACDDivergence)

**定义**: 价格创出新低，但 MACD 未创新低。

**识别条件**:
```python
# 价格创20日新低
price_new_low = close[-1] == close[-20:].min()

# MACD 未创新低 (或比前期低点高)
macd_value = ema12 - ema26  # MACD 线
macd_not_new_low = macd_value[-1] > macd_value[-20:].min() * 1.05  # 留5%余量

macd_divergence = price_new_low and macd_not_new_low
```

### 2.8 EventFactor (事件驱动因子)

将离散的市场事件对齐到日线数据中。

#### 2.8.1 龙虎榜 (DragonTigerList)

**数据来源**: 上交所/深交所公布的龙虎榜数据

**事件字段**:
- `lhb_buy_amount`: 机构席位买入金额
- `lhb_sell_amount`: 机构席位卖出金额
- `lhb_net_amount`: 机构席位净买入额
- `ymt_buy_amount`: 知名游资买入金额
- `ymt_net_amount`: 游资净买入额

**对齐逻辑**:
```python
# 将事件表按日期左连接到日线
# 若某股票某日上了龙虎榜，则 lhb_* 字段有值，否则为 NaN
merged = daily_k.merge(events, on=['date', 'symbol'], how='left')
```

### 2.9 LabelGenerator (标签生成器)

生成训练目标标签。

**target_type 选项**:

| 类型 | 说明 | 适用目标 |
|------|------|----------|
| `regression` | 未来 N 日收益率 | `reg:squarederror` |
| `binary` | 涨跌 (1/0) | `binary:logistic` |
| `classification_3` | 大涨/平盘/大跌 (1/0/-1) | `multi:softmax` |
| `excess_return_binary` | 是否跑赢基准 | `binary:logistic` |
| `rank_pct` | 截面收益率排名 (0.0~1.0) | `rank:pairwise` |

**配置示例**:
```yaml
- name: "LabelGenerator"
  params:
    horizon: 3                    # 预测未来3日
    target_type: "rank_pct"       # 截面百分比排名
```

**rank_pct 计算逻辑**:
```python
# 对每个日期的股票按收益率排名
# 返回 0.0 ~ 1.0 的百分比排名
label = grouped['return'].rank(pct=True)
```

---

## 三、横截面处理 (Cross-Sectional Processing)

在输入模型前，必须对因子进行横截面处理以消除市场噪音。

### 3.1 MAD 去极值

**原理**: 剔除超过中位数绝对偏差 3 倍的异常值。

```python
def mad_clip(series, threshold=3.0):
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return series
    upper = median + threshold * mad * 1.4826  # 1.4826 使 MAD 渐变于标准差
    lower = median - threshold * mad * 1.4826
    return series.clip(lower, upper)
```

### 3.2 Z-Score 标准化

**原理**: 将因子值转换为标准正态分布，消除大盘 Beta 影响。

```python
def zscore(series):
    mean = series.mean()
    std = series.std()
    if std == 0:
        return series * 0  # 返回0
    return (series - mean) / std
```

### 3.3 处理流程

```python
# 对每个日期的横截面分别处理
for date in dates:
    cross_section = features[features.index == date]

    for col in feature_cols:
        cross_section[col] = mad_clip(cross_section[col])
        cross_section[col] = zscore(cross_section[col])

    features[features.index == date] = cross_section
```

---

## 四、动态过滤 (Dynamic Filter)

### 4.1 僵尸股过滤

**定义**: 流动性枯竭的股票，日均成交额低于阈值。

```python
def filter_zombie(df, min_avg_turnover=10000000):
    # 计算过去20日日均成交额
    avg_turnover = df['amount'][-20:].mean()
    return avg_turnover >= min_avg_turnover
```

### 4.2 次新股过滤

**定义**: 上市时间不满指定天数的股票。

```python
def filter_new_stock(df, min_listed_days=120):
    listed_date = df['symbol'].iloc[0]  # 需要从 basics 表获取
    days_since_listed = (today - listed_date).days
    return days_since_listed >= min_listed_days
```

---

## 五、特征缓存 (Feature Caching)

### 5.1 缓存键生成

```python
import hashlib
import json

def get_config_hash(features_config):
    """生成配置的 MD5 指纹"""
    config_str = json.dumps(features_config, sort_keys=True)
    return hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]
```

### 5.2 缓存路径

```
data/local_lake/features/
└── <config_hash>/                    # 如 "a1b2c3d4"
    ├── sh.600000_20230101_20230630_with_label.parquet
    ├── sz.000001_20230101_20230630_with_label.parquet
    └── ...
```

### 5.3 缓存命中逻辑

```python
cache_path = f"data/local_lake/features/{config_hash}/{symbol}_{start}_{end}_{label_flag}.parquet"

if os.path.exists(cache_path):
    features_df = pd.read_parquet(cache_path)
else:
    features_df = compute_features(...)
    features_df.to_parquet(cache_path)  # 保存缓存
```

---

## 六、自定义因子开发指南

### 6.1 创建新因子

```python
from .base_factor import BaseFactor
import pandas as pd

class MyCustomFactor(BaseFactor):
    def calculate(self, df):
        """
        实现你的因子计算逻辑

        Args:
            df: 包含 OHLCV + symbol 的 DataFrame

        Returns:
            pd.Series: 因子值（索引与 df 相同）
        """
        # 你的计算逻辑
        result = df['close'].pct_change()

        # 确保返回 Series 且索引对齐
        return result
```

### 6.2 注册因子

在 `features/factors/__init__.py` 或 `FACTOR_MAP` 中注册。

### 6.3 在配置中使用

```yaml
features:
  - name: "MyCustomFactor"
    params:
      param1: value1
```
