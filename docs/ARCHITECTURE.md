# 架构设计文档 (Architecture Design)

## 一、设计目标

本框架的设计目标是为 A 股量化研究提供一个**工业级、可复现、配置驱动**的完整闭环。核心设计原则：

1. **配置优于代码**: 任何策略参数调整、因子切换、数据范围修改，都应通过修改 YAML 配置完成，无需改动代码
2. **离线数据湖**: 所有数据预先同步到本地，避免回测时即时下载导致的不稳定
3. **防作弊机制**: 内置多种过滤器（ST、僵尸股、次新股、截面标准化）防止过拟合到未来数据
4. **A 股特色**: 完整支持 T+1、涨跌停、印花税、大盘风控等 A 股特有机制

## 二、模块架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Runner (runner.py)                     │
│                  配置解析 + 流水线编排                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │  Data Layer │  │Feature Layer│  │  Model Layer│
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │Repository   │  │Pipeline     │  │XGBoost      │
    │PoolManager  │  │Factors      │  │Sklearn      │
    └─────────────┘  └─────────────┘  └─────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
    ┌─────────────────────┐       ┌─────────────────────┐
    │  Strategy Layer      │       │  Backtest Layer      │
    │  MLStrategy          │       │  BacktestEngine      │
    │  MomentumStrategy    │       │  T+1 Execution       │
    └─────────────────────┘       └─────────────────────┘
```

## 三、各模块职责

### 3.1 数据层 (Data Layer)

**DataRepository** (`data/repository.py`)
- 统一的数据访问入口
- 从 Local Data Lake (Parquet 文件) 加载数据
- 提供 `get_daily_data(symbol, start_date, end_date)` 接口
- 不再包含网络下载逻辑（已迁移到 `scripts/sync_data.py`）

**StockPoolManager** (`data/pool_manager.py`)
- 根据板块（主板/创业板/科创板）、交易所（沪/深/北）、ST 状态过滤股票
- 返回符合条件的目标股票列表

**数据目录结构**:
```
data/local_lake/
├── basics/              # 股票基本信息 (symbol, name, list_date, ...)
├── daily_k/             # 日线数据 (*.parquet per stock)
│   ├── sh.600000.parquet  # 包含全量历史日线数据
│   ├── sz.000001.parquet
│   └── ...
├── events/             # 事件数据 (龙虎榜等)
└── features/           # 特征缓存 (MD5 指纹)
    └── <config_hash>/
        ├── sh.600000_20230101_20230630_with_label.parquet
        └── ...
```

### 3.2 特征工程层 (Feature Layer)

**FeaturePipeline** (`features/pipeline.py`)
- 串联所有因子计算
- 输入: 原始 OHLCV 数据 + symbol
- 输出: 原始数据 + 所有因子值

**因子类型**:

| 因子类 | 文件 | 说明 |
|--------|------|------|
| PandasTAFactor | `pandas_ta_factor.py` | 130+ 技术指标 (MACD, RSI, Bollinger Bands 等) |
| FinancialFactor | `financial.py` | ROE、净利润增长率等财务指标 |
| FundFlowFactor | `fund_flow.py` | 主力/超大单资金净流入 |
| MarketFactor | `market.py` | 与大盘基准的对比特征 |
| BoardFactor | `fundamental.py` | 行业/概念编码 |
| SubjectiveFactor | `subjective.py` | 涨停溢价、量价背离、弱转强等主观交易逻辑 |
| PatternFactor | `pattern.py` | 均线多头、箱体突破、红三兵、MACD 背离等形态 |
| EventFactor | `event_driven.py` | 龙虎榜机构/游资净买入对齐 |
| LabelGenerator | `technical.py` | 生成训练标签（收益率/排名） |

**CrossSectionalProcessor** (`features/processor.py`)
- MAD 去极值: 剔除超过 3σ 的异常值
- Z-Score 标准化: 消除大盘 Beta 影响

**DynamicFilter** (`features/processor.py`)
- 剔除僵尸股: 近 20 日日均成交额 < 阈值
- 剔除次新股: 上市不满指定天数

### 3.3 模型层 (Model Layer)

**XGBoostWrapper** (`models/xgboost_model.py`)
- 支持增量学习 (`partial_train`)
- 支持 `rank:pairwise` 排序目标
- 支持原生类别特征
- 模型保存/加载

**SklearnWrapper** (`models/machine_learning.py`)
- 通用 sklearn 模型封装

### 3.4 策略层 (Strategy Layer)

**MLStrategy** (`strategies/ml_strategy.py`)
- 基于 ML 模型预测分数选股
- 大盘风控: 沪深300跌破20日线时空仓
- 换手率控制: 限制每次调仓比例
- 权重分配: 等权或按分数加权

**MomentumStrategy** (`strategies/momentum_strategy.py`)
- 传统动量策略示例
- 按历史收益率排名选股

### 3.5 回测层 (Backtest Layer)

**BacktestEngine** (`backtest/engine.py`)
- **T+1 执行**: T 日信号 → T+1 开盘价成交
- **涨跌停判断**: 触及涨跌停时跳过买卖
- **交易成本**:
  - 佣金: 万分之三
  - 印花税: 万分之五（仅卖出）
  - 滑点: 千分之二
- **账户清算**: 每日收盘后按收盘价盯市

## 四、核心数据流

### 4.1 训练流程

```
1. 读取配置 → 确定时间窗口和股票池
         │
         ▼
2. 获取股票列表 (StockPoolManager)
         │
         ▼
3. 对每只股票循环:
   a. 尝试从特征缓存加载
   b. 若无缓存 → 调用 DataRepository 获取原始数据
   c. → DynamicFilter 过滤
   d. → FeaturePipeline 计算因子
   e. → 保存特征缓存
         │
         ▼
4. 合并所有股票的特征矩阵
         │
         ▼
5. CrossSectionalProcessor (MAD + Z-Score)
         │
         ▼
6. 标签处理 (若是 rank_pct → 截面百分比排名)
         │
         ▼
7. Drop Middle 采样 (可选)
         │
         ▼
8. XGBoost 训练 (或增量训练)
         │
         ▼
9. 保存模型
```

### 4.2 回测流程

```
1. 读取训练好的模型
         │
         ▼
2. 构建 Inference Pipeline (无 LabelGenerator)
         │
         ▼
3. 按日期循环:
   │
   ├─ Day T:
   │   a. 获取持仓股票当日收盘价 (Mark to Market)
   │   b. Strategy.select_stocks(T) → 生成 T+1 交易信号
   │   c. 保存信号到 pending_signals
   │
   └─ Day T+1:
       a. 执行 pending_signals (T日收盘生成的信号)
       b. T+1 开盘价 × 涨跌停判断 × 滑点 × 佣金/印花税
       │
       ▼
4. 汇总计算:
   - 总收益率
   - 年化收益率
   - 最大回撤
   - 夏普比率 (待实现)
```

## 五、关键设计决策

### 5.1 为什么用 Parquet 而非 HDF5/CSV?

- **列式存储**: 读取特定列（如仅 `close`）远快于行式存储
- **压缩效率**: Parquet 通常比 CSV 小 5-10 倍
- **类型保留**: 保留 datetime 类型，无需重新解析
- **跨语言**: Python、R、Julia 均可读取

### 5.2 为什么不使用数据库 (MySQL/PostgreSQL)?

- **回测场景**: 需要频繁读取单只股票的全量历史，数据库随机读取性能差
- **成本**: Parquet + Pandas 的内存切片足够快，无需引入数据库运维复杂度
- **未来扩展**: 可引入 DuckDB 加速横截面查询

### 5.3 特征缓存的 MD5 指纹设计

缓存路径: `data/local_lake/features/<config_hash>/<symbol>_<start>_<end>_<label_flag>.parquet`

- `config_hash`: 由 `pipeline_config.yaml` 中 `features` 部分的 JSON 序列化后的 MD5 前8位
- 同一只股票、同一时间段、同一特征配置，第二次运行直接读取缓存
- 若修改了任何因子参数，缓存自动失效（因为 config_hash 变化）

### 5.4 T+1 实现的信号延迟机制

```python
pending_signals = {}  # T 日收盘后填充

for current_date in dates:
    # 1. 执行 T-1 日生成的信号 (T 日开盘)
    if pending_signals:
        execute_trades(current_date, pending_signals)

    # 2. 收盘后生成 T+1 信号
    signals = strategy.select_stocks(T, data_loader)
    pending_signals = signals  # 延迟到 T+1 执行
```

## 六、扩展指南

### 6.1 添加新的因子

1. 在 `features/factors/` 下创建新因子类，继承 `BaseFactor`
2. 实现 `calculate(df)` 方法
3. 在 `FACTOR_MAP` 中注册
4. 在 YAML 配置中添加因子配置

### 6.2 添加新的数据源

1. 在 `data/source/` 下创建新的 Source 类
2. 实现 `get_daily_data(symbol, start, end)` 方法
3. 在 `scripts/sync_data.py` 中调用新数据源同步到 Local Data Lake

### 6.3 添加新的策略

1. 在 `strategies/` 下创建新策略类，继承 `BaseStrategy`
2. 实现 `select_stocks(date, data_loader, current_positions)` 方法
3. 返回股票列表或权重字典

---

## 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| V4.0 | 2026-04 | 引入 Local Data Lake + Feature Caching + Drop-Middle 采样 |
| V3.0 | 2026-03 | 引入 T+1 回测引擎 + 涨跌停机制 |
| V2.0 | 2026-02 | 引入 XGBoost + 排序学习 |
| V1.0 | 2026-01 | 初版实现 |
