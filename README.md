# A-Share Quant (A股量化选股框架)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A-Share Quant 是一个基于 Python 的工业级、配置驱动 (Config-driven)、高度模块化的 A 股量化交易与回测框架。

本项目旨在为量化研究员提供一个从数据获取、特征工程、模型训练到回测评估的完整闭环，尤其强化了对机器学习（如 XGBoost）和前沿量化方法（如 Learning to Rank）的支持。

---

## 核心特性 (Features)

### 配置驱动
- **YAML 配置驱动**: 通过修改 `configs/pipeline_config.yaml` 即可切换数据源、调整时间窗口、开启/关闭因子、修改模型参数，实现极简的策略实验。
- **多种运行模式**: 支持 `train`（全量训练+回测）、`backtest_only`（加载已有模型仅回测）、`predict_only`（实盘推理）、`incremental_train`（增量训练）。

### 数据管理
- **离线数据湖 (Local Data Lake)**: 彻底剥离即时网络下载，采用独立的 `sync_data.py` 脚本进行数据每日增量/全量同步。底层采用 Parquet 格式进行按股票全历史存储。支持基础 K 线、大盘指数、行业板块分类、季度财务数据、资金流向及龙虎榜事件数据。
- **纯本地极速读取**: 所有特征计算已完全解耦网络请求，统一由 `DataRepository` 从本地数据湖加载。
- **极速特征缓存**: 内置基于配置 MD5 指纹的特征缓存机制。相同配置下，第二次运行跳过特征重复计算。

### 特征工程
- **丰富因子库**: 45 种精选 `pandas-ta` 技术指标（curated 策略，去冗余）、13 项财务基本面因子（ROE、净利率、毛利率、PE/PSTTM等）、资金流向因子、大盘宏观特征及反转信号因子。
- **时序特征编码**: 对每个数值特征自动生成 5日绝对变化 (_d5)、5日百分比变化 (_r5)、20日滚动标准差 (_s20)，让 XGBoost 学习"RSI 从 30 涨到 65"vs"RSI 从 80 跌到 65"这类方向性信息。
- **主观逻辑量化**: `SubjectiveFactor` 将短线游资打法（打板溢价、换手突破、量价背离、弱转强等）翻译为精确的量价因子。
- **经典形态识别**: `PatternFactor` 识别均线多头排列、箱体平台突破、龙头首阴反包、MACD底背离、红三兵。
- **事件驱动整合**: `EventFactor` 将龙虎榜机构与知名游资的净买入额对齐到日K线。
- **截面去噪与标准化**: `CrossSectionalProcessor` 在输入模型前进行横截面 MAD 去极值和 Z-Score 标准化。
- **多周期衰减标签**: `LabelGenerator` 支持 `decay_weighted` 目标类型，用 [1d,3d,5d,7d]×[0.4,0.3,0.2,0.1] 的复合标签缓解单一周期盲区。采用 T+1 开盘价作为交易成本基准。

### 模型训练
- **多进程并行加速**: 训练和推理均支持 `ProcessPoolExecutor` 多进程批量特征计算。
- **Learning to Rank**: 支持 `rank:pairwise` 目标函数，配合衰减加权标签优化相对排序。
- **XGBoost 正则化**: L1/L2 正则化、行列采样、min_child_weight 等多重防过拟合措施。
- **增量学习**: 支持增量训练，无需全量重训练即可更新模型。

### 策略与回测
- **严谨的 T+1 交易引擎**: T 日收盘后生成信号，T+1 日开盘价执行。
- **A 股特色机制**: 涨跌停过滤（涨停不买、跌停不卖），内置滑点与印花税计算。
- **ATR 自适应止损**: 基于 ATR(14) 的动态止损，高波动股给更宽空间，低波动股收更紧。
- **移动止盈**: 盈利超阈值后激活 tighter trailing stop，锁住利润。
- **波动率倒数加权**: `inv_vol` 资金分配方式，低波动股获得更高权重。
- **大盘风控**: 指数跌破 20 日线时空仓，逆风期自动降低仓位。
- **头寸风控**: 单日暴跌检测、累计亏损止损、逆风期加速止损。

### 防雷机制
- **静态过滤**: `StockPoolManager` 默认剔除 ST、*ST 及退市整理期股票。
- **动态过滤**: `DynamicFilter` 剔除流动性枯竭的"僵尸股"（日均成交额<1000万）及次新股。

---

## 项目结构 (Project Structure)

```text
ashare_quant/
├── configs/                    # 配置文件目录
│   ├── pipeline_config.yaml    # 全量训练 + 回测配置
│   ├── backtest_only_config.yaml  # 仅回测模式（加载已有模型）
│   ├── incremental_train_config.yaml  # 增量训练配置
│   └── predict_config.yaml     # 实盘推理配置
├── data/                       # 数据接入与管理层
│   ├── repository.py           # 统一数据仓库入口
│   ├── pool_manager.py         # 股票池过滤器
│   ├── market_manager.py       # 大盘指数数据管理
│   ├── board_manager.py        # 板块/行业管理
│   ├── local_lake/             # 本地离线数据湖 (Parquet)
│   │   ├── basics/             # 股票基础信息
│   │   ├── daily_k/            # 个股日线全量历史数据
│   │   ├── events/             # 事件数据池（龙虎榜等）
│   │   └── features/           # 基于 MD5 配置指纹的特征缓存
│   ├── backtest/               # 回测交割单输出目录
│   └── source/                 # 数据源实现
├── features/                    # 特征工程层
│   ├── pipeline.py             # 特征流水线（含时序编码）
│   ├── processor.py            # 横截面 MAD/Z-Score 处理器
│   └── factors/                # 因子库
│       ├── base_factor.py      # 因子基类
│       ├── pandas_ta_factor.py # 技术指标因子（curated 策略）
│       ├── financial.py        # 财务基本面因子（13 项）
│       ├── fund_flow.py        # 资金流向因子
│       ├── market.py           # 大盘宏观因子
│       ├── fundamental.py      # 行业/概念类别因子
│       ├── subjective.py       # 主观交易逻辑因子
│       ├── pattern.py          # 经典形态识别因子
│       ├── event_driven.py     # 事件驱动因子
│       ├── reversal.py         # 反转信号因子
│       └── technical.py        # Label 生成器（decay_weighted）
├── models/                     # 机器学习模型层
│   ├── base_model.py           # 模型基类
│   ├── xgboost_model.py        # XGBoost 包装器
│   └── machine_learning.py     # Sklearn 通用包装器
├── strategies/                 # 交易策略层
│   ├── base_strategy.py        # 策略基类
│   ├── ml_strategy.py          # ML 选股策略（ATR止损/止盈/风控）
│   └── momentum_strategy.py    # 传统动量策略示例
├── backtest/                   # 回测引擎层
│   └── engine.py               # 事件驱动的 T+1 回测撮合
├── scripts/                    # 实用脚本
│   └── sync_data.py            # 离线数据湖同步脚本（多进程）
├── utils/                      # 工具函数
├── runner.py                   # 核心引擎：配置解析与流水线执行
├── main.py                     # 简单 CLI
├── analyze_importance.py       # 因子重要性分析工具
└── requirements.txt            # 依赖列表
```

---

## 快速开始 (Quick Start)

### 1. 环境准备

```bash
git clone <repository_url>
cd ashare_quant
pip install pandas numpy xgboost scikit-learn baostock akshare pandas-ta pyyaml pyarrow
```

### 2. 同步本地数据湖

```bash
# 全量同步（耗时较长，建议盘后执行）
python scripts/sync_data.py

# 增量同步
python scripts/sync_data.py --start_date 2026-01-01

# 只更新事件数据（龙虎榜）
python scripts/sync_data.py --events_only

# 快速测试（只同步前 50 只）
python scripts/sync_data.py --limit 50
```

### 3. 训练模型并回测

```bash
# 全量训练 + 回测
python runner.py configs/pipeline_config.yaml

# 仅回测（加载已有模型）
python runner.py configs/backtest_only_config.yaml

# 增量训练
python runner.py configs/incremental_train_config.yaml

# 实盘预测
python runner.py configs/predict_config.yaml
```

### 4. 特征重要性分析

```bash
python analyze_importance.py
```

---

## 配置指南 (Configuration Guide)

### 数据与股票池

```yaml
data:
  cache_dir: "data/local_lake"
  symbols: []  # 为空则使用 pool 过滤
  pool:
    board: "all"
    exchange: "all"
    exclude_st: true
    max_count: 0  # 0 = 不限制
```

### 训练优化

```yaml
training_optimization:
  sample_rate: 1.0        # 1.0 = 全量数据
  drop_middle: false      # 是否丢弃中间平庸样本
```

### 预处理与防雷

```yaml
preprocessing:
  dynamic_filter:
    enable: true
    min_avg_turnover: 10000000
    min_listed_days: 120
  mad_clip: true
  z_score: true
```

### 特征配置

```yaml
features:
  - name: "MarketFactor"
    params: { index_symbol: "sh.000300" }
  - name: "SubjectiveFactor"
  - name: "PatternFactor"
  - name: "EventFactor"
  - name: "ReversalFactor"
  - name: "BoardFactor"
    params: { encode_method: "category" }
  - name: "FinancialFactor"
  - name: "FundFlowFactor"
  - name: "PandasTAFactor"
    params: { strategy: "curated" }
  - name: "LabelGenerator"
    params:
      horizon: 7
      target_type: "decay_weighted"  # [1d,3d,5d,7d] × [0.4,0.3,0.2,0.1]
```

### 模型配置

```yaml
model:
  name: "XGBoostWrapper"
  params:
    objective: "rank:pairwise"
    max_depth: 3
    eta: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
    min_child_weight: 5
    lambda_l2: 1.0
    alpha_l1: 0.5
  num_boost_round: 100
  save_path: "models/saved/config_xgb.json"
```

### 策略与风控

```yaml
strategy:
  name: "MLStrategy"
  params:
    top_k: 4
    target_position_ratio: 0.6
    rebalance_period: 3
    use_market_filter: true
    max_turnover: 0.5
    weight_method: "inv_vol"      # equal / score / inv_vol
    atr_stop_mult: 2.5            # ATR 自适应止损倍数
    atr_daily_drop_mult: 2.0      # 单日暴跌 ATR 倍数
    atr_trail_mult: 3.0           # 移动止损 ATR 倍数
    take_profit_activate: 0.15    # 止盈激活阈值 (15% 盈利)
    take_profit_trail: 0.08       # 止盈回撤线 (8% from peak)
  initial_capital: 100000.0
```

---

## 数据流 (Data Flow)

```
配置文件 (YAML)
       │
       ▼
┌─────────────────┐
│  PipelineRunner │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ DataRepository  │────▶│   Local Data Lake │
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ StockPoolManager│     │  FeaturePipeline │
│ (股票池过滤)     │     │  (因子+时序编码)   │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         │              ┌────────┼────────────┐
         │              ▼        ▼            ▼
         │        PandasTA  Financial   Pattern/Event
         │        (45 curated) (13项)   /Reversal...
         │              │        │            │
         │              └────────┼────────────┘
         │                       ▼
         │              ┌──────────────────┐
         │              │ Temporal Encoding│
         │              │ (_d5/_r5/_s20)   │
         │              └────────┬─────────┘
         │                       ▼
         │              ┌──────────────────┐
         │              │CrossSectional    │
         │              │Processor         │
         │              └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│   XGBoostModel  │◀────│  训练特征矩阵 X   │
│  rank:pairwise  │     │  decay_weighted y │
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐
│  MLStrategy     │   ATR止损/止盈/逆风期风控/inv_vol加权
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BacktestEngine │◀──── T+1 开盘价执行
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   回测报告       │
│ 收益/回撤/Sharpe │
└─────────────────┘
```

---

## 关键文件说明

| 文件 | 说明 |
|------|------|
| `runner.py` | 核心执行器，解析 YAML 配置，支持 train/backtest_only/predict_only 模式 |
| `data/repository.py` | 数据仓库，统一从 Local Data Lake 加载数据 |
| `data/pool_manager.py` | 股票池管理，支持板块/交易所过滤和 ST 剔除 |
| `features/pipeline.py` | 特征流水线，串联因子计算 + 时序编码 (_d5/_r5/_s20) |
| `features/processor.py` | 横截面处理器，MAD 去极值 + Z-Score 标准化 |
| `features/factors/technical.py` | Label 生成器，支持 decay_weighted 多周期衰减标签 |
| `features/factors/financial.py` | 财务因子，13 项基本面指标 + PE/PSTTM 估值 |
| `features/factors/pandas_ta_factor.py` | 技术指标，curated 策略精选 45 个非冗余指标 |
| `strategies/ml_strategy.py` | ML 选股策略，ATR 自适应止损 + 移动止盈 + 逆风期风控 |
| `backtest/engine.py` | 回测引擎，T+1 交易、涨跌停、印花税、滑点 |
| `models/xgboost_model.py` | XGBoost 封装，支持 rank:pairwise/增量学习/特征对齐 |

---

## 免责声明 (Disclaimer)

本项目仅供量化研究与学术交流使用。实盘交易需谨慎评估风险，开发者对使用本项目造成的任何投资损失概不负责。
