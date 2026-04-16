# A-Share Quant (A股量化选股框架)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A-Share Quant 是一个基于 Python 的工业级、配置驱动 (Config-driven)、高度模块化的 A 股量化交易与回测框架。

本项目旨在为量化研究员提供一个从数据获取、特征工程、模型训练到回测评估的完整闭环，尤其强化了对机器学习（如 XGBoost）和前沿量化方法（如 Learning to Rank）的支持。

---

##  核心特性 (Features)

### 配置驱动
- **YAML 配置驱动**: 通过修改 `configs/pipeline_config.yaml` 即可切换数据源、调整时间窗口、开启/关闭因子、修改模型参数，实现极简的策略实验。

### 数据管理
- **离线数据湖 (Local Data Lake)**: 彻底剥离即时网络下载，采用独立的 `sync_data.py` 脚本进行数据每日增量/全量同步。底层采用 Parquet 格式进行按股票全历史存储。不仅同步基础 K 线，还包括大盘指数、行业板块分类、季度财务数据以及资金流向数据。
- **纯本地极速读取**: 所有的特征计算（包括宏观因子、财务因子、资金流向等）已完全解耦网络请求，统一由 `DataRepository` 从本地数据湖加载，彻底消除了网络 I/O 阻塞。
- **极速特征缓存 (Feature Caching)**: 内置基于配置 MD5 指纹的特征缓存机制。相同配置下，第二次运行将跳过 130+ 种技术指标与自定义因子的重复计算，直接从本地读取。

### 特征工程
- **丰富的因子库**: 支持 130+ 种 `pandas-ta` 技术指标、财务基本面因子（ROE、净利润增长等）、资金流向因子及大盘宏观特征。
- **主观逻辑量化**: 独家内置 `SubjectiveFactor`，将短线游资打法（打板溢价、换手突破、量价背离、弱转强等）翻译为精确的量价因子。
- **经典形态识别**: 内置 `PatternFactor`，通过向量化代码精准识别A股高胜率形态，如均线多头排列、箱体平台突破、龙头首阴反包、MACD底背离以及红三兵。
- **事件驱动整合**: 内置 `EventFactor`，将离散的市场事件（如龙虎榜机构与知名游资的净买入额）自动对齐到日K线中。
- **截面去噪与标准化**: 内置 `CrossSectionalProcessor`，在输入模型前强制进行按日期的横截面 MAD 去极值（Clipping）和 Z-Score 标准化。
- **消除未来函数**: `LabelGenerator` 采用 T+1 开盘价作为真实交易成本基准计算收益。

### 训练与推理优化
- **多进程并行加速**: 无论是模型训练还是实盘推理（Inference），均支持基于 `ProcessPoolExecutor` 的多进程批量特征计算，能跑满多核 CPU。
- **数据采样**: 支持随机降采样 (Random Downsampling) 以防止过拟合，以及 Drop Middle (截面极值采样) 直接丢弃中间表现平庸的样本，迫使模型聚焦于寻找"牛股"与"熊股"的本质区别。
- **增量学习**: 支持增量训练，无需全量重训练即可更新模型。
- **Learning to Rank**: 支持 `rank:pairwise` 目标函数，配合截面收益率排名标签 (rank_pct) 效果最佳。

### 策略与回测
- **严谨的 T+1 交易引擎**: 信号在 T 日收盘后生成，并在 T+1 日开盘价执行。
- **A 股特色机制**: 完美支持涨跌停（Limit Up/Down）过滤（涨停不买、跌停不卖），内置滑点（Slippage）与 A股真实印花税（单边万分之五）计算。
- **大盘风控**: 支持大盘趋势过滤（如跌破20日线空仓）、换手率控制及基于预测分数的资金加权分配。

### 防雷机制
- **静态过滤**: `StockPoolManager` 默认剔除 ST、*ST 及退市整理期股票。
- **动态过滤**: `DynamicFilter` 基于时序数据，动态剔除流动性枯竭的"僵尸股"（如近20日日均成交额<1000万）以及上市不满半年的次新股。

---

##  项目结构 (Project Structure)

```text
ashare_quant/
├── configs/                    # 配置文件目录
│   ├── pipeline_config.yaml    # 训练与回测配置文件
│   └── predict_config.yaml     # 实盘推理配置文件
├── data/                       # 数据接入与管理层
│   ├── __init__.py
│   ├── repository.py           # 统一的数据仓库入口（从 local_lake 加载）
│   ├── pool_manager.py         # 股票池过滤器（板块、交易所过滤）
│   ├── market_manager.py       # 大盘指数与宏观数据管理
│   ├── board_manager.py        # 板块/行业管理
│   ├── local_lake/             # 本地离线数据湖 (Parquet)
│   │   ├── basics/             # 股票基础信息
│   │   ├── daily_k/            # 个股日线全量历史数据
│   │   ├── events/             # 事件数据池（如龙虎榜数据）
│   │   └── features/           # 基于 MD5 配置指纹的特征缓存池
│   └── source/                 # 数据源实现（已弃用，数据从 local_lake 加载）
│       ├── baostock_source.py
│       ├── akshare_source.py
│       └── mock_source.py
├── features/                    # 特征工程层
│   ├── __init__.py
│   ├── pipeline.py             # 特征处理流水线
│   ├── processor.py            # 横截面处理器（MAD去极值、Z-Score标准化）
│   └── factors/                # 因子库
│       ├── __init__.py
│       ├── base_factor.py      # 因子基类
│       ├── pandas_ta_factor.py # 技术指标因子（基于 pandas-ta）
│       ├── financial.py        # 财务基本面因子
│       ├── fund_flow.py        # 资金流向因子
│       ├── market.py           # 大盘宏观因子
│       ├── fundamental.py      # 行业/概念类别因子
│       ├── subjective.py       # 主观交易逻辑因子
│       ├── pattern.py          # 经典形态识别因子
│       ├── event_driven.py     # 事件驱动因子
│       └── technical.py        # Label 生成器
├── models/                     # 机器学习模型层
│   ├── base_model.py           # 模型基类
│   ├── xgboost_model.py        # XGBoost 包装器
│   └── machine_learning.py     # Sklearn 通用包装器
├── strategies/                 # 交易策略层
│   ├── base_strategy.py        # 策略基类
│   ├── ml_strategy.py          # 基于 ML 预测分数的选股策略
│   └── momentum_strategy.py    # 传统动量策略示例
├── backtest/                   # 回测引擎层
│   └── engine.py               # 事件驱动的回测与撮合逻辑
├── scripts/                    # 实用脚本目录
│   └── sync_data.py            # 离线数据湖同步脚本
├── utils/                      # 工具函数
├── runner.py                   # 核心引擎：配置解析与流水线执行器
├── main.py                     # 简单 CLI（动量策略回测）
├── analyze_importance.py       # 因子重要性分析工具
└── requirements.txt            # 依赖列表
```

---

##  快速开始 (Quick Start)

### 1. 环境准备

```bash
# 克隆代码
git clone <repository_url>
cd ashare_quant

# 安装依赖
pip install pandas numpy xgboost scikit-learn baostock akshare pandas-ta pyyaml pyarrow
```

### 2. 同步本地数据湖

```bash
# 全量同步所有 A 股日线数据及龙虎榜事件（耗时较长，建议盘后执行）
python scripts/sync_data.py

# 指定开始日期同步数据（增量或指定区间同步）
python scripts/sync_data.py --start_date 2023-01-01

# 只单独更新事件数据（如龙虎榜）
python scripts/sync_data.py --events_only

# 为了快速测试，只同步前 50 只股票
python scripts/sync_data.py --limit 50
```

### 3. 运行模型训练与回测

编辑 `configs/pipeline_config.yaml`，设定您想要的数据范围、因子组合和模型参数，然后执行：

```bash
python runner.py configs/pipeline_config.yaml
```

系统将自动：
1. 从本地 Data Lake 加载所需数据
2. 生成全量特征矩阵
3. 训练 XGBoost 模型并保存到 `models/saved/`
4. 在样本外时间段执行模拟回测并输出收益报告

### 4. 特征重要性分析

在跑完一次"全量特征"训练后，您可以分析哪些因子真正有效：

```bash
python analyze_importance.py
```

### 5. 生成实盘预测

当模型训练完毕后，使用推理配置文件进行最新数据的预测：

```bash
python runner.py configs/predict_config.yaml
```

---

##  配置指南 (Configuration Guide)

### 数据与股票池

```yaml
data:
  sources: ["BaostockSource"]  # 已弃用，数据统一从 Local Data Lake 加载
  cache_dir: "data/local_lake"

  # 股票池配置（如果指定了 symbols，则优先使用 symbols，忽略 pool 配置）
  symbols: []  # 为空则使用 pool 过滤
  pool:
    board: "all"      # 可选: 'main'(主板), 'chinext'(创业板), 'star'(科创板), 'all'
    exchange: "all"   # 可选: 'sh', 'sz', 'bj', 'all'
    exclude_st: true  # 是否剔除 ST/*ST/退市股
    max_count: 10     # 限制股票数量（0 表示不限制）
```

### 训练提速与数据降噪

```yaml
training_optimization:
  sample_rate: 0.5            # 随机降采样：仅使用 50% 的数据训练
  drop_middle: true           # 截面极值采样：丢弃中间 40% 表现平庸的样本
  drop_middle_threshold: 0.4  # 丢弃阈值范围
```

### 数据预处理与防雷

```yaml
preprocessing:
  dynamic_filter:       # 动态过滤僵尸股与次新股
    enable: true
    min_avg_turnover: 10000000  # 剔除近20日日均成交额低于 1000万 的僵尸股
    min_listed_days: 120        # 剔除上市不满半年的次新股
  mad_clip: true                # 开启横截面 MAD 去极值
  z_score: true                 # 开启横截面 Z-Score 标准化
```

### 特征配置

```yaml
features:
  - name: "MarketFactor"        # 大盘基准因子
    params:
      index_symbol: "sh.000300"  # 沪深300作为基准
  - name: "SubjectiveFactor"    # 主观交易逻辑因子
    params: {}
  - name: "PatternFactor"        # 形态识别因子
    params: {}
  - name: "EventFactor"          # 事件驱动因子
    params: {}
  - name: "BoardFactor"          # 行业/概念因子
    params:
      encode_method: "category"
  - name: "FinancialFactor"      # 财务基本面因子
    params: {}
  - name: "FundFlowFactor"       # 资金流向因子
    params: {}
  - name: "PandasTAFactor"       # 技术指标因子
    params:
      strategy: "custom"
      features: ['MACD_12_26_9', 'RSI_14', 'BBU_5_2.0_2.0', ...]  # 可选技术指标
  - name: "LabelGenerator"       # 标签生成器
    params:
      horizon: 3                # 预测未来3天的表现
      target_type: "rank_pct"    # 截面收益率百分比排名（配合 rank:pairwise）
```

**target_type 选项说明：**
- `regression`: 预测具体收益率数值（对应 `objective: reg:squarederror`）
- `binary`: 预测涨跌（对应 `objective: binary:logistic`）
- `classification_3`: 预测大涨(1)、平盘(0)、大跌(-1)（对应 `objective: multi:softmax`）
- `excess_return_binary`: 预测是否跑赢沪深300
- `rank_pct`: 截面收益率百分比排名 (0.0~1.0)，**强烈推荐**配合 `rank:pairwise` 使用

### 模型配置

```yaml
model:
  name: "XGBoostWrapper"
  params:
    objective: "rank:pairwise"  # 使用排序学习目标
    max_depth: 4
    eta: 0.1
    # 其他 XGBoost 参数...
  save_path: "models/saved/config_xgb.json"
```

### 策略与风控

```yaml
strategy:
  name: "MLStrategy"
  params:
    top_k: 10                   # 每日买入排名前 10 的股票
    rebalance_period: 1          # 调仓周期（天）
    use_market_filter: true      # 大盘风控：指数跌破 20 日线时自动空仓
    max_turnover: 0.5           # 换手率控制：每日最多调仓 50%
    weight_method: "score"       # 资金分配：按预测分数加权买入
  initial_capital: 100000.0     # 初始资金
```

---

##  数据流 (Data Flow)

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
│ (数据仓库入口)   │     │  (Parquet 文件)   │
└────────┬────────┘     └──────────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐     ┌──────────────────┐
│ StockPoolManager│     │  FeaturePipeline │
│ (股票池过滤)     │     │  (特征工程流水线)  │
└────────┬────────┘     └────────┬─────────┘
         │                      │
         │         ┌────────────┼────────────┐
         │         ▼            ▼            ▼
         │   ┌──────────┐ ┌──────────┐ ┌──────────┐
         │   │PandasTA  │ │Financial │ │Pattern   │
         │   │Factor    │ │Factor    │ │Factor    │
         │   └──────────┘ └──────────┘ └──────────┘
         │         │            │            │
         │         └────────────┼────────────┘
         │                      ▼
         │            ┌──────────────────┐
         │            │CrossSectional    │
         │            │Processor         │
         │            │(MAD+Z-Score)     │
         │            └────────┬─────────┘
         │                     │
         ▼                     ▼
┌─────────────────┐     ┌──────────────────┐
│   XGBoostModel  │◀────│  训练特征矩阵 X   │
│   (模型训练)     │     │  标签向量 y       │
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐
│  BacktestEngine │◀──── Signals (T日收盘信号)
│  (T+1执行回测)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   回测报告       │
│ (收益、最大回撤)  │
└─────────────────┘
```

---

##  核心概念 (Core Concepts)

### T+1 交易机制

A股实行 T+1 交易制度，即当日买入的股票不能在当日卖出。本框架严格遵循此规则：
- **信号生成**: T 日收盘后，根据当日数据生成交易信号
- **交易执行**: T+1 日开盘价执行买卖操作
- **涨跌停处理**: 若 T+1 日开盘价触及涨跌停，则跳过该笔交易

### 截面处理 (Cross-Sectional Processing)

为了消除大盘 Beta 波动对因子的影响，本框架在每天的横截面上对因子进行处理：

1. **MAD 去极值**: 对每个因子，计算其中位数绝对偏差 (MAD)，剔除超过 3σ 的异常值
2. **Z-Score 标准化**: 将因子值转换为标准正态分布，公式: `z = (x - μ) / σ`

### 特征缓存 (Feature Caching)

特征计算（尤其是 130+ 技术指标）是耗时最长的步骤之一。本框架使用配置指纹的 MD5 哈希值作为缓存键：
- 相同配置下，第二次运行直接读取缓存，速度提升 100 倍以上
- 缓存目录: `data/local_lake/features/<config_hash>/`

### 排序学习 (Learning to Rank)

相比传统的回归或分类目标，排序学习更适合量化选股场景：
- **目标**: `rank:pairwise`，优化 pairwise 排序损失
- **标签**: `rank_pct`，截面收益率百分比排名 (0.0~1.0)
- **优势**: 解决二分类梯度为0的问题，直接优化相对排序

---

##  关键文件说明

| 文件 | 说明 |
|------|------|
| `runner.py` | 核心执行器，解析 YAML 配置并协调整个训练/回测流程 |
| `data/repository.py` | 数据仓库，统一从 Local Data Lake 加载数据 |
| `data/pool_manager.py` | 股票池管理，支持板块、交易所过滤和 ST 剔除 |
| `features/pipeline.py` | 特征流水线，串联所有因子计算 |
| `features/processor.py` | 横截面处理器，执行 MAD 去极值和 Z-Score 标准化 |
| `strategies/ml_strategy.py` | ML 选股策略，包含大盘风控和换手率控制 |
| `backtest/engine.py` | 回测引擎，实现 T+1 交易、涨跌停、印花税、滑点 |
| `models/xgboost_model.py` | XGBoost 模型封装，支持增量学习和排序目标 |

---

##  免责声明 (Disclaimer)

本项目仅供量化研究与学术交流使用。由于公共数据源（如 Baostock, AkShare）可能存在网络延迟或数据缺失，实盘交易需谨慎评估风险。开发者对使用本项目造成的任何投资损失概不负责。
