# A-Share Quant (A股量化选股框架)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A-Share Quant 是一个基于 Python 的工业级、配置驱动 (Config-driven)、高度模块化的 A 股量化交易与回测框架。

本项目旨在为量化研究员提供一个从数据获取、特征工程、模型训练到回测评估的完整闭环，尤其强化了对机器学习（如 XGBoost）和前沿量化方法（如 Learning to Rank）的支持。

---

## 🌟 核心特性 (Features)

- **⚙️ YAML 配置驱动**: 告别硬编码！通过修改 `configs/pipeline_config.yaml` 即可切换数据源、调整时间窗口、开启/关闭因子、修改模型参数，实现极简的策略实验。
- **🔌 模块化架构**:
  - **离线数据湖 (Local Data Lake)**: 彻底剥离即时网络下载请求，采用独立的 `sync_data.py` 脚本进行数据每日增量/全量同步。底层采用 Parquet 格式进行按股票全历史存储，回测时在内存中极速切片，加载速度提升万倍，不再受限于网络波动。
  - **🚀 极速特征缓存 (Feature Caching)**: 内置基于配置 MD5 指纹的特征缓存机制。相同配置下，第二次运行将跳过 130+ 种技术指标与自定义因子的重复计算，直接从本地读取，训练提速 100 倍以上。
  - **📉 训练降噪采样 (Data Sampling)**: 支持在训练阶段进行**随机降采样 (Random Downsampling)**以防止过拟合，以及**截面极值采样 (Drop Middle)**直接丢弃中间 40% 表现平庸的样本，迫使模型聚焦于寻找“牛股”与“熊股”的本质区别。
  - **特征工程 (Features)**: 
    - **丰富的因子库**: 支持 130+ 种 `pandas-ta` 技术指标、财务基本面因子（ROE、净利润增长等）、资金流向因子及大盘宏观特征。
    - **🌟 主观逻辑量化**: 独家内置 `SubjectiveFactor`，将短线游资打法（打板溢价、换手突破、量价背离、弱转强等）翻译为精确的量价因子。
    - **🌟 经典形态识别**: 内置 `PatternFactor`，通过向量化代码精准识别A股高胜率形态，如**均线多头排列**、**箱体平台突破**、**龙头首阴反包**、**MACD底背离**以及**红三兵**。
    - **🌟 事件驱动整合**: 内置 `EventFactor`，将离散的市场事件（如龙虎榜机构与知名游资的净买入额）自动对齐到日K线中，赋予模型洞察“资金共识”的能力。
    - **截面去噪与标准化**: 内置 `CrossSectionalProcessor`，在输入模型前强制进行按日期的横截面 MAD 去极值（Clipping）和 Z-Score 标准化，消除大盘 Beta 波动与极端异动的噪音。
    - **消除未来函数**: `LabelGenerator` 采用 T+1 开盘价作为真实交易成本基准计算收益。
  - **动态防雷与垃圾股过滤**:
    - **静态过滤**: `StockPoolManager` 默认剔除 ST、*ST 及退市整理期股票，建立基础隔离墙。
    - **动态过滤**: `DynamicFilter` 基于时序数据，动态剔除流动性枯竭的“僵尸股”（如近20日日均成交额<1000万）以及上市不满半年的次新股，防止模型被资金操纵和流动性陷阱反噬。
  - **模型层 (Models)**: 原生集成 XGBoost，支持**增量学习 (Incremental Learning)**、**类别特征原生处理 (Categorical Support)** 以及专业的**排序学习 (Learning to Rank - pairwise)**。
  - **策略与回测 (Strategy & Backtest)**: 
    - **严谨的 T+1 交易引擎**: 信号在 T 日收盘后生成，并在 T+1 日开盘价执行。
    - **A 股特色机制**: 完美支持涨跌停（Limit Up/Down）过滤（涨停不买、跌停不卖），内置滑点（Slippage）与 A股真实印花税（单边万分之五）计算，挤出回测水分。
    - **进阶风控**: 支持大盘风控过滤（如跌破20日线空仓）、换手率控制及基于预测分数的资金加权分配。
- **🔮 实盘预测模式**: 提供专门的 `configs/predict_config.yaml` 和推理模式，一键输出“明日十大金股”。

---

## 🏗️ 架构概览 (Architecture)

```text
ashare_quant/
├── configs/                    # ⚙️ 配置文件目录
│   ├── pipeline_config.yaml    # 训练与回测配置文件
│   └── predict_config.yaml     # 实盘推理配置文件
├── data/                       # 数据接入与管理层
│   ├── local_lake/             # 🚀 本地离线数据湖 (Parquet)
│   │   ├── basics/             # 股票基础信息
│   │   ├── daily_k/            # 个股日线全量历史数据
│   │   └── events/             # 离线事件数据池 (如龙虎榜数据)
│   ├── source/                 # 具体数据源实现 (Baostock, AkShare)
│   ├── repository.py           # 统一的数据仓库入口 (从 local_lake 极速加载)
│   ├── pool_manager.py         # 股票池过滤器 (板块、交易所过滤)
│   └── market_manager.py       # 大盘指数与宏观数据管理
├── features/                   # 特征工程层 (Pipeline)
│   ├── factors/                # 因子库
│   │   ├── pandas_ta_factor.py # 技术指标因子 (基于 pandas-ta)
│   │   ├── fundamental.py      # 行业/概念类别因子
│   │   ├── financial.py        # 财务基本面因子
│   │   ├── fund_flow.py        # 资金流向因子
│   │   ├── market.py           # 大盘宏观因子
│   │   ├── subjective.py       # 🌟 主观交易逻辑因子 (涨停溢价/弱转强/量价背离)
│   │   ├── pattern.py          # 🌟 经典形态识别因子 (均线多头/平台突破/反包/底背离)
│   │   ├── event_driven.py     # 🌟 事件驱动因子 (龙虎榜资金等)
│   │   └── technical.py        # Label生成器 (回归、二分类、排序等)
│   └── pipeline.py             # 特征处理流水线
├── models/                     # 机器学习模型层
│   ├── xgboost_model.py        # XGBoost 包装器
│   └── machine_learning.py     # Sklearn 通用包装器
├── strategies/                 # 交易策略层
│   ├── ml_strategy.py          # 基于 ML 预测分数的选股策略
│   └── momentum_strategy.py    # 传统动量策略示例
├── backtest/                   # 回测引擎层
│   └── engine.py               # 事件驱动的回测与撮合逻辑
├── scripts/                    # 实用脚本目录
│   └── sync_data.py            # 🚀 离线数据湖同步脚本
├── runner.py                   # 🌟 核心引擎：配置解析与流水线执行器
└── analyze_importance.py       # 因子重要性分析工具
```

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
建议使用 Python 3.8 及以上版本。
```bash
# 克隆代码
git clone <repository_url>
cd ashare_quant

# 安装依赖
pip install pandas numpy xgboost scikit-learn baostock akshare pandas-ta pyyaml pyarrow
```

### 2. 同步本地数据湖
由于废弃了边跑边下的旧模式，运行回测前必须通过 `sync_data.py` 将最新数据拉取到本地数据湖中。
```bash
# 全量同步所有 A 股日线数据及龙虎榜事件 (耗时较长，建议盘后执行)
python scripts/sync_data.py

# 只单独更新最新的事件数据（如龙虎榜）
python scripts/sync_data.py --events_only

# 或者为了快速测试，只同步前 50 只股票
python scripts/sync_data.py --limit 50
```

### 3. 运行模型训练与回测
编辑 `configs/pipeline_config.yaml`，设定您想要的数据范围、因子组合和模型参数，然后执行：
```bash
python runner.py configs/pipeline_config.yaml
```
系统将自动：
1. 瞬间从本地 Data Lake 加载所需数据并进行内存切片
2. 生成全量特征矩阵
3. 训练 XGBoost 模型并保存到 `models/saved/`
4. 在样本外时间段执行模拟回测并输出收益报告

### 4. 特征重要性分析 (因子提纯)
在跑完一次“全量特征”训练后，您可以分析哪些因子真正有效：
```bash
python analyze_importance.py
```
这会输出特征重要性排名（并保存在 `data/analysis/` 下）。您可以挑选 Top 30 的因子，将它们填入 YAML 的 `custom` 策略中进行第二轮精简训练，从而提高模型鲁棒性。

### 5. 生成实盘预测 (Inference)
当模型训练完毕后，使用推理配置文件进行最新数据的预测：
```bash
python runner.py configs/predict_config.yaml
```

---

## ⚙️ 配置指南 (Configuration Guide)

`configs/pipeline_config.yaml` 是整个框架的“遥控器”。以下是关键配置项说明：

### 数据与股票池 (`data`)
```yaml
data:
  sources: ["BaostockSource"]
  pool:
    board: "chinext"    # 可选: 'main'(主板), 'chinext'(创业板), 'star'(科创板), 'all'
    exchange: "sz"      # 可选: 'sh', 'sz', 'bj', 'all'
    max_count: 50       # 测试用，限制股票数量。实盘设为 0
```

### 数据预处理与防雷 (`preprocessing`)
```yaml
preprocessing:
  dynamic_filter:
    enable: true
    min_avg_turnover: 10000000  # 剔除近20日日均成交额低于 1000万 的僵尸股
    min_listed_days: 120        # 剔除上市不满半年的次新股
  mad_clip: true                # 开启横截面 MAD 去极值
  z_score: true                 # 开启横截面 Z-Score 标准化
```

### 建模目标 (`LabelGenerator`)
```yaml
- name: "LabelGenerator"
  params: 
    horizon: 3  # 预测未来3天的表现
    target_type: "rank_pct" # 强烈推荐：截面收益率百分比排名 (配合 rank:pairwise)
```

### 策略与风控 (`strategy`)
```yaml
strategy:
  name: "MLStrategy"
  params:
    top_k: 10                   # 每日买入排名前 10 的股票
    use_market_filter: true     # 大盘风控：指数跌破 20 日线时自动空仓
    max_turnover: 0.5           # 换手率控制：每日最多调仓 50%
    weight_method: "score"      # 资金分配：按预测分数加权买入 (而非等权)
```

---

## ⚠️ 免责声明 (Disclaimer)
本项目仅供量化研究与学术交流使用。由于公共数据源（如 Baostock, AkShare）可能存在网络延迟或数据缺失，实盘交易需谨慎评估风险。开发者对使用本项目造成的任何投资损失概不负责。
