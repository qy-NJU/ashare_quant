# A-Share Quant (A股量化选股框架)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A-Share Quant 是一个基于 Python 的工业级、配置驱动 (Config-driven)、高度模块化的 A 股量化交易与回测框架。

本项目旨在为量化研究员提供一个从数据获取、特征工程、模型训练到回测评估的完整闭环，尤其强化了对机器学习（如 XGBoost）和前沿量化方法（如 Learning to Rank）的支持。

---

## 🌟 核心特性 (Features)

- **⚙️ YAML 配置驱动**: 告别硬编码！通过修改 `pipeline_config.yaml` 即可切换数据源、调整时间窗口、开启/关闭因子、修改模型参数，实现极简的策略实验。
- **🔌 模块化架构**:
  - **数据层 (Data)**: 支持 Baostock、AkShare 多源数据接入，内置本地 Parquet 高速缓存与基于 `StockPoolManager` 的灵活板块过滤（如全市场、沪深300、创业板）。
  - **特征工程 (Features)**: 支持 130+ 种 `pandas-ta` 技术指标、财务基本面因子（ROE、净利润增长等）、资金流向因子及大盘宏观特征。
  - **模型层 (Models)**: 原生集成 XGBoost，支持**增量学习 (Incremental Learning)**、**类别特征原生处理 (Categorical Support)** 以及专业的**排序学习 (Learning to Rank - pairwise)**。
  - **策略与回测 (Strategy & Backtest)**: 内置基于预测分数的 Top-K 选股策略，支持大盘风控过滤、换手率控制及基于预测分数的资金加权分配。
- **🔮 实盘预测模式**: 提供专门的 `predict_config.yaml` 和推理模式，一键输出“明日十大金股”。

---

## 🏗️ 架构概览 (Architecture)

```text
ashare_quant/
├── data/                       # 数据接入与管理层
│   ├── source/                 # 具体数据源实现 (Baostock, AkShare)
│   ├── cache/                  # Parquet 格式的本地数据缓存
│   ├── repository.py           # 统一的数据仓库入口
│   ├── pool_manager.py         # 股票池过滤器 (板块、交易所过滤)
│   └── market_manager.py       # 大盘指数与宏观数据管理
├── features/                   # 特征工程层 (Pipeline)
│   ├── factors/                # 因子库
│   │   ├── pandas_ta_factor.py # 技术指标因子 (基于 pandas-ta)
│   │   ├── fundamental.py      # 行业/概念类别因子
│   │   ├── financial.py        # 财务基本面因子
│   │   ├── fund_flow.py        # 资金流向因子
│   │   ├── market.py           # 大盘宏观因子
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
├── runner.py                   # 🌟 核心引擎：配置解析与流水线执行器
├── pipeline_config.yaml        # 训练与回测配置文件
└── predict_config.yaml         # 实盘推理配置文件
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

### 2. 运行模型训练与回测
编辑 `pipeline_config.yaml`，设定您想要的数据范围、因子组合和模型参数，然后执行：
```bash
python runner.py pipeline_config.yaml
```
系统将自动：
1. 下载并缓存数据
2. 生成全量特征矩阵
3. 训练 XGBoost 模型并保存到 `models/saved/`
4. 在样本外时间段执行模拟回测并输出收益报告

### 3. 特征重要性分析 (因子提纯)
在跑完一次“全量特征”训练后，您可以分析哪些因子真正有效：
```bash
python analyze_importance.py
```
这会输出特征重要性排名（并保存在 `data/analysis/` 下）。您可以挑选 Top 30 的因子，将它们填入 YAML 的 `custom` 策略中进行第二轮精简训练，从而提高模型鲁棒性。

### 4. 生成实盘预测 (Inference)
当模型训练完毕后，使用推理配置文件进行最新数据的预测：
```bash
python runner.py predict_config.yaml
```
系统会拉取最近的数据，跳过训练，直接输出明日的预测打分及推荐买入列表。

---

## ⚙️ 配置指南 (Configuration Guide)

`pipeline_config.yaml` 是整个框架的“遥控器”。以下是关键配置项说明：

### 数据与股票池 (`data`)
```yaml
data:
  sources: ["BaostockSource"]
  pool:
    board: "chinext"    # 可选: 'main'(主板), 'chinext'(创业板), 'star'(科创板), 'all'
    exchange: "sz"      # 可选: 'sh', 'sz', 'bj', 'all'
    max_count: 50       # 测试用，限制股票数量。实盘设为 0
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
