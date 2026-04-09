# A股量化模型去噪与垃圾股过滤优化计划

## 一、 现状分析 (Current State Analysis)

根据对当前代码的探索，框架在数据清洗和特征去噪方面存在以下薄弱环节：
1. **股票池未做风险过滤**：目前的 `StockPoolManager` (`data/pool_manager.py`) 仅仅支持按照板块（主板、创业板等）和交易所过滤，**没有剔除 ST 股、*ST 股和退市整理期股票**。这会导致模型学习到被资金高度控盘或即将退市的垃圾股的异常波动。
2. **特征缺乏截面去极值与标准化**：在 `runner.py` 中，各个股票独立计算完技术指标后，直接被 `concat` 拼接到一起喂给 XGBoost 模型。A 股市场经常出现极端异动（如连板或闪崩），如果不对每天的特征数据做横截面（Cross-sectional）的去极值（MAD）和 Z-Score 标准化，模型极易被极少数的异常值带偏，且容易受到大盘整体涨跌波动的干扰（把 Beta 当 Alpha）。
3. **标签去噪**：目前的 `LabelGenerator` 已经使用了 `rank_pct`（截面排名）作为预测目标，这在一定程度上已经剥离了大盘波动噪音，非常棒！但特征输入端依然是绝对数值，需要改进。

## 二、 提议的优化 (Proposed Changes)

为了提高模型的鲁棒性（Robustness）和抗噪能力，计划从“源头”和“特征”两个层面进行优化：

### 1. 股票池防雷：剔除 ST 与退市股
- **目标文件**: `data/pool_manager.py`
- **改动说明**: 
  - 在 `StockPoolManager.get_filtered_symbols` 方法中，读取 `stock_list.parquet` 时，获取股票的 `name` 字段。
  - 增加过滤逻辑：默认剔除名称中包含 `ST`、`*ST`、`退` 的股票。
  - 在配置文件中暴露 `exclude_st` 参数，允许用户开关此功能。

### 2. 截面特征清洗：MAD去极值 + Z-Score标准化
- **目标文件**: 新增 `features/processor.py` (或直接在 `features/pipeline.py` 中扩展) 以及 `runner.py`
- **改动说明**:
  - 实现一个 `CrossSectionalProcessor` 类。
  - **MAD 去极值 (Winsorization)**：对每天（groupby date）的每个因子，计算中位数（Median）和绝对中位差（MAD），将超出 $[Median - 3 \times 1.4826 \times MAD, Median + 3 \times 1.4826 \times MAD]$ 范围的极端异常值强行拉回（Clipping）。
  - **Z-Score 标准化**：去极值后，将每个因子在每天的横截面上减去均值并除以标准差，使得每天的因子分布都服从标准正态分布。
  - 在 `runner.py` 的训练阶段和推理阶段，在调用 `model.train` 和 `model.predict` 之前，将拼接好的全市场数据传入 `CrossSectionalProcessor` 进行处理。

### 3. 配置文件升级
- **目标文件**: `configs/pipeline_config.yaml`
- **改动说明**:
  - 在 `pool` 下面增加 `exclude_st: true`。
  - 在 `features` 同级或 `pipeline` 级别增加 `preprocessing` 配置项，允许开启或关闭 `mad_clip` 和 `z_score`。

## 三、 假设与决策 (Assumptions & Decisions)
- **计算时机决策**：特征的截面标准化必须在所有股票的数据合并（`pd.concat`）之后、喂给模型之前进行（即按 `date` 分组计算）。
- **性能假设**：由于数据已经缓存在本地 Data Lake，在 Pandas 中进行基于 `groupby(level=0)`（日期作为 index）的 `transform` 操作性能是可以接受的。

## 四、 验证步骤 (Verification Steps)
1. 运行 `runner.py configs/pipeline_config.yaml`。
2. 观察控制台输出，确认 `StockPoolManager` 打印出“剔除了 X 只 ST/退市股票”。
3. 确认日志中输出了“Applying cross-sectional MAD clipping and Z-score normalization...”的信息。
4. 确保回测能够正常完成，并且不报错，验证去噪后模型的表现是否更加平稳。