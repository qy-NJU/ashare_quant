# 常见问题与排查指南 (Troubleshooting Guide)

## 一、数据问题

### 1.1 数据加载失败

**症状**: 回测时报错 `No data available` 或空 DataFrame。

**排查步骤**:

1. 检查本地数据湖是否存在:
```bash
ls -la data/local_lake/daily_k/ | head -20
```

2. 检查数据是否覆盖目标时间段:
```python
import pandas as pd
df = pd.read_parquet('data/local_lake/daily_k/sh.600000.parquet')
print(f"数据范围: {df.index.min()} ~ {df.index.max()}")
```

3. 检查同步脚本是否正常执行:
```bash
python scripts/sync_data.py --limit 5 --start_date 2023-01-01
```

**解决方案**:
- 运行全量同步: `python scripts/sync_data.py`
- 或增量同步: `python scripts/sync_data.py --start_date 2023-01-01`

---

### 1.2 特征缓存导致的数据不一致

**症状**: 修改了因子参数但回测结果没变化。

**原因**: 旧的特征缓存仍然被使用。

**解决方案**:
```bash
# 删除旧的特征缓存 (根据配置的 MD5 指纹)
rm -rf data/local_lake/features/*

# 或者只删除特定配置的缓存
# 首先找到 config_hash
# 然后删除对应目录
rm -rf data/local_lake/features/<config_hash>/
```

---

### 1.3 股票代码格式不匹配

**症状**: 报错 `Symbol not found` 或数据为空。

**原因**: 不同数据源的股票代码格式不同。

| 数据源 | 格式示例 | 说明 |
|--------|----------|------|
| Baostock | `sh.600000`, `sz.000001` | 带前缀 |
| AkShare | `600000`, `000001` | 不带前缀 |
| Local Lake | `sh.600000`, `sz.000001` | 与 Baostock 一致 |

**排查**:
```python
# 检查 repository 返回的股票代码格式
repo = DataRepository()
df = repo.get_daily_data("sh.600000", "20230101", "20230131")
print(df.head())
```

**解决方案**: 确保 YAML 配置中的 `index_symbol` 等参数使用正确的格式 (`sh.000300` 而非 `000300`)。

---

## 二、模型问题

### 2.1 模型训练后预测全是相同值

**症状**: 所有股票的预测分数相同或非常接近。

**可能原因**:

1. **特征全为 NaN 或常数**: 检查特征是否正确计算。
2. **目标变量 (label) 全为相同值**: 某些股票在目标时间段停牌。
3. **Drop Middle 过度采样**: 把太多样本删掉了。

**排查**:
```python
# 检查特征方差
print(X_full.std().sort_values().head(10))

# 检查标签分布
print(y_full.describe())
```

**解决方案**:
- 减少 `drop_middle_threshold` (如从 0.4 降到 0.2)
- 增加 `sample_rate`
- 检查 `preprocessing.mad_clip` 和 `z_score` 是否过于严格

---

### 2.2 增量训练效果变差

**症状**: 增量训练后模型性能下降。

**原因**: XGBoost 的增量训练是在已有模型基础上继续优化，如果新数据分布与旧数据差异大，可能导致"灾难性遗忘"。

**解决方案**:
- 降低 `num_boost_round` (如从 50 降到 5-10)
- 调整学习率 `eta`
- 考虑全量重训练而非增量训练

---

### 2.3 rank:pairwise 目标训练报错

**症状**: 报错 `group size must be > 1` 或排序相关错误。

**原因**: 排序学习需要每个 query (日期) 至少有 2 个样本。

**排查**:
```python
# 检查每日的股票数量
groups = X_full.groupby(X_full.index).size()
print(groups[groups < 2])
```

**解决方案**:
- 增加股票池数量 (`pool.max_count` 设为更大值)
- 过滤掉股票数量太少的日期
- 确认 `min_avg_turnover` 和 `min_listed_days` 过滤条件不要太严格

---

## 三、回测问题

### 3.1 回测结果为 0 或 NaN

**症状**: 回测报告的收益率全是 0，或 `portfolio_history` 为空。

**可能原因**:

1. **信号未能生成**: MLStrategy 无法选出股票。
2. **涨跌停导致无法买入**: 所有候选股票都涨停。
3. **数据时间范围不匹配**: 回测时间段早于或晚于数据范围。

**排查**:
```python
# 检查回测时间范围
print(f"Backtest period: {b_start} to {b_end}")

# 检查数据是否覆盖回测期
df = pd.read_parquet('data/local_lake/daily_k/sh.600000.parquet')
print(f"Available: {df.index.min()} ~ {df.index.max()}")
```

**解决方案**:
- 确保 `windows.backtest.start` 和 `end` 在数据范围内
- 检查 `pipeline_config.yaml` 中的 `pool.max_count` 是否太小
- 尝试增加 `top_k` 数量

---

### 3.2 最大回撤异常大

**症状**: 最大回撤超过 50% 或显示为 NaN。

**可能原因**:
1. 某只股票长期停牌，但持仓按最后收盘价计算
2. 涨跌停连续，无法卖出
3. 初始资金太小，个股仓位占比过高

**解决方案**:
- 启用大盘风控 (`use_market_filter: true`)
- 限制单只股票最大权重
- 增加初始资金

---

### 3.3 佣金/印花税计算错误

**症状**: 收益明显低于预期。

**排查**: 检查 `backtest/engine.py` 中的费率设置:

```python
self.commission = 0.0003      # 万三佣金 (买卖都要交)
slippage = 0.002              # 千二滑点
stamp_duty = 0.0005           # 万五印花税 (仅卖出)
```

**A股真实费率**:
- 佣金: 通常万三，最低 5 元/笔
- 印花税: 万五，仅卖出时收取
- 过户费: 万分之 0.1 (仅沪市)

---

## 四、配置问题

### 4.1 YAML 格式错误

**症状**: 报错 `yaml.scanner.ScannerError` 或配置读取失败。

**排查**:
```bash
# 检查 YAML 语法
python -c "import yaml; yaml.safe_load(open('configs/pipeline_config.yaml'))"
```

**常见错误**:
- 缩进不一致 (YAML 对缩进敏感)
- 使用了 Tab 而非空格
- 字符串中包含未转义的特殊字符

---

### 4.2 配置参数不生效

**症状**: 修改了配置但没效果。

**可能原因**:
1. **特征缓存**: 同 1.2 节。
2. **配置路径错误**: 使用了相对路径而非绝对路径。
3. **Mode 设置错误**: 确认 `mode` 是 `train` 还是 `inference`。

---

## 五、性能问题

### 5.1 训练速度太慢

**可能原因**:
1. 特征缓存未命中，每次都重新计算 130+ 指标。
2. 股票数量太多 (pool.max_count 设为 0 表示全市场)。
3. 时间窗口太长。

**解决方案**:
- 确保特征缓存生效 (同 1.2 节)
- 先用小样本测试 (`pool.max_count: 50`)
- 缩短时间窗口
- 启用随机降采样 (`sample_rate: 0.5`)

---

### 5.2 内存不足 (OOM)

**症状**: 训练时内存溢出。

**原因**: 全市场股票 × 全时间段 × 全因子 = 巨大矩阵。

**解决方案**:
- 减少股票数量 (`pool.max_count: 100`)
- 减少时间窗口
- 使用随机降采样 (`sample_rate: 0.3`)
- 启用 Drop Middle 减少样本量

---

## 六、调试技巧

### 6.1 打印中间结果

在 `runner.py` 中添加调试输出:

```python
# 打印股票列表
print(f"Target symbols: {symbols[:10]}...")

# 打印特征数量
print(f"Feature columns: {len(feature_cols)}")

# 打印每批数据大小
print(f"X_batch shape: {X_full.shape}")
```

### 6.2 保存中间结果

```python
# 保存特征矩阵
X_full.to_parquet('debug_features.parquet')

# 保存标签
y_full.to_parquet('debug_labels.parquet')

# 保存预测结果
pred_df.to_csv('debug_predictions.csv')
```

### 6.3 检查日志

运行回测时注意观察日志输出，特别是:
- `[date] ML Selected: [...]` - 每日选股结果
- `[date] Buy/Sell: ...` - 每日交易记录
- `Limit Up/Down! Cannot buy/sell` - 涨跌停跳过

---

## 七、错误代码速查

| 错误信息 | 原因 | 解决方案 |
|----------|------|----------|
| `No symbols to process` | 股票池过滤后为空 | 检查 `pool.board`, `pool.exclude_st` 设置 |
| `No data available for this window` | 数据不覆盖训练期 | 同步数据或调整时间窗口 |
| `group size must be > 1` | 排序学习样本不足 | 增加股票数量 |
| `No module named 'xxx'` | 缺少依赖 | `pip install xxx` |
| `File not found: models/saved/xxx` | 模型文件不存在 | 先运行训练模式 |
| `Cache file is corrupted` | 缓存文件损坏 | 删除缓存重新计算 |
