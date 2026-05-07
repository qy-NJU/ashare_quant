# 回测性能优化计划

## 现状瓶颈

```
每日调仓回测: 80个交易日, 每个交易日:
  ┌─ ProcessPoolExecutor(8进程)          ← 创建成本高
  ├─ 每进程 import pandas_ta → numba JIT  ← 编译2-3秒/进程  
  ├─ 每进程加载500天原始数据×~220只       ← I/O密集
  ├─ 每进程运行11个因子模块 + 时序衍生    ← CPU密集
  ├─ 主进程 concat 1764行 → MAD/ZScore   ← 内存峰值
  └─ 进程池销毁                           
  总: ~30秒/天 × 80天 = 40分钟
```

## 优化方案（三层）

### 第一层：特征预计算 + 磁盘缓存 [预期 -95% 耗时]

核心思路：复用训练阶段已实现的批量特征计算，在回测循环前一次性算完所有日期的全截面特征。

```
优化前: backtest_loop { select_stocks(day_i) → ProcessPoolExecutor → compute }
优化后: precompute_all_dates_batch → cache to disk
        backtest_loop { select_stocks(day_i) → load cached DataFrame }
```

**改动点**：
- `runner.py`: 在 `backtest_only` 模式中，回测循环前增加 `_precompute_backtest_cross_sections()`
- `ml_strategy.py`: 添加 `precomputed_snapshots` 参数，命中时跳过 ProcessPoolExecutor

**预期效果**：40分钟 → 2-3分钟（批量特征计算~2分钟 + 回测循环查表30秒）

### 第二层：减少候选股票池 [预期 -30% 后续开销]

回测时不需要每天对全部1764只股票预测。预筛选200-500只：
- 流通市值前500
- 或日均成交额 > 阈值

已实现的 MaxDailySwaps 只需要前N名，缩小候选池不影响结果。

### 第三层：内存优化

- 磁盘缓存替代内存全量持有（大DataFrame按需加载）
- 回测结束后自动清理临时缓存

## 实施

1. `ml_strategy.py`: 添加 `precomputed_cache_dir` 参数，`select_stocks` 优先从缓存读取
2. `runner.py`: `backtest_only` 模式中预计算并写入缓存
3. 批量计算复用训练阶段的 `_compute_features_for_stocks` 模式
