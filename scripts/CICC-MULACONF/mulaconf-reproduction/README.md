# MuLaConformal 多标签共形基线复刻

本项目为 WWW 2027 意图识别实验实现两种固定标签集合共形基线：

- `powerset`：将训练集中每一种已经观察到的多意图标签组合作为一个类别，在这些标签集合上构造 split-conformal 预测区域。
- `tree_fixed_alpha`：根据标签集合之间的 Hamming 距离进行平均链接层次聚类，形成标签集合树；随后沿根节点到叶节点的路径执行 split-conformal 检验，并使用分层 Bonferroni 阈值控制族错误率。

树结构方法参考 Tyagi 和 Guo 的论文 *Multi-label Classification under Uncertainty: A Tree-based Conformal Prediction Approach*（COPA/PMLR 2023，arXiv 2404.19472）。项目名称沿用论文方案中的“mulaconf”基线名称，但不依赖同名的 `mulaconf` PyPI 软件包；该软件包中的 powerset 实现对应后续的结构惩罚研究，并不是本项目引用文献 `[33]` 的原始树结构方法。

## 算法流程

两个方法都从训练集构建免训练 TF-IDF 检索评分器，并只在校准集上确定阈值。`powerset` 把训练中出现过的标签组合作为单个类别构造共形区域；`tree_fixed_alpha` 按标签集合距离构建层次树，沿树路径以固定显著性水平检验。测试时将区域映射回意图集合，随后计算多标签、校准和拒识指标。训练中未出现的标签组合不在候选类别中，解释覆盖率时须考虑这一限制。

## 数据集

项目直接读取父级项目中的数据，不重复复制数据文件：

| 数据集 | 训练集 | 验证集 | 测试集 | 标签语义 |
| --- | ---: | ---: | ---: | --- |
| MixATIS clean | 13,162 | 759 | 828 | 原生 1–3 个意图的标签集合 |
| MixSNIPS clean | 39,776 | 2,198 | 2,199 | 原生 1–3 个意图的标签集合 |
| CLINC150 | 11,438（含 1,313 OOS，仅 10,125 ID 建索引） | 3,812（3,375 ID + 437 OOS） | 5,500（4,050 ID + 1,450 OOS） | 域内标签表示为单元素标签集合 |

CLINC150 不是原生多标签数据集。本项目使用它检验多标签方法退化到单标签场景时的表现，并利用其原生 OOS 数据评估域外检测能力。构建检索索引和标签集合词表时只使用训练集，不使用任何测试样本，防止测试数据泄漏。

## 环境创建

在本目录执行；先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。项目支持 Python 3.11–3.14；`uv sync` 根据 `pyproject.toml` 和 `uv.lock` 创建本目录的 `.venv`，后续 `uv run` 自动使用该环境。若系统 Python 版本不符，先执行 `uv python install 3.11`，再执行 `uv sync --python 3.11`。运行前确认父目录中的 MixATIS/MixSNIPS 原始数据和 CLINC150 处理后划分已经就绪。

```powershell
uv sync
```

环境包含 NumPy、SciPy 和 pytest。基础 TF-IDF 评分器不包含通过梯度训练得到的参数；SciPy 只用于构建标签集合层次树。

## 验证数据集和自动化测试

```powershell
uv run python .\scripts\validate_datasets.py
uv run pytest
```

第一条命令检查三个数据集的样本数量、标签数量和标签基数分布，第二条命令运行全部自动化测试。

## Smoke 快速验证

```powershell
uv run python .\scripts\run_mulaconf.py `
  --config .\configs\mulaconf_all.json `
  --dataset clinc150 `
  --smoke
```

Smoke 模式只使用缩小后的数据子集和随机种子 42，用于快速验证数据加载、检索、共形校准、指标计算和结果保存流程。Smoke 结果不能作为论文正式实验结果。

## 完整三轮实验

```powershell
uv run python .\scripts\run_mulaconf.py `
  --config .\configs\mulaconf_all.json `
  --dataset clinc150
```

完整实验使用随机种子 42、43 和 44，运行三轮并在控制台打印均值、样本标准差和 95% t 置信区间。每轮实验结果保存在：

控制台会先按 Seed 42、43、44 分别打印 L1–L5 的完整标量指标，最后打印三轮的 Mean、Std 和 95% CI 汇总。可靠性分箱和预测区域大小分布等明细保存在 JSON 文件中，不在控制台逐项展开。

```text
results/mulaconf_all/<dataset>/<method>/seed_<seed>/
```

## 分别运行三个数据集

使用 `--dataset` 指定本次需要运行的数据集。单独运行时仍然使用相同的两种方法、三个随机种子和全部指标。

```powershell
# 只运行 MixATIS
uv run python .\scripts\run_mulaconf.py --config .\configs\mulaconf_all.json --dataset mixatis

# 只运行 MixSNIPS
uv run python .\scripts\run_mulaconf.py --config .\configs\mulaconf_all.json --dataset mixsnips

# 只运行 CLINC150
uv run python .\scripts\run_mulaconf.py --config .\configs\mulaconf_all.json --dataset clinc150
```

`--dataset` 为必填参数；每条命令只运行并打印一套数据集的一张统一口径表。

可以在任意单数据集命令后添加 `--smoke`，执行缩小数据和单个随机种子的快速验证。三个数据集的独立汇总文件分别保存为：

- `summary_mixatis.json`
- `summary_mixsnips.json`
- `summary_clinc150.json`

因此，依次运行三个数据集不会造成汇总结果相互覆盖。

## 配对 McNemar 检验

```powershell
uv run python .\scripts\compare_mcnemar.py `
  .\results\mulaconf_all\mixsnips\powerset\seed_42\predictions.jsonl `
  .\results\mulaconf_all\mixsnips\tree_fixed_alpha\seed_42\predictions.jsonl
```

检验通过统一的 `sample_id` 对齐相同域内测试样本，并使用统一口径下拒识后主意图是否正确作为二元配对结果；可直接比较三个项目同一 seed 的预测文件。不能在不同数据划分之间比较。

## 结果文件

每个 `<dataset>/<method>/seed_<seed>` 目录包含：

- `metrics.json`：该轮实验的 L1–L5 指标。
- `predictions.jsonl`：逐样本的原始文本、真实标签、预测标签、共形区域、置信度、OOS 分数和决策结果。

根结果目录中的 `summary.json` 保存全部数据集的三轮汇总；单数据集运行则生成对应的 `summary_<dataset>.json`。

## 指标范围

- L1 多意图识别：Exact Match、意图集合 IoU、Macro-F1、Hit@1/3/5。
- L2 OOS：AUROC、AUPRC、FPR95 和 F1-OOS；只在具有原生 OOS 的 CLINC150 上计算。
- L3 校准：ECE、Brier、精确标签集合覆盖率、Jaccard≥0.5 覆盖率、名义覆盖率和平均预测区域大小。
- L4 效率：延迟 P50/P99、近似 tokens/样本、可训练参数量和索引构建时间。
- L5 决策：拒识率、错误拒识率、澄清率、集合级 Jaccard 错误率以及 95% OOS 召回率工作点。

## 解释边界和注意事项

- 只有训练集中出现过的标签集合会成为 powerset 类别或树的叶节点。测试集中未见过的标签组合会构成真实的端到端覆盖失败。实验同时报告全测试覆盖率和仅针对训练阶段已见标签集合的条件覆盖率。
- `tree_fixed_alpha` 复刻 fixed-alpha 分层 FWER 控制过程。论文中的 adaptive-alpha 调优版本尚未实现，项目不会将当前结果标记为 adaptive-alpha 结果。
- MixATIS 和 MixSNIPS 没有 OOS 划分，因此 L2 OOS 指标只在 CLINC150 上报告。
- 当前三个数据集不包含 hard-negative OOS 和“是否需要澄清”的真实标签，这两项会明确标记为未评估，而不会使用人工构造数值代替。
- `mean_clarification_rounds` 当前是单轮代理指标。三个数据集都不包含真实的多轮澄清交互轨迹，因此不能将其解释为真实多轮 turn-valid 实验结果。
