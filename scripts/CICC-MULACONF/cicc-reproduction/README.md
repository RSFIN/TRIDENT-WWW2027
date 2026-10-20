# WWW 2027 Intent Experiments

当前实现聚焦论文方案中的基线“单标签共形（CICC 复刻）”：用免训练检索分数代替已训练分类器的 softmax 分数，构造 split-conformal 意图集合，并按集合大小执行放行、澄清或拒识。

该目录是独立实验根目录。CLINC150 使用论文统一口径 v1.0 的固定处理后划分：`../intent-conformal_20260929/processed/clinc150/`。

项目使用 `uv` 管理环境，支持 Python 3.11–3.14。核心实验代码只依赖 Python 标准库；开发依赖仅包含 `pytest`。

## 算法流程

先用域内训练文本构建 TF-IDF 意图原型并计算各意图检索分数；再用域内校准集的真实标签分数确定 split-conformal 阈值，形成候选意图集合。根据集合大小执行放行或澄清，并用校准集中的域外样本选择拒识阈值。测试集只用于最终评测。这里复现的是 CICC 决策框架，评分器是本项目的免训练检索实现。

## 数据集

本单标签 CICC 基线使用处理后的 **CLINC150** JSONL。

- 固定划分总规模为 train 11,438、calib 3,812、test 5,500；其中 ID 为 10,125/3,375/4,050，OOS 为 1,313/437/1,450；
- 实验只从 ID train 构建检索索引，ID calib 用于 split-conformal（α=0.1），calib 的 ID/OOS 用于 OOS 阈值，test 只做最终评测；
- MixATIS 和 MixSNIPS 是多意图数据集，不属于本次单标签 CICC 主实验。

## uv 环境与运行（PowerShell）

在本目录执行。首次使用 uv 时先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)；`uv sync` 根据 `pyproject.toml` 和 `uv.lock` 创建本目录的 `.venv`，`uv run` 自动使用它。如果当前 Python 不在 3.11–3.14 范围内，先执行 `uv python install 3.11`，再用 `uv sync --python 3.11`。运行前确认 `../intent-conformal_20260929/processed/clinc150/` 已准备好。

完整实验：

```powershell
uv sync
uv run python .\scripts\run_cicc.py --config .\configs\cicc_clinc150.json --dataset clinc150
```

快速冒烟实验：

```powershell
uv run python .\scripts\run_cicc.py --config .\configs\cicc_clinc150.json --dataset clinc150 --smoke
```

测试：

```powershell
uv run pytest -q
```

## 方法边界

- CICC 复刻的是其框架：有限样本分位数、单标签 conformal set、集合大小驱动的放行/澄清/拒识，以及用 OOS 校准数据选择 `alpha` 和最大澄清集大小。
- 原 CICC 使用训练分类器概率；本项目按论文方案改为训练集上的意图级 TF-IDF 原型检索分数。这是“CICC on retrieval scores”，不是官方仓库逐行复现。
- 默认 `aps` 使用按分数降序累积的自适应集合；配置可切换为 `marginal`，复刻 `1 - p_y` 非一致性分数。
- CLARA 原论文研究视觉检索。本项目只迁移两个可比较原则：把候选模式展示成可直接选择的卡片，以及在多轮选择引起分布偏移时使用选择似然比加权的 split conformal。没有正确指定选择模型时，代码不会宣称多轮 coverage 保证。
- CLINC150 没有歧义/澄清必要性人工标注，因此其 `clarification_trigger_accuracy` 输出为 `null`。CAsT/Needle 数据接入后才计算该指标。
- 当前目录没有 hard-negative OOS 数据；配置中的 `hard_negative_path` 默认为 `null`，结果会明确标为未评测。接入 JSONL（每行至少含 `text`）后会单独报告 hard-negative AUROC/AUPRC 与拒识召回。

## 数据协议

- `train`：仅用于构建无梯度意图原型索引。
- `calib`：ID 部分用于 conformal calibration；ID/OOS 同时用于固定目标召回率的 OOS 阈值估计。
- `test` 与 `oos_test`：只用于最终评测。
- 默认种子为 42、43、44，共 3 次重复；汇总均值、样本标准差和基于 3 次重复的 t 分布 95% 置信区间。
- 预测文件保留逐样本结果，可与其他基线做成对 McNemar 检验。

## 指标

| 层级 | 已实现指标 |
| --- | --- |
| 统一口径 v1.0 | Macro-F1、Set Exact、Set IoU、Hit@1/3、Cov(J≥0.5)、Cov(含)、Avg \|P\|、ECE、误拒率、OOS拒识率、澄清率 |

`src/cicc_baseline/statistics.py` 还实现了精确 McNemar 检验。只有存在同一测试样本上的另一算法预测时才应运行该检验。

## 参考

- den Hengst et al. Conformal Intent Classification and Clarification. Findings of NAACL 2024. https://aclanthology.org/2024.findings-naacl.156/
- Tran et al. Show, Don’t Ask: Generative Visual Disambiguation for Composed Image Retrieval with Turn-Valid Coverage. arXiv:2606.18992v1. https://arxiv.org/abs/2606.18992

