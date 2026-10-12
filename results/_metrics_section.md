## 指标定义（统一口径 v1.0）

### 统一口径核心规则（所有方法必须遵守）

1. **全集合口径**：所有集合指标在"完整意图集合"上计算——P = 系统最终输出的完整候选集合，Y = 真实全部意图；禁止拆分 aux 子集计算。
2. **reject = 空集 ∅**：决策为拒绝的样本 P = ∅，且计入所有指标分母：Exact Match=0、Set IoU=0、Hit@k=0、set size 计 0、主意图 F1 计预测错误。
3. **ECE 统一为"主意图 top-1 置信度"**（15 bins）：Ours 用 top1-fused confidence = 1 − min fused score；CICC / MULACONF / QP 用各自模型 top-1 输出概率；各方法必须注明置信度来源。
4. **误拒率唯一合法定义** = in-domain 样本中被判"拒绝 / 空集"的比例；禁止用"集合与标签无交集比例"替代。
5. **OOS 指标仅 CLINC150 报告**（MixATIS / MixSNIPS 无 OOS 类，写 N/A，不写 0）。
6. **α = 0.1**（名义覆盖率 1−α = 0.90）、split conformal；报告 3 次重复实验的"均值 ± 标准差"（Ours 如时间不允许至少注明单次）。

### 指标精确定义（L1–L5）

| 层级 | 指标 | 精确定义 | 备注 |
|---|---|---|---|
| L1 闭集 | Accuracy / Macro-F1 | 主意图预测 m（P 内最高置信度者；reject 时 m=∅ 计错误）与真实主意图比较；Macro-F1 用 sklearn 口径，zero_division=0 | Accuracy 仅 CLINC150；多标签数据集主意图取标注的 main_intent |
| L1 多意图 | Set Exact Match | #{P == Y} / N | 全集合 |
| L1 多意图 | Set IoU | mean( \|P∩Y\| / \|P∪Y\| )；P=Y=∅ 记 1，仅一方为 ∅ 记 0 | 全集合；禁止只在非拒子集上平均 |
| L1 多意图 | Hit@k | #{Y ⊆ top-k(R)} / N，R 为按置信度降序的预测排序 | 各方法需提供排序输出 R |
| L2 OOS | AUROC / AUPRC / FPR95 | 基于 OOS 打分（1 − top1_sim）与标签的阈值无关指标 | 仅 CLINC150 |
| L3 校准 | ECE（15 bins） | 主意图 top-1 置信度 vs 主意图正确率 | 见核心规则 3 |
| L3 校准 | 集合级经验覆盖率 | Coverage(J≥0.5) = #{J(P,Y) ≥ 0.5} / N，对比名义 1−α | 共形保证检验；多意图并列报 Coverage(含) = #{P∩Y≠∅} / N 作诊断 |
| L3 校准 | 平均预测集大小 | mean( \|P\| ) | 紧致性（自适应澄清集收益） |
| L5 决策 | 误拒率 | #{x ∈ D_in : P = ∅} / \|D_in\| | 见核心规则 4 |
| L5 决策 | OOS 拒识率 | #{x ∈ D_oos : P = ∅} / \|D_oos\| | 仅 CLINC150；多意图 N/A |
| L5 决策 | 澄清触发率 | #{决策为 clarify} / N | Ours 三态决策；基线机制不同须注明判定规则 |
| L5 决策 | 拒识率@95% OOS 召回 | 标定阈值使 OOS 拒识率 = 95%，报告此时 in-domain 误拒率 | 仅 CLINC150 |

### 各方法调整清单（按此重算后回填 T2 表）

| 方法 | 必改项 |
|---|---|
| Ours（本文） | Set Exact Match / Set IoU 改为 full 级（true = 全部意图，pred = 完整候选集 pred_full）；补算 Hit@1/3/5（fused 排序）；ECE 已按 top1-fused 保持 |
| CICC（R5） | 说明 CP 输出集合构成（avg set size=78.7）；按 full 级重算 Exact / IoU / Hit@k；ECE 注明置信度来源；误拒率按"拒绝 / 空集"口径重算 |
| MULACONF（R12） | 澄清 MixATIS 误拒率=1.0、MixSNIPS Powerset=0.819 的原始定义并重算；IoU 必须含被拒样本（P=∅ 计 0）；Tree fixed α 集合产出方式说明；ECE 注明置信度来源 |
| QP-OneModel（R12 变体） | 1/5/10-shot 均按 full 级重算；set size 固定 5 的说明保留；ECE 注明置信度来源 |

### 输出模板（每个数据集一张表：CLINC150 / MixATIS / MixSNIPS）

| 方法 | Macro-F1 | Set Exact | Set IoU | Hit@1 | Hit@3 | Cov(J≥0.5) | Cov(含) | Avg \|P\| | ECE | 误拒率 | OOS拒识率 | 澄清率 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

> 本规范与代码工程 `docs/统一指标口径规范_v1.md` 一致，实现细节以该文件为准。
