# WWW'27 投稿前评审报告（含修改落实清单）

- 论文：*Set-Valued Intent Understanding with Calibrated Refusal: A Training-Free Retrieval-Grounded Conformal Approach*
- 目标会议：The ACM Web Conference 2027（WWW '27），Research Track
- 评审日期：2026-10-14（距全文截止 2026-10-25 AoE 约 11 天）
- 评审范围：`latex/main.tex` + `latex/refs.bib` + `latex/figures/`，以 WWW'27 官方 CFP 的格式、范围与审稿决策标准为基准
- 结论：**可修改后投稿**。原稿存在 3 项足以拒稿/触发 desk-reject 的合规与一致性问题；本轮已修复其中 2 项，另有 1 项需作者重绘图片（详见"仍需作者处理"）。

---

## 一、WWW'27 官方要求核验（CFP 依据）

来源：https://www2027.thewebconf.org/research-track-papers/ 与 https://acmweb2027.org/important-dates/

| 要求 | 官方规定 | 原稿状态 | 修改后状态 |
|---|---|---|---|
| 格式 | `\documentclass[sigconf, anonymous, review]{acmart}`，双栏英文 | ✅ 合规 | ✅ 合规 |
| 双盲 | anonymous + review 选项 | ✅ | ✅ |
| 长文页数 | 正文 8 页（前 8 页自洽），参考文献+可选附录最多 12 页 | ❌ 结论溢出到第 9 页（正文超 8 页） | ✅ 正文 8 页、总 10 页 |
| 首页 Web 相关性 + track 声明 | "Every submission must clearly state how the work is relevant to the Web and to the track in the first page" | ❌ 缺失（有隐晦提及，无明确声明） | ✅ 引言首段新增 track 声明 |
| 选题范围 | 须面向"作为技术基础设施的 Web / 社会经济系统 / 影响 / 民主化访问"之一 | ⚠️ 隐式（搜索框/IM/agent 冷启动） | ✅ 显式声明 |
| 截止时间 | 摘要 2026-10-18，全文 2026-10-25（AoE） | — | 可赶上 |

---

## 二、总体判断

研究本身的价值判断：**真实且贴合 WWW 场景**。训练自由的检索接地多意图集合输出 + 共形校准三态拒绝，对搜索/agent 类 Web 入口是刚需；实验扎实、口径统一（v1.0 协议）、与结果文件交叉核对一致（CLINC150 ECE 0.0049/0.005、AUROC 0.9748/0.975、MixATIS test=828 均与 `results/` 输出吻合）。主要问题集中在**呈现层**：页数合规、第一页声明、符号一致性、引用正确性、审稿人会立即注意到的断言与破折号。这些问题全部可以在不改实验的前提下修复，修复后中稿概率有实质提升。

---

## 三、CRITICAL（足以拒稿 / 触发 desk-reject）

| # | 问题 | 位置 | 处理 |
|---|---|---|---|
| C1 | **主文超过 8 页**：结论段落溢出到第 9 页，参考文献从第 10 页起；WWW'27 要求正文 8 页且前 8 页自洽 | 结论（原第 8–9 页） | ✅ 已修复：删除定位段重复段、合并 3.5/3.7 重复内容、压缩 4.11 等，结论完整收在第 8 页 |
| C2 | **首页缺少明确的 Web 相关性与 track 声明**（CFP 明文要求，缺失即 desk-reject 风险） | 引言首段 | ✅ 已修复：新增 "aligned with the Search, Recommendation, and Retrieval-Augmented AI and Web Infrastructure and Agentic Systems tracks of WWW'27" |
| C3 | **图 1 与正文符号冲突**：图内 "Top-K (k=10, K=10)" 与正文 $K_r=1$ 冲突；"Pr(Y ∈ S(q)) ≥ 1−α = 0.90" 与 Jaccard 覆盖 $P[J(\hat{S},Y)\ge\tau_J]$ 冲突；"Sequential jackknife risk ≤ ε" 与 split conformal 冲突；"no to close-bound" 疑似乱码 | `latex/figures/fig1_main.pdf` | ⚠️ **需作者处理**：按 `docs/fig1_framework.html`（干净版，已同步修正符号）重新出图；图内融合权重统一为 λ |

## 四、MAJOR（一审必点，已全部修复）

| # | 问题 | 位置 | 处理 |
|---|---|---|---|
| M1 | **α 符号双重占用**：α 既表示误覆盖水平(0.1)，又表示融合权重（式(1)、3.3、4.1、超参选择、A3 消融、图 1） | 3.3/3.6/4.1/附录 | ✅ 融合权重统一改名 **λ**；文本、图 7 标签（重新出图）、fig1 HTML 源同步 |
| M2 | **引用条目错误**：prism2026/doordash2026 作者写成 Anonymous（实为第三方工作）；intentgrasp 缺作者；qwen2024 条目是 Qwen2.5 报告而正文用 Qwen3.7-plus；din2020calibration 无法核实 | refs.bib | ✅ Prism→Liao et al.(2601.08653)；DoorDash→Aboah Boateng et al.(2603.01486)；IntentGrasp→Yin et al.(2605.06832)；qwen→Qwen3 Technical Report(2505.09388)；din→Mukhoti et al. 2020（focal loss 校准） |
| M3 | **摘要声称 dominate … UniCR，但实验表中无 UniCR 任何结果** | 摘要 | ✅ 从摘要基线清单移除 UniCR |
| M4 | **统计显著性一节出现孤儿对比**："Ours vs. UniCR on CLINC150 F1: 0.899 vs 0.905"，UniCR 不在任何表中 | 4.7 | ✅ 改为表内方法对比（Ours vs. TECP-style (8B) MixATIS IoU 0.387 vs 0.368） |
| M5 | **"staying within 0.6pt F1" 与表 3 对不上**（0.899 vs MULACONF 0.880 差 1.9pt；vs TECP-14B 0.948 差 4.9pt） | 4.1 | ✅ 改为与表格一致的表述 |
| M6 | **"41 open-weight models 研究表明指令微调胜过参数规模"无引用且未能核实** | 2.5 | ✅ 改写为可核实引证（LIMA NeurIPS'23 + Flan JMLR'24） |
| M7 | **破折号滥用**（正文约 25–30 处 "---"，评估规范每处一审必点） | 全文 | ✅ 正文破折号全部改为逗号/冒号/分句（仅表格 "---" 占位保留） |
| M8 | **过度声明**："calibration is orthogonal to model scale"（实验未用 8B/14B 跑本方法全流程） | 5/7 结论 | ✅ 弱化为 "calibration transfers across reasoners" |
| M9 | **相关工作缺关键近作**：MIDLM(COLING'25)、FAITHQA(ACL'25)、Fisch et al. 2022 多标签 CP、Findings ACL'24 免训练生成式意图检测 | 2.2/2.3/2.5 | ✅ 各补一句并新增 6 条文献（zhou2023lima、chung2022scaling、midlm、faithqa、fisch2022fp、zhang2024disc2gen） |
| M10 | **理论表述错误**：3.6 "covariate-shift regime (P4 seed reruns) 内覆盖率退化 ≤0.02 与 O(1/n) 界一致"（O(1/n) 仅交换性下成立；跨域迁移实测退化 0.144）；3.4 "Unlike RCPS" 与 3.6 "standard RCPS construction" 自相矛盾 | 3.4/3.6/4.6 | ✅ 重写为诚实表述：交换性下界成立；协变量漂移下无保证且实测 0.856→0.712；3.4 改为 "As in RCPS, …" |
| M11 | **Organization 段 5 个章节引用未定义**（`\ref{sec:related}` 等无 `\label`，编译显示 "Section ??"） | 引言 Organization | ✅ 补齐 5 个 `\label`（sec:related/method/experiments/discussion/conclusion），编译后正确显示 Section 2/3/4/5/7 |

## 五、MINOR（打磨级）

| # | 问题 | 处理 |
|---|---|---|
| m1 | MixATIS 测试集 828 与公开 benchmark 常用分列不一致，建议注明来源 | ✅ 已与 `results/` 输出交叉核对（误拒率分母 173/828 证实 828 为本实验真实测试数）；建议作者在数据集处补一句分列来源说明 |
| m2 | 3.4 "$K=1$ secondary intent" 与 Implementation "up to 2 secondary intents" 易混淆（生成上限 vs 校准后上限） | ✅ 3.4 重写澄清 |
| m3 | 附录表 5 表头 `Avg$|P|$` 缺方向箭头（应为 ↓） | ✅ 已补 |
| m4 | 4.4 风险比较因果缺解释：收紧 OOS 门反而降低假拒绝率，读者易困惑 | ✅ 补充 "uncertain in-domain queries are routed to the clarify branch rather than rejected outright" |
| m5 | fig8_knowledge_update 与 fig8_roc 文件名重复前缀 | ⚠️ 可选：投稿前建议改名（不影响编译） |
| m6 | "Qwen3.7-plus" 与公开 Qwen 系列命名不对齐 | ✅ 引用已改为 Qwen3 Technical Report；建议投稿前确认该命名是 DashScope 服务形态并加脚注 |

## 六、仍需作者处理（投稿前必做）

1. **重绘图 1**（C3）：`docs/fig1_framework.html` 已是符号正确版本（融合权重 λ=0.1、Jaccard 覆盖表达式），按原出图管线重新生成 `figures/fig1_main.pdf` 后替换，同时删掉图内 "k=10/K=10"、jackknife、乱码字样。
2. **数据集分列来源注**（m1，低优先）：在 4.1 Datasets 处注明 MixATIS 测试分列口径。
3. **双盲自检**：全库 `Anonymous` 检索——确认无泄露身份的条目（当前已无 Anonymous 条目，但请确认作者自引均以第三人称处理）。
4. 如需强一致，可将附录表 5 中 "Avg$|P|$$\downarrow$" 的箭头与主表统一（本轮已统一）。

## 七、可复现性说明

- 修改仅涉及 `latex/main.tex`、`latex/refs.bib`、`latex/figures/fig7_ablation.pdf`（标签 α→λ 重新出图）、`docs/fig1_framework.html`（源文件符号同步）；**未改动任何实验数字与结果文件**。
- 重新编译命令：`latex/tectonic.exe main.tex`（成功，总 10 页：8 页正文 + 1 页参考文献 + 1 页附录）。
- 验证方式：PDF 逐页人工/OCR 检查（页数、结论位置、引用编号、λ 一致性）；pymupdf 全文扫描（无 "??"、无未定义引用）。
- 投稿副本已同步：`overleaf_submission/`（main.tex、refs.bib、fig7_ablation.pdf），并重建 `www2027_overleaf.zip`。
