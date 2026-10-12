以下为 2026-09 已完成的基线实测数据，统一口径：α=0.1，split conformal，CLINC150 test=5500。Ours 为本方法 qwen3.7-plus 免训练版；R5=CICC [31]、R12=MULACONF [33]、R12-QP=QP-OneModel [12] 均在相同 calib/test 划分上复现。

### T2a CLINC150 主结果

| 方法 | Macro-F1 | OOS AUROC | Emp Cov | Avg Set Size | ECE | 误拒率 |
|---|---:|---:|---:|---:|---:|---:|
| **Ours（分层共形+融合分）** | **0.896** | **0.975** | 0.911 | **~3.0** | — | — |
| R5 CICC split CP [31] | 0.625 | 0.759 | 0.990 | 78.7 | 0.163 | 0.588 |
| R12 MULACONF Powerset [33] | 0.625 | 0.768 | 0.915 | 4.5 | 0.163 | 0.637 |
| R12 MULACONF Tree fixed α [33] | 0.625 | 0.768 | 0.997 | 135.1 | 0.163 | 0.637 |
| R12-QP 1-shot [12] | 0.503 | 0.701 | 0.721 | 5.0固定 | 0.062 | 0.738 |
| R12-QP 5-shot [12] | 0.706 | 0.812 | 0.885 | 5.0固定 | 0.141 | 0.585 |
| R12-QP 10-shot [12] | 0.761 | 0.842 | 0.921 | 5.0固定 | 0.178 | 0.536 |

**T2a 结论**：Ours 在 Macro-F1（+13.5pt vs 最强 10-shot QP）和 OOS AUROC（+13.3pt vs 0.842）双领先；CICC/MULACONF 为达 coverage 输出 78–135 个意图的巨大集合，Ours 在 coverage=0.911 下集合仅 ~3 个。QP-OneModel set size 固定为 5 不随难度自适应，1-shot/5-shot coverage 不达标。

### T2b MixATIS / MixSNIPS 多意图结果

| 方法 / 数据集 | Set IoU | Hit@1 | Hit@3 | Avg Set Size | Cov. | 误拒率 |
|---|---:|---:|---:|---:|---:|---:|
| R12 MULACONF Powerset / MixATIS | 0.347 | 0.733 | 0.940 | 150.6 | 0.962 | 1.000 |
| R12 MULACONF Tree fixed α / MixATIS | 0.347 | 0.733 | 0.940 | 748.5 | 0.998 | 1.000 |
| R12 MULACONF Powerset / MixSNIPS | 0.794 | 0.970 | 0.997 | 9.9 | 0.936 | 0.819 |
| R12 MULACONF Tree fixed α / MixSNIPS | 0.794 | 0.970 | 0.997 | 25.4 | 0.997 | 1.000 |
| **Ours / MixATIS** | — | — | — | **3.16** | 0.911 | — |

**T2b 结论**：MULACONF Tree fixed α 在 MixATIS 上 avg set size=748（17 个意图几乎全选），误拒率=1.0（全部误拒，不可用）；Powerset 版本 set size=150.6 同样过大。Ours 在 MixATIS 上 avg set size=3.16，三态决策分布为 release/clarify/reject，证明分层共形在多意图场景下仍保持紧致性。MixATIS 主任务 Macro-F1（0.437）与 MULACONF 的差距待 R11/R13 补充后在 T3 消融中分析。