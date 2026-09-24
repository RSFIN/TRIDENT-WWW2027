# 实验任务清单（按优先级）

## 阶段 1：跑通 pipeline（本周）
- [ ] `01_download_data.py`：下载三个数据集，确认 schema 正确
- [ ] `02_build_index.py`：构建 FAISS 索引，CPU 编码 bge-large
- [ ] `03_run_llm.py --dataset clinc150 --split test --limit 50`：先跑 50 条调试
- [ ] `04_calibrate_and_eval.py --dataset clinc150`：端到端指标

## 阶段 2：核心验证（下周）
- [ ] **O9 单调性实验**：f_LLM 分 10 bin × 实际正确率，Spearman 相关
- [ ] **empirical validity**：α∈{0.1,0.05,0.01} 报经验覆盖率，要求 α±1/n
- [ ] 分层 loss vs 单一 Jaccard 0.5 的主意图错误率对比
- [ ] OOS dual threshold：CLINC150 留意图 OOS，报 AUROC/联合错误率

## 阶段 3：消融（第 3-4 周）
- [ ] K_r ∈ {1,3,5,10}
- [ ] α ∈ {0.3,0.5,0.7} 融合权重
- [ ] τ' ∈ {0.5,0.75,1.0}
- [ ] 知识库每意图示例数 {3,5,10}
- [ ] 跨域：CLINC150 校准 → MixSNIPS 测试
- [ ] 知识更新模拟：加新意图条目 vs 重训
- [ ] 轻微调对照：SetFit / LoRA（CPU 可跑 SetFit 小样本）

## 阶段 4：基线对比
- [ ] CICC 复刻（单标签共形）
- [ ] mulaconf 复刻（powerset 多标签共形）
- [ ] 温度缩放
- [ ] LLM zero-shot（无检索）
- [ ] McNemar 显著性检验（3 seeds）
