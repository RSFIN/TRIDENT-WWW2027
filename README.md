# Intent Conformal (WWW 2027 选题 A)

免训练知识库索引 + 检索增强多意图推理 + 集合级共形校准拒识。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY

# 3. 下载并预处理数据
python scripts/01_download_data.py

# 4. 构建检索索引
python scripts/02_build_index.py

# 5. 跑 LLM 推理
python scripts/03_run_llm.py

# 6. 共形校准 + 评测
python scripts/04_calibrate_and_eval.py
```

## 项目结构

```
configs/        配置文件（数据、LLM、共形）
src/data/       数据下载、统一 schema、train/calibration/test 划分
src/retriever/  嵌入模型 + FAISS 意图/示例索引
src/llm/        DashScope OpenAI 兼容 client + prompt 模板
src/conformal/  非一致性分数、rank 归一化、RCPS、OOS gate、三态决策
src/baselines/  SetFit、zero-shot、温度缩放、CICC/mulaconf 复刻
src/eval/       L1-L5 指标、McNemar
scripts/        端到端脚本（按编号顺序跑）
experiments/    消融实验配置与结果
data/           下载后的数据（gitignore）
```

## 核心方法

- **风险度量（分层）**：`L = 1[主意图错] + 1[主意图对且次意图 Jaccard < τ']`
- **非一致性分数**：`s(q,i) = α·rank(f_ret) + (1-α)·rank(f_LLM)`
- **RCPS**：split-conformal 在分层 loss 下取分位数 λ̂，`Ĉ = {i: s(q,i) ≤ λ̂}`
- **OOS dual threshold**：in-domain RCPS + 检索相似度 gate（FPR95 标定）

## 子任务（派发给同学）

### R11: TECP-style token entropy 共形
```bash
# 1. 采样 LLM 5 次（先 500 条子样本试跑）
python scripts/11_run_llm_samples.py --dataset clinc150 --split calib --num-samples 5 --limit 500
python scripts/11_run_llm_samples.py --dataset clinc150 --split test  --num-samples 5 --limit 500
# 2. 算 entropy + split CP 评测
python scripts/11_eval_tecp.py --dataset clinc150 --alpha 0.1
```
参考：arXiv 2509.00461。输出 `data/predictions/{ds}/tecp_{split}.jsonl`。

### R13: SetFit + MC Dropout 混合路由
```bash
pip install setfit torch
python scripts/13_setfit_mc_dropout.py --dataset clinc150 --shots 16 --mc-samples 30
```
参考：arXiv 2410.01627。CPU 即可跑，不需要 GPU。
