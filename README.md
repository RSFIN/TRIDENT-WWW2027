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
