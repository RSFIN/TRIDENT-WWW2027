# TRIDENT: Training-Free Conformal Prediction for Set-Valued Intent Understanding with Calibrated Refusal

This repository contains the code for **TRIDENT**, a training-free framework for single-turn intent understanding that couples multi-intent set output with calibrated refusal. We name it **TRIDENT** because its three decision states — *release*, *clarify*, and *reject* — mirror the three prongs of a trident.

## Key Idea

TRIDENT unifies three capabilities in a single training-free pipeline:
1. **Set-valued intent prediction**: outputs an ordered set of multiple intents per query
2. **Conformal calibration**: provides finite-sample coverage guarantees on variable-length intent sets
3. **Calibrated refusal**: three-state decision (release / clarify / reject) with controlled false-positive rate (FPR95)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
cp .env.example .env
# Edit .env and add your DASHSCOPE_API_KEY
```

### 3. Data Preparation
```bash
python scripts/01_download_data.py
```

### 4. Build Retrieval Index
```bash
python scripts/02_build_index.py
```

### 5. Run LLM Inference
```bash
python scripts/03_run_llm.py
```

### 6. Conformal Calibration & Evaluation
```bash
python scripts/04_calibrate_and_eval.py
```

## Project Structure

```
configs/        Configuration files (data, LLM, conformal parameters)
src/data/       Data download, unified schema, train/calibration/test splits
src/retriever/  Embedding model (BGE-M3) + FAISS intent/example index
src/llm/        DashScope OpenAI-compatible client + prompt templates
src/conformal/  Nonconformity scores, rank normalization, RCPS, OOS gate, three-state decision
src/baselines/  Baseline implementations (CICC, MULACONF, TECP-style, QP-OneModel, SetFit+MC Dropout)
src/eval/       Evaluation metrics (set IoU, F1, ECE, OOS rejection rate), McNemar's test
scripts/        End-to-end scripts (run in numerical order)
experiments/    Ablation study configurations and results
data/           Downloaded datasets (gitignored)
```

## Core Method

### Pipeline Overview
1. **Retrieval-Augmented Reasoning**: Zero-annotation intent knowledge base (definitions + example utterances) is indexed into intent-level prototypes. Top-$K_r$ candidates are retrieved and fed into an LLM, which outputs a primary intent plus an ordered set of secondary intents.

2. **Fused Nonconformity Score**: Retrieval statistics (top-$K_r$ similarities, margin) are combined with LLM self-assessment confidence, rank-normalized to a common scale:
   ```
   s(q, i) = α · rank(f_retrieval) + (1-α) · rank(f_LLM)
   ```

3. **Set-Valued Conformal Calibration**: Split-conformal calibration is performed on a set-level Jaccard risk, extending standard conformal prediction from point/fixed-label-set coverage to variable-length multi-intent set coverage:
   ```
   Ĉ = {i : s(q, i) ≤ λ̂}
   ```
   where λ̂ is the quantile of the stratified nonconformity scores on the calibration set.

4. **Three-State Decision**:
   - **Release**: Clear query → return single candidate
   - **Clarify**: Ambiguous query → return 2-3 candidates for user confirmation
   - **Reject**: Out-of-scope input → empty set (OOS gate calibrated at FPR95)

## Datasets

We evaluate on three standard intent recognition benchmarks:
- **CLINC150**: 150 intents with native OOS samples
- **MixATIS**: Multi-label intent dataset synthesized from ATIS (1-3 intents per query)
- **MixSNIPS**: Multi-label intent dataset synthesized from SNIPS (1-3 intents per query)

## Main Results (α=0.1)

### CLINC150
| Method | F1 ↑ | IoU ↑ | ECE ↓ | OOS Rejection ↑ | Avg Set Size ↓ |
|--------|------|-------|-------|-----------------|----------------|
| **TRIDENT (Ours)** | **0.899** | **0.856** | **0.005** | **100%** | **0.87** |
| Few-shot LLM (14B) | 0.956 | - | - | 83.3% | - |
| TECP-style (14B) | 0.948 | - | 0.045 | 8.3% | - |
| MULACONF | 0.880 | 0.840 | 0.060 | - | - |
| CICC | 0.852 | 0.820 | 0.085 | - | - |
| QP-OneModel (10-shot) | 0.203 | 0.027 | 0.471 | 13.7% | 5.0 |

### Key Findings
- **100% OOS rejection** with ECE as low as 0.005 (6-8× better than baselines)
- Under equal-risk operating point, F1 gap to few-shot LLM shrinks to **~2.7pt** while preserving finite-sample coverage guarantees
- Average set size: **0.87 / 1.52 / 1.80** on CLINC150 / MixATIS / MixSNIPS
- Training-free: zero gradient updates on our side, only a small calibration set and off-the-shelf LLM API

## Contact

For questions or issues, please open a GitHub issue.
