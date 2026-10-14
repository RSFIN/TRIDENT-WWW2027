# TRIDENT: Training-Free Conformal Prediction for Set-Valued Intent Understanding with Calibrated Refusal

This repository contains the code for the TRIDENT framework, a training-free approach for calibrated multi-intent understanding with out-of-scope (OOS) refusal.

## Key Idea

TRIDENT unifies three capabilities in a single training-free pipeline:
1. **Set-valued intent prediction**: outputs multiple intents per query
2. **Conformal calibration**: provides finite-sample coverage guarantees
3. **Calibrated refusal**: three-state decision (release / clarify / reject) with controlled false positive rate

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
src/retriever/  Embedding model + FAISS intent/example index
src/llm/        DashScope OpenAI-compatible client + prompt templates
src/conformal/  Nonconformity scores, rank normalization, RCPS, OOS gate, three-state decision
src/baselines/  Baseline implementations (SetFit, zero-shot, temperature scaling, CICC, MULACONF)
src/eval/       Evaluation metrics (L1-L5), McNemar's test
scripts/        End-to-end scripts (run in numerical order)
experiments/    Ablation study configurations and results
data/           Downloaded datasets (gitignored)
```

## Core Method

### Risk Measure (Stratified)
```
L = 1[primary intent wrong] + 1[primary intent correct AND secondary intent Jaccard < τ']
```

### Nonconformity Score
```
s(q, i) = α · rank(f_retrieval) + (1-α) · rank(f_LLM)
```

### Risk-Controlling Prediction Sets (RCPS)
Split-conformal calibration under stratified loss yields quantile λ̂:
```
Ĉ = {i : s(q, i) ≤ λ̂}
```

### OOS Dual Threshold
- In-domain: RCPS prediction set
- OOS: retrieval similarity gate calibrated at FPR95

## Datasets

We evaluate on three standard intent recognition benchmarks:
- **CLINC150**: 150 intents with native OOS samples
- **MixATIS**: Multi-label intent dataset from ATIS
- **MixSNIPS**: Multi-label intent dataset from SNIPS

## Main Results

| Dataset | Method | F1 | IoU | ECE | OOS Rejection |
|---------|--------|----|-----|-----|---------------|
| CLINC150 | Ours | 0.899 | 0.856 | 0.005 | 100% |
| CLINC150 | Few-shot LLM | 0.956 | - | - | 83.3% |
| CLINC150 | CICC | 0.852 | 0.820 | 0.085 | - |
| CLINC150 | MULACONF | 0.880 | 0.840 | 0.060 | - |

## Citation

If you find this work useful, please cite our paper:
```
@inproceedings{trident2027,
  title={TRIDENT: Training-Free Conformal Prediction for Set-Valued Intent Understanding with Calibrated Refusal},
  booktitle={Proceedings of the ACM Web Conference 2027 (WWW '27)},
  year={2027}
}
```
