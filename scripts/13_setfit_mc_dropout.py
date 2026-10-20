"""SetFit + MC Dropout hybrid routing baseline.

流程：
1. 每类 16 例 few-shot 训练 SetFit（BGE-small backbone）
2. MC Dropout 30 次前向算 predictive entropy
3. entropy < tau 直接用 SetFit；否则 fallback 到已有的 LLM predictions

用法：
    python scripts/13_setfit_mc_dropout.py --dataset clinc150 --shots 16 --mc-samples 30
"""
import sys, json, argparse, time
from pathlib import Path
from collections import Counter
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--shots", type=int, default=16)
    ap.add_argument("--mc-samples", type=int, default=30)
    ap.add_argument("--backbone", default="BAAI/bge-small-en-v1.5")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments
    from sklearn.metrics import roc_auc_score

    # ---- 1. 读数据，few-shot 采样 ----
    from src.data.prepare import load_jsonl
    train = load_jsonl(ROOT / "data" / "processed" / args.dataset / "train.jsonl")
    test = load_jsonl(ROOT / "data" / "processed" / args.dataset / "test.jsonl")
    calib = load_jsonl(ROOT / "data" / "processed" / args.dataset / "calib.jsonl")

    by_label = {}
    for s in train:
        if s.oos: continue
        by_label.setdefault(s.main_intent, []).append(s.utterance)
    few_shot = []
    for lab, utts in by_label.items():
        for u in utts[:args.shots]:
            few_shot.append({"text": u, "label": lab})
    print(f"[setfit] few-shot: {len(few_shot)} examples, {len(by_label)} classes")
    ds = Dataset.from_list(few_shot)

    # ---- 2. 训练 SetFit ----
    model = SetFitModel.from_pretrained(args.backbone, labels=list(by_label.keys()))
    args_train = TrainingArguments(
        output_dir="/tmp/setfit",
        num_epochs=1,
        batch_size=16,
        logging_steps=10,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args_train, train_dataset=ds)
    trainer.train()
    model.model_body.eval()
    print("[setfit] training done")

    # ---- 3. MC Dropout 推理 ----
    def mc_predict(texts, n_samples=30):
        """打开 dropout，前向 n_samples 次，返回 (preds, entropies)。"""
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
        all_probs = []
        with torch.no_grad():
            for _ in range(n_samples):
                probs = model.predict_proba(texts)
                if hasattr(probs, "numpy"): probs = probs.numpy()
                all_probs.append(probs)
        P = np.stack(all_probs)  # (n_samples, N, C)
        mean_p = P.mean(axis=0)
        # predictive entropy
        eps = 1e-12
        ent = -(mean_p * np.log(mean_p + eps)).sum(axis=1)
        preds = mean_p.argmax(axis=1)
        return preds, ent, mean_p

    # ---- 4. calib 上定 entropy 阈值 ----
    calib_texts = [s.utterance for s in calib if not s.oos]
    _, calib_ent, _ = mc_predict(calib_texts, args.mc_samples)
    # alpha=0.1: 接受 90% 的 in-distribution
    tau = np.quantile(calib_ent, 0.9)
    print(f"[mc] entropy threshold tau = {tau:.4f}")

    # ---- 5. test 评测 ----
    test_texts = [s.utterance for s in test]
    preds_idx, ents, mean_p = mc_predict(test_texts, args.mc_samples)
    id2label = model.model_head.classes_ if hasattr(model.model_head, "classes_") else list(by_label.keys())
    pred_labels = [id2label[i] for i in preds_idx]

    # fallback 到 LLM predictions
    llm_pred_path = ROOT / "data" / "predictions" / args.dataset / "test.jsonl"
    llm_map = {}
    if llm_pred_path.exists():
        for line in open(llm_pred_path, encoding="utf-8"):
            r = json.loads(line)
            llm_map[r["utterance"]] = r.get("llm_main", "")

    n = len(test); llm_called = 0; correct = 0; oos_scores = []; oos_labels = []
    for i, s in enumerate(test):
        oos_scores.append(float(ents[i])); oos_labels.append(1 if s.oos else 0)
        if ents[i] >= tau or s.oos:
            # fallback LLM
            llm_called += 1
            pred = llm_map.get(s.utterance, "")
        else:
            pred = pred_labels[i]
        if pred == s.main_intent: correct += 1

    auroc = roc_auc_score(oos_labels, oos_scores)
    print(f"\n=== SetFit+MC Dropout Router ({args.dataset}) ===")
    print(f"Acc: {correct/n:.4f}")
    print(f"LLM call rate: {llm_called/n:.1%}")
    print(f"OOS AUROC (entropy): {auroc:.4f}")


if __name__ == "__main__":
    main()
