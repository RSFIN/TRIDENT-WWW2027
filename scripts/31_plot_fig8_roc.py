import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(r"E:\PY-Project\WWW2027\code\intent-conformal")
OUT = ROOT / "docs" / "figs"
data = json.loads((ROOT / "results" / "_f8_roc.json").read_text(encoding="utf-8"))

ret = data["retrieval"]
llm = data["llm_conf"]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(4.2, 3.6))

ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random")
ax.plot(ret["fpr"], ret["tpr"], color="#C0392B", lw=2,
        label=f"Retrieval-fused score (AUC={ret['auroc']:.3f})")
ax.plot(llm["fpr"], llm["tpr"], color="#3B6FB6", lw=1.5, ls=":",
        label=f"LLM self-confidence (AUC={llm['auroc']:.3f})")

ax.set_xlabel("False Positive Rate (OOS accepted)")
ax.set_ylabel("True Positive Rate (in-domain recalled)")
ax.set_title("OOS Detection ROC on CLINC150", fontsize=12)
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)
ax.legend(loc="lower right", fontsize=9, frameon=False)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "fig8_roc.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig8_roc.png", dpi=200, bbox_inches="tight")
print("saved fig8_roc")
