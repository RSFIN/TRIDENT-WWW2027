import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(r"E:\PY-Project\WWW2027\code\intent-conformal")
OUT = ROOT / "docs" / "figs"
data = json.loads((ROOT / "results" / "_f8_roc.json").read_text(encoding="utf-8"))

ret = data["retrieval"]
llm = data["llm_conf"]

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": "#eef0f2", "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(4.6, 3.8))

ax.plot([0, 1], [0, 1], "--", color="#7F8C8D", lw=1.2, label="Random")
ax.plot(ret["fpr"], ret["tpr"], color="#C0392B", lw=2.0,
        label=f"Retrieval-fused score (AUC={ret['auroc']:.3f})")
ax.plot(llm["fpr"], llm["tpr"], color="#2C5F8D", lw=1.6, ls=":",
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
