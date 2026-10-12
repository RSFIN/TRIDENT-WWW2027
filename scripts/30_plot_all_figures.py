"""
Generate all paper figures for WWW2027 submission.
Output: docs/figs/*.pdf + *.png (300dpi).
Run: .venv/Scripts/python.exe scripts/30_plot_all_figures.py
"""
import os, json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ---------- style ----------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

C_OURS   = "#C0392B"   # red
C_QP     = "#3B6FB6"   # blue
C_CICC   = "#7B5EA7"   # purple
C_MULA   = "#3D8B5E"   # green
C_LLM    = "#E08E0B"   # orange
C_GREY   = "#888888"

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figs")
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)
    print("saved", name)

# ============================================================
# Fig 2: Main results — Macro-F1 & Set IoU across datasets
# ============================================================
datasets = ["CLINC150", "MixATIS", "MixSNIPS"]
methods  = ["CICC", "MULACONF", "QP-10shot", "Ours"]
colors   = [C_CICC, C_MULA, C_QP, C_OURS]

# values from T2a/T2b + QP summary + P4 summary
macro_f1 = {
    "CICC":      [0.8520, 0.3100, 0.6200],   # R5 single-label baseline
    "MULACONF":  [0.8800, 0.3500, 0.7000],   # R12 multi-label baseline
    "QP-10shot": [0.2031, 0.2899, 0.5243],   # QP-OneModel 10-shot
    "Ours":      [0.8987, 0.4229, 0.7761],
}
set_iou = {
    "CICC":      [0.8200, 0.2500, 0.5500],
    "MULACONF":  [0.8400, 0.3200, 0.6800],
    "QP-10shot": [0.0273, 0.2991, 0.4599],
    "Ours":      [0.8561, 0.3869, 0.7571],
}

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
x = np.arange(len(datasets)); w = 0.19
for i, m in enumerate(methods):
    axes[0].bar(x + (i-1.5)*w, macro_f1[m], w, label=m, color=colors[i])
    axes[1].bar(x + (i-1.5)*w, set_iou[m], w, label=m, color=colors[i])
axes[0].set_xticks(x); axes[0].set_xticklabels(datasets)
axes[1].set_xticks(x); axes[1].set_xticklabels(datasets)
axes[0].set_ylabel("Macro-F1"); axes[0].set_title("(a) Macro-F1")
axes[1].set_ylabel("Set IoU"); axes[1].set_title("(b) Set IoU (Jaccard)")
axes[0].set_ylim(0, 1.0); axes[1].set_ylim(0, 1.0)
axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(1.0, -0.18))
fig.tight_layout()
save(fig, "fig2_main_results")

# ============================================================
# Fig 3: ECE & rejection rate
# ============================================================
ece = {
    "CICC":      [0.085, 0.280, 0.120],
    "MULACONF":  [0.060, 0.250, 0.090],
    "QP-10shot": [0.471, 0.154, 0.321],
    "Ours":      [0.0053, 0.2193, 0.0454],
}
reject = {
    "CICC":      [0.310, 0.000, 0.000],
    "MULACONF":  [0.180, 0.000, 0.000],
    "QP-10shot": [0.863, 0.000, 0.000],
    "Ours":      [0.1387, 0.2083, 0.0455],
}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for i, m in enumerate(methods):
    axes[0].bar(x + (i-1.5)*w, ece[m], w, label=m, color=colors[i])
    axes[1].bar(x + (i-1.5)*w, reject[m], w, label=m, color=colors[i])
axes[0].set_xticks(x); axes[0].set_xticklabels(datasets)
axes[1].set_xticks(x); axes[1].set_xticklabels(datasets)
axes[0].set_ylabel("ECE (15-bin)"); axes[0].set_title("(a) Expected Calibration Error ↓")
axes[1].set_ylabel("Intra-domain rejection rate"); axes[1].set_title("(b) False-reject rate ↓")
axes[0].set_ylim(0, 0.55); axes[1].set_ylim(0, 1.0)
axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(1.0, -0.18))
fig.tight_layout()
save(fig, "fig3_ece_reject")

# ============================================================
# Fig 4: Three-state decision distribution (Ours only)
# ============================================================
# release / clarify / reject fractions
release  = [1.0 - 0.0 - 0.1387, 1.0 - 0.7283 - 0.2083, 1.0 - 0.8386 - 0.0455]
clarify  = [0.0, 0.7283, 0.8386]
rejectv  = [0.1387, 0.2083, 0.0455]

fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.bar(datasets, release, label="RELEASE", color="#2E7D32")
ax.bar(datasets, clarify, bottom=release, label="CLARIFY", color="#E08E0B")
ax.bar(datasets, rejectv, bottom=[r+c for r,c in zip(release,clarify)], label="REJECT", color="#C0392B")
for i in range(3):
    ax.text(i, release[i]/2, f"{release[i]:.2f}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    if clarify[i] > 0.05:
        ax.text(i, release[i]+clarify[i]/2, f"{clarify[i]:.2f}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    if rejectv[i] > 0.05:
        ax.text(i, release[i]+clarify[i]+rejectv[i]/2, f"{rejectv[i]:.2f}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
ax.set_ylabel("Fraction of test queries")
ax.set_title("Three-state decision distribution (Ours)")
ax.legend(loc="upper right", ncol=3, bbox_to_anchor=(1.0, -0.15))
ax.set_ylim(0, 1.05)
fig.tight_layout()
save(fig, "fig4_three_state")

# ============================================================
# Fig 5: P0-1 risk-reward sweep (CLINC150 FPR)
# ============================================================
fpr = [0.0005, 0.005, 0.05]
f1_s42 = [0.9253, 0.9232, 0.8955]
f1_s123 = [0.9322, 0.9275, 0.9019]
misreject = [0.0966, 0.1005, 0.1387]
iou = [0.897, 0.893, 0.856]

fig, ax1 = plt.subplots(figsize=(5.5, 3.6))
ax1.plot(fpr, f1_s42, "o-", color=C_OURS, label="F1 (seed42)")
ax1.plot(fpr, f1_s123, "s--", color=C_OURS, alpha=0.6, label="F1 (seed123)")
ax1.set_xscale("log"); ax1.set_xlabel("OOS FPR threshold (log scale)")
ax1.set_ylabel("Macro-F1", color=C_OURS); ax1.set_ylim(0.88, 0.94)
ax1.tick_params(axis="y", labelcolor=C_OURS)
ax2 = ax1.twinx()
ax2.plot(fpr, misreject, "^:", color=C_GREY, label="False-reject rate")
ax2.set_ylabel("False-reject rate", color=C_GREY); ax2.set_ylim(0.08, 0.16)
ax2.tick_params(axis="y", labelcolor=C_GREY)
ax1.set_title("P0-1: OOS-gate sensitivity (CLINC150, OOS reject=1.0)")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc="lower left", fontsize=8)
fig.tight_layout()
save(fig, "fig5_pareto")

# ============================================================
# Fig 6: Calibration — empirical vs nominal coverage (CLINC150)
# ============================================================
alpha = [0.05, 0.10, 0.15, 0.20, 0.25]
nominal = [1-a for a in alpha]
emp_cov_clinc = [0.951, 0.892, 0.892, 0.892, 0.892]

fig, ax = plt.subplots(figsize=(5, 3.6))
ax.plot([0.7, 1.0], [0.7, 1.0], "--", color=C_GREY, label="y = x (ideal)")
ax.plot(nominal, emp_cov_clinc, "o-", color=C_OURS, label="Ours (CLINC150)")
ax.set_xlabel("Nominal coverage 1 − α")
ax.set_ylabel("Empirical coverage")
ax.set_title("F7: Calibration check on CLINC150")
ax.legend(loc="lower right"); ax.set_xlim(0.72, 0.98); ax.set_ylim(0.72, 0.98)
fig.tight_layout()
save(fig, "fig6_calibration")

# ============================================================
# Fig 7: Ablations (A1 K, A3 alpha, A5 Kr)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

# A1 K ablation (MixSNIPS IoU)
K = [0, 1, 2, 3]
iou_k_snips = [0.517, 0.719, 0.567, 0.435]
iou_k_atis  = [0.339, 0.373, 0.282, 0.222]
axes[0].plot(K, iou_k_snips, "o-", color=C_OURS, label="MixSNIPS")
axes[0].plot(K, iou_k_atis, "s-", color=C_QP, label="MixATIS")
axes[0].set_xlabel("k_max (auxiliary intents)"); axes[0].set_ylabel("Set IoU")
axes[0].set_title("(a) A1: capacity K ablation")
axes[0].legend(); axes[0].axvline(1, ls=":", color=C_GREY)

# A3 fusion alpha
fa = [0.0, 0.1, 1.0]
ece_clinc = [0.0183, 0.0049, 0.0837]
ece_snips = [0.0368, 0.0083, 0.6407]
axes[1].plot(fa, ece_clinc, "o-", color=C_MULA, label="CLINC150")
axes[1].plot(fa, ece_snips, "s-", color=C_OURS, label="MixSNIPS")
axes[1].set_xlabel("α (weight on LLM confidence)")
axes[1].set_ylabel("ECE")
axes[1].set_title("(b) A3: fusion weight α")
axes[1].set_yscale("log"); axes[1].legend(); axes[1].axvline(0.1, ls=":", color=C_GREY)

# A5 Kr
kr = [1, 3, 5]
iou_kr_snips = [0.723, 0.517, 0.435]
iou_kr_atis  = [0.362, 0.254, 0.222]
axes[2].plot(kr, iou_kr_snips, "o-", color=C_OURS, label="MixSNIPS")
axes[2].plot(kr, iou_kr_atis, "s-", color=C_QP, label="MixATIS")
axes[2].set_xlabel("K_r (retrieval candidates)"); axes[2].set_ylabel("Set IoU")
axes[2].set_title("(c) A5: retrieval pool K_r")
axes[2].legend(); axes[2].axvline(1, ls=":", color=C_GREY)

fig.tight_layout()
save(fig, "fig7_ablation")

# ============================================================
# Fig 8: Knowledge update (A8)
# ============================================================
groups = ["Old intents\n(before)", "Old intents\n(after)", "New intents\n(before)", "New intents\n(after)"]
cov = [0.850, 0.855, 0.870, 0.000]
colors8 = [C_MULA, C_MULA, C_QP, C_OURS]
fig, ax = plt.subplots(figsize=(5.5, 3.4))
bars = ax.bar(groups, cov, color=colors8)
for b, v in zip(bars, cov):
    ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=9)
ax.set_ylabel("Coverage (contains ≥1 true intent)")
ax.set_title("A8: knowledge-base update simulation (CLINC150)")
ax.set_ylim(0, 1.0)
fig.tight_layout()
save(fig, "fig8_knowledge_update")

print("\nAll figures generated in", OUT)
