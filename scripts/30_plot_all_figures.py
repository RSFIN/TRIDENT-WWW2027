"""
Generate all paper figures for WWW2027 submission.
Output: docs/figs/*.pdf + *.png (300dpi).
Run: .venv/Scripts/python.exe scripts/30_plot_all_figures.py
"""
import os, json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ---------- academic style (NeurIPS/ICML) ----------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.grid": True,
    "grid.color": "#eef0f2",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.9,
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# low-saturation, print-safe palette
C_OURS   = "#C0392B"   # brick red
C_QP     = "#2C5F8D"   # deep steel blue
C_CICC   = "#6A4C93"   # muted purple
C_MULA   = "#2E7D5B"   # forest green
C_LLM    = "#D9822B"   # amber
C_GREY   = "#7F8C8D"
C_LGREY  = "#BDC3C7"

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figs")
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)
    print("saved", name)

# ============================================================
# Fig 2: Main results — Macro-F1 & Set IoU (rev 198 data, WWW two-col layout)
# ============================================================
datasets = ["CLINC150", "MixATIS", "MixSNIPS"]
methods  = ["TECP-14B", "SetFit+MC", "QP-10shot", "Ours"]
colors   = [C_CICC, C_MULA, C_QP, C_OURS]

macro_f1 = {
    "TECP-14B":  [0.9483, 0.3735, 0.6614],
    "SetFit+MC": [0.9341, 0.4971, 0.5677],
    "QP-10shot": [0.2031, 0.2899, 0.5243],
    "Ours":      [0.8987, 0.4229, 0.7761],
}
set_iou = {
    "TECP-14B":  [0.9541, 0.3843, 0.5286],
    "SetFit+MC": [0.1898, 0.3023, 0.4745],
    "QP-10shot": [0.0273, 0.2991, 0.4599],
    "Ours":      [0.8561, 0.3869, 0.7571],
}

fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7), dpi=300)
fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.22, wspace=0.25)
x = np.arange(len(datasets)); w = 0.18
for i, m in enumerate(methods):
    axes[0].bar(x + (i-1.5)*w, macro_f1[m], w, color=colors[i],
                edgecolor="white", linewidth=0.6)
    axes[1].bar(x + (i-1.5)*w, set_iou[m], w, color=colors[i],
                edgecolor="white", linewidth=0.6)
from matplotlib.patches import Patch
handles = [Patch(color=colors[i], label=methods[i]) for i in range(len(methods))]
fig.legend(handles=handles, ncol=4, loc="lower center",
           bbox_to_anchor=(0.5, 0.005), frameon=False,
           fontsize=7.5, handlelength=1.2, columnspacing=1.2)
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
axes[0].set_ylabel("Macro-F1", fontsize=8.5); axes[0].set_title("(a) Macro-F1", fontsize=9)
axes[1].set_ylabel("Set IoU", fontsize=8.5); axes[1].set_title("(b) Set IoU (Jaccard)", fontsize=9)
axes[0].set_ylim(0, 1.0); axes[1].set_ylim(0, 1.0)
save(fig, "fig2_main_results")

# ============================================================
# Fig 3: ECE & rejection rate (rev 198 data)
# ============================================================
ece = {
    "TECP-14B":  [0.0447, 0.4710, 0.1522],
    "SetFit+MC": [0.6951, 0.3983, 0.3738],
    "QP-10shot": [0.4709, 0.1539, 0.3214],
    "Ours":      [0.0053, 0.2193, 0.0454],
}
reject = {
    "TECP-14B":  [0.0037, 0.0483, 0.0236],
    "SetFit+MC": [0.0482, 0.0471, 0.0227],
    "QP-10shot": [0.8631, 0.0000, 0.0000],
    "Ours":      [0.1387, 0.2083, 0.0455],
}
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7), dpi=300)
fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.22, wspace=0.25)
for i, m in enumerate(methods):
    axes[0].bar(x + (i-1.5)*w, ece[m], w, color=colors[i],
                edgecolor="white", linewidth=0.6)
    axes[1].bar(x + (i-1.5)*w, reject[m], w, color=colors[i],
                edgecolor="white", linewidth=0.6)
fig.legend(handles=handles, ncol=4, loc="lower center",
           bbox_to_anchor=(0.5, 0.005), frameon=False,
           fontsize=7.5, handlelength=1.2, columnspacing=1.2)
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
axes[0].set_ylabel("ECE", fontsize=8.5); axes[0].set_title("(a) Expected Calibration Error", fontsize=9)
axes[1].set_ylabel("False-reject rate", fontsize=8.5); axes[1].set_title("(b) Intra-domain refusal", fontsize=9)
axes[0].set_ylim(0, 0.75); axes[1].set_ylim(0, 1.0)
save(fig, "fig3_ece_reject")

# ============================================================
# Fig 4: Three-state decision distribution (Ours only)
# ============================================================
# release / clarify / reject fractions
release  = [1.0 - 0.0 - 0.1387, 1.0 - 0.7283 - 0.2083, 1.0 - 0.8386 - 0.0455]
clarify  = [0.0, 0.7283, 0.8386]
rejectv  = [0.1387, 0.2083, 0.0455]

fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.bar(datasets, release, label="RELEASE", color="#2E7D32",
       edgecolor="white", linewidth=0.7)
ax.bar(datasets, clarify, bottom=release, label="CLARIFY", color=C_LLM,
       edgecolor="white", linewidth=0.7)
ax.bar(datasets, rejectv, bottom=[r+c for r,c in zip(release,clarify)],
       label="REJECT", color=C_OURS, edgecolor="white", linewidth=0.7)
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
ax1.plot(fpr, f1_s42, "o-", color=C_OURS, lw=1.8, ms=6,
         markeredgecolor="black", markeredgewidth=0.5, label="F1 (seed42)")
ax1.plot(fpr, f1_s123, "s--", color=C_OURS, alpha=0.6, lw=1.8, ms=6,
         markeredgecolor="black", markeredgewidth=0.5, label="F1 (seed123)")
ax1.set_xscale("log"); ax1.set_xlabel("OOS FPR threshold (log scale)")
ax1.set_ylabel("Macro-F1", color=C_OURS); ax1.set_ylim(0.88, 0.94)
ax1.tick_params(axis="y", labelcolor=C_OURS)
ax2 = ax1.twinx()
ax2.plot(fpr, misreject, "^:", color=C_GREY, lw=1.6, ms=6,
         markeredgecolor="black", markeredgewidth=0.5, label="False-reject rate")
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
ax.plot([0.7, 1.0], [0.7, 1.0], "--", color=C_GREY, lw=1.2, label="y = x (ideal)")
ax.plot(nominal, emp_cov_clinc, "o-", color=C_OURS, lw=1.8, ms=7,
        markeredgecolor="black", markeredgewidth=0.5, label="Ours (CLINC150)")
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
axes[0].plot(K, iou_k_snips, "o-", color=C_OURS, lw=1.7, ms=6,
             markeredgecolor="black", markeredgewidth=0.5, label="MixSNIPS")
axes[0].plot(K, iou_k_atis, "s-", color=C_QP, lw=1.7, ms=6,
             markeredgecolor="black", markeredgewidth=0.5, label="MixATIS")
axes[0].set_xlabel("k_max (auxiliary intents)"); axes[0].set_ylabel("Set IoU")
axes[0].set_title("(a) A1: capacity K ablation")
axes[0].legend(); axes[0].axvline(1, ls=":", color=C_GREY)

# A3 fusion alpha
fa = [0.0, 0.1, 1.0]
ece_clinc = [0.0183, 0.0049, 0.0837]
ece_snips = [0.0368, 0.0083, 0.6407]
axes[1].plot(fa, ece_clinc, "o-", color=C_MULA, lw=1.7, ms=6,
             markeredgecolor="black", markeredgewidth=0.5, label="CLINC150")
axes[1].plot(fa, ece_snips, "s-", color=C_OURS, lw=1.7, ms=6,
             markeredgecolor="black", markeredgewidth=0.5, label="MixSNIPS")
axes[1].set_xlabel("α (weight on LLM confidence)")
axes[1].set_ylabel("ECE")
axes[1].set_title("(b) A3: fusion weight α")
axes[1].set_yscale("log"); axes[1].legend(); axes[1].axvline(0.1, ls=":", color=C_GREY)

# A5 Kr
kr = [1, 3, 5]
iou_kr_snips = [0.723, 0.517, 0.435]
iou_kr_atis  = [0.362, 0.254, 0.222]
axes[2].plot(kr, iou_kr_snips, "o-", color=C_OURS, lw=1.7, ms=6,
             markeredgecolor="black", markeredgewidth=0.5, label="MixSNIPS")
axes[2].plot(kr, iou_kr_atis, "s-", color=C_QP, lw=1.7, ms=6,
             markeredgecolor="black", markeredgewidth=0.5, label="MixATIS")
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
bars = ax.bar(groups, cov, color=colors8, edgecolor="white", linewidth=0.8)
for b, v in zip(bars, cov):
    ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=9)
ax.set_ylabel("Coverage (contains ≥1 true intent)")
ax.set_title("A8: knowledge-base update simulation (CLINC150)")
ax.set_ylim(0, 1.0)
fig.tight_layout()
save(fig, "fig8_knowledge_update")

print("\nAll figures generated in", OUT)
