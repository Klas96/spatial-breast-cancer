"""
Generate graphical abstract for STAR Protocols submission.
Self-contained — creates schematic visualisations, no data files needed.
Output: report/graphical_abstract.png  (300 dpi, 183 x 247 mm)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from pathlib import Path

RNG = np.random.default_rng(42)
OUT = Path("report/graphical_abstract.png")

# ── Palette ──────────────────────────────────────────────────────────────────
BG      = "#F7F9FC"
STEP_BG = "#FFFFFF"
BORDER  = "#CBD5E1"
ARROW   = "#64748B"
TITLE   = "#0F172A"
CAPTION = "#475569"

STEP_COLORS = ["#1D4ED8", "#0891B2", "#059669", "#D97706", "#DC2626", "#7C3AED"]
CELL_COLORS = {
    "Cancer\nEpithelial": "#E11D48",
    "CAFs":               "#F59E0B",
    "T-cells":            "#10B981",
    "Myeloid":            "#3B82F6",
    "B-cells":            "#8B5CF6",
    "Other":              "#94A3B8",
}

# ── Canvas ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10), facecolor=BG)

# Three rows: title | pipeline boxes | bottom caption
gs_outer = GridSpec(3, 1, figure=fig,
                    height_ratios=[0.07, 0.82, 0.11],
                    hspace=0.04)

# ── Title ─────────────────────────────────────────────────────────────────────
ax_title = fig.add_subplot(gs_outer[0])
ax_title.set_axis_off()
ax_title.text(0.5, 0.55,
              "Spatial Transcriptomics Pipeline for Breast Cancer Tumour Microenvironment Analysis",
              ha="center", va="center", fontsize=14, fontweight="bold", color=TITLE,
              transform=ax_title.transAxes)
ax_title.text(0.5, 0.05,
              "10x Genomics Visium  ·  Cell2location  ·  Squidpy  ·  PyTorch Geometric",
              ha="center", va="center", fontsize=9, color=CAPTION,
              transform=ax_title.transAxes)

# ── Pipeline boxes ─────────────────────────────────────────────────────────────
n_steps = 6
gs_pipe = GridSpec(1, n_steps, figure=fig,
                   left=0.03, right=0.97,
                   bottom=0.12, top=0.90,
                   wspace=0.06)

step_labels = [
    "Stage 1\nQC & Preprocessing",
    "Stage 2\nClustering",
    "Stage 3\nDeconvolution",
    "Stage 4\nSpatially Variable\nGenes",
    "Stage 5\nNeighbourhood\nEnrichment",
    "Stage 6\nGNN Domain\nPrediction",
]

axes = [fig.add_subplot(gs_pipe[0, i]) for i in range(n_steps)]


def box_border(ax, color):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.5)
    ax.set_facecolor(STEP_BG)


# ── Helper to draw a rounded header strip ────────────────────────────────────
def header(ax, label, color, lines=2):
    lh = 0.14 * lines
    rect = FancyBboxPatch((0, 1 - lh), 1, lh,
                          boxstyle="round,pad=0.01",
                          fc=color, ec="none",
                          transform=ax.transAxes, clip_on=False, zorder=3)
    ax.add_patch(rect)
    ax.text(0.5, 1 - lh / 2, label,
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="white", transform=ax.transAxes, zorder=4,
            multialignment="center")


# ─── Stage 1 : QC histograms ──────────────────────────────────────────────────
ax = axes[0]
box_border(ax, STEP_COLORS[0])
header(ax, step_labels[0], STEP_COLORS[0])

data = [RNG.exponential(4000, 200), RNG.exponential(2500, 200),
        RNG.exponential(5, 200)]
labels_qc = ["Total counts", "Genes detected", "% MT"]
colors_qc = ["#BFDBFE", "#93C5FD", "#60A5FA"]
y_positions = [0.74, 0.46, 0.18]
for i, (d, lab, col, yp) in enumerate(zip(data, labels_qc, colors_qc, y_positions)):
    inner = ax.inset_axes([0.08, yp, 0.84, 0.2])
    inner.hist(d, bins=20, color=col, edgecolor="white", linewidth=0.3)
    inner.axvline(np.percentile(d, 10), color=STEP_COLORS[0],
                  lw=1.4, ls="--")
    inner.set_axis_off()
    inner.text(-0.02, 0.5, lab, transform=inner.transAxes,
               fontsize=6, ha="right", va="center", color=CAPTION)
ax.set_axis_off()
ax.text(0.5, 0.01, "Filter low-quality spots\nNormalise · Log-transform · PCA",
        ha="center", va="bottom", fontsize=6.5, color=CAPTION,
        transform=ax.transAxes, multialignment="center")

# ─── Stage 2 : UMAP clusters ─────────────────────────────────────────────────
ax = axes[1]
box_border(ax, STEP_COLORS[1])
header(ax, step_labels[1], STEP_COLORS[1])

inner = ax.inset_axes([0.05, 0.18, 0.9, 0.72])
cluster_cols = ["#1D4ED8", "#0891B2", "#059669", "#D97706", "#9333EA",
                "#DC2626", "#F472B6", "#6B7280"]
n_clusters = len(cluster_cols)
centers = RNG.uniform(0.1, 0.9, (n_clusters, 2))
for ci, (cx, cy) in enumerate(centers):
    pts = RNG.normal([cx, cy], 0.05, (60, 2)).clip(0, 1)
    inner.scatter(pts[:, 0], pts[:, 1], s=6, c=cluster_cols[ci],
                  alpha=0.8, linewidths=0)
inner.set_xlim(0, 1)
inner.set_ylim(0, 1)
inner.set_axis_off()
inner.set_title("UMAP", fontsize=7, color=CAPTION, pad=2)
ax.set_axis_off()
ax.text(0.5, 0.01, "k-NN graph · UMAP · Leiden clustering",
        ha="center", va="bottom", fontsize=6.5, color=CAPTION,
        transform=ax.transAxes, multialignment="center")

# ─── Stage 3 : Deconvolution spatial map ─────────────────────────────────────
ax = axes[2]
box_border(ax, STEP_COLORS[2])
header(ax, step_labels[2], STEP_COLORS[2])

inner = ax.inset_axes([0.05, 0.18, 0.9, 0.72])
grid_n = 18
xx, yy = np.meshgrid(np.linspace(0, 1, grid_n), np.linspace(0, 1, grid_n))
spots = np.column_stack([xx.ravel(), yy.ravel()])
# simulate dominant cell type per spot
cx = [0.25, 0.75, 0.5, 0.2, 0.8, 0.5]
cy = [0.5, 0.5, 0.75, 0.2, 0.2, 0.3]
ct_cols = list(CELL_COLORS.values())
labels_arr = np.argmin(
    np.array([np.hypot(spots[:, 0] - x, spots[:, 1] - y)
              for x, y in zip(cx, cy)]), axis=0)
for ci, col in enumerate(ct_cols):
    mask = labels_arr == ci
    inner.scatter(spots[mask, 0], spots[mask, 1], s=22, c=col,
                  alpha=0.9, linewidths=0, marker="s")
inner.set_xlim(-0.04, 1.04)
inner.set_ylim(-0.04, 1.04)
inner.set_axis_off()
inner.set_title("Spatial cell-type map", fontsize=7, color=CAPTION, pad=2)

legend_handles = [mpatches.Patch(color=c, label=l)
                  for l, c in CELL_COLORS.items()]
inner.legend(handles=legend_handles, fontsize=4.5, loc="lower right",
             framealpha=0.8, ncol=2, handlelength=1)
ax.set_axis_off()
ax.text(0.5, 0.01, "Cell2location · Wu et al. 2021 reference",
        ha="center", va="bottom", fontsize=6.5, color=CAPTION,
        transform=ax.transAxes, multialignment="center")

# ─── Stage 4 : Spatially variable genes ──────────────────────────────────────
ax = axes[3]
box_border(ax, STEP_COLORS[3])
header(ax, step_labels[3], STEP_COLORS[3], lines=3)

inner = ax.inset_axes([0.05, 0.18, 0.9, 0.68])
gene_vals = np.zeros((grid_n, grid_n))
for gx, gy, sig in [(0.3, 0.5, 0.15), (0.7, 0.5, 0.12), (0.5, 0.8, 0.1)]:
    gene_vals += np.exp(
        -((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sig ** 2))
gene_vals += RNG.normal(0, 0.04, gene_vals.shape)
inner.imshow(gene_vals, cmap="viridis", origin="lower", aspect="auto")
inner.set_axis_off()
inner.set_title("IFI6  (Moran's I = 0.769)", fontsize=6.5, color=CAPTION, pad=2)
ax.set_axis_off()
ax.text(0.5, 0.01, "Moran's I spatial autocorrelation\nTop 12 SVGs visualised",
        ha="center", va="bottom", fontsize=6.5, color=CAPTION,
        transform=ax.transAxes, multialignment="center")

# ─── Stage 5 : Neighbourhood enrichment ──────────────────────────────────────
ax = axes[4]
box_border(ax, STEP_COLORS[4])
header(ax, step_labels[4], STEP_COLORS[4], lines=3)

inner = ax.inset_axes([0.08, 0.18, 0.84, 0.68])
n_ct = len(CELL_COLORS)
zmat = RNG.normal(0, 1.5, (n_ct, n_ct))
np.fill_diagonal(zmat, RNG.uniform(2, 4, n_ct))
zmat = (zmat + zmat.T) / 2
im = inner.imshow(zmat, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
ct_short = ["CanEpi", "CAFs", "T", "Mye", "B", "Other"]
inner.set_xticks(range(n_ct))
inner.set_xticklabels(ct_short, fontsize=4.5, rotation=45, ha="right")
inner.set_yticks(range(n_ct))
inner.set_yticklabels(ct_short, fontsize=4.5)
inner.set_title("Co-localisation z-score", fontsize=6.5, color=CAPTION, pad=2)
plt.colorbar(im, ax=inner, fraction=0.046, pad=0.04).ax.tick_params(labelsize=4)
ax.set_axis_off()
ax.text(0.5, 0.01, "Permutation-based enrichment\nPairs above/below chance",
        ha="center", va="bottom", fontsize=6.5, color=CAPTION,
        transform=ax.transAxes, multialignment="center")

# ─── Stage 6 : GNN ───────────────────────────────────────────────────────────
ax = axes[5]
box_border(ax, STEP_COLORS[5])
header(ax, step_labels[5], STEP_COLORS[5], lines=3)

inner = ax.inset_axes([0.05, 0.18, 0.9, 0.68])
# Draw a small spatial graph
n_nodes = 30
node_xy = RNG.uniform(0.08, 0.92, (n_nodes, 2))
node_ct = RNG.choice(len(ct_cols), n_nodes,
                     p=[0.45, 0.15, 0.15, 0.1, 0.08, 0.07])
# Edges: connect nearby nodes
for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        d = np.hypot(*(node_xy[i] - node_xy[j]))
        if d < 0.22:
            inner.plot([node_xy[i, 0], node_xy[j, 0]],
                       [node_xy[i, 1], node_xy[j, 1]],
                       color="#CBD5E1", lw=0.7, zorder=1)
inner.scatter(node_xy[:, 0], node_xy[:, 1],
              c=[ct_cols[c] for c in node_ct],
              s=60, zorder=2, edgecolors="white", linewidths=0.6)
inner.set_xlim(0, 1)
inner.set_ylim(0, 1)
inner.set_axis_off()
inner.set_title("GCN  ·  75 – 90 % accuracy", fontsize=6.5, color=CAPTION, pad=2)
ax.set_axis_off()
ax.text(0.5, 0.01, "2-layer GCN · PCA node features\nValidated on 2 datasets",
        ha="center", va="bottom", fontsize=6.5, color=CAPTION,
        transform=ax.transAxes, multialignment="center")

# ── Arrows between boxes ──────────────────────────────────────────────────────
for i in range(n_steps - 1):
    x_left  = axes[i].get_position().x1
    x_right = axes[i + 1].get_position().x0
    x_mid   = (x_left + x_right) / 2
    y_mid   = (axes[i].get_position().y0 + axes[i].get_position().y1) / 2
    fig.add_artist(FancyArrowPatch(
        posA=(x_left + 0.003, y_mid),
        posB=(x_right - 0.003, y_mid),
        arrowstyle="-|>",
        color=ARROW,
        lw=1.5,
        mutation_scale=12,
        transform=fig.transFigure,
        clip_on=False,
    ))

# ── Input label (left of Stage 1) ────────────────────────────────────────────
x0 = axes[0].get_position().x0
y_mid = (axes[0].get_position().y0 + axes[0].get_position().y1) / 2
fig.text(x0 - 0.01, y_mid + 0.06,
         "10x Visium\nInput", ha="right", va="center",
         fontsize=8, fontweight="bold", color=TITLE,
         multialignment="center")
fig.text(x0 - 0.01, y_mid - 0.06,
         "FFPE &\nFresh-Frozen", ha="right", va="center",
         fontsize=7, color=CAPTION, multialignment="center")
fig.add_artist(FancyArrowPatch(
    posA=(x0 - 0.005, y_mid),
    posB=(x0 + 0.003, y_mid),
    arrowstyle="-|>", color=ARROW, lw=1.5, mutation_scale=12,
    transform=fig.transFigure, clip_on=False,
))

# ── Bottom caption ─────────────────────────────────────────────────────────────
ax_cap = fig.add_subplot(gs_outer[2])
ax_cap.set_axis_off()
ax_cap.text(0.5, 0.7,
            "Fully reproducible · uv-managed Python environment · "
            "GPU-accelerated (NVIDIA RTX 4070) · "
            "github.com/Klas96/spatial-breast-cancer",
            ha="center", va="center", fontsize=8, color=CAPTION,
            transform=ax_cap.transAxes)

OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {OUT}  ({OUT.stat().st_size // 1024} KB)")
