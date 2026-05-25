import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

"""
python scripts/batch_explore.py \
    --input data/tasic_preprocessed.h5ad \
    --outdir results/batch_explore \
    --batch_keys sample_prefix1 sample_prefix2
"""

def make_palette(categories):
    n = len(categories)
    if n <= 20:
        colors = sns.color_palette("tab20", n)
    elif n <= 40:
        colors = sns.color_palette("tab20b", n)
    else:
        colors = sns.color_palette("husl", n)
    return {cat: colors[idx] for idx, cat in enumerate(categories)}


def plot_scatter(coords, labels, palette, title, outpath):
    plt.figure(figsize=(6, 5))
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=4,
            alpha=0.8,
            c=[palette[label]],
            label=label,
            edgecolors="none",
            rasterized=True,
        )
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    png_path = outpath
    pdf_path = outpath.with_suffix(".pdf")
    plt.savefig(png_path, dpi=600)
    plt.savefig(pdf_path, dpi=600)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Explore batch effects with PCA/UMAP.")
    parser.add_argument("--input", required=True, help="Path to preprocessed .h5ad file")
    parser.add_argument("--outdir", required=True, help="Output directory for plots")
    parser.add_argument("--batch_keys", nargs="+", default=["sample_prefix1", "sample_prefix2"], help="Obs columns to color by and compare")
    parser.add_argument("--n_pcs", type=int, default=50, help="Number of PCs to compute if missing")
    parser.add_argument("--umap_neighbors", type=int, default=30, help="UMAP n_neighbors")
    parser.add_argument("--umap_min_dist", type=float, default=0.3, help="UMAP min_dist")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for UMAP")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading {args.input}...")
    adata = sc.read_h5ad(args.input)

    if "X_pca" not in adata.obsm or adata.obsm["X_pca"].shape[1] < 2:
        print(f"[*] Computing PCA ({args.n_pcs} comps)...")
        sc.pp.pca(adata, n_comps=args.n_pcs, svd_solver="arpack")
    pca_coords = np.asarray(adata.obsm["X_pca"])[:, :2]

    print("[*] Computing UMAP...")
    sc.pp.neighbors(adata, n_neighbors=args.umap_neighbors, use_rep="X_pca")
    sc.tl.umap(adata, min_dist=args.umap_min_dist, random_state=args.seed)
    umap_coords = np.asarray(adata.obsm["X_umap"])

    for bkey in args.batch_keys:
        if bkey not in adata.obs:
            print(f"[!] batch_key '{bkey}' not found in adata.obs, skipping...")
            continue

        print(f"[*] Generating plots for {bkey}...")
        labels = adata.obs[bkey].astype(str).to_numpy()
        categories = np.unique(labels)
        palette = make_palette(categories)

        plot_scatter(
            pca_coords,
            labels,
            palette,
            f"PCA colored by {bkey}",
            outdir / f"batch_pca_{bkey}.png",
        )

        plot_scatter(
            umap_coords,
            labels,
            palette,
            f"UMAP colored by {bkey}",
            outdir / f"batch_umap_{bkey}.png",
        )

    print(f"[*] Saved plots to {outdir}")


if __name__ == "__main__":
    main()
