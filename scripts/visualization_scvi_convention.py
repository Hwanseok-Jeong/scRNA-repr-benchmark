import argparse
import os
import scanpy as sc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

def embedding_quality(X, Z, classes, knn=10, knn_classes=10, subsetsize=1000):
    nbrs1 = NearestNeighbors(n_neighbors=knn).fit(X)
    ind1 = nbrs1.kneighbors(return_distance=False)

    nbrs2 = NearestNeighbors(n_neighbors=knn).fit(Z)
    ind2 = nbrs2.kneighbors(return_distance=False)

    intersections = 0.0
    for i in range(X.shape[0]):
        intersections += len(set(ind1[i]) & set(ind2[i]))
    mnn = intersections / X.shape[0] / knn

    cl, cl_inv = np.unique(classes, return_inverse=True)
    C = cl.size
    mu1 = np.zeros((C, X.shape[1]))
    mu2 = np.zeros((C, Z.shape[1]))
    for c in range(C):
        mu1[c, :] = np.mean(X[cl_inv == c, :], axis=0)
        mu2[c, :] = np.mean(Z[cl_inv == c, :], axis=0)

    nbrs1 = NearestNeighbors(n_neighbors=knn_classes).fit(mu1)
    ind1 = nbrs1.kneighbors(return_distance=False)
    nbrs2 = NearestNeighbors(n_neighbors=knn_classes).fit(mu2)
    ind2 = nbrs2.kneighbors(return_distance=False)

    intersections = 0.0
    for i in range(C):
        intersections += len(set(ind1[i]) & set(ind2[i]))
    mnn_global = intersections / C / knn_classes

    subset = np.random.choice(X.shape[0], size=min(subsetsize, X.shape[0]), replace=False)
    d1 = pdist(X[subset, :])
    d2 = pdist(Z[subset, :])
    rho = spearmanr(d1[:, None], d2[:, None]).correlation
    return (mnn, mnn_global, rho)

def metric_text(quality_tuple):
    # mnn (KNN), mnn_global (KNC), rho (CPD)
    return f"KNN:\nKNC:\nCPD:\n{quality_tuple[0]:.2f}\n{quality_tuple[1]:.2f}\n{quality_tuple[2]:.2f}"

def sns_styleset():
    sns.set_context("paper")
    sns.set_style("ticks")
    matplotlib.rcParams["axes.linewidth"] = 0.75
    matplotlib.rcParams["xtick.major.width"] = 0.75
    matplotlib.rcParams["ytick.major.width"] = 0.75
    matplotlib.rcParams["font.size"] = 7
    matplotlib.rcParams["axes.titlesize"] = 7
    matplotlib.rcParams["axes.labelsize"] = 7
    matplotlib.rcParams["legend.fontsize"] = 7

def main():
    parser = argparse.ArgumentParser(description="Visualize Scanpy convention UMAPs across dimensions.")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of h5ad files to compare")
    parser.add_argument("--dims", nargs="+", type=int, required=True, help="List of dimensions corresponding to inputs")
    parser.add_argument("--outdir", required=True, help="Directory to save figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sns_styleset()

    # Sort files by dimension so that panels go from low to high dim
    dim_files = sorted(zip(args.dims, args.inputs), key=lambda x: x[0])
    
    variants = [
        ("X_umap_n15_md01_scanpy", "UMAP (n_neighbors=15, min_dist=0.1)"),
        ("X_umap_n30_md05_scanpy", "UMAP (n_neighbors=30, min_dist=0.5)")
    ]

    letters = "abcdefghij"

    # Create one figure per UMAP variant
    for variant_key, title_prefix in variants:
        n_panels = len(dim_files)
        plt.figure(figsize=(2.5 * n_panels, 2.5))
        
        for i, (dim, filepath) in enumerate(dim_files):
            adata = sc.read_h5ad(filepath)
            X = adata.obsm[variant_key]
            
            labels = adata.obs['cluster_label'].astype(str).values
            colors_data = adata.obs['cluster_color'].astype(str).values if 'cluster_color' in adata.obs else None

            plt.subplot(1, n_panels, i + 1)
            plt.gca().set_aspect("equal", adjustable="datalim")
            
            if colors_data is not None:
                # Map colors based on unique labels safely
                unique_labels, indices = np.unique(labels, return_index=True)
                color_map = {lbl: colors_data[idx] for lbl, idx in zip(unique_labels, indices)}
                for lbl in unique_labels:
                    idx = labels == lbl
                    plt.scatter(X[idx, 0], X[idx, 1], c=[color_map[lbl]], s=1, alpha=0.8, edgecolors="none")
            else:
                plt.scatter(X[:, 0], X[:, 1], s=1, alpha=0.8, edgecolors="none")

            # Calculate and display metrics
            latent_key = f"X_latent_scvi_n{dim}"
            if latent_key not in adata.obsm:
                latent_key = "X_latent_scvi"
            if latent_key in adata.obsm:
                high_dim_X = adata.obsm[latent_key]
                cluster_ids = adata.obs["cluster_id"].to_numpy() if "cluster_id" in adata.obs else labels
                quality = embedding_quality(high_dim_X, X, cluster_ids)
                
                # Split KNN/KNC/CPD text
                metrics_labels = "KNN:\nKNC:\nCPD:"
                metrics_values = f"{quality[0]:.2f}\n{quality[1]:.2f}\n{quality[2]:.2f}"
                plt.text(0.75, 0.02, metrics_labels, transform=plt.gca().transAxes, fontsize=6)
                plt.text(0.87, 0.02, metrics_values, transform=plt.gca().transAxes, fontsize=6)

            plt.title(f"Method: scVI (Latent Dim: {dim})")
            plt.text(0, 1.05, letters[i], transform=plt.gca().transAxes, fontsize=8, fontweight="bold")
            plt.xticks([])
            plt.yticks([])

        sns.despine(left=True, bottom=True)
        # Add latent information top-left
        plt.figtext(0.01, 1.05, f"Latent: SCVI | {title_prefix}", fontsize=9, fontweight="bold", color="black", ha="left", va="top")
        plt.tight_layout()
        
        clean_name = variant_key.replace("X_umap_", "")
        stem = os.path.join(args.outdir, f"umap_{clean_name}_{n_panels}panels")
        
        # Save as high-res PNG and PDF
        plt.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
        plt.savefig(f"{stem}.pdf", dpi=600, bbox_inches="tight")
        plt.close()

    print(f"[*] Visualizations (Scanpy convention) saved to {args.outdir}")

if __name__ == "__main__":
    main()
