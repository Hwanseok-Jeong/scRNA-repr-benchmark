import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import MDS


def sns_styleset():
    sns.set_context("paper")
    sns.set_style("ticks")
    matplotlib.rcParams["axes.linewidth"] = 0.75
    matplotlib.rcParams["xtick.major.width"] = 0.75
    matplotlib.rcParams["ytick.major.width"] = 0.75
    matplotlib.rcParams["xtick.major.size"] = 3
    matplotlib.rcParams["ytick.major.size"] = 3
    matplotlib.rcParams["xtick.minor.size"] = 2
    matplotlib.rcParams["ytick.minor.size"] = 2
    matplotlib.rcParams["font.size"] = 7
    matplotlib.rcParams["axes.titlesize"] = 7
    matplotlib.rcParams["axes.labelsize"] = 7
    matplotlib.rcParams["legend.fontsize"] = 7
    matplotlib.rcParams["xtick.labelsize"] = 7
    matplotlib.rcParams["ytick.labelsize"] = 7


def save_current_figure(outdir, stem):
    plt.savefig(os.path.join(outdir, f"{stem}.png"), dpi=150)
    plt.savefig(os.path.join(outdir, f"{stem}.pdf"), dpi=300)
    plt.savefig(os.path.join(outdir, f"{stem}-600.pdf"), dpi=600)


def get_cluster_metadata(adata):
    cluster_ids = adata.obs["cluster_id"].to_numpy()
    cluster_labels = adata.obs["cluster_label"].astype(str).to_numpy()
    cluster_colors = adata.obs["cluster_color"].astype(str).to_numpy()

    unique_ids = np.unique(cluster_ids)
    ordered_names = []
    ordered_colors = []
    ordered_sizes = []
    for cluster_id in unique_ids:
        mask = cluster_ids == cluster_id
        ordered_names.append(str(adata.obs.loc[mask, "cluster_label"].iloc[0]))
        ordered_colors.append(str(adata.obs.loc[mask, "cluster_color"].iloc[0]))
        ordered_sizes.append(int(mask.sum()))

    return cluster_ids, cluster_labels, cluster_colors, np.asarray(ordered_names), np.asarray(ordered_colors), np.asarray(ordered_sizes)

def metric_text(metric_dict):
    return "KNN:\nKNC:\nCPD:\n{KNN}\n{KNC}\n{CPD}".format(
        KNN=metric_dict.get("KNN", "N/A"),
        KNC=metric_dict.get("KNC", "N/A"),
        CPD=metric_dict.get("CPD", "N/A"),
    )


def plot_tasic_variants(adata, metrics, outdir):
    sns_styleset()

    X = np.asarray(adata.obsm["X_pca"])
    cluster_ids, _, cluster_colors, cluster_names, ordered_colors, cluster_sizes = get_cluster_metadata(adata)

    cluster_means = np.zeros((cluster_names.size, X.shape[1]))
    for cluster_index, cluster_name in enumerate(cluster_names):
        mask = adata.obs["cluster_label"].astype(str).to_numpy() == cluster_name
        cluster_means[cluster_index, :] = np.mean(X[mask, :], axis=0)

    z_mds = MDS(n_components=2, max_iter=100, n_init=1000, random_state=42).fit_transform(cluster_means)
    tsne_metrics = metrics.get("tsne", {})
    tsne_order = adata.uns.get("tsne_variant_order", ["default", "n100", "pca_init", "multiscale_pca_high_lr"])
    tsne_variants = [(variant_name, adata.obsm[f"X_tsne_{variant_name}"]) for variant_name in tsne_order if f"X_tsne_{variant_name}" in adata.obsm]

    titles = [
        "MDS on class means",
        "PCA",
        "Default t-SNE\n(perpexity 30, random init., $\\eta=200$)",
        "Perplexity $n/100$",
        "PCA initialisation",
        "Multi-scale, PCA initialisation,\nhigh learning rate ($\\eta=n/12$)",
    ]
    letters = "abcdef"

    plt.figure(figsize=(7.2, 5))
    plt.subplot(231)
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.scatter(z_mds[:, 0], z_mds[:, 1], c=ordered_colors, edgecolor="none", s=cluster_sizes / 10)
    plt.title(titles[0], va="center")
    plt.xticks([])
    plt.yticks([])
    plt.text(0, 1.05, letters[0], transform=plt.gca().transAxes, fontsize=8, fontweight="bold")

    for index, (variant_name, variant) in enumerate(tsne_variants):
        plt.subplot(2, 3, 2 + index)
        plt.gca().set_aspect("equal", adjustable="datalim")
        plt.scatter(variant[:, 0], variant[:, 1], s=1, c=cluster_colors, edgecolor="none", rasterized=True)
        plt.title(titles[index + 1], va="center")
        plt.text(0.75, 0.02, "KNN:\nKNC:\nCPD:", transform=plt.gca().transAxes, fontsize=6)
        plt.text(0.87, 0.02, metric_text(tsne_metrics.get(variant_name, {})), transform=plt.gca().transAxes, fontsize=6)
        plt.text(0, 1.05, letters[index + 1], transform=plt.gca().transAxes, fontsize=8, fontweight="bold")
        plt.xticks([])
        plt.yticks([])

    plt.sca(plt.gcf().get_axes()[3])
    classes = {
        "Lamp5": [-35, -12, "Lamp5 Lsp1"],
        "Vip": [15, 30, "Vip Rspo4 Rxfp1 Chat"],
        "Pvalb": [-4, 32, "Pvalb Reln Tac1"],
        "Sst": [-32, 18, "Sst Myh8 Fibin"],
        "L2/3 IT": [-30, -30, "L2/3 IT ALM Sla"],
        "L5 IT": [-10, -25, "L5 IT ALM Tnc"],
        "L6 IT": [6, -4, "L6 IT VISp Penk Col27a1"],
        "L5 PT": [23, -25, "L5 PT ALM Hpgd"],
        "L5 NP": [-23, -40, "L5 NP VISp Trhr Cpne7"],
        "L6 CT": [30, 12, "L6 CT VISp Nxph2 Wls"],
        "L6b": [35, -11, "L6b P2ry12"],
        "Non-neurons": [20, -20, "Astro Aqp4"],
    }
    for class_name, (x_pos, y_pos, representative_label) in classes.items():
        matches = np.where(cluster_names == representative_label)[0]
        if matches.size:
            plt.text(x_pos, y_pos, class_name, fontsize=6, color=ordered_colors[matches[0]])

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    save_current_figure(outdir, "tasic-variants")
    plt.close()


def plot_tasic_subsets_pca(adata, outdir):
    sns_styleset()
    X = np.asarray(adata.obsm["X_pca"])
    cluster_ids = adata.obs["cluster_id"].to_numpy()
    cluster_colors = adata.obs["cluster_color"].astype(str).to_numpy()
    _, _, _, ordered_names, _, _ = get_cluster_metadata(adata)
    cluster_index = cluster_ids - cluster_ids.min()

    first_excitatory_cluster = np.where(ordered_names == "L2/3 IT VISp Rrad")[0][0]
    last_excitatory_cluster = np.where(ordered_names == "L6b Hsd17b2")[0][0]
    first_nonneural_cluster = np.where(ordered_names == "Astro Aqp4")[0][0]
    inh_neurons = cluster_index < first_excitatory_cluster
    exc_neurons = (cluster_index >= first_excitatory_cluster) & (cluster_index <= last_excitatory_cluster)
    non_neurons = cluster_index >= first_nonneural_cluster
    subsets = [inh_neurons, exc_neurons, non_neurons]

    subset_zs = []
    for subset in subsets:
        subset_x = X[subset, :]
        subset_x = subset_x - subset_x.mean(axis=0)
        u, s, v = np.linalg.svd(subset_x, full_matrices=False)
        u[:, np.sum(v, axis=1) < 0] *= -1
        subset_x = np.dot(u, np.diag(s))
        subset_x = subset_x[:, np.argsort(s)[::-1]][:, :50]
        subset_zs.append(subset_x[:, :2])

    titles = ["Inhibitory neurons", "Excitatory neurons", "Non-neurons"]
    letters = "abc"
    plt.figure(figsize=(7.2, 2.5))
    for subset_index, z in enumerate(subset_zs):
        plt.subplot(1, 3, subset_index + 1)
        plt.gca().set_aspect("equal", adjustable="datalim")
        plt.scatter(z[:, 0], z[:, 1], s=1, edgecolor="none", rasterized=True, c=cluster_colors[subsets[subset_index]])
        plt.title(titles[subset_index])
        plt.text(0, 1.05, letters[subset_index], transform=plt.gca().transAxes, fontsize=8, fontweight="bold")
        plt.xticks([])
        plt.yticks([])

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    save_current_figure(outdir, "tasic-subsets-pca")
    plt.close()


def plot_umap_small(adata, metrics, outdir):
    sns_styleset()
    cluster_colors = adata.obs["cluster_color"].astype(str).to_numpy()
    umap_metrics = metrics.get("umap", {})
    umap_order = adata.uns.get("umap_variant_order", ["n15_md01", "n30_md03"])
    z_tasic1 = adata.obsm[f"X_umap_{umap_order[0]}"]
    z_tasic2 = adata.obsm[f"X_umap_{umap_order[1]}"]

    plt.figure(figsize=(6, 3))
    plt.subplot(121)
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.scatter(z_tasic1[:, 0], z_tasic1[:, 1], s=1, rasterized=True, c=cluster_colors, edgecolor="none")
    plt.text(0.75, 0.02, "KNN:\nKNC:\nCPD:", transform=plt.gca().transAxes, fontsize=6)
    plt.text(0.87, 0.02, metric_text(umap_metrics.get(umap_order[0], {})), transform=plt.gca().transAxes, fontsize=6)
    plt.xticks([])
    plt.yticks([])
    plt.title("Tasic et al. 2018\nmin_dist=0.1, n_neighbors=15", va="center")
    plt.text(0, 1.05, "c", transform=plt.gca().transAxes, fontsize=8, fontweight="bold")

    plt.subplot(122)
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.scatter(z_tasic2[:, 0], z_tasic2[:, 1], s=1, rasterized=True, c=cluster_colors, edgecolor="none")
    plt.text(0.75, 0.02, "KNN:\nKNC:\nCPD:", transform=plt.gca().transAxes, fontsize=6)
    plt.text(0.87, 0.02, metric_text(umap_metrics.get(umap_order[1], {})), transform=plt.gca().transAxes, fontsize=6)
    plt.xticks([])
    plt.yticks([])
    plt.title("Tasic et al. 2018\nmin_dist=0.3, n_neighbors=30", va="center")
    plt.text(0, 1.05, "d", transform=plt.gca().transAxes, fontsize=8, fontweight="bold")

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    save_current_figure(outdir, "umap-small")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize Embeddings with Metrics")
    parser.add_argument("--input", required=True, help="Path to .h5ad with 2D embeddings")
    parser.add_argument("--method", choices=["pca", "ae", "vae", "scvi"], required=True, help="Latent model to visualize")
    parser.add_argument("--metrics", required=True, help="Path to metrics JSON file")
    parser.add_argument("--outdir", default="results/figures/", help="Directory to save figures")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    adata = sc.read_h5ad(args.input)
    
    with open(args.metrics, 'r') as f:
        metrics = json.load(f)
    adata.uns["variant_metrics"] = metrics
        
    if args.method == "pca":
        plot_tasic_variants(adata, metrics, args.outdir)
        plot_tasic_subsets_pca(adata, args.outdir)
        plot_umap_small(adata, metrics, args.outdir)
    else:
        labels = adata.obs['cluster_label'].astype(str).values
        colors = adata.obs['cluster_color'].astype(str).values if 'cluster_color' in adata.obs else [None] * len(labels)

        for emb_name, title in [("tsne", f"t-SNE (Latent: {args.method.upper()}; Kobak variants)"), ("umap", f"UMAP (Latent: {args.method.upper()})")]:
            variant_keys = sorted([key for key in adata.obsm_keys() if key.startswith(f"X_{emb_name}_")])
            if not variant_keys:
                continue
            X = adata.obsm[variant_keys[0]]
            variant_name = variant_keys[0].replace(f"X_{emb_name}_", "")
            quality = metrics.get(emb_name, {}).get(variant_name, {})
            plt.figure(figsize=(6, 6))
            unique_labels = np.unique(labels)
            color_map = {lbl: col for lbl, col in zip(labels, colors)}
            for lbl in unique_labels:
                idx = labels == lbl
                plt.scatter(X[idx, 0], X[idx, 1], c=[color_map[lbl]], s=1, alpha=0.8, edgecolors="none")
            plt.title(title, fontsize=14)
            plt.xticks([])
            plt.yticks([])
            plt.gca().set_aspect("equal", adjustable="datalim")
            plt.text(0.95, 0.05, metric_text(quality), horizontalalignment="right", verticalalignment="bottom", transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"fig_{args.method}_{emb_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()

    print(f"[*] Visualizations saved to {args.outdir}")

if __name__ == "__main__":
    main()
