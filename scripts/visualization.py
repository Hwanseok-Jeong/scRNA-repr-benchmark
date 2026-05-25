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
    global SUFFIX
    stem_with = f"{stem}{SUFFIX}" if SUFFIX else stem
    plt.savefig(os.path.join(outdir, f"{stem_with}.png"), dpi=600)
    plt.savefig(os.path.join(outdir, f"{stem_with}.pdf"), dpi=600)


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
    def fmt(value):
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    return "{KNN}\n{KNC}\n{CPD}".format(
        KNN=fmt(metric_dict.get("KNN", "N/A")),
        KNC=fmt(metric_dict.get("KNC", "N/A")),
        CPD=fmt(metric_dict.get("CPD", "N/A")),
    )


def metric_labels_text():
    return "KNN:\nKNC:\nCPD:"


def embedding_quality(X, Z, classes, knn=10, knn_classes=10, subsetsize=1000):
    nbrs1 = sklearn_neighbors = None
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

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


def place_cluster_center_labels(ax, coords, cell_labels, cluster_names, ordered_colors, labels):
    for class_name, representative_label in labels:
        matches = np.where(cluster_names == representative_label)[0]
        if not matches.size:
            continue
        mask = cell_labels == representative_label
        if not np.any(mask):
            continue
        x_pos, y_pos = np.median(coords[mask], axis=0)
        ax.text(x_pos, y_pos, class_name, fontsize=6, color=ordered_colors[matches[0]])


def plot_tasic_variants(adata, metrics, outdir):
    sns_styleset()

    X = np.asarray(adata.obsm["X_pca"])
    cluster_ids, _, cluster_colors, cluster_names, ordered_colors, cluster_sizes = get_cluster_metadata(adata)
    cell_labels = adata.obs["cluster_label"].astype(str).to_numpy()
    pca_2d = X[:, :2]
    pca_quality = embedding_quality(X, pca_2d, cluster_ids)

    cluster_means = np.zeros((cluster_names.size, X.shape[1]))
    for cluster_index, cluster_name in enumerate(cluster_names):
        mask = adata.obs["cluster_label"].astype(str).to_numpy() == cluster_name
        cluster_means[cluster_index, :] = np.mean(X[mask, :], axis=0)

    z_mds = MDS(n_components=2, max_iter=100, n_init=1000, random_state=42).fit_transform(cluster_means)
    tsne_metrics = metrics.get("tsne", {})
    tsne_order = adata.uns.get("tsne_variant_order", ["default", "n100", "pca_init", "multiscale_pca_high_lr"])
    tsne_variants = [(variant_name, adata.obsm[f"X_tsne_{variant_name}"]) for variant_name in tsne_order if f"X_tsne_{variant_name}" in adata.obsm]
    tsne_variant_map = dict(tsne_variants)

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

    plt.subplot(232)
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.scatter(pca_2d[:, 0], pca_2d[:, 1], s=1, c=cluster_colors, edgecolor="none", rasterized=True)
    plt.title(titles[1], va="center")
    plt.text(0.75, 0.02, metric_labels_text(), transform=plt.gca().transAxes, fontsize=6)
    plt.text(0.87, 0.02, metric_text({"KNN": pca_quality[0], "KNC": pca_quality[1], "CPD": pca_quality[2]}), transform=plt.gca().transAxes, fontsize=6)
    plt.xticks([])
    plt.yticks([])
    plt.text(0, 1.05, letters[1], transform=plt.gca().transAxes, fontsize=8, fontweight="bold")

    for index, (variant_name, variant) in enumerate(tsne_variants):
        plt.subplot(2, 3, 3 + index)
        plt.gca().set_aspect("equal", adjustable="datalim")
        plt.scatter(variant[:, 0], variant[:, 1], s=1, c=cluster_colors, edgecolor="none", rasterized=True)
        plt.title(titles[index + 2], va="center")
        plt.text(0.75, 0.02, metric_labels_text(), transform=plt.gca().transAxes, fontsize=6)
        plt.text(0.87, 0.02, metric_text(tsne_metrics.get(variant_name, {})), transform=plt.gca().transAxes, fontsize=6)
        plt.text(0, 1.05, letters[index + 2], transform=plt.gca().transAxes, fontsize=8, fontweight="bold")
        plt.xticks([])
        plt.yticks([])

    default_tsne = tsne_variant_map.get("default", tsne_variants[0][1] if tsne_variants else None)
    if default_tsne is not None:
        plt.sca(plt.gcf().get_axes()[3])
        cluster_labels = [
            ("Lamp5", "Lamp5 Lsp1"),
            ("Vip", "Vip Rspo4 Rxfp1 Chat"),
            ("Pvalb", "Pvalb Reln Tac1"),
            ("Sst", "Sst Myh8 Fibin"),
            ("L2/3 IT", "L2/3 IT ALM Sla"),
            ("L5 IT", "L5 IT ALM Tnc"),
            ("L6 IT", "L6 IT VISp Penk Col27a1"),
            ("L5 PT", "L5 PT ALM Hpgd"),
            ("L5 NP", "L5 NP VISp Trhr Cpne7"),
            ("L6 CT", "L6 CT VISp Nxph2 Wls"),
            ("L6b", "L6b P2ry12"),
            ("Non-neurons", "Astro Aqp4"),
        ]
        place_cluster_center_labels(plt.gca(), default_tsne, cell_labels, cluster_names, ordered_colors, cluster_labels)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    save_current_figure(outdir, "tasic-variants")
    plt.close()


def plot_tasic_subsets_pca(adata, outdir):
    sns_styleset()
    counts = adata.layers["counts"]
    library_sizes = np.asarray(counts.sum(axis=1)).ravel()
    cluster_colors = adata.obs["cluster_color"].astype(str).to_numpy()
    cluster_ids = adata.obs["cluster_id"].to_numpy() - 1
    _, _, _, ordered_names, _, _ = get_cluster_metadata(adata)

    first_excitatory_cluster = np.where(ordered_names == "L2/3 IT VISp Rrad")[0][0]
    last_excitatory_cluster = np.where(ordered_names == "L6b Hsd17b2")[0][0]
    first_nonneural_cluster = np.where(ordered_names == "Astro Aqp4")[0][0]
    inh_neurons = cluster_ids < first_excitatory_cluster
    exc_neurons = (cluster_ids >= first_excitatory_cluster) & (cluster_ids <= last_excitatory_cluster)
    non_neurons = cluster_ids >= first_nonneural_cluster
    subsets = [inh_neurons, exc_neurons, non_neurons]

    subset_zs = []
    for subset in subsets:
        subset_x = counts[subset, :]
        if hasattr(subset_x, "toarray"):
            subset_x = subset_x.toarray()
        subset_x = np.asarray(subset_x)
        subset_x = np.log2(subset_x / library_sizes[subset][:, None] * 1e6 + 1)
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
    umap_order = list(adata.uns.get("umap_variant_order", ["n15_md01", "n30_md05"]))
    if len(umap_order) < 2:
        raise ValueError("[!] Need at least two UMAP variants to plot in plot_umap_small().")

    missing_keys = [name for name in umap_order[:2] if f"X_umap_{name}" not in adata.obsm]
    if missing_keys:
        raise ValueError(
            "[!] Missing UMAP embeddings for: "
            + ", ".join(missing_keys)
            + ". Check embed_umap variants and rerun embedding/evaluation."
        )

    z_tasic1 = adata.obsm[f"X_umap_{umap_order[0]}"]
    z_tasic2 = adata.obsm[f"X_umap_{umap_order[1]}"]

    plt.figure(figsize=(6, 3))
    plt.subplot(121)
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.scatter(z_tasic1[:, 0], z_tasic1[:, 1], s=1, rasterized=True, c=cluster_colors, edgecolor="none")
    plt.text(0.75, 0.02, metric_labels_text(), transform=plt.gca().transAxes, fontsize=6)
    plt.text(0.87, 0.02, metric_text(umap_metrics.get(umap_order[0], {})), transform=plt.gca().transAxes, fontsize=6)
    plt.xticks([])
    plt.yticks([])
    plt.title("Tasic et al. 2018\nmin_dist=0.1, n_neighbors=15", va="center")
    plt.text(0, 1.05, "c", transform=plt.gca().transAxes, fontsize=8, fontweight="bold")

    plt.subplot(122)
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.scatter(z_tasic2[:, 0], z_tasic2[:, 1], s=1, rasterized=True, c=cluster_colors, edgecolor="none")
    plt.text(0.75, 0.02, metric_labels_text(), transform=plt.gca().transAxes, fontsize=6)
    plt.text(0.87, 0.02, metric_text(umap_metrics.get(umap_order[1], {})), transform=plt.gca().transAxes, fontsize=6)
    plt.xticks([])
    plt.yticks([])
    plt.title("Tasic et al. 2018\nmin_dist=0.5, n_neighbors=30", va="center")
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
    parser.add_argument("--suffix", default="", help="Optional suffix appended to generated figure filenames")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    global SUFFIX
    SUFFIX = args.suffix
    
    adata = sc.read_h5ad(args.input)
    
    with open(args.metrics, 'r') as f:
        metrics = json.load(f)
    adata.uns["variant_metrics"] = metrics

    required_latent_key = f"X_latent_{args.method}"
    if required_latent_key not in adata.obsm:
        raise ValueError(f"[!] {required_latent_key} missing in adata.obsm. Rerun latent_model.py.")
        
    if args.method == "pca":
        for key in ["cluster_id", "cluster_label", "cluster_color"]:
            if key not in adata.obs:
                raise ValueError(f"[!] Required obs column '{key}' not found for PCA plots.")
        # Ensure embeddings + metrics exist for all variants referenced by the plots.
        expected_tsne = list(adata.uns.get("tsne_variant_order", []))
        expected_umap = list(adata.uns.get("umap_variant_order", []))
        missing_tsne_embeddings = [name for name in expected_tsne if f"X_tsne_{name}" not in adata.obsm]
        missing_umap_embeddings = [name for name in expected_umap if f"X_umap_{name}" not in adata.obsm]
        missing_tsne_metrics = [name for name in expected_tsne if name not in metrics.get("tsne", {})]
        missing_umap_metrics = [name for name in expected_umap if name not in metrics.get("umap", {})]
        if missing_tsne_embeddings or missing_umap_embeddings or missing_tsne_metrics or missing_umap_metrics:
            raise ValueError(
                "[!] Missing embeddings/metrics for variants. "
                f"t-SNE embeddings: {missing_tsne_embeddings} | UMAP embeddings: {missing_umap_embeddings} | "
                f"t-SNE metrics: {missing_tsne_metrics} | UMAP metrics: {missing_umap_metrics}. "
                "Re-run evaluation.py and ensure variants match embedding outputs."
            )
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
            plt.text(0.86, 0.05, metric_labels_text(), horizontalalignment="right", verticalalignment="bottom", transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))
            plt.text(0.95, 0.05, metric_text(quality), horizontalalignment="right", verticalalignment="bottom", transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))
            plt.tight_layout()
            stem = os.path.join(args.outdir, f"fig_{args.method}_{emb_name}")
            plt.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
            plt.savefig(f"{stem}.pdf", dpi=600, bbox_inches="tight")
            plt.close()

    print(f"[*] Visualizations saved to {args.outdir}")

if __name__ == "__main__":
    main()
