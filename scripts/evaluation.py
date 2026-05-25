import argparse
import scanpy as sc
import numpy as np
import json
import os
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist

def embedding_quality(X, Z, classes, knn=10, knn_classes=10, subsetsize=1000):
    """
    Kobak/Berens embedding quality metrics.
    Returns (KNN, KNC, CPD).
    """
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate 2D Embeddings using KNN, KNC, CPD")
    parser.add_argument("--input", required=True, help="Path to .h5ad with 2D embeddings")
    parser.add_argument("--method", choices=["pca", "ae", "vae", "scvi"], required=True, help="Latent model evaluated")
    parser.add_argument("--outdir", default="results/metrics/", help="Directory to save metric JSONs")
    parser.add_argument("--suffix", default="", help="Optional suffix to append to output filenames")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[*] Loading AnnData from {args.input}...")
    adata = sc.read_h5ad(args.input)
    
    # Get High-D and Low-D (Using PCA 50 or Latent representation as High-D reference here)
    X_high = adata.obsm[f"X_latent_{args.method}"]
    labels = adata.obs['cluster_id'].values if 'cluster_id' in adata.obs else (
             adata.obs['cluster_label'].values if 'cluster_label' in adata.obs else None)

    if labels is None:
        print("[!] No 'cluster_id' or 'cluster_label' found in adata.obs. KNC will be skipped.")

    results = {
        "__meta__": {
            "method": args.method,
            "variant_orders": {
                "tsne": list(adata.uns.get("tsne_variant_order", [])),
                "umap": list(adata.uns.get("umap_variant_order", [])),
            },
        },
        "tsne": {},
        "umap": {},
    }
    for emb_name in ["tsne", "umap"]:
        variant_order_key = f"{emb_name}_variant_order"
        if variant_order_key in adata.uns:
            emb_keys = [f"X_{emb_name}_{variant_name}" for variant_name in adata.uns[variant_order_key] if f"X_{emb_name}_{variant_name}" in adata.obsm]
        else:
            emb_keys = sorted([key for key in adata.obsm_keys() if key.startswith(f"X_{emb_name}_")])
        for emb_key in emb_keys:
            variant_name = emb_key.replace(f"X_{emb_name}_", "")
            print(f"\n[*] Evaluating {emb_name.upper()} variant '{variant_name}'...")
            X_low = adata.obsm[emb_key]
            knn, knc, cpd = embedding_quality(X_high, X_low, labels, knn=10, knn_classes=10, subsetsize=1000) if labels is not None else ("N/A", "N/A", "N/A")

            if knn != "N/A":
                print(f"    - KNN (k=10): {knn:.3f}")
            if knc != "N/A":
                print(f"    - KNC (k=10): {knc:.3f}")
            if cpd != "N/A":
                print(f"    - CPD       : {cpd:.3f}")

            results[emb_name][variant_name] = {
                "KNN": round(knn, 3) if knn != "N/A" else "N/A",
                "KNC": round(knc, 3) if knc != "N/A" else "N/A",
                "CPD": round(cpd, 3) if cpd != "N/A" else "N/A"
            }

    out_file = os.path.join(args.outdir, f"metrics_{args.method}{args.suffix}.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[*] Evaluation complete. Saved to {out_file}")

if __name__ == "__main__":
    main()
