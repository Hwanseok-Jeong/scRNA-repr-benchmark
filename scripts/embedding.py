import argparse
import os
import json
import sys
from pathlib import Path

import scanpy as sc
import numpy as np
import umap

sys.path.append(str(Path(__file__).resolve().parents[1] / "FIt-SNE"))
from fast_tsne import fast_tsne


def resolve_value(value, n_cells):
    if isinstance(value, str):
        normalized = value.strip().upper().replace(" ", "")
        if normalized == "N/100":
            return max(1, int(n_cells / 100))
        if normalized == "N/12":
            return n_cells / 12.0
        if normalized == "PCA":
            return "pca"
        if normalized == "RANDOM":
            return "random"
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    if isinstance(value, list):
        return [resolve_value(item, n_cells) for item in value]
    if isinstance(value, dict):
        return {key: resolve_value(item, n_cells) for key, item in value.items()}
    return value


def main():
    """
    Downstream 2D Embedding script.
    Projects 50D Latent Representations (PCA, AE, VAE, scVI) into 2D using t-SNE and UMAP.
    Specifically applies Kobak & Berens (2019) heuristics:
      - PCA Initialization (preserves global topology)
      - Perplexity = N / 100
      - Learning Rate = N / 12
    """
    parser = argparse.ArgumentParser(description="2D Embedding of Latent Spaces (t-SNE/UMAP)")
    parser.add_argument("--input", required=True, help="Path to .h5ad with latent space")
    parser.add_argument("--output", required=True, help="Path to save 2D embedded .h5ad")
    parser.add_argument("--method", choices=["pca", "scvi"], required=True, help="Which latent space to embed")
    parser.add_argument("--tsne_variants", default="[]", help="JSON list of t-SNE variant configs")
    parser.add_argument("--umap_variants", default="[]", help="JSON list of UMAP variant configs")
    parser.add_argument("--kobak_outdir", default=None, help="Directory to save Kobak-style .npy outputs")
    args = parser.parse_args()
    
    print(f"[*] Loading AnnData from {args.input}...")
    adata = sc.read_h5ad(args.input)
    latent_key = f"X_latent_{args.method}"
    
    if latent_key not in adata.obsm:
        raise ValueError(f"[!] {latent_key} not found in AnnData object. Run latent_model.py first.")
        
    X_latent = np.asarray(adata.obsm[latent_key])
    n_cells = X_latent.shape[0]

    if not np.isfinite(X_latent).all():
        bad_count = np.size(X_latent) - np.isfinite(X_latent).sum()
        raise ValueError(f"[!] {latent_key} has {bad_count} non-finite values (NaN/Inf). Fix latent model output first.")
    
    tsne_variants = json.loads(args.tsne_variants)
    umap_variants = json.loads(args.umap_variants)
    adata.uns["tsne_variant_order"] = [variant["name"] for variant in tsne_variants]
    adata.uns["umap_variant_order"] = [variant["name"] for variant in umap_variants]

    # -------------------------------------------------------------
    # 1. t-SNE variants
    # -------------------------------------------------------------
    for variant in tsne_variants:
        variant = resolve_value(variant, n_cells)
        variant_name = variant["name"]
        init_mode = variant.get("initialization", "random")
        perplexity = variant.get("perplexity")
        perplexity_list = variant.get("perplexity_list")
        learning_rate = variant.get("learning_rate", 200)
        seed = variant.get("seed", 42)

        if init_mode == "pca":
            initialization = "pca"
        elif init_mode == "random":
            # FIt-SNE defaults to PCA init; pass explicit random to avoid fallback.
            initialization = "random"
        else:
            initialization = init_mode

        print(f"\n[*] Running FItSNE variant '{variant_name}' on {latent_key}...")
        if perplexity_list is not None:
            print(f"    -> Perplexity list: {perplexity_list}")
        else:
            print(f"    -> Perplexity: {perplexity}")
        print(f"    -> Learning Rate: {learning_rate}")
        print(f"    -> Initialization: {init_mode}")

        kwargs = {"learning_rate": learning_rate}
        if perplexity_list is not None:
            kwargs["perplexity_list"] = perplexity_list
        elif perplexity is not None:
            kwargs["perplexity"] = perplexity
        if initialization is not None:
            kwargs["initialization"] = initialization
        if seed is not None and init_mode == "random":
            kwargs["seed"] = seed

        X_tsne = fast_tsne(X_latent, **kwargs)
        adata.obsm[f"X_tsne_{variant_name}"] = np.asarray(X_tsne)

    # -------------------------------------------------------------
    # 2. UMAP variants
    # -------------------------------------------------------------
    for variant in umap_variants:
        variant = resolve_value(variant, n_cells)
        variant_name = variant["name"]
        n_neighbors = variant.get("n_neighbors", 30)
        min_dist = variant.get("min_dist", 0.3)
        init_mode = variant.get("init", None)

        print(f"\n[*] Running UMAP variant '{variant_name}' on {latent_key}...")
        print(f"    -> Neighbors: {n_neighbors}, Min Dist: {min_dist}, Init: {init_mode}")

        umap_obj = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=1,
            **({"init": init_mode} if init_mode is not None else {}),
        )
        X_umap = umap_obj.fit_transform(X_latent)
        adata.obsm[f"X_umap_{variant_name}"] = X_umap

    if args.kobak_outdir:
        method_outdir = os.path.join(args.kobak_outdir, args.method)
        os.makedirs(method_outdir, exist_ok=True)
        
        dim = X_latent.shape[1]
        if args.method == "pca":
            np.save(os.path.join(method_outdir, f"tasic-pca{dim}.npy"), X_latent)
        else:
            np.save(os.path.join(method_outdir, f"tasic-latent{dim}-{args.method}.npy"), X_latent)

        if "cluster_color" in adata.obs:
            np.save(os.path.join(method_outdir, "tasic-colors.npy"), adata.obs["cluster_color"].to_numpy())
        else:
            print("[!] cluster_color not found in adata.obs; skipping tasic-colors.npy")

        if "cluster_label" in adata.obs:
            np.save(os.path.join(method_outdir, "tasic-ttypes.npy"), adata.obs["cluster_label"].to_numpy())
        elif "cluster_id" in adata.obs:
            np.save(os.path.join(method_outdir, "tasic-ttypes.npy"), adata.obs["cluster_id"].to_numpy())
        else:
            print("[!] cluster_label/cluster_id not found in adata.obs; skipping tasic-ttypes.npy")
    
    print(f"\n[*] Saving 2D embedded AnnData to {args.output}...")
    adata.write(args.output)
    print("\n[*] 2D Embedding complete!")

if __name__ == "__main__":
    main()
