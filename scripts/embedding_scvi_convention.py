import argparse
import scanpy as sc
import os

def main():
    parser = argparse.ArgumentParser(description="Embed scVI latent spaces using standard Scanpy workflow.")
    parser.add_argument("--input", required=True, help="Input h5ad from latent_model.py")
    parser.add_argument("--output", required=True, help="Output h5ad file")
    parser.add_argument("--dim", type=int, required=True, help="Latent dimensionality")
    parser.add_argument("--method", type=str, default="scvi", help="Name of the method (e.g. scvi)")
    args = parser.parse_args()

    print(f"[*] Processing {args.input} for Scanpy workflow (dim={args.dim})")
    adata = sc.read_h5ad(args.input)

    latent_key = f"X_latent_{args.method}_n{args.dim}"
    if latent_key not in adata.obsm:
        latent_key = f"X_latent_{args.method}" # Fallback
        
    if latent_key not in adata.obsm:
        raise ValueError(f"[!] Could not find {latent_key} in adata.obsm.")

    # 1. Variant 1: n15_md01
    print("[*] Computing KNN graph (n_neighbors=15) and UMAP (min_dist=0.1)...")
    sc.pp.neighbors(adata, use_rep=latent_key, n_neighbors=15, key_added='neighbors_n15')
    sc.tl.umap(adata, min_dist=0.1, neighbors_key='neighbors_n15')
    adata.obsm["X_umap_n15_md01_scanpy"] = adata.obsm["X_umap"].copy()

    # 2. Variant 2: n30_md05
    print("[*] Computing KNN graph (n_neighbors=30) and UMAP (min_dist=0.5)...")
    sc.pp.neighbors(adata, use_rep=latent_key, n_neighbors=30, key_added='neighbors_n30')
    sc.tl.umap(adata, min_dist=0.5, neighbors_key='neighbors_n30')
    adata.obsm["X_umap_n30_md05_scanpy"] = adata.obsm["X_umap"].copy()

    # Clean up default generated keys to avoid ambiguity later
    if "X_umap" in adata.obsm:
        del adata.obsm["X_umap"]

    print(f"[*] Saving Scanpy-processed h5ad to {args.output}...")
    adata.write(args.output)
    print("[*] Scanpy embedding complete.")

if __name__ == "__main__":
    main()
