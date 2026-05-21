import argparse
import scanpy as sc
import numpy as np
from openTSNE import TSNE
import umap

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
    parser.add_argument("--method", choices=["pca", "ae", "vae", "scvi"], required=True, help="Which latent space to embed")
    args = parser.parse_args()
    
    print(f"[*] Loading AnnData from {args.input}...")
    adata = sc.read_h5ad(args.input)
    latent_key = f"X_latent_{args.method}"
    
    if latent_key not in adata.obsm:
        raise ValueError(f"[!] {latent_key} not found in AnnData object. Run latent_model.py first.")
        
    X_latent = adata.obsm[latent_key]
    n_cells = X_latent.shape[0]
    
    # -------------------------------------------------------------
    # 1. t-SNE (Kobak's Rules)
    # -------------------------------------------------------------
    # Rule 1: Perplexities = [30, N/100]
    # Rule 2: Learning Rate = N/12
    # Rule 3: PCA initialization scaled by std(X[:,0]) * 1e-4

    perplexities = [30, int(n_cells / 100)]
    learning_rate = n_cells / 12.0
    pca_init = X_latent[:, :2] / np.std(X_latent[:, 0]) * 0.0001
    
    print(f"\n[*] Running openTSNE on {latent_key}...")
    print(f"    -> Perplexities: {perplexities}")
    print(f"    -> Learning Rate: {learning_rate}")
    print(f"    -> Initialization: PCA (scaled uniformly)")
    
    tsne_obj = TSNE(
        perplexity=perplexities,
        initialization=pca_init,
        metric="euclidean",
        n_jobs=-1,
        random_state=42,
        learning_rate=learning_rate
    )
    
    X_tsne = tsne_obj.fit(X_latent)
    adata.obsm[f"X_tsne_{args.method}"] = np.array(X_tsne)

    # -------------------------------------------------------------
    # 2. UMAP 
    # -------------------------------------------------------------
    # UMAP naturally scales well and preserves global structure reasonably well 
    # out-of-the-box thanks to its spectral initialization (similar to PCA).
    print(f"\n[*] Running UMAP on {latent_key} for comparison...")
    print(f"    -> Neighbors: 30, Min Dist: 0.5, Init: PCA")
    
    umap_obj = umap.UMAP(
        n_neighbors=30,
        min_dist=0.5,
        init='pca',
        random_state=42
    )
    X_umap = umap_obj.fit_transform(X_latent)
    adata.obsm[f"X_umap_{args.method}"] = X_umap
    
    print(f"\n[*] Saving 2D embedded AnnData to {args.output}...")
    adata.write(args.output)
    print("\n[*] 2D Embedding complete!")

if __name__ == "__main__":
    main()
