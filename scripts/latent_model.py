import argparse
import scanpy as sc
import numpy as np
from scipy import sparse
import torch
import scvi
from sklearn.decomposition import PCA

def main():
    parser = argparse.ArgumentParser(description="Latent Representation Learning for scRNA-seq")
    parser.add_argument("--input", required=True, help="Path to preprocessed .h5ad file")
    parser.add_argument("--output", required=True, help="Path to save .h5ad with latent representations")
    parser.add_argument("--method", choices=["pca", "scvi"], required=True, help="Latent model to use")
    parser.add_argument("--dim", type=int, default=50, help="Latent dimensionality")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for DL models")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    args = parser.parse_args()

    print(f"[*] Loading preprocessed data from {args.input}...")
    adata = sc.read_h5ad(args.input)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # --- PCA ---
    if args.method == "pca":
        print(f"[*] Running PCA ({args.dim} dimensions)...")
        X = adata.X
        if sparse.issparse(X):
            X = X.toarray()
        pca = PCA(n_components=args.dim, svd_solver="full")
        X_pca = pca.fit_transform(X)
        sign_flip = np.sum(pca.components_, axis=1) < 0
        X_pca[:, sign_flip] *= -1
        adata.obsm["X_pca"] = X_pca.copy()
        adata.obsm[f"X_latent_{args.method}"] = X_pca.copy()

    # --- scVI ---
    elif args.method == "scvi":
        print(f"[*] Running scVI ({args.dim} dimensions)...")
        # scVI uses raw counts from layered data
        scvi.model.SCVI.setup_anndata(adata, layer="counts")
        model = scvi.model.SCVI(adata, n_latent=args.dim, gene_likelihood="zinb")
        # Use simple progress bar for cleaner logs
        model.train(max_epochs=args.epochs, batch_size=args.batch_size, plan_kwargs={"n_epochs_kl_warmup": min(400, args.epochs)})
        adata.obsm[f"X_latent_{args.method}"] = model.get_latent_representation()

    # Save
    print(f"[*] Saving AnnData with {args.method} latent space to {args.output}...")
    adata.write(args.output)
    print("[*] Done!")

if __name__ == "__main__":
    main()
