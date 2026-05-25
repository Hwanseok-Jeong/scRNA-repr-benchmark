import argparse
import os
import scanpy as sc
import numpy as np
from scipy import sparse
import torch
import scvi
from sklearn.decomposition import PCA

# Run:
#   conda run -n tasic_benchmark python scripts/latent_model.py --input <in.h5ad> --output <out.h5ad> --method pca
#   conda run -n tasic_benchmark python scripts/latent_model.py --input <in.h5ad> --output <out.h5ad> --method scvi --batch_key sample_prefix2

def main():
    parser = argparse.ArgumentParser(description="Latent Representation Learning for scRNA-seq")
    parser.add_argument("--input", required=True, help="Path to preprocessed .h5ad file")
    parser.add_argument("--output", required=True, help="Path to save .h5ad with latent representations")
    parser.add_argument("--method", choices=["pca", "scvi"], required=True, help="Latent model to use")
    # --- PCA params ---
    parser.add_argument("--dim", type=int, default=50, help="Latent dimensionality")
    # --- scVI params ---
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum training epochs for scVI")
    parser.add_argument("--early_stopping", type=str, default="true", help="Enable early stopping (true/false)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for scVI training")
    parser.add_argument("--batch_key", default=None, help="Obs column to use as scVI batch key")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of hidden layers for scVI")
    parser.add_argument("--n_hidden", type=int, default=128, help="Hidden units per layer for scVI")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for scVI")
    parser.add_argument("--gene_likelihood", default="zinb", help="Gene likelihood for scVI")
    parser.add_argument("--layer", default="counts", help="AnnData layer to use for scVI counts")
    parser.add_argument("--dispersion", default="gene", help="Dispersion setting for scVI")
    parser.add_argument("--kobak_outdir", default=None, help="Directory to save Kobak-style .npy outputs (tasic-pca50.npy)")
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
        adata.obsm[f"X_latent_{args.method}_n{args.dim}"] = X_pca.copy()
        # Optionally save Kobak-style PCA numpy array for reference comparison
        if args.kobak_outdir:
            method_outdir = os.path.join(args.kobak_outdir, "pca")
            os.makedirs(method_outdir, exist_ok=True)
            outpath = os.path.join(method_outdir, f"tasic-pca{args.dim}.npy")
            np.save(outpath, X_pca)

    # --- scVI ---
    elif args.method == "scvi":
        print(f"[*] Running scVI ({args.dim} dimensions)...")
        setup_kwargs = {"layer": args.layer}
        if args.batch_key:
            if args.batch_key not in adata.obs:
                raise ValueError(f"[!] batch_key '{args.batch_key}' not found in adata.obs")
            setup_kwargs["batch_key"] = args.batch_key
        scvi.model.SCVI.setup_anndata(adata, **setup_kwargs)
        model = scvi.model.SCVI(
            adata,
            n_latent=args.dim,
            n_layers=args.n_layers,
            n_hidden=args.n_hidden,
            dropout_rate=args.dropout_rate,
            gene_likelihood=args.gene_likelihood,
            dispersion=args.dispersion,
        )
        # Use simple progress bar for cleaner logs
        early_stopping = str(args.early_stopping).lower() in {"1", "true", "yes"}
        model.train(
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            early_stopping=early_stopping,
            early_stopping_patience=args.patience,
            plan_kwargs={
                "n_epochs_kl_warmup": min(400, args.max_epochs),
                "lr": args.learning_rate,
            },
        )
        adata.obsm[f"X_latent_{args.method}_n{args.dim}"] = model.get_latent_representation()

    # Save
    print(f"[*] Saving AnnData with {args.method} latent space to {args.output}...")
    adata.write(args.output)
    print("[*] Done!")

if __name__ == "__main__":
    main()
