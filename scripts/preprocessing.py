import scanpy as sc
import numpy as np
import argparse
from scipy import sparse

def geneSelection(data, threshold=32, atleast=10,
                  yoffset=0.02, xoffset=5, decay=1.5, n=None,
                  plot=False, verbose=1):
    """
    Kobak/Berens gene selection function (ported from rnaseqTools.py).
    Selects genes based on mean log2 nonzero expression and zero-rate curve.
    """
    if sparse.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data > threshold).mean(axis=0)))
        A = data.multiply(data > threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:, detected].mean(axis=0))) / (1 - zeroRate[detected])
    else:
        zeroRate = 1 - np.mean(data > threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        mask = data[:, detected] > threshold
        logs = np.zeros_like(data[:, detected]) * np.nan
        logs[mask] = np.log2(data[:, detected][mask])
        meanExpr[detected] = np.nanmean(logs, axis=0)

    lowDetection = np.array(np.sum(data > threshold, axis=0)).squeeze() < atleast
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan

    if n is not None:
        up = 10
        low = 0
        for _ in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
            if np.sum(selected) == n:
                break
            if np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low) / 2
            else:
                low = xoffset
                xoffset = (xoffset + up) / 2
        if verbose > 0:
            print(f"Chosen offset: {xoffset:.2f}")
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset

    return selected

def main():
    """
    Kobak/Berens-style preprocessing for Tasic 2018.
    - Gene selection with the original heuristic (threshold=32, n=3000).
    - Library size normalization to CPM (1e6) followed by log2(x + 1).
    - Raw counts preserved in adata.layers["counts"].
    """
    parser = argparse.ArgumentParser(description="General scRNA-seq preprocessing module.")
    parser.add_argument("--input", required=True, help="Path to raw input .h5ad file")
    parser.add_argument("--output", required=True, help="Path to save processed .h5ad file")
    parser.add_argument("--n_hvg", type=int, default=3000, help="Number of highly variable genes to select")
    parser.add_argument("--target_sum", type=float, default=1e6, help="Target sum for normalization (e.g. 1e6 for CPM)")
    args = parser.parse_args()

    print(f"[*] Loading raw data from {args.input}...")
    adata = sc.read_h5ad(args.input)

    # 1. Gene selection (Kobak/Berens heuristic)
    print(f"[*] Selecting {args.n_hvg} genes with Kobak/Berens heuristic...")
    selected = geneSelection(adata.X, n=args.n_hvg, threshold=32, plot=False, verbose=1)
    adata = adata[:, selected].copy()

    # 2. Preserve raw counts for deep generative models (e.g., scVI)
    print("[*] Preserving raw counts in adata.layers['counts'] for downstream deep learning models...")
    adata.layers["counts"] = adata.X.copy()

    # 3. Normalization (Library size scaling)
    print(f"[*] Normalizing library size to target_sum={args.target_sum}...")
    sc.pp.normalize_total(adata, target_sum=args.target_sum)

    # 4. Log2 Transformation (exactly log2(CPM + 1))
    print("[*] Applying log2(X + 1) transformation...")
    sc.pp.log1p(adata, base=2)

    # Output save
    print(f"[*] Saving processed AnnData to {args.output}...")
    adata.write(args.output)
    print("[*] Preprocessing complete.")

if __name__ == "__main__":
    main()
