import scanpy as sc
import numpy as np
import argparse
from scipy import sparse


def geneSelection(data, threshold=32, atleast=10,
                  yoffset=0.02, xoffset=5, decay=1.5, n=None,
                  plot=False, verbose=1, return_scores=False):
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

    # Margin from the selection curve; lower values are weaker selected genes.
    scores = np.full_like(zeroRate, fill_value=-np.inf, dtype=float)
    curve = np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
    scores[nonan] = zeroRate[nonan] - curve

    if return_scores:
        return selected, scores

    return selected


def parse_marker_genes(raw_value):
    if raw_value is None:
        return []
    stripped = raw_value.strip()
    if not stripped:
        return []
    return [item.strip() for item in stripped.split(",") if item.strip()]


def ensure_prefix_columns(adata):
    """Populate sample_prefix1/sample_prefix2 from sample_name when available.

    sample_prefix1 extracts just the first part before the first underscore.
    sample_prefix2 removes the trailing well/location token by splitting on the
    last underscore. This is the batch key used by scVI in this project.
    """
    if "sample_name" not in adata.obs:
        return

    sample_names = adata.obs["sample_name"].astype(str)
    if "sample_prefix1" not in adata.obs:
        adata.obs["sample_prefix1"] = sample_names.str.split("_").str[0]
    if "sample_prefix2" not in adata.obs:
        adata.obs["sample_prefix2"] = sample_names.str.split("_").str[:2].str.join("_")

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
    parser.add_argument("--marker_genes", type=str, default="", help="Comma-separated marker gene symbols")
    parser.add_argument("--no_force_include_markers", action="store_true", help="Do not force marker genes into selected genes")
    args = parser.parse_args()

    print(f"[*] Loading raw data from {args.input}...")
    adata = sc.read_h5ad(args.input)

    ensure_prefix_columns(adata)

    print('Number of cells:', adata.n_obs)
    if 'area' in adata.obs:
        print('Number of cells from ALM:', int(np.sum(adata.obs['area'] == 'ALM')))
        print('Number of cells from VISp:', int(np.sum(adata.obs['area'] == 'VISp')))
    if 'cluster_id' in adata.obs:
        print('Number of clusters:', np.unique(adata.obs['cluster_id']).size)
    print('Number of genes:', adata.n_vars)
    print('Fraction of zeros in the data matrix: {:.2f}'.format(
        adata.X.size / np.prod(adata.X.shape)
    ))

    # Keep the full-cell library sizes from the raw matrix.
    # The Kobak notebook normalizes selected genes by these original totals,
    # not by the post-selection library sizes.
    original_library_sizes = np.asarray(adata.X.sum(axis=1)).ravel()

    # 1. Gene selection (Kobak/Berens heuristic)
    print(f"[*] Selecting {args.n_hvg} genes with Kobak/Berens heuristic...")
    selected, selection_scores = geneSelection(adata.X, n=args.n_hvg, threshold=32, plot=False, verbose=1, return_scores=True)

    marker_genes = parse_marker_genes(args.marker_genes)
    if marker_genes:
        var_names = adata.var_names.astype(str).to_numpy()
        marker_set = set(marker_genes)
        present_markers = [gene for gene in marker_genes if gene in set(var_names)]
        missing_markers = [gene for gene in marker_genes if gene not in set(var_names)]
        print(f"[*] Marker genes requested: {len(marker_genes)} | present: {len(present_markers)} | missing: {len(missing_markers)}")
        if missing_markers:
            print(f"[!] Missing marker genes: {', '.join(missing_markers)}")

        if present_markers and not args.no_force_include_markers:
            marker_mask = np.isin(var_names, present_markers)
            to_add = marker_mask & ~selected
            add_count = int(np.sum(to_add))
            if add_count > 0:
                drop_candidates = np.where(selected & ~marker_mask)[0]
                if drop_candidates.size >= add_count:
                    drop_order = drop_candidates[np.argsort(selection_scores[drop_candidates])]
                    selected[drop_order[:add_count]] = False
                    selected[to_add] = True
                    print(f"[*] Forced inclusion of {add_count} marker genes (replaced weakest non-marker selections).")
                else:
                    selected[to_add] = True
                    print(f"[!] Included {add_count} additional marker genes; selected genes exceed n_hvg due to insufficient non-marker candidates.")

    adata = adata[:, selected].copy()

    print('Selected genes:', adata.n_vars)
    print('Selected matrix shape:', adata.X.shape)

    # 2. Preserve raw counts for deep generative models (e.g., scVI)
    print("[*] Preserving raw counts in adata.layers['counts'] for downstream deep learning models...")
    adata.layers["counts"] = adata.X.copy()

    # 3. Normalization (Library size scaling)
    print(f"[*] Normalizing library size to target_sum={args.target_sum}...")
    if sparse.issparse(adata.X):
        adata.X = adata.X.multiply((args.target_sum / original_library_sizes)[:, None]).tocsr()
    else:
        adata.X = adata.X / original_library_sizes[:, None] * args.target_sum

    # 4. Log2 Transformation (exactly log2(CPM + 1))
    print("[*] Applying log2(X + 1) transformation...")
    sc.pp.log1p(adata, base=2)

    # Output save
    print(f"[*] Saving processed AnnData to {args.output}...")
    adata.write(args.output)
    print("[*] Preprocessing complete.")

if __name__ == "__main__":
    main()
