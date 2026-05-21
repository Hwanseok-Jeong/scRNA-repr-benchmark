import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import scanpy as sc
from scipy import sparse
import argparse

def download_file(url, save_path):
    if not os.path.exists(save_path):
        print(f"[*] Downloading {save_path}...")
        urllib.request.urlretrieve(url, save_path)
    else:
        print(f"[*] {save_path} already exists. Skipping download.")

def extract_zip(zip_path, extract_to):
    print(f"[*] Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def sparseload(filenames):
    """
    Function directly adopted from Kobak's notebook to handle massive CSV files efficiently.
    Uses Pandas chunking and scipy sparse matrices.
    """
    genes = []
    sparseblocks = []
    areas = []
    cells = []
    for chunk1, chunk2 in zip(pd.read_csv(filenames[0], chunksize=1000, index_col=0, na_filter=False),
                              pd.read_csv(filenames[1], chunksize=1000, index_col=0, na_filter=False)):
        if len(cells) == 0:
            cells = np.concatenate((chunk1.columns, chunk2.columns))
            areas = [0]*chunk1.columns.size + [1]*chunk2.columns.size # 0: ALM, 1: VISp
        
        genes.extend(list(chunk1.index))
            
        sparseblock1 = sparse.csr_matrix(chunk1.values.astype(float))
        sparseblock2 = sparse.csr_matrix(chunk2.values.astype(float))
        sparseblock = sparse.hstack((sparseblock1, sparseblock2), format='csr')
        sparseblocks.append([sparseblock])
        print('.', end='', flush=True)
    print(' done')
    counts = sparse.bmat(sparseblocks)
    return counts.T, np.array(genes), cells, np.array(areas)

def main():
    parser = argparse.ArgumentParser(description="Download and format Tasic 2018 dataset into raw .h5ad")
    parser.add_argument("--outdir", default="data/raw", help="Directory to save raw tasic data")
    parser.add_argument("--output", default="data/tasic_raw.h5ad", help="Final raw h5ad path")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1. Download URLs from Allen Brain Institute & Reference Repo
    url_visp = "http://celltypes.brain-map.org/api/v2/well_known_file_download/694413985"
    url_alm = "http://celltypes.brain-map.org/api/v2/well_known_file_download/694413179"
    url_metadata = "https://raw.githubusercontent.com/berenslab/rna-seq-tsne/master/data/tasic-sample_heatmap_plot_data.csv"
    
    zip_visp = os.path.join(args.outdir, "VISp.zip")
    zip_alm = os.path.join(args.outdir, "ALM.zip")
    meta_csv = os.path.join(args.outdir, "sample_heatmap_plot_data.csv")
    
    download_file(url_visp, zip_visp)
    download_file(url_alm, zip_alm)
    download_file(url_metadata, meta_csv)
    
    extract_zip(zip_visp, args.outdir)
    extract_zip(zip_alm, args.outdir)

    # File paths generated after extraction
    f_visp = os.path.join(args.outdir, "mouse_VISp_2018-06-14_exon-matrix.csv")
    f_alm = os.path.join(args.outdir, "mouse_ALM_2018-06-14_exon-matrix.csv")
    f_genes = os.path.join(args.outdir, "mouse_VISp_2018-06-14_genes-rows.csv")

    # 2. Parse using Kobak's sparse loading method
    print("[*] Parsing extremely large CSV expression matrices into sparse format...")
    counts, genes, cells, areas = sparseload([f_visp, f_alm])

    # 3. Map Gene IDs to Symbols
    print("[*] Mapping Gene IDs to Symbols...")
    genesDF = pd.read_csv(f_genes)
    id2symbol = dict(zip(genesDF['gene_entrez_id'].astype(str).tolist(), genesDF['gene_symbol'].tolist()))
    # Some genes in the count matrix might be strings, clean them depending on data frame formats
    genes = np.array([id2symbol.get(str(g), str(g)) for g in genes])

    # 4. Filter for 'Good Cells' based on metadata & Assign Cluster Labels
    print("[*] Filtering for QC-passed 'Good Cells' and assigning cluster labels...")
    clusterInfo = pd.read_csv(meta_csv)
    goodCells  = clusterInfo['sample_name'].values
    ids        = clusterInfo['cluster_id'].values
    labels     = clusterInfo['cluster_label'].values
    colors     = clusterInfo['cluster_color'].values

    # Find intersections
    cell_to_idx = {c: i for i, c in enumerate(cells)}
    valid_indices = []
    valid_good_cells = []
    
    for c in goodCells:
        if c in cell_to_idx:
            valid_indices.append(cell_to_idx[c])
            valid_good_cells.append(c)

    # Subset everything to Good Cells
    counts = counts[valid_indices, :]
    areas = areas[valid_indices]
    
    # Create valid metadata mapping
    obs_df = pd.DataFrame({'cell_id': valid_good_cells})
    # Merge with clusterInfo to keep order
    obs_df = obs_df.merge(clusterInfo, left_on='cell_id', right_on='sample_name', how='left')
    obs_df['area'] = ['VISp' if a == 1 else 'ALM' for a in areas]

    # 5. Create final AnnData Object
    print("[*] Creating AnnData object...")
    adata = sc.AnnData(X=counts)
    adata.obs = obs_df
    adata.obs.index = adata.obs['cell_id'].values
    adata.var_names = genes

    # 6. Save Base Raw Object
    print(f"[*] Saving Tasic raw dataset to {args.output}...")
    adata.write(args.output)
    print("[*] SUCCESS: Raw data formatting completed. Ready for downstream preprocessing.")

if __name__ == "__main__":
    main()
