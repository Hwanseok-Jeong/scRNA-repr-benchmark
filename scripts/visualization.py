import argparse
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

def draw_plot(X, labels, colors, title, metric_text, save_path):
    plt.figure(figsize=(6, 6))
    
    # Identify unique clusters
    unique_labels = pd.unique(labels)
    # Reconstruct color dictionary roughly
    if len(colors) == len(labels):
        color_map = {lbl: col for lbl, col in zip(labels, colors)}
    else:
        # Fallback to standard matplotlib colormap if colors not properly extracted
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20', len(unique_labels))
        color_map = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    # Scatter per cluster
    for lbl in unique_labels:
        idx = labels == lbl
        plt.scatter(X[idx, 0], X[idx, 1], c=[color_map[lbl]], s=1, label=None, alpha=0.8, edgecolors='none')
        
    plt.title(title, fontsize=14)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', adjustable='datalim')
    
    # Add Metrics Text Bottom Right
    plt.text(0.95, 0.05, metric_text, 
             horizontalalignment='right', 
             verticalalignment='bottom', 
             transform=plt.gca().transAxes, 
             fontsize=12,
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize Embeddings with Metrics")
    parser.add_argument("--input", required=True, help="Path to .h5ad with 2D embeddings")
    parser.add_argument("--method", choices=["pca", "ae", "vae", "scvi"], required=True, help="Latent model to visualize")
    parser.add_argument("--metrics", required=True, help="Path to metrics JSON file")
    parser.add_argument("--outdir", default="results/figures/", help="Directory to save figures")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    adata = sc.read_h5ad(args.input)
    
    with open(args.metrics, 'r') as f:
        metrics = json.load(f)
        
    labels = adata.obs['cluster_label'].values
    colors = adata.obs['cluster_color'].values if 'cluster_color' in adata.obs else [None]*len(labels)
    
    # 1. t-SNE Plot
    tsne_key = f"X_tsne_{args.method}"
    m_tsne = metrics.get('tsne', {})
    txt_tsne = f"KNN: {m_tsne.get('KNN', 'N/A')}\nKNC: {m_tsne.get('KNC', 'N/A')}\nCPD: {m_tsne.get('CPD', 'N/A')}"
    draw_plot(adata.obsm[tsne_key], labels, colors, 
              f"t-SNE (Latent: {args.method.upper()})", 
              txt_tsne, 
              os.path.join(args.outdir, f"fig_{args.method}_tsne.png"))
              
    # 2. UMAP Plot
    umap_key = f"X_umap_{args.method}"
    m_umap = metrics.get('umap', {})
    txt_umap = f"KNN: {m_umap.get('KNN', 'N/A')}\nKNC: {m_umap.get('KNC', 'N/A')}\nCPD: {m_umap.get('CPD', 'N/A')}"
    draw_plot(adata.obsm[umap_key], labels, colors, 
              f"UMAP (Latent: {args.method.upper()})", 
              txt_umap, 
              os.path.join(args.outdir, f"fig_{args.method}_umap.png"))

    print(f"[*] Visualizations saved to {args.outdir}")

if __name__ == "__main__":
    main()
