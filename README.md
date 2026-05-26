# scRNA-seq Representation Learning Benchmark: A Comparative Study of Linear and Non-Linear Latent Spaces

## Executive Summary
I compared three ways of turning scRNA-seq data into a 2D visualization:
- PCA + Kobak-style t-SNE / UMAP as the linear baseline.
- scVI latents fed into the same Kobak-style embedding recipe.
- scVI latents visualized with Scanpy's graph-based UMAP workflow.

The main takeaway is simple: PCA-style initialization works well for PCA latents, but it is a weaker match for scVI latents. For scVI, the graph-based UMAP workflow is the more natural fit.

## Key Comparison Figures
Use these three PDFs as the first visual comparison:
- [PCA baseline](results/figures/pca/n50/tasic-variants.pdf)
- [scVI Kobak-style result](results/figures/scvi/n50/tasic-variants.pdf)
- [scVI Scanpy workflow result](results/scvi_scanpy_workflow/figures/umap_n30_md05_scanpy_4panels.pdf)

If I were polishing this README further, I would keep these three figure links near the top and move the longer motivation / implementation text below them.

## Figure 1 — Visual comparison (click thumbnails for PDFs)
[![PCA baseline](results/figures/pca/n50/tasic-variants.png)](results/figures/pca/n50/tasic-variants.pdf)  
*PCA baseline — Kobak-style variants (50 PCs).*  

[![scVI Kobak-style](results/figures/scvi/n50/tasic-variants.png)](results/figures/scvi/n50/tasic-variants.pdf)  
*scVI latents + Kobak-style embedding (array-based t-SNE/UMAP).*  

[![scVI Scanpy workflow](results/scvi_scanpy_workflow/figures/umap_n30_md05_scanpy_4panels.png)](results/scvi_scanpy_workflow/figures/umap_n30_md05_scanpy_4panels.pdf)  
*scVI latents visualized with Scanpy graph-based UMAP (n=30, min_dist=0.5).*  

## 1. Introduction & Motivation
High-dimensional single-cell RNA-sequencing (scRNA-seq) data is usually explored with nonlinear dimensionality reduction methods such as t-SNE and UMAP. In *The art of using t-SNE for single-cell transcriptomics* [Kobak & Berens, 2019](https://www.nature.com/articles/s41467-019-13055-y), PCA initialization and careful parameter choice are enough to recover both local cluster structure and a coarse global layout.

I started this project after reading Kobak's lectures and paper, and I wanted a benchmark that is small enough to run on a local machine but still large enough to expose the failure modes of different embeddings. The Tasic dataset was a practical choice for that reason: it is big enough to be interesting, but still manageable without a dedicated GPU during prototyping.

The main question in this repository is simple: if I replace the usual linear PCA latent space with a count-aware deep generative latent space such as scVI, do the downstream embeddings become better, or do they break because the embedding algorithm makes the wrong assumptions?

I deliberately leave AE/VAE out of the main benchmark. In a raw-count setting, a plain Gaussian reconstruction loss is not a great fit for scRNA-seq noise, so the comparison becomes less clean than PCA vs scVI. scVI is the more appropriate count-aware baseline because it models overdispersion and zero inflation directly.

This repository builds on the Kobak lab reference implementation [rna-seq-tsne](https://github.com/berenslab/rna-seq-tsne) and uses [FIt-SNE](https://github.com/KlugerLab/FIt-SNE) for the t-SNE runs.

## 2. Quick Start
1. Create the environment from the provided spec: `conda env create -f environment.yml`
2. Activate it: `conda activate tasic_benchmark`
3. Check that the bundled t-SNE binary exists at `FIt-SNE/bin/fast_tsne`. If it is missing, rebuild the upstream FIt-SNE project in that folder.
4. Run the pipeline with Nextflow: `nextflow run main.nf -resume`

I keep the environment definition in `environment.yml` so the repo can be reproduced on another machine with a single conda command. The file already includes the main Python stack (`scanpy`, `anndata`, `scvi-tools`-compatible dependencies, `umap-learn`, `scikit-learn`) plus `nextflow`.

## 2. Aims and Objectives
1. **Baseline reproduction:** I first check that PCA with 50 dimensions, PCA initialization, and Kobak-style FIt-SNE settings reproduces the expected cortical separation patterns from the reference paper.
2. **Core benchmark:** I then replace the linear PCA latent with scVI and compare the downstream t-SNE/UMAP outputs against PCA.
3. **Generalization:** I keep the preprocessing code modular so it can be reused for raw-count datasets beyond Smart-seq2, including UMI-based data.
4. **Dual track comparison:** I test both a fixed 50-dimensional track and a model-specific latent track.
5. **Quantitative evaluation:** I score embeddings with KNN retention, KNC purity, CPD, and, in the next pass, Trustworthiness and Continuity.
6. **Automation:** I orchestrate the whole workflow with Nextflow so the benchmark can be reproduced locally and later moved to HPC without rewriting the core scripts.

## 3. Biological Context (Tasic et al. 2018)
To evaluate structure preservation, I use the ~24,000 mouse cortical cells dataset sequenced with Smart-seq2 in [Tasic et al. 2018](https://www.cell.com/cell/fulltext/S0092-8674(18)30751-4), sampled from the visual (VISp) and anterior lateral motor (ALM) cortices.

The dataset gives me a useful biological reference:
- **GABAergic (Inhibitory) neurons** are highly conserved transcriptomically across both VISp and ALM regions (expected to intermingle in embedding space).
- **Glutamatergic (Excitatory) neurons** possess distinct, region-specific signatures (expected to segregate cleanly into VISp and ALM branches).
The benchmark asks which latent representation most faithfully captures this known biological manifold.

## 4. Methodology
### 4.1 Data Preprocessing
Using the Kobak & Berens reference workflow:
- Outlier filtering based on provided metadata (masking 'Good cells').
- Library size normalization (CPM).
- Log-transformation: $\log_2(CPM + 1)$.
- Feature selection: Top 3,000 Highly Variable Genes (HVGs).
*(Note: scVI utilizes raw counts from the selected HVGs with a ZINB likelihood, while PCA uses the log-normalized data.)*

Implementation notes:
- `scripts/preprocessing.py` ports the Kobak gene-selection heuristic from `rnaseqTools.geneSelection(...)` with `threshold=32`, `n=3000`, `decay=1.5`, `yoffset=0.02`, and the same binary search over `xoffset`.
- The `Chosen offset: 6.56` style line is not a fixed parameter; it is the final `xoffset` found by that binary search when the requested number of genes is reached.
- The CPM plus log transform is implemented as `normalize_total(target_sum=1e6)` followed by `log1p(base=2)`, which matches the notebook's `np.log2(CPM + 1)`.
- `scripts/latent_model.py` uses `sklearn.decomposition.PCA(n_components=50, svd_solver='full')` and the same PC sign-flip convention as the Kobak notebook.

### 4.2 Latent Representations
- **PCA:** Linear baseline (reference point).
- **scVI:** [scvi-tools](https://scvi-tools.org/) baseline implementation with a count-aware likelihood to account for technical variance inherent to scRNA-seq counts.

#### AE vs VAE (concise note)
Autoencoders (AE) and Variational Autoencoders (VAE) are often proposed as flexible nonlinear latent models, but care is required for scRNA-seq counts:
- **AE:** deterministic encoder/decoder trained with a pointwise reconstruction loss (commonly MSE). This produces a single-point embedding per cell but assumes Gaussian errors — a poor fit for overdispersed, zero-inflated count data.
- **VAE:** probabilistic encoder that learns an approximate posterior (mean + log-variance) and optimizes the ELBO (reconstruction + KL). VAEs model uncertainty, but when the reconstruction likelihood is Gaussian they still mis-specify counts.
- **To make AE/VAE appropriate for scRNA-seq:** replace Gaussian reconstruction with a count-aware likelihood (for example Negative Binomial or ZINB), account for library size or size-factors, and include overdispersion modeling. That is the part scVI handles for me. Without these changes, AE/VAE reconstructions and learned latents can be biased or misleading for raw counts.


### 4.3 Dimension Reduction (Embeddings)
- **t-SNE implementation:** `scripts/embedding.py` uses the Kobak FIt-SNE wrapper interface and the same recommended settings: PCA-style initialization, learning rate $N/12$, and multi-scale perplexity `[30, N/100]`.
- **UMAP implementation:** The reference notebook `REFERENCE-rna-seq-tsne/umap-comparison.ipynb` shows two Tasic UMAP settings. I use the second one as the default because it improves KNC and CPD on this benchmark: `random_state=1`, `n_neighbors=30`, `min_dist=0.5`.
- **Rationale:** The first UMAP setting stays as a historical reference, but the second setting gives me the cleaner comparison for this dataset.

Kobak-derived pieces that are intentionally preserved in this pipeline:
- Gene selection heuristic and its binary-search thresholding behavior.
- CPM normalization and base-2 log transform.
- PCA with `svd_solver='full'` and sign correction.
- FIt-SNE recommendation of PCA initialization, $N/12$ learning rate, and multi-scale perplexity.
- UMAP benchmark settings from the supplementary notebook, with the second setting chosen for the default pipeline.

### 4.4 Quantitative Evaluation
I keep the metric definitions before the results so the comparison table reads cleanly.

- **KNN Preservation:** overlap of exact neighborhoods across the original, latent, and embedded spaces.
- **KNC Purity:** preservation of cluster label coherence.
- **CPD:** Spearman correlation between pairwise distances in the original gene space and the reduced space.
- **Trustworthiness:** how often neighbors in the 2D embedding are also neighbors in the original space.
- **Continuity:** how often original-space neighbors remain neighbors after projection.

Trustworthiness and Continuity are the next metrics I would add for a fairer comparison between PCA-style and graph-based embeddings.

## 5. Results & Discussion

### 5.1 What I Expected
My initial expectation was that scVI would produce a cleaner latent space than PCA because it models scRNA-seq counts more directly. That part is true in the latent space itself, but it does not automatically mean that a PCA-optimized embedding recipe will behave better downstream.

### 5.2 What the Current Numbers Say
The current metric summaries show that the PCA-initialized Kobak recipe still does a very good job on PCA latents, while the same recipe transfers less cleanly to scVI latents.

| Track | Embedding | KNN | KNC | CPD | Short note |
| --- | --- | ---: | ---: | ---: | --- |
| PCA baseline | t-SNE, PCA init | 0.383 | 0.605 | 0.578 | Strong global preservation |
| PCA baseline | UMAP, n30/md0.5 | 0.208 | 0.634 | 0.559 | Good global structure, weaker KNN |
| scVI Kobak-style | t-SNE, PCA init | 0.424 | 0.565 | 0.377 | Better local retention, weaker global match |
| scVI Kobak-style | UMAP, n30/md0.5 | 0.213 | 0.623 | 0.350 | More balanced than t-SNE, still below PCA CPD |

These values are from the completed array-based benchmark runs. I would keep the Scanpy graph-based scVI workflow in a separate row once its metrics are finalized, because it is a different embedding pipeline rather than just another parameter choice.

### 5.3 What I Would Show in the README
For the main README, I would not use the Nextflow metro-map as the primary result figure. It is useful, but it explains the workflow rather than the benchmark outcome.

My preferred layout is:
1. one compact comparison figure with three panels: PCA baseline, scVI Kobak-style result, and scVI Scanpy graph-based result,
2. one metrics table underneath it, and
3. one workflow diagram or metro-map as a smaller methods figure in the appendix or implementation section.

That gives the reader the result first, then the pipeline, which is much easier to scan than a long block of prose.

## 6. Work Log (Updated 2026-05-26)
- Implemented PCA-initialized t-SNE benchmark variants in the pipeline.
- Integrated the scVI deep generative baseline.
- Found that PCA initialization is a poor match for some scVI embeddings, especially when I force the Kobak-style t-SNE recipe onto them.
- Added a Scanpy graph-based UMAP workflow for scVI so I can compare a more natural manifold-aware embedding path.
- Updated `.gitignore` so lightweight result artifacts stay tracked while heavy `.h5ad` files stay out.
- Kept the README focused on the benchmark outcome instead of the implementation details alone.

---
*Developed for application to the Kobak Lab PhD Program and reproducible research in single-cell benchmarking.*

## References
- Kobak, D. & Berens, P. *The art of using t-SNE for single-cell transcriptomics.* [Nature Communications (2019)](https://www.nature.com/articles/s41467-019-13055-y)
- Tasic, B. et al. *Shared and distinct transcriptomic cell types across neocortical areas.* [Cell (2018)](https://www.cell.com/cell/fulltext/S0092-8674(18)30751-4)
- Lopez, R. et al. *Deep generative modeling for single-cell transcriptomics.* [Nature Methods (2018)](https://www.nature.com/articles/s41592-018-0229-2)
- Wolf, F. A. et al. *Scanpy: large-scale single-cell gene expression data analysis.* [Genome Biology (2018)](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1382-0)
- McInnes, L., Healy, J. & Melville, J. *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.* [arXiv (2018)](https://arxiv.org/abs/1802.03426)
- Linderman, G. C. et al. *Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data.* [Nature Methods (2019)](https://www.nature.com/articles/s41592-018-0308-4)
