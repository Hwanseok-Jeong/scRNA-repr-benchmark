# scRNA-seq Representation Learning Benchmark: A Comparative Study of Linear and Non-Linear Latent Spaces

> Note: This README is a working draft under active revision. The benchmark interpretation and wording may be updated as additional results are finalized.

## Executive Summary
This study compares three 2D embedding strategies for scRNA-seq representation analysis:
- PCA with Kobak-style t-SNE/UMAP as the linear reference.
- scVI latent representations embedded with the same array-based Kobak-style embedding setup.
- scVI latent representations embedded with Scanpy's graph-based UMAP workflow.

The current results indicate that PCA-style initialization is highly effective for PCA latents, whereas transfer of the same initialization strategy to scVI latents is less consistent. For scVI, the graph-based UMAP workflow is evaluated here as the methodologically better-matched approach.

## Key Comparison Figures
The following PDFs provide the primary visual comparison used in this report:
- [PCA baseline](results/figures/pca/n50/tasic-variants.pdf)
- [scVI Kobak-style result](results/figures/scvi/n50/tasic-variants.pdf)
- [scVI Scanpy workflow result](results/scvi_scanpy_workflow/figures/umap_n30_md05_scanpy_4panels.pdf)

In the current document structure, these figures are intentionally placed before the long-form methodological narrative to make the core comparison visible at first pass.

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
1. Create the environment from the provided spec.

	```bash
	conda env create -f environment.yml
	```

2. Activate the environment.

	```bash
	conda activate tasic_benchmark
	```

3. Check that the bundled t-SNE binary exists.

	```bash
	ls FIt-SNE/bin/fast_tsne
	```

	If this file is missing, use the upstream FIt-SNE repository and follow its build instructions:
	<https://github.com/KlugerLab/FIt-SNE>

4. Run the pipeline with Nextflow.

	```bash
	nextflow run main.nf -resume
	```

The environment definition is stored in `environment.yml` so the repository can be reproduced on another machine with a single conda command. The file already includes the main Python stack (`scanpy`, `anndata`, `scvi-tools`-compatible dependencies, `umap-learn`, `scikit-learn`) plus `nextflow`.

## 2. Aims and Objectives
The study is organized around the following objectives:
1. **Baseline reproduction:** To reproduce the cortical separation patterns reported in the reference paper using PCA with 50 dimensions, PCA initialization, and Kobak-style FIt-SNE settings.
2. **Core benchmark:** To compare PCA with scVI by replacing the linear PCA latent space with scVI latents and evaluating the resulting t-SNE/UMAP embeddings.
3. **Scope of applicability:** To maintain a preprocessing pipeline that can be extended to additional raw-count datasets, while validating the present implementation on the Smart-seq2-based Tasic benchmark.
4. **Dual-track comparison:** To evaluate both a fixed 50-dimensional setting and model-specific latent dimensions.
5. **Quantitative evaluation:** To assess embeddings using KNN retention, KNC purity, and CPD.
6. **Reproducibility:** To implement the workflow in Nextflow so it can be reproduced locally and, if needed, transferred to HPC environments.

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
*(Note: scVI is trained on the raw HVG count matrix with a ZINB likelihood; it does not use the CPM/log-transformed input used for PCA.)*

Implementation notes:
- `scripts/preprocessing.py` ports the Kobak gene-selection heuristic from `rnaseqTools.geneSelection(...)` with `threshold=32`, `n=3000`, `decay=1.5`, `yoffset=0.02`, and the same binary search over `xoffset`.
- The `Chosen offset: 6.56` style line is not a fixed parameter; it is the final `xoffset` found by that binary search when the requested number of genes is reached.
- The CPM plus log transform is implemented as `normalize_total(target_sum=1e6)` followed by `log1p(base=2)`, which matches the notebook's `np.log2(CPM + 1)`.
- `scripts/latent_model.py` uses `sklearn.decomposition.PCA(n_components=50, svd_solver='full')` and the same PC sign-flip convention as the Kobak notebook.

### 4.2 Latent Representations
- **PCA:** Linear baseline (reference point).
- **scVI:** [scvi-tools](https://scvi-tools.org/) latent representation learned from the raw HVG count matrix with a count-aware ZINB likelihood.


### 4.3 Dimensionality Reduction
- **t-SNE implementation:** `scripts/embedding.py` uses the Kobak FIt-SNE wrapper interface and the same recommended settings: PCA-style initialization, learning rate $N/12$, and multi-scale perplexity `[30, N/100]`.
- **UMAP implementation:** The reference notebook `REFERENCE-rna-seq-tsne/umap-comparison.ipynb` compares two Tasic UMAP parameterizations. The second configuration, `random_state=1`, `n_neighbors=30`, `min_dist=0.5`, is used as the default in this benchmark because it is the configuration reproduced in the reference material and is the one carried through the downstream comparisons.

Kobak-derived pieces that are intentionally preserved in this pipeline:
- Gene selection heuristic and its binary-search thresholding behavior.
- CPM normalization and base-2 log transform.
- PCA with `svd_solver='full'` and sign correction.
- FIt-SNE recommendation of PCA initialization, $N/12$ learning rate, and multi-scale perplexity.
- UMAP benchmark settings from the supplementary notebook, with the second setting chosen for the default pipeline.

### 4.4 Quantitative Evaluation
- **KNN Preservation:** local neighborhood retention between the original space and the reduced embedding.
- **KNC Purity:** preservation of class-level neighborhood coherence, which captures whether biologically similar cells remain adjacent after projection.
- **CPD:** Spearman correlation between pairwise distances in the original gene space and the reduced space, used here as a proxy for global structure preservation.

## 5. Results & Discussion

### 5.1 What I Expected
The initial expectation was that scVI would produce a cleaner latent space than PCA because it models scRNA-seq counts more directly. That remains plausible at the latent-space level, but it does not automatically imply improved performance under a PCA-optimized embedding setup.

### 5.2 What the Current Numbers Say
The current metric summaries show that the PCA-initialized Kobak settings still perform well on PCA latents, while the same settings transfer less cleanly to scVI latents.

The comparison between the array-based Kobak workflow and the Scanpy graph-based workflow is driven by the geometry of the learned scVI latent space: the former operates directly on Euclidean coordinates, whereas the latter constructs a neighborhood graph before applying UMAP. In the figures produced for this repository, that graph-based workflow is the more appropriate visualization path for scVI.

| Track | Embedding | KNN | KNC | CPD | Short note |
| --- | --- | ---: | ---: | ---: | --- |
| PCA baseline | t-SNE, PCA init | 0.383 | 0.605 | 0.578 | Strong global preservation |
| PCA baseline | UMAP, n30/md0.5 | 0.208 | 0.634 | 0.559 | Good global structure, weaker KNN |
| scVI Kobak-style | t-SNE, PCA init | 0.424 | 0.565 | 0.377 | Better local retention, weaker global match |
| scVI Kobak-style | UMAP, n30/md0.5 | 0.213 | 0.623 | 0.350 | More balanced than t-SNE, still below PCA CPD |

These values are from the completed array-based benchmark runs.

## References
- Kobak, D. & Berens, P. *The art of using t-SNE for single-cell transcriptomics.* [Nature Communications (2019)](https://www.nature.com/articles/s41467-019-13055-y)
- Tasic, B. et al. *Shared and distinct transcriptomic cell types across neocortical areas.* [Cell (2018)](https://www.cell.com/cell/fulltext/S0092-8674(18)30751-4)
- Linderman, G. C. et al. *Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data.* [Nature Methods (2019)](https://www.nature.com/articles/s41592-018-0308-4)
