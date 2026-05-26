# scRNA-seq Representation Learning Benchmark: A Comparative Study of Linear and Non-Linear Latent Spaces

## 1. Introduction & Motivation
High-dimensional single-cell RNA-sequencing (scRNA-seq) data is routinely visualized using nonlinear dimensionality reduction techniques like t-SNE and UMAP. As elegantly demonstrated in *“The art of using t-SNE for single-cell transcriptomics” (Kobak & Berens, 2019)*, combining PCA initialization with appropriate parameter tuning (e.g., perplexity of $N/100$) successfully preserves both local and global biological structures. 

*Personal Motivation:* This project was heavily inspired by Prof. Dmitry Kobak's Machine Learning introductory lectures at the University of Tübingen and his remarkable 2019 Nature Communications paper regarding representation manifolds. Driven by a deep interest in representation learning, particularly how Autoencoders capture complex topologies, and recognizing that dimensionality reduction is arguably the most critical step in bioinformatics, this pipeline was developed as a scalable portfolio piece. To accommodate local desktop constraints without a dedicated GPU during the prototyping phase, the moderately sized ~24,000 cell Tasic dataset was chosen.

Traditionally, t-SNE is computed on the first 50 Principal Components (PCs) of the log-normalized expression matrix. Here we focus on a count-aware deep generative baseline (scVI) alongside the linear PCA baseline to test whether a ZINB-based latent space preserves global topology better than PCA.
We intentionally exclude AE/VAE in this benchmark because the simple Gaussian reconstruction losses used in our AE/VAE do not model scRNA-seq count noise (zero inflation, overdispersion), and tend to be mis-specified for raw count data compared to scVI's ZINB likelihood.

This repository serves as a systematic benchmark pipeline to evaluate how different latent representations (PCA vs. AE vs. VAE vs. scVI) influence downstream non-linear embeddings (t-SNE, UMAP) and topological evaluations. *Developed with the assistance of GitHub Copilot, it builds directly upon the Kobak lab reference implementation (https://github.com/berenslab/rna-seq-tsne).*
We used this t-SNE implementation: https://github.com/KlugerLab/FIt-SNE.

## 2. Aims and Objectives
1. **Phase 1: Baseline Reproduction:** First, ensure that PCA (50 dimensions) followed by t-SNE with Kobak's exact settings (PCA initialization, Perplexity $N/100$) perfectly reproduces the topological structures (e.g., VISp vs ALM cortical segregation) showcased in the 2019 reference paper.
2. **Phase 2: Core Benchmark (DL Latent Models):** We substitute the linear 50D PCA latent space with a scVI latent representation and compare downstream t-SNE/UMAP layouts and topology metrics against PCA.
3. **Generalization & Scalability:** The pipeline is intentionally modular. `preprocessing.py` can parse and standardize raw counts from diverse platforms (like 10x Genomics UMI drops) alongside the Smart-seq2 standard used here, facilitating expansive future benchmarking.
4. **Dual-Track Evaluation:** Evaluate whether performance optimally peaks at the standard fixed 50 dimensions constraint (Track 1) or individual dimensionalities intrinsically optimal to each model's reconstruction loss (Track 2).
3. **Robust Evaluation:** Extensively quantify topology preservation using K-Nearest Neighbor (KNN) retention, K-Nearest Class (KNC) purity, and Correlative Preservation of Distances (CPD).
4. **Scalability & Automation:** Containerize the entire pipeline via Docker and orchestrate it with Nextflow to allow for scalable and reproducible execution (eventually transferring from local setups to HPC).

## 3. Biological Context (Tasic et al. 2018)
To evaluate structure preservation, we utilize the ~24,000 mouse cortical cells dataset sequenced with Smart-seq2 (*Tasic et al. 2018*), sampled from the visual (VISp) and anterior lateral motor (ALM) cortices.

The dataset provides a ground-truth biological topology:
- **GABAergic (Inhibitory) neurons** are highly conserved transcriptomically across both VISp and ALM regions (expected to intermingle in embedding space).
- **Glutamatergic (Excitatory) neurons** possess distinct, region-specific signatures (expected to segregate cleanly into VISp and ALM branches).
The benchmark explicitly tests which latent representation most faithfully captures this known biological manifold.

## 4. Methodology
### 4.1 Data Preprocessing
Mimicking the Kobak & Berens repo:
- Outlier filtering based on provided metadata (masking 'Good cells').
- Library size normalization (CPM).
- Log-transformation: $\log_2(CPM + 1)$.
- Feature selection: Top 3,000 Highly Variable Genes (HVGs).
*(Note: scVI utilizes raw counts from the selected HVGs with a ZINB likelihood, while PCA uses the log-normalized data.)*

Implementation notes:
- `scripts/preprocessing.py` ports the Kobak gene-selection heuristic from `rnaseqTools.geneSelection(...)` with `threshold=32`, `n=3000`, `decay=1.5`, `yoffset=0.02`, and the same binary search over `xoffset`.
- The `Chosen offset: 6.56` style line is not a fixed parameter; it is the final `xoffset` found by that binary search when the requested number of genes is reached.
- The CPM + log transform is implemented as `normalize_total(target_sum=1e6)` followed by `log1p(base=2)`, which matches the notebook's `np.log2(CPM + 1)`.
- `scripts/latent_model.py` now uses `sklearn.decomposition.PCA(n_components=50, svd_solver='full')` and the same PC sign flip convention as the Kobak notebook.

### 4.2 Latent Representations
- **PCA:** Linear baseline (reference point).
- **scVI:** `scvi-tools` baseline implementation (ZINB/NB likelihood) to account for technical variance inherent to scRNA-seq counts.

#### AE vs VAE (concise note)
Autoencoders (AE) and Variational Autoencoders (VAE) are often proposed as flexible nonlinear latent models, but care is required for scRNA-seq counts:
- **AE:** deterministic encoder/decoder trained with a pointwise reconstruction loss (commonly MSE). This produces a single-point embedding per cell but assumes Gaussian errors — a poor fit for overdispersed, zero-inflated count data.
- **VAE:** probabilistic encoder that learns an approximate posterior (mean + log-variance) and optimizes the ELBO (reconstruction + KL). VAEs model uncertainty, but when the reconstruction likelihood is Gaussian they still mis-specify counts.
- **To make AE/VAE appropriate for scRNA-seq:** replace Gaussian reconstruction with a count-aware likelihood (e.g., Negative Binomial or ZINB), account for library size / size-factors, and include overdispersion modeling — essentially what scVI implements. Without these changes, AE/VAE reconstructions and learned latents can be biased or misleading for raw counts.


### 4.3 Dimension Reduction (Embeddings)
- **t-SNE implementation:** `scripts/embedding.py` uses the Kobak FIt-SNE wrapper interface and the same recommended settings: PCA-style initialization, learning rate $N/12$, and multi-scale perplexity `[30, N/100]`.
- **UMAP implementation:** The reference notebook `REFERENCE-rna-seq-tsne/umap-comparison.ipynb` shows two Tasic UMAP settings. We select the second one as the default because it improves KNC and CPD on the Tasic benchmark: `random_state=1`, `n_neighbors=30`, `min_dist=0.5`.
- **Rationale:** The first UMAP setting is kept as a historical reference, but the second setting better preserves cluster-level and global structure for this dataset, so it is the one used in the pipeline.

Kobak-derived pieces that are intentionally preserved in this pipeline:
- Gene selection heuristic and its binary-search thresholding behavior.
- CPM normalization and base-2 log transform.
- PCA with `svd_solver='full'` and sign correction.
- FIt-SNE recommendation of PCA initialization, $N/12$ learning rate, and multi-scale perplexity.
- UMAP benchmark settings from the supplementary notebook, with the second setting chosen for the default pipeline.

### 4.4 Quantitative Evaluation
To avoid subjective visual bias, embeddings are scored using:
- **KNN Preservation:** Overlap of exact neighborhoods across original (high-dimensional) vs. latent vs. embedded spaces.
- **KNC Purity (Silhouette approximation):** Preservation of cluster label coherence.
- **Correlative Preservation of Distances (CPD):** Spearman correlation between pairwise distances in the original gene space and the reduced space.

## 5. Results & Discussion

### 5.1 Research Aim & Initial Hypothesis
**Aim:** To evaluate whether substituting a traditional linear baseline (PCA) with a more purified, count-aware deep generative latent space (scVI) improves the preservation of local and global biological topologies in 2D embeddings (t-SNE/UMAP).

**Initial Hypothesis:** We initially hypothesized that since scVI explicitly models the zero-inflation and overdispersion of scRNA-seq counts via a ZINB likelihood, its latent space would be inherently "cleaner" and therefore yield a superior 2D embedding representation compared to PCA. 

**Findings & Rejection of the Hypothesis (Kobak Workflow):**
Our findings revealed a critical nuance: **direct application of scVI into t-SNE heuristics optimized for PCA utterly collapses the global structure.** 
- **PCA-based t-SNE** highly benefits from PCA initialization, leaping in Correlative Preservation of Distances (CPD) from 0.231 (random init) to an impressive **0.578** (PCA init). Because PCA is a linear representation, Euclidean distances in PCA space directly correspond to data variance, matching the assumptions of standard t-SNE.
- **scVI-based t-SNE**, however, is a highly non-linear probabilistic space. Imposing a linear PCA initialization on scVI latents conflicts with its topological structure, resulting in a significantly degraded CPD of **0.377** (and 0.310 in multiscale mode). Thus, the initial hypothesis is rejected under this specific workflow: a better latent space does not guarantee a better embedding if the reduction algorithm's assumptions (linearity, Euclidean distance) are mismatched.

### 5.2 The Solution: Scanpy Workflow vs. Kobak Array-based Workflow
To properly visualize scVI, we introduced a second evaluation track (**scVI Scanpy Workflow**):
1. **Kobak Array-Based t-SNE/UMAP:** Feeds the raw 50D latent coordinates directly into embedding algorithms using Euclidean distance. Highly optimal for PCA, but toxic for non-linear generative models.
2. **Scanpy Graph-Based Workflow:** Computes a K-Nearest Neighbor (KNN) affinity graph (`sc.pp.neighbors`) on the scVI latent space *prior* to embedding, mapping the non-linear manifold topologically. UMAP is then run directly on this graph (`sc.tl.umap`).

*(Note on UMAP vs t-SNE for scVI: The scverse ecosystem predominantly recommends graph-based UMAP. Standard t-SNE implementations calculate affinities directly from the raw Euclidean array, making it less natural and rarely used in tandem with pre-computed non-linear KNN graphs)*

### 5.3 Metric Definitions
To avoid subjective visual bias, embeddings are scored using:
- **KNN (K-Nearest Neighbors Preservation):** The fraction of local neighborhoods preserved from the high-dimensional space into the 2D projection. (Measures local structure).
- **KNC (K-Nearest Class Purity):** The fraction of nearest neighbors that belong to the same biological class/cluster. (Measures cluster coherence).
- **CPD (Correlative Preservation of Distances):** Spearman correlation ($\rho$) between pairwise distances in the original space and 2D space. (Measures global structure preservation).

## 5.4 Tasic et al. 2018 — paper context and why t-SNE was used

The Tasic et al. 2018 study is a large single-cell transcriptomic survey of mouse cortical neurons (VISp and ALM regions) aimed at generating a detailed taxonomy of cortical cell types rather than explicitly tracing developmental trajectories. The primary goals were to
- characterize cell types and marker genes, and
- build a reproducible reference atlas of cortical cell taxonomy.

Because the paper's main goal is classifying and visualizing discrete cell types and their relationships, t-SNE (with PCA preprocessing) is a natural choice in that work: t-SNE excels at producing visually separable clusters which helps in identifying and labeling cell types. t-SNE with PCA initialization further helps maintain a coarse global arrangement (inter-cluster relationships) while emphasizing local cluster structure.

If your downstream analysis goal were different (e.g., inferring continuous developmental trajectories or preserving long-range manifold geometry), then representations and visualization algorithms that emphasize global topology (diffusion maps, PHATE, or PCA-based summaries) would be more appropriate.

## 5.5 Choosing a representation and embedding based on the analysis goal
Pick the latent and visualization method that matches your biological question:

- **Cell type identification / marker discovery (clustering focus):** prioritize local structure and cluster purity — use UMAP (graph-based) or t-SNE; for scVI latents prefer the Scanpy graph-based UMAP workflow to compute KNN on the learned manifold first.
- **Trajectory / continuous processes (global topology):** prioritize global structure — use PCA-derived summaries, diffusion maps, or PHATE. If using t-SNE, use PCA initialization and tune multi-scale perplexity to better preserve global relations.
- **Differential neighborhood testing / local cell–cell comparisons:** use representations and embeddings that preserve local neighbor ranks (Trustworthiness, KNN metrics) and validate with multiple metrics.

### Quick guide
- If you want visually crisp clusters for annotation: `sc.pp.neighbors(use_rep=...)` -> `sc.tl.umap()` (graph-based). 
- If you want to preserve global distances/trajectories: run PCA or diffusion map first, then visualize (PCA + t-SNE with PCA init or PHATE).

## 6. Work Log (Updated 2026-05-26)
- Implemented PCA-initialized t-SNE benchmark variants in the pipeline.
- Fully integrated scVI deep generative baseline tests.
- Discovered that PCA initialization severely limits highly non-linear scVI latents (CPD scores dropped from 0.578 in PCA to 0.377 in scVI).
- Introduced a dedicated **Scanpy Graph-based UMAP Workflow** for scVI to properly resolve the latent manifold prior to 2D projection.
- Configured `.gitignore` to track lightweight metrics and high-res visualizations while excluding heavy `.h5ad` models.
- Integrated multi-panel automatic drawing for both Kobak variants (6-panel) and Scanpy convention tests via Nextflow branching.

---
*Developed for application to the Kobak Lab PhD Program and reproducible research in single-cell benchmarking.*
