#!/usr/bin/env nextflow

/*
 * scRNA-seq Representation Benchmark Pipeline
 * Matches Dmitry Kobak's t-SNE methodology while introducing Deep Generative Models.
 */

nextflow.enable.dsl=2

// Pipeline Parameters (can be overridden by configs/params.yml or CLI)
params.data_dir = "${projectDir}/data"
params.results_dir = "${projectDir}/results"
params.methods = ['pca', 'ae', 'vae', 'scvi']
params.n_hvg = 3000

// -------------------------------------------------------------
// 1. Fetch & Parse Raw Tasic Data
// -------------------------------------------------------------
process FETCH_DATA {
    publishDir "${params.data_dir}", mode: 'copy'

    output:
    path "tasic_raw.h5ad", emit: raw_h5ad

    script:
    """
    mkdir -p "${params.data_dir}"
    python "${projectDir}/scripts/fetch_tasic.py" \\
        --outdir "${params.data_dir}/raw" \\
        --output tasic_raw.h5ad
    """
}

// -------------------------------------------------------------
// 2. Preprocessing (General)
// -------------------------------------------------------------
process PREPROCESS {
    publishDir "${params.data_dir}", mode: 'copy'

    input:
    path raw_h5ad

    output:
    path "tasic_preprocessed.h5ad", emit: preprocessed_h5ad

    script:
    """
    python "${projectDir}/scripts/preprocessing.py" \\
        --input "${raw_h5ad}" \\
        --output tasic_preprocessed.h5ad \\
        --n_hvg ${params.n_hvg}
    """
}

// -------------------------------------------------------------
// 3. Latent Representation (Parallel by Method)
// -------------------------------------------------------------
process LATENT_MODEL {
    tag "Latent: ${method}"
    publishDir "${params.results_dir}/embeddings", mode: 'copy'

    input:
    tuple val(method), path(preprocessed_h5ad)

    output:
    tuple val(method), path("tasic_latent_${method}.h5ad"), emit: latent_h5ad

    script:
    """
    python "${projectDir}/scripts/latent_model.py" \\
        --input "${preprocessed_h5ad}" \\
        --output tasic_latent_${method}.h5ad \\
        --method ${method} \\
        --dim 50
    """
}

// -------------------------------------------------------------
// 4. 2D Embedding (t-SNE & UMAP)
// -------------------------------------------------------------
process EMBEDDING {
    tag "Embed: ${method}"
    publishDir "${params.results_dir}/embeddings", mode: 'copy'

    input:
    tuple val(method), path(latent_h5ad)

    output:
    tuple val(method), path("tasic_embedded_${method}.h5ad"), emit: embedded_h5ad

    script:
    """
    python "${projectDir}/scripts/embedding.py" \\
        --input "${latent_h5ad}" \\
        --output tasic_embedded_${method}.h5ad \\
        --method ${method}
    """
}

// -------------------------------------------------------------
// 5. Evaluation (KNN, KNC, CPD)
// -------------------------------------------------------------
process EVALUATION {
    tag "Eval: ${method}"
    publishDir "${params.results_dir}/metrics", mode: 'copy'

    input:
    tuple val(method), path(embedded_h5ad)

    output:
    tuple val(method), path(embedded_h5ad), path("metrics_${method}.json"), emit: metrics_tuple

    script:
    """
    python "${projectDir}/scripts/evaluation.py" \\
        --input "${embedded_h5ad}" \\
        --method ${method} \\
        --outdir .
    """
}

// -------------------------------------------------------------
// 6. Visualization (Scatter Plots + Scores)
// -------------------------------------------------------------
process VISUALIZATION {
    tag "Plot: ${method}"
    publishDir "${params.results_dir}/figures", mode: 'copy'

    input:
    tuple val(method), path(embedded_h5ad), path(metric_json)

    output:
    path "fig_${method}_*.png"

    script:
    """
    python "${projectDir}/scripts/visualization.py" \\
        --input "${embedded_h5ad}" \\
        --method ${method} \\
        --metrics "${metric_json}" \\
        --outdir .
    """
}

// -------------------------------------------------------------
// Workflow Execution
// -------------------------------------------------------------
workflow {
    // 1. Download
    raw_data = FETCH_DATA()

    // 2. Preprocess
    prepped_data = PREPROCESS(raw_data)

    // 3. Create a channel for methods to branch parallel execution
    methods_ch = Channel.fromList(params.methods)
    
    // Combine each method with the prepped data: [ 'pca', data ], [ 'scvi', data ]...
    latent_input_ch = methods_ch.combine(prepped_data)

    // 4. Run Latent extraction in parallel
    latent_data = LATENT_MODEL(latent_input_ch)

    // 5. Run 2D Embeddings in parallel
    embedded_data = EMBEDDING(latent_data)

    // 6. Run Evaluations in parallel
    metrics_data = EVALUATION(embedded_data)

    // 7. Run Visualizations in parallel
    VISUALIZATION(metrics_data)
}
