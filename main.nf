#!/usr/bin/env nextflow

/*
 * scRNA-seq Representation Benchmark Pipeline
 * Matches Dmitry Kobak's t-SNE methodology while introducing Deep Generative Models.
 */

nextflow.enable.dsl=2

// Pipeline Parameters (can be overridden by configs or CLI)
params.data_dir = "${projectDir}/data"
params.results_dir = "${projectDir}/results"
params.python_cmd = params.python_cmd ?: 'conda run -n tasic_benchmark python'
params.methods = params.methods ?: ['pca', 'scvi']
params.n_hvg = params.n_hvg ?: (params.preprocessing?.n_hvg ?: 3000)
params.latent_dim = params.latent_dim ?: (params.latent_models?.latent_dim ?: 50)

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
    ${params.python_cmd} "${projectDir}/scripts/fetch_tasic.py" \
        --outdir "${params.data_dir}/raw" \
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
    mkdir -p "${params.data_dir}"
    ${params.python_cmd} "${projectDir}/scripts/preprocessing.py" \
        --input "${raw_h5ad}" \
        --output tasic_preprocessed.h5ad \
        --n_hvg ${params.n_hvg} \
        --marker_genes '${(params.preprocessing?.marker_genes ?: []).join(',')}'
    """
}

// -------------------------------------------------------------
// 2b. Batch Explore (Optional)
// -------------------------------------------------------------
process BATCH_EXPLORE {
    publishDir "${params.results_dir}/batch_explore", mode: 'copy'

    input:
    path preprocessed_h5ad

    output:
    path "*.png"
    path "*.pdf"

    script:
    """
    ${params.python_cmd} "${projectDir}/scripts/batch_explore.py" \
        --input "${preprocessed_h5ad}" \
        --outdir . \
        --batch_keys sample_prefix1 sample_prefix2
    """
}

// -------------------------------------------------------------
// 3. Latent Representation (Parallel by Method)
// -------------------------------------------------------------
process LATENT_MODEL {
    publishDir "${params.results_dir}/embeddings", mode: 'copy'

    input:
    tuple val(method), val(dim), path(preprocessed_h5ad)

    output:
    tuple val(method), val(dim), path("tasic_latent_${method}_n${dim}.h5ad"), emit: latent_h5ad

    script:
    def scvi_args = method == 'scvi' ? [
        "--max_epochs ${params.latent_models?.scvi?.max_epochs ?: 100}",
        params.latent_models?.scvi?.early_stopping != null ? "--early_stopping ${params.latent_models.scvi.early_stopping}" : "",
        params.latent_models?.scvi?.patience ? "--patience ${params.latent_models.scvi.patience}" : "",
        "--batch_size ${params.latent_models?.scvi?.batch_size ?: 128}",
        "--learning_rate ${params.latent_models?.scvi?.learning_rate ?: 0.001}",
        params.latent_models?.scvi?.batch_key ? "--batch_key ${params.latent_models.scvi.batch_key}" : "",
        "--n_layers ${params.latent_models?.scvi?.n_layers ?: 2}",
        "--n_hidden ${params.latent_models?.scvi?.n_hidden ?: 128}",
        "--dropout_rate ${params.latent_models?.scvi?.dropout_rate ?: 0.1}",
        "--gene_likelihood ${params.latent_models?.scvi?.gene_likelihood ?: 'zinb'}",
        "--layer ${params.latent_models?.scvi?.layer ?: 'counts'}",
        params.latent_models?.scvi?.dispersion ? "--dispersion ${params.latent_models.scvi.dispersion}" : ""
    ].findAll { it }.join(" \\\n+        ") : ""
    def scvi_args_block = scvi_args ? "${scvi_args} \\\n+        " : ""

    """
    ${params.python_cmd} "${projectDir}/scripts/latent_model.py" \
        --input "${preprocessed_h5ad}" \
        --output tasic_latent_${method}_n${dim}.h5ad \
        --method ${method} \
        --dim ${dim} \
        ${scvi_args_block}--kobak_outdir "${params.results_dir}/embeddings/kobak_npy"
    """
}

// -------------------------------------------------------------
// 4. 2D Embedding (t-SNE & UMAP)
// -------------------------------------------------------------
process EMBEDDING {
    tag "Embed: ${method}:${dim} | Kobak variants"
    publishDir "${params.results_dir}/embeddings", mode: 'copy'

    input:
    tuple val(method), val(dim), path(latent_h5ad)

    output:
    tuple val(method), val(dim), path("tasic_embedded_${method}_n${dim}.h5ad"), emit: embedded_h5ad
    script:
    """
    ${params.python_cmd} "${projectDir}/scripts/embedding.py" \
        --input "${latent_h5ad}" \
        --output tasic_embedded_${method}_n${dim}.h5ad \
        --method ${method} \
        --dim ${dim} \
        --tsne_variants '${groovy.json.JsonOutput.toJson(params.embed_tsne?.variants ?: [])}' \
        --umap_variants '${groovy.json.JsonOutput.toJson(params.embed_umap?.variants ?: [])}' \
        --kobak_outdir "${params.results_dir}/embeddings/kobak_npy"
    """
}

// -------------------------------------------------------------
// 5. Evaluation (KNN, KNC, CPD)
// -------------------------------------------------------------
process EVALUATION {
    tag "Eval: ${method}:${dim} embeddings"
    publishDir "${params.results_dir}/metrics", mode: 'copy'

    input:
    tuple val(method), val(dim), path(embedded_h5ad)

    output:
    tuple val(method), val(dim), path(embedded_h5ad), path("metrics_${method}_n${dim}.json"), emit: metrics_tuple
    script:
    """
    ${params.python_cmd} "${projectDir}/scripts/evaluation.py" \
        --input "${embedded_h5ad}" \
        --method ${method} \
        --outdir . \
        --suffix _n${dim}
    """
}

// -------------------------------------------------------------
// 6. Visualization (Scatter Plots + Scores)
// -------------------------------------------------------------
process VISUALIZATION {
    tag "Plot: ${method}:${dim} embeddings"
    publishDir "${params.results_dir}/figures", mode: 'copy'

    input:
    tuple val(method), val(dim), path(embedded_h5ad), path(metric_json)

    output:
    path "${method}/n${dim}/*"
    script:
    """
    outdir="${method}/n${dim}"
    mkdir -p "\$outdir"
    ${params.python_cmd} "${projectDir}/scripts/visualization.py" \
        --input "${embedded_h5ad}" \
        --method ${method} \
        --metrics "${metric_json}" \
        --outdir "\$outdir" \
        --suffix ""
    """
}

// -------------------------------------------------------------
// Workflow Execution
// -------------------------------------------------------------
workflow {
    // 1. Download
    def raw_h5ad_file = file("${params.data_dir}/tasic_raw.h5ad")
    if( raw_h5ad_file.exists() ) {
        log.info "Found existing raw dataset at ${raw_h5ad_file}; skipping FETCH_DATA."
        raw_data = Channel.value(raw_h5ad_file)
    }
    else {
        raw_data = FETCH_DATA()
    }

    // 2. Preprocess
    prepped_data = PREPROCESS(raw_data)

    // 2b. Batch Explore (optional)
    if (params.run_batch_explore) {
        BATCH_EXPLORE(prepped_data)
    }

    // 3. Create a channel for methods to branch parallel execution
    def methods_list
    if( params.methods instanceof List ) {
        methods_list = params.methods
    }
    else if( params.methods instanceof String ) {
        methods_list = params.methods.split(',').collect { it.trim().toLowerCase() }.findAll { it }
    }
    else {
        methods_list = ['pca', 'scvi']
    }

    // Expand methods into entries; for scvi expand over configured n_latent_list
    def method_entries = []
    methods_list.each { m ->
        if(m == 'scvi' && params.latent_models?.scvi?.n_latent_list) {
            params.latent_models.scvi.n_latent_list.each { nl ->
                method_entries << [m, nl]
            }
        }
        else {
            method_entries << [m, params.latent_dim]
        }
    }

    // Build a channel from the method entries and convert each entry to a Nextflow Tuple
    // to avoid List (java.util.LinkedList) items which cause process invocation errors.
    // Create methods channel as Tuple items
    methods_ch = Channel.fromList(method_entries).map { e -> tuple(e[0], e[1]) }

    // Pair methods with the preprocessed h5ad to feed the latent model.
    // combine() performs a Cartesian product and flattens tuple items.
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
