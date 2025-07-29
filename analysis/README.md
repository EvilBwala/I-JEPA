# I-JEPA Analysis Code

This directory contains organized analysis code for studying I-JEPA representations.

## Directory Structure

```
analysis/
├── dim_reduction/       # Dimensionality reduction analysis
│   ├── extract_embeddings.py
│   ├── reduce_embeddings.py
│   ├── visualize_embeddings.py
│   ├── embeddings/      # Extracted embeddings (configurable)
│   ├── reduced_embeddings/  # Reduced embeddings (configurable)
│   └── embedding_plots/     # Visualization plots (configurable)
│
├── decoder/            # Decoder analysis
│   ├── run_decoder_analysis.py      # Logistic regression decoder
│   ├── run_nn_decoder_analysis.py   # Neural network decoder
│   ├── run_sgd_decoder_analysis.py  # SGD classifier decoder
│   ├── compare_decoder_approaches.py # Compare all approaches
│   ├── plot_decoder_comparison.py   # Quick plotting script
│   ├── decoder_results/             # Results from decoder analysis
│   ├── nn_decoder_results/          # Neural network results
│   └── sgd_decoder_results/         # SGD classifier results
│
├── rsa/                # Representational Similarity Analysis
│   ├── rsa_analysis.py              # Core RSA functions
│   ├── run_rsa.py                   # Run RSA analysis
│   └── rsa_results/                 # RSA results and plots
│
└── sparsity/           # Sparsity analysis (empty - for future work)
```

## Usage Examples

### 1. Extract Embeddings

```bash
# Extract embeddings from a checkpoint
python -m analysis.dim_reduction.extract_embeddings \
    --checkpoint path/to/checkpoint.ckpt \
    --data_path tiny-imagenet-200 \
    --output_dir analysis/dim_reduction/embeddings
```

### 2. Reduce Dimensionality

```bash
# Apply PCA, UMAP, and Laplacian Eigenmaps
python -m analysis.dim_reduction.reduce_embeddings \
    --embeddings_dir analysis/dim_reduction/embeddings \
    --output_dir analysis/dim_reduction/reduced_embeddings \
    --methods pca umap laplacian
```

### 3. Visualize Embeddings

```bash
# Create visualization plots
python -m analysis.dim_reduction.visualize_embeddings \
    --reduced_dir analysis/dim_reduction/reduced_embeddings \
    --output_dir analysis/dim_reduction/embedding_plots
```

### 4. Run Decoder Analysis

```bash
# Logistic regression decoder
python -m analysis.decoder.run_decoder_analysis \
    --embeddings_dir embeddings \
    --epochs 0 1 2 3 4 \
    --samples_per_class 100

# Neural network decoder
python -m analysis.decoder.run_nn_decoder_analysis \
    --embeddings_dir embeddings \
    --epochs 0 1 2 3 4

# SGD classifier (fast, full dataset)
python -m analysis.decoder.run_sgd_decoder_analysis \
    --embeddings_dir embeddings \
    --epochs 0 1 2 3 4
```

### 5. Run RSA Analysis

```bash
# From within the rsa directory
cd analysis/rsa
python run_rsa.py \
    --split val \
    --epochs 0 1 2 3 4 \
    --samples_per_class 50 \
    --embeddings_dir ../../embeddings
```

## Key Features

1. **Configurable Paths**: All scripts accept command-line arguments for input/output directories
2. **Default Paths**: Scripts have sensible defaults that follow the organized structure
3. **Modular Design**: Each analysis type is self-contained in its own directory
4. **Consistent Interface**: All scripts follow similar command-line patterns

## Dependencies

- PyTorch and torchvision (for extract_embeddings)
- scikit-learn (for decoder analysis)
- driada (for dimensionality reduction - optional)
- matplotlib, seaborn (for visualization)
- numpy, scipy (general computation)

## Notes

- The embeddings directory can be shared across analyses or specified separately
- All output directories are created automatically if they don't exist
- Results are saved as JSON files for easy loading and comparison
- Plots are saved as high-resolution PNG files