# Dimensionality Reduction Analysis

This folder contains all scripts and results for dimensionality reduction analysis of I-JEPA embeddings.

## Directory Structure

```
dim_reduction/
├── extract_embeddings.py          # Main embedding extraction script
├── reduce_embeddings.py           # Dimensionality reduction script
├── reduce_exp_embeddings.py       # Reduction for exp folder embeddings
├── visualize_embeddings.py        # Visualization script
├── extract_exp_embeddings.sh      # Batch extraction for exp folder
├── run_multi_mask_extraction.sh   # Extract with multiple masks
├── run_exp_dim_reduction.sh       # Batch reduction for exp folder
├── embeddings/                    # Extracted embeddings (checkpoint 4)
├── embeddings_exp/                # Extracted embeddings (all 10 epochs)
├── reduced_embeddings/            # Reduced embeddings (checkpoint 4)
├── reduced_embeddings_exp/        # Reduced embeddings (all 10 epochs)
└── embedding_plots/               # Visualization outputs
```

## Usage

### Extract embeddings from a checkpoint:
```bash
python extract_embeddings.py --checkpoint ../../checkpoints/ijepa-64px-epoch=04.ckpt --data_path ../../datasets/tiny-imagenet-200
```

### Extract embeddings from all exp folder checkpoints:
```bash
./extract_exp_embeddings.sh
```

### Run dimensionality reduction:
```bash
python reduce_embeddings.py --num_classes 50
```

### Run dimensionality reduction on exp folder:
```bash
python reduce_exp_embeddings.py --num_classes 50
```

## Note on Paths

- Scripts within this folder use relative paths
- Scripts in other analysis folders may use absolute paths from project root
- Both approaches work correctly due to Python's path resolution