# I-JEPA for TinyImageNet

This repository contains an implementation of the Image Joint-Embedding Predictive Architecture (I-JEPA) for self-supervised learning on the TinyImageNet dataset.

## Overview

I-JEPA is a self-supervised learning approach that learns representations by predicting the embeddings of image regions (targets) from the embeddings of other regions (contexts). This implementation is adapted for the TinyImageNet dataset, which consists of 200 classes with 500 training images and 50 validation images per class, all of size 64x64.

## Files

- `model_new.py`: Contains the I-JEPA model architecture and TinyImageNet dataset implementation
- `pretrain_IJEPA_new.py`: Script for pretraining the I-JEPA model on TinyImageNet
- `run_pretrain.py`: Helper script for running pretraining with predefined configurations
- `eval_IJEPA_new.py`: Script for evaluating a pretrained model

## Requirements

- PyTorch
- PyTorch Lightning
- x-transformers
- einops
- scikit-learn (for evaluation)
- matplotlib (for visualization)
- tqdm

You can install the requirements with:

```bash
pip install torch torchvision pytorch-lightning x-transformers einops scikit-learn matplotlib tqdm
```

## Dataset

The TinyImageNet dataset should be placed in the `tiny-imagenet-200` directory. The dataset can be downloaded from [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip).

## Pretraining

### Using the run_pretrain.py script

The easiest way to pretrain the model is to use the `run_pretrain.py` script, which provides predefined configurations:

```bash
# For a small model (faster training, lower memory)
python run_pretrain.py small

# For a medium-sized model (balanced performance)
python run_pretrain.py medium

# For a large model (best performance, but slower and more memory-intensive)
python run_pretrain.py large

# For CPU training (very small model for testing)
python run_pretrain.py cpu
```

You can also pass additional arguments to the pretraining script:

```bash
python run_pretrain.py small --save_final_model --use_wandb
```

### Using pretrain_IJEPA_new.py directly

For more control, you can use the `pretrain_IJEPA_new.py` script directly:

```bash
python pretrain_IJEPA_new.py \
    --img_size 64 \
    --patch_size 8 \
    --embed_dim 192 \
    --enc_heads 8 \
    --enc_depth 12 \
    --decoder_depth 4 \
    --batch_size 32 \
    --lr 1e-3 \
    --max_epochs 50 \
    --checkpoint_dir checkpoints/custom \
    --save_final_model
```

Run `python pretrain_IJEPA_new.py --help` to see all available options.

## Evaluation

After pretraining, you can evaluate the model using the `eval_IJEPA_new.py` script:

```bash
python eval_IJEPA_new.py \
    --checkpoint checkpoints/medium/ijepa-64px-final.ckpt \
    --visualize \
    --evaluate_clustering
```

This will:
1. Load the pretrained model
2. Extract features from the validation set
3. Visualize the features using PCA and t-SNE (if `--visualize` is specified)
4. Evaluate clustering performance (if `--evaluate_clustering` is specified)

Run `python eval_IJEPA_new.py --help` to see all available options.

## Model Architecture

The I-JEPA model consists of:

- A student encoder that encodes context regions
- A teacher encoder (momentum-updated copy of the student) that encodes target regions
- A predictor that predicts target embeddings from context embeddings

The model is trained by minimizing the mean squared error between predicted and actual target embeddings.

## Hyperparameters

Key hyperparameters for the TinyImageNet implementation:

- Image size: 64x64 (native TinyImageNet size)
- Patch size: 8x8 (resulting in 8x8=64 patches per image)
- Embedding dimension: 192 (default for medium model)
- Encoder depth: 12 layers (default for medium model)
- Decoder depth: 4 layers (default for medium model)
- Learning rate: 1e-3
- Weight decay: 0.05
- Batch size: 32 (default for medium model)

## Acknowledgments

This implementation is based on the I-JEPA paper:
- [Joint-Embedding Predictive Architecture for Vision](https://arxiv.org/abs/2301.08243) by Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Florian Bordes, Pascal Vincent, Armand Joulin, Michael Rabbat, Nicolas Ballas 