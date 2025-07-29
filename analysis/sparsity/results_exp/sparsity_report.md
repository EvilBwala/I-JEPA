# Sparsity Analysis Report: I-JEPA Embeddings Across 10 Epochs

## Executive Summary

This report presents a comprehensive sparsity analysis of I-JEPA embeddings extracted from 10 training epochs (epochs 0-9) in the exp folder. The analysis focuses on comparing student embeddings (with I-JEPA masking applied) against teacher embeddings across multiple sparsity metrics.

## Key Findings

### 1. Overall Sparsity Trends
- **L1 Sparsity**: Both student and teacher networks show moderate sparsity levels
  - Student (masked): 0.286 ± 0.034
  - Teacher: 0.290 ± 0.038
  - Both networks exhibit increasing sparsity from epoch 0 to epoch 4, then stabilize

### 2. Activation Patterns
- **Activation Ratio**: Shows significant decrease over training
  - Initial (epoch 0): ~62% of neurons significantly active
  - Final (epoch 9): ~19-22% of neurons significantly active
  - This indicates progressive specialization of representations

### 3. Dead Neurons
- **Zero dead neurons detected** across all epochs for both networks
- All embedding dimensions remain active throughout training
- This suggests effective initialization and training dynamics

### 4. Gini Coefficient
- Moderate inequality in activation magnitudes
  - Student: 0.472 ± 0.020
  - Teacher: 0.476 ± 0.025
  - Values increase slightly during training, indicating growing specialization

## Detailed Analysis by Epoch

### Early Training (Epochs 0-2)
- Lower sparsity levels (L1: 0.21-0.27)
- High activation ratios (35-62%)
- Networks learn broad, distributed representations

### Mid Training (Epochs 3-5)
- Peak sparsity achieved (L1: 0.30-0.33)
- Significant drop in activation ratio (16-27%)
- Emergence of specialized feature detectors

### Late Training (Epochs 6-9)
- Stabilized sparsity levels (L1: 0.30-0.32)
- Low, stable activation ratios (15-24%)
- Consolidated sparse representations

## Impact of Masking

The analysis reveals minimal differences between student (masked) and teacher embeddings:
- Average L1 sparsity difference: -0.003
- Average Gini coefficient difference: -0.003
- Average activation ratio difference: +0.001

This suggests that the student network successfully learns to produce similar sparse representations despite receiving masked inputs.

## Technical Details

- **Dataset**: Tiny-ImageNet validation set
- **Masking Strategy**: I-JEPA block masking (M=4 blocks, 15-20% scale)
- **Samples**: 500 samples per mask × 5 masks × 10 epochs = 25,000 total samples
- **Embedding Dimension**: 256
- **Patch Configuration**: 8×8 patches of size 8×8 pixels (64×64 images)

## Conclusions

1. **Successful Sparse Learning**: Both networks develop increasingly sparse representations during training
2. **No Dead Neurons**: The architecture maintains all 256 dimensions active
3. **Masking Robustness**: Student network produces comparable sparsity despite masked inputs
4. **Progressive Specialization**: Clear trend from distributed to specialized representations

## Recommendations for Future Work

1. Investigate the relationship between sparsity and downstream task performance
2. Analyze sparsity patterns at the patch level (before pooling)
3. Compare with other self-supervised methods (MAE, DINO)
4. Study the impact of different masking ratios on sparsity evolution