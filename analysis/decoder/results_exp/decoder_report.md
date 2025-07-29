# Decoder Analysis Report: I-JEPA Embeddings Across 10 Epochs

## Executive Summary

This report presents decoder analysis results for I-JEPA embeddings extracted from 10 training epochs (0-9). Both linear (logistic regression) and non-linear (neural network) decoders were trained with α=0.001 regularization and a 50/50 train-test split to evaluate the quality of learned representations.

## Key Findings

### 1. Overall Performance (50/50 split)
- **Linear Decoders**:
  - Student (masked): 0.174 ± 0.018 average accuracy
  - Teacher: 0.188 ± 0.014 average accuracy
  
- **Non-linear Decoders**: 
  - Student (masked): 0.155 ± 0.021 average accuracy  
  - Teacher: 0.176 ± 0.028 average accuracy

### 2. Linear vs Non-linear Performance
- Linear decoders outperform non-linear decoders for student embeddings (+1.9%)
- Teacher embeddings show similar performance (1.2% difference)
- The 50/50 split reduced overfitting in non-linear models

### 3. Student vs Teacher Comparison
- Teacher embeddings consistently outperform student:
  - Linear: +1.4% advantage
  - Non-linear: +2.1% advantage
- The gap is small, indicating effective masked learning

### 4. Training Dynamics
- Performance peaks around epochs 4-6 for both networks
- Late epochs (7-9) show performance decline
- This suggests possible overfitting to the pretext task

## Detailed Analysis by Epoch

### Early Training (Epochs 0-2)
- Linear decoder accuracy: 16-19%
- Student and teacher perform similarly
- Basic feature learning phase

### Mid Training (Epochs 3-6)
- Best performance achieved:
  - Epoch 5: Teacher linear 21.2%
  - Epoch 6: Student linear 20.0%
- Most discriminative representations

### Late Training (Epochs 7-9)
- Notable performance drop:
  - Epoch 7: Student linear drops to 15.3%
  - Epoch 8: Lowest overall performance
- Possible representation collapse or overfitting

## Comparison: 50/50 vs 70/30 Split

| Metric | 70/30 Split | 50/50 Split | Difference |
|--------|-------------|-------------|------------|
| Linear Student | 19.9% | 17.4% | -2.5% |
| Linear Teacher | 21.2% | 18.8% | -2.4% |
| NN Student | 15.3% | 15.5% | +0.2% |
| NN Teacher | 16.8% | 17.6% | +0.8% |

The 50/50 split:
- Reduces linear decoder performance (less training data)
- Slightly improves non-linear decoder performance (less overfitting)
- Maintains relative ordering of methods

## Technical Details

- **Regularization**: α = 0.001 (C = 1000 for sklearn)
- **Train/Test Split**: 50/50 stratified
- **Decoder Architecture**:
  - Linear: Multinomial logistic regression with L2 penalty
  - Non-linear: MLP with 128 hidden units, early stopping
- **Evaluation**: 5 mask samples per epoch, averaged
- **Dataset**: 200 classes, 256-dimensional embeddings

## Statistical Analysis

### Performance Stability
- Standard deviation across masks:
  - Student: 0.006-0.034 (higher variance)
  - Teacher: 0.000 (no variance - same input)
  
### Epoch-wise Correlation
- Student-Teacher accuracy correlation: r = 0.71
- Strong positive correlation indicates consistent learning

## Visualizations

The analysis generated two key plots:
1. **decoder_evolution.png**: Shows performance trends across epochs
2. **decoder_summary.png**: Compares average performance across all conditions

## Conclusions

1. **Effective Masked Learning**: Student achieves 92% of teacher performance despite masking

2. **Linear Separability**: Embeddings are well-suited for linear classification

3. **Optimal Training Window**: Best representations emerge at epochs 4-6

4. **Representation Degradation**: Late training may hurt downstream performance

## Recommendations

1. **Early Stopping**: Consider stopping at epoch 5-6 for best downstream performance
2. **Architecture Search**: Test deeper non-linear decoders with dropout
3. **Multi-task Evaluation**: Assess on tasks beyond classification
4. **Masking Strategy**: Analyze impact of different masking ratios

## Summary

The decoder analysis confirms that I-JEPA learns meaningful representations throughout training, with student networks successfully learning from masked inputs. The 50/50 split provides a more conservative estimate of performance while reducing overfitting in non-linear models. The optimal checkpoint for downstream tasks appears to be around epoch 5-6.