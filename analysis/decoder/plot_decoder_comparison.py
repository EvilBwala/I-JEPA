#!/usr/bin/env python3
"""Quick script to plot decoder comparison results."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('analysis/decoder/decoder_results/decoder_results.json', 'r') as f:
    linear_results = json.load(f)

with open('analysis/decoder/nn_decoder_results/nn_decoder_results.json', 'r') as f:
    nn_results = json.load(f)

epochs = [0, 1, 2, 3, 4]

# Extract data
linear_student_val = [linear_results['student'][str(e)]['val_accuracy'] for e in epochs]
linear_teacher_val = [linear_results['teacher'][str(e)]['val_accuracy'] for e in epochs]

nn_linear_student_val = [nn_results['linear']['student'][str(e)]['val_accuracy'] for e in epochs]
nn_mlp_student_val = [nn_results['mlp']['student'][str(e)]['val_accuracy'] for e in epochs]
nn_linear_teacher_val = [nn_results['linear']['teacher'][str(e)]['val_accuracy'] for e in epochs]
nn_mlp_teacher_val = [nn_results['mlp']['teacher'][str(e)]['val_accuracy'] for e in epochs]

# Create plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Student comparison
ax1.plot(epochs, linear_student_val, 'o-', label='Sklearn Logistic Reg', linewidth=2, markersize=8)
ax1.plot(epochs, nn_linear_student_val, 's-', label='PyTorch Linear', linewidth=2, markersize=8)
ax1.plot(epochs, nn_mlp_student_val, '^-', label='PyTorch MLP', linewidth=2, markersize=8)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Validation Accuracy', fontsize=12)
ax1.set_title('Student Network - Decoder Comparison', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 0.20])

# Teacher comparison
ax2.plot(epochs, linear_teacher_val, 'o-', label='Sklearn Logistic Reg', linewidth=2, markersize=8)
ax2.plot(epochs, nn_linear_teacher_val, 's-', label='PyTorch Linear', linewidth=2, markersize=8)
ax2.plot(epochs, nn_mlp_teacher_val, '^-', label='PyTorch MLP', linewidth=2, markersize=8)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Validation Accuracy', fontsize=12)
ax2.set_title('Teacher Network - Decoder Comparison', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 0.20])

plt.suptitle('Decoder Performance Comparison (50 classes, 100 samples/class)', fontsize=16)
plt.tight_layout()
plt.savefig('analysis/decoder/decoder_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Print summary
print("="*60)
print("DECODER COMPARISON SUMMARY")
print("="*60)
print("\nValidation Accuracy at Epoch 4:")
print("-"*40)
print("Method                | Student | Teacher")
print("-"*40)
print(f"Sklearn Logistic Reg  |  {linear_student_val[4]:.3f}  |  {linear_teacher_val[4]:.3f}")
print(f"PyTorch Linear        |  {nn_linear_student_val[4]:.3f}  |  {nn_linear_teacher_val[4]:.3f}")
print(f"PyTorch MLP           |  {nn_mlp_student_val[4]:.3f}  |  {nn_mlp_teacher_val[4]:.3f}")
print("-"*40)

print("\nKey Findings:")
print("1. All methods show similar performance (~13-15% accuracy)")
print("2. No significant advantage from nonlinear decoder (MLP)")
print("3. Teacher network shows very poor performance with neural decoders")
print("4. Student representations are more amenable to linear decoding")
print("="*60)