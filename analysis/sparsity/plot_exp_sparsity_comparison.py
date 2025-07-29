#!/usr/bin/env python3
"""
Create detailed comparison plots for sparsity analysis of exp folder embeddings.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_results():
    """Load the sparsity analysis results."""
    results_file = Path('analysis/sparsity/results_exp/exp_epochs_sparsity_analysis.json')
    with open(results_file, 'r') as f:
        return json.load(f)

def create_comparison_plots():
    """Create detailed comparison plots."""
    results = load_results()
    
    # Extract data
    epochs = list(range(10))
    metrics = ['l1_sparsity', 'gini_coefficient', 'activation_ratio']
    metric_names = ['L1 Sparsity', 'Gini Coefficient', 'Activation Ratio']
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Metric comparison plots
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx//2, idx%2]
        
        student_values = []
        teacher_values = []
        
        for epoch in epochs:
            if f'epoch_{epoch}' in results:
                student_values.append(results[f'epoch_{epoch}']['student'][metric])
                teacher_values.append(results[f'epoch_{epoch}']['teacher'][metric])
        
        # Plot with error bands
        ax.plot(epochs[:len(student_values)], student_values, 'o-', 
               color='#2E86AB', linewidth=2.5, markersize=10, label='Student (masked)', alpha=0.8)
        ax.plot(epochs[:len(teacher_values)], teacher_values, 's-', 
               color='#A23B72', linewidth=2.5, markersize=10, label='Teacher', alpha=0.8)
        
        # Add difference shading
        if len(student_values) == len(teacher_values):
            ax.fill_between(epochs[:len(student_values)], student_values, teacher_values, 
                           alpha=0.2, color='gray', label='Difference')
        
        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel(name, fontsize=14, fontweight='bold')
        ax.set_title(f'{name} Comparison', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(epochs)
    
    # 4. Difference plot
    ax = axes[1, 1]
    
    differences = {
        'L1 Sparsity': [],
        'Gini Coefficient': [],
        'Activation Ratio': []
    }
    
    for epoch in epochs:
        if f'epoch_{epoch}' in results:
            for metric, name in zip(metrics, metric_names):
                diff = results[f'epoch_{epoch}']['student'][metric] - results[f'epoch_{epoch}']['teacher'][metric]
                differences[name].append(diff)
    
    # Plot differences
    x = np.arange(len(epochs))
    width = 0.25
    
    for i, (name, values) in enumerate(differences.items()):
        offset = (i - 1) * width
        ax.bar(x[:len(values)] + offset, values, width, label=name, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Student - Teacher Difference', fontsize=14, fontweight='bold')
    ax.set_title('Metric Differences (Student - Teacher)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_xticks(epochs)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.suptitle('Sparsity Analysis: Student (Masked) vs Teacher Embeddings\nAcross 10 Training Epochs', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = Path('analysis/sparsity/results_exp/exp_sparsity_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {output_path}")
    
    # Create a summary statistics plot
    create_summary_stats_plot(results)

def create_summary_stats_plot(results):
    """Create a summary statistics plot."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Calculate mean and std across epochs
    metrics = ['l1_sparsity', 'gini_coefficient', 'activation_ratio']
    metric_names = ['L1 Sparsity', 'Gini Coefficient', 'Activation Ratio']
    
    student_means = []
    student_stds = []
    teacher_means = []
    teacher_stds = []
    
    for metric in metrics:
        student_values = []
        teacher_values = []
        
        for epoch in range(10):
            if f'epoch_{epoch}' in results:
                student_values.append(results[f'epoch_{epoch}']['student'][metric])
                teacher_values.append(results[f'epoch_{epoch}']['teacher'][metric])
        
        student_means.append(np.mean(student_values))
        student_stds.append(np.std(student_values))
        teacher_means.append(np.mean(teacher_values))
        teacher_stds.append(np.std(teacher_values))
    
    # Create grouped bar plot
    x = np.arange(len(metric_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, student_means, width, yerr=student_stds, 
                    label='Student (masked)', color='#2E86AB', alpha=0.8, capsize=10)
    bars2 = ax.bar(x + width/2, teacher_means, width, yerr=teacher_stds,
                    label='Teacher', color='#A23B72', alpha=0.8, capsize=10)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
    
    autolabel(bars1)
    autolabel(bars2)
    
    ax.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
    ax.set_title('Average Sparsity Metrics Across All Epochs\n(Error bars show standard deviation)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_path = Path('analysis/sparsity/results_exp/exp_sparsity_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved to: {output_path}")

if __name__ == "__main__":
    create_comparison_plots()