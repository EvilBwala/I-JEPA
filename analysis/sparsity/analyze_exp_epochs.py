#!/usr/bin/env python3
"""
Analyze sparsity across all 10 epochs from exp folder embeddings.
"""

import numpy as np
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sys.path.append(str(Path(__file__).parent))
from sparsity_analysis import SparsityAnalyzer


def analyze_exp_epochs():
    """Analyze sparsity across all epochs in exp folder."""
    
    base_dir = Path('analysis/dim_reduction/embeddings_exp')
    output_dir = Path('analysis/sparsity/results_exp')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = SparsityAnalyzer(str(base_dir), str(output_dir))
    
    print("="*60)
    print("Sparsity Analysis of Exp Folder Embeddings (10 Epochs)")
    print("="*60)
    
    # Storage for results
    all_epochs_results = {}
    evolution_data = {
        'epochs': list(range(10)),
        'student': {
            'l1_sparsity': [],
            'gini_coefficient': [],
            'activation_ratio': [],
            'dead_ratio': []
        },
        'teacher': {
            'l1_sparsity': [],
            'gini_coefficient': [],
            'activation_ratio': [],
            'dead_ratio': []
        }
    }
    
    # Process each epoch
    for epoch in tqdm(range(10), desc="Processing epochs"):
        epoch_dir = base_dir / f"ijepa-64px-epoch={epoch:02d}" / "masked" / "val" / "ijepa_75"
        
        if not epoch_dir.exists():
            print(f"Warning: Directory not found for epoch {epoch}")
            continue
            
        print(f"\n--- Epoch {epoch} ---")
        
        # Collect data from all mask samples
        all_student_masked = []
        all_teacher = []
        
        mask_samples = sorted([d for d in epoch_dir.iterdir() if d.is_dir() and d.name.startswith('mask_')])
        
        for mask_dir in mask_samples[:5]:  # Use all 5 mask samples
            student_path = mask_dir / 'student_embeddings_masked_pooled.npy'
            teacher_path = mask_dir / 'teacher_embeddings_pooled.npy'
            
            if student_path.exists() and teacher_path.exists():
                all_student_masked.append(np.load(student_path))
                all_teacher.append(np.load(teacher_path))
        
        if not all_student_masked:
            print(f"No data found for epoch {epoch}")
            continue
            
        # Concatenate all samples
        student_masked = np.vstack(all_student_masked)
        teacher_emb = np.vstack(all_teacher)
        
        print(f"  Loaded {len(all_student_masked)} mask samples")
        print(f"  Total samples: {student_masked.shape[0]}")
        
        # Compute metrics
        epoch_results = {
            'student': {},
            'teacher': {},
            'n_samples': student_masked.shape[0],
            'n_mask_samples': len(all_student_masked)
        }
        
        # Student metrics (masked)
        epoch_results['student']['l1_sparsity'] = analyzer.compute_l1_sparsity(student_masked)
        epoch_results['student']['gini_coefficient'] = analyzer.compute_gini_coefficient(student_masked)
        epoch_results['student']['activation_ratio'] = analyzer.compute_activation_ratio(student_masked)
        epoch_results['student']['dead_neurons'] = analyzer.analyze_dead_neurons(student_masked)
        
        # Teacher metrics
        epoch_results['teacher']['l1_sparsity'] = analyzer.compute_l1_sparsity(teacher_emb)
        epoch_results['teacher']['gini_coefficient'] = analyzer.compute_gini_coefficient(teacher_emb)
        epoch_results['teacher']['activation_ratio'] = analyzer.compute_activation_ratio(teacher_emb)
        epoch_results['teacher']['dead_neurons'] = analyzer.analyze_dead_neurons(teacher_emb)
        
        # Store results
        all_epochs_results[f'epoch_{epoch}'] = epoch_results
        
        # Track evolution
        for network in ['student', 'teacher']:
            evolution_data[network]['l1_sparsity'].append(epoch_results[network]['l1_sparsity'])
            evolution_data[network]['gini_coefficient'].append(epoch_results[network]['gini_coefficient'])
            evolution_data[network]['activation_ratio'].append(epoch_results[network]['activation_ratio'])
            evolution_data[network]['dead_ratio'].append(epoch_results[network]['dead_neurons']['dead_ratio'])
        
        # Print summary for this epoch
        print(f"\n  Student (masked) metrics:")
        print(f"    L1 Sparsity: {epoch_results['student']['l1_sparsity']:.4f}")
        print(f"    Gini Coefficient: {epoch_results['student']['gini_coefficient']:.4f}")
        print(f"    Activation Ratio: {epoch_results['student']['activation_ratio']:.4f}")
        print(f"    Dead Neurons: {epoch_results['student']['dead_neurons']['dead_ratio']:.1%}")
        
        print(f"\n  Teacher metrics:")
        print(f"    L1 Sparsity: {epoch_results['teacher']['l1_sparsity']:.4f}")
        print(f"    Gini Coefficient: {epoch_results['teacher']['gini_coefficient']:.4f}")
        print(f"    Activation Ratio: {epoch_results['teacher']['activation_ratio']:.4f}")
        print(f"    Dead Neurons: {epoch_results['teacher']['dead_neurons']['dead_ratio']:.1%}")
    
    # Save all results
    with open(output_dir / 'exp_epochs_sparsity_analysis.json', 'w') as f:
        json.dump(all_epochs_results, f, indent=2)
    
    # Create visualizations
    create_evolution_plots(evolution_data, output_dir)
    
    # Print overall summary
    print("\n" + "="*60)
    print("Summary Across All Epochs")
    print("="*60)
    
    metrics = ['l1_sparsity', 'gini_coefficient', 'activation_ratio', 'dead_ratio']
    metric_names = ['L1 Sparsity', 'Gini Coefficient', 'Activation Ratio', 'Dead Neuron Ratio']
    
    for metric, name in zip(metrics, metric_names):
        print(f"\n{name}:")
        student_values = evolution_data['student'][metric]
        teacher_values = evolution_data['teacher'][metric]
        
        if student_values:
            print(f"  Student (masked): {np.mean(student_values):.4f} ± {np.std(student_values):.4f}")
            print(f"  Teacher:         {np.mean(teacher_values):.4f} ± {np.std(teacher_values):.4f}")
    
    return all_epochs_results, evolution_data


def create_evolution_plots(evolution_data, output_dir):
    """Create evolution plots for sparsity metrics across epochs."""
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics = ['l1_sparsity', 'gini_coefficient', 'activation_ratio', 'dead_ratio']
    titles = ['L1 Sparsity (Hoyer)', 'Gini Coefficient', 'Activation Ratio', 'Dead Neuron Ratio']
    
    epochs = evolution_data['epochs']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        # Plot student and teacher
        student_values = evolution_data['student'][metric]
        teacher_values = evolution_data['teacher'][metric]
        
        if student_values:  # Only plot if we have data
            ax.plot(epochs[:len(student_values)], student_values, 'o-', 
                   color='#1f77b4', linewidth=2, markersize=8, label='Student (masked)')
            ax.plot(epochs[:len(teacher_values)], teacher_values, 's-', 
                   color='#ff7f0e', linewidth=2, markersize=8, label='Teacher')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis to show all epochs
        ax.set_xticks(range(10))
    
    plt.suptitle('Sparsity Metrics Evolution Across 10 Epochs (Exp Folder)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'exp_epochs_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nEvolution plot saved to: {output_path}")


if __name__ == "__main__":
    analyze_exp_epochs()