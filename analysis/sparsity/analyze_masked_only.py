#!/usr/bin/env python3
"""
Analyze sparsity on existing masked embeddings only.
"""

import numpy as np
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from sparsity_analysis import SparsityAnalyzer


def main():
    # Setup analyzer
    analyzer = SparsityAnalyzer()
    
    # Load masked embeddings directly
    epoch = 4
    mask_dir = Path('analysis/dim_reduction/embeddings/ijepa-64px-epoch=04/masked/val/ijepa_75')
    
    print("="*60)
    print("Sparsity Analysis of Masked Embeddings")
    print("="*60)
    
    # Load all mask samples
    masked_data = {}
    mask_samples = sorted([d for d in mask_dir.iterdir() if d.is_dir() and d.name.startswith('mask_')])
    
    print(f"\nFound {len(mask_samples)} mask samples")
    
    for mask_sample_dir in mask_samples:
        mask_id = mask_sample_dir.name
        
        # Check if all required files exist
        required_files = [
            'student_embeddings_masked_pooled.npy',
            'teacher_embeddings_pooled.npy', 
            'labels.npy',
            'mask_pattern.npy'
        ]
        
        if all((mask_sample_dir / f).exists() for f in required_files):
            masked_data[mask_id] = {
                'student': np.load(mask_sample_dir / 'student_embeddings_masked_pooled.npy'),
                'teacher': np.load(mask_sample_dir / 'teacher_embeddings_pooled.npy'),
                'labels': np.load(mask_sample_dir / 'labels.npy'),
                'mask_pattern': np.load(mask_sample_dir / 'mask_pattern.npy')
            }
            print(f"  Loaded {mask_id}: {masked_data[mask_id]['student'].shape[0]} samples")
        else:
            print(f"  Skipping {mask_id}: missing files")
    
    # Analyze masked embeddings
    print("\nAnalyzing sparsity metrics...")
    masked_analysis = analyzer.analyze_masked_embeddings(masked_data)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS: Averaged over {} mask samples".format(masked_analysis['n_mask_samples']))
    print("="*60)
    
    print("\nStudent Network (with masking):")
    print("-"*40)
    for metric in ['l0_sparsity', 'l1_sparsity', 'gini_coefficient', 'activation_ratio']:
        value = masked_analysis['student'][metric]
        std = masked_analysis['mask_variance'][metric]['student_std']
        print(f"  {metric:20s}: {value:.4f} ± {std:.4f}")
    
    print("\nTeacher Network (no masking):")
    print("-"*40)
    for metric in ['l0_sparsity', 'l1_sparsity', 'gini_coefficient', 'activation_ratio']:
        value = masked_analysis['teacher'][metric]
        std = masked_analysis['mask_variance'][metric]['teacher_std']
        print(f"  {metric:20s}: {value:.4f} ± {std:.4f}")
    
    print("\nComparison (Teacher - Student):")
    print("-"*40)
    for metric in ['l0_sparsity', 'l1_sparsity', 'gini_coefficient', 'activation_ratio']:
        diff = masked_analysis['differences'][metric]
        print(f"  {metric:20s}: {diff['absolute']:+.4f} ({diff['relative']:+.1%})")
    
    print(f"\nActivation Correlation: {masked_analysis['activation_correlation']['mean']:.4f} ± {masked_analysis['activation_correlation']['std']:.4f}")
    
    # Additional statistics
    print("\nAdditional Statistics:")
    print("-"*40)
    print(f"  Dead neurons (Student): {masked_analysis['student']['dead_neurons']['n_dead']:.1f} / {masked_analysis['student']['dead_neurons']['n_total']}")
    print(f"  Dead neurons (Teacher): {masked_analysis['teacher']['dead_neurons']['n_dead']:.1f} / {masked_analysis['teacher']['dead_neurons']['n_total']}")
    print(f"  Lifetime sparsity (Student): {masked_analysis['student']['lifetime_sparsity_stats']['mean']:.4f}")
    print(f"  Lifetime sparsity (Teacher): {masked_analysis['teacher']['lifetime_sparsity_stats']['mean']:.4f}")
    
    # Save results
    output_file = analyzer.output_dir / f'masked_embeddings_analysis_epoch_{epoch}.json'
    with open(output_file, 'w') as f:
        json.dump(masked_analysis, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()