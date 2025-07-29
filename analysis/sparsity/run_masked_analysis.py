#!/usr/bin/env python3
"""
Run sparsity analysis on masked embeddings.
This script creates synthetic masked embeddings for testing the analysis pipeline.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict
import argparse


def create_synthetic_masked_embeddings(epoch: int = 4, output_dir: str = 'analysis/dim_reduction/embeddings'):
    """Create synthetic masked embeddings for testing analysis."""
    
    # Load existing embeddings
    epoch_dir = Path(output_dir) / f"ijepa-64px-epoch={epoch:02d}"
    val_dir = epoch_dir / 'val'
    
    if not val_dir.exists():
        print(f"Error: No embeddings found at {val_dir}")
        return False
    
    # Load original embeddings
    student_emb = np.load(val_dir / 'student_embeddings_pooled.npy')
    teacher_emb = np.load(val_dir / 'teacher_embeddings_pooled.npy')
    labels = np.load(val_dir / 'labels.npy')
    
    print(f"Loaded embeddings: student shape={student_emb.shape}, teacher shape={teacher_emb.shape}")
    
    # Check if masked embeddings already exist
    masked_dir = epoch_dir / 'masked' / 'val' / 'ijepa_75'
    if masked_dir.exists():
        print(f"Masked embeddings already exist at {masked_dir}")
        return True
        
    # Create masked directory
    masked_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 5 mask samples
    num_mask_samples = 5
    samples_per_mask = min(500, len(labels))
    
    for mask_idx in range(num_mask_samples):
        mask_sample_dir = masked_dir / f"mask_{mask_idx:02d}"
        mask_sample_dir.mkdir(exist_ok=True)
        
        # Sample indices
        indices = np.random.choice(len(labels), samples_per_mask, replace=False)
        
        # Create masked student embeddings (simulate masking effect)
        # Apply random scaling to simulate effect of masking
        mask_effect = np.random.uniform(0.7, 0.9, size=(samples_per_mask, student_emb.shape[1]))
        masked_student = student_emb[indices] * mask_effect
        
        # Teacher sees unmasked input
        masked_teacher = teacher_emb[indices]
        
        # Save embeddings
        np.save(mask_sample_dir / 'student_embeddings_masked_pooled.npy', masked_student)
        np.save(mask_sample_dir / 'teacher_embeddings_pooled.npy', masked_teacher)
        np.save(mask_sample_dir / 'labels.npy', labels[indices])
        
        # Create synthetic mask pattern (8x8 grid)
        mask_pattern = np.random.choice([True, False], size=(8, 8), p=[0.25, 0.75])
        np.save(mask_sample_dir / 'mask_pattern.npy', mask_pattern)
        
        print(f"  Created mask sample {mask_idx}: {samples_per_mask} samples")
    
    # Save metadata
    metadata = {
        'mask_type': 'ijepa',
        'num_mask_samples': num_mask_samples,
        'samples_per_mask': samples_per_mask,
        'mask_results': {
            f'mask_{i:02d}': {
                'mask_ratio': 0.75,
                'num_masked_patches': int(64 * 0.75),
                'num_visible_patches': int(64 * 0.25),
                'total_samples': samples_per_mask,
                'embedding_dim': student_emb.shape[1]
            }
            for i in range(num_mask_samples)
        }
    }
    
    with open(masked_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nCreated synthetic masked embeddings in: {masked_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run sparsity analysis on masked embeddings')
    parser.add_argument('--create_synthetic', action='store_true',
                       help='Create synthetic masked embeddings for testing')
    parser.add_argument('--epoch', type=int, default=4,
                       help='Epoch to analyze')
    parser.add_argument('--embeddings_dir', type=str,
                       default='analysis/dim_reduction/embeddings',
                       help='Directory containing embeddings')
    
    args = parser.parse_args()
    
    if args.create_synthetic:
        success = create_synthetic_masked_embeddings(args.epoch, args.embeddings_dir)
        if not success:
            return
    
    # Run sparsity analysis with masked embeddings
    import sys
    sys.path.append(str(Path(__file__).parent))
    from sparsity_analysis import SparsityAnalyzer
    
    print("\n" + "="*60)
    print("Running Sparsity Analysis with Masked Embeddings")
    print("="*60)
    
    analyzer = SparsityAnalyzer(args.embeddings_dir)
    
    # Analyze single epoch with masked embeddings
    results = analyzer.analyze_epoch(args.epoch, split='val', analyze_masked=True)
    
    # Save results
    output_file = analyzer.output_dir / f'masked_sparsity_epoch_{args.epoch}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    if results.get('masked_analysis'):
        masked = results['masked_analysis']
        print(f"\nMasked Analysis Summary (averaged over {masked['n_mask_samples']} masks):")
        print("-" * 50)
        
        metrics = ['l1_sparsity', 'gini_coefficient', 'activation_ratio']
        
        print("\nStudent Network:")
        print("  Metric          | Unmasked → Masked  | Change")
        print("  " + "-"*45)
        for metric in metrics:
            unmasked = results['student'][metric]
            masked_val = masked['student'][metric]
            change = masked_val - unmasked
            print(f"  {metric:15s} | {unmasked:.4f} → {masked_val:.4f} | {change:+.4f}")
        
        print("\nTeacher Network (sees unmasked input):")
        print("  Metric          | Value")
        print("  " + "-"*25)
        for metric in metrics:
            teacher_val = masked['teacher'][metric]
            print(f"  {metric:15s} | {teacher_val:.4f}")
        
        print(f"\nActivation Correlation: {masked['activation_correlation']['mean']:.4f} ± {masked['activation_correlation']['std']:.4f}")
        
        print("\nVariance across masks (std):")
        for metric in metrics:
            student_std = masked['mask_variance'][metric]['student_std']
            teacher_std = masked['mask_variance'][metric]['teacher_std']
            print(f"  {metric:15s} | Student: {student_std:.4f} | Teacher: {teacher_std:.4f}")


if __name__ == "__main__":
    main()