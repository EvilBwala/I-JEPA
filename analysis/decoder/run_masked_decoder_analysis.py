#!/usr/bin/env python3
"""
Run decoder analysis on masked embeddings to evaluate reconstruction quality.
Tests multiple mask realizations to assess robustness.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def load_masked_embeddings(mask_idx: int, base_dir: Path):
    """Load embeddings for a specific mask realization."""
    mask_dir = base_dir / f"mask_{mask_idx:02d}"
    
    data = {
        'student': np.load(mask_dir / 'student_embeddings_masked_pooled.npy'),
        'teacher': np.load(mask_dir / 'teacher_embeddings_pooled.npy'),
        'labels': np.load(mask_dir / 'labels.npy'),
        'mask_pattern': np.load(mask_dir / 'mask_pattern.npy')
    }
    
    # Calculate mask statistics
    mask_ratio = 1.0 - data['mask_pattern'].sum() / 64
    
    return data, mask_ratio


def train_decoder(X_train, y_train, X_test, y_test, C=1.0):
    """Train logistic regression decoder."""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train decoder
    decoder = LogisticRegression(C=C, max_iter=1000, random_state=42)
    decoder.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = decoder.score(X_train_scaled, y_train)
    test_acc = decoder.score(X_test_scaled, y_test)
    y_pred = decoder.predict(X_test_scaled)
    
    return {
        'decoder': decoder,
        'scaler': scaler,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'y_pred': y_pred
    }


def analyze_multiple_masks(num_masks=3, num_classes=50):
    """Analyze decoder performance across multiple mask realizations."""
    
    base_dir = Path("analysis/dim_reduction/embeddings/ijepa-64px-epoch=04/masked/val/ijepa_75")
    results_dir = Path("analysis/decoder/masked_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Storage for results
    all_results = {
        'mask_results': {},
        'summary': {}
    }
    
    print("="*60)
    print("Decoder Analysis on Masked Embeddings")
    print("="*60)
    
    # Analyze each mask
    for mask_idx in range(num_masks):
        print(f"\n--- Analyzing Mask {mask_idx} ---")
        
        try:
            # Load data
            data, mask_ratio = load_masked_embeddings(mask_idx, base_dir)
            
            # Filter to first num_classes
            class_mask = data['labels'] < num_classes
            data['student'] = data['student'][class_mask]
            data['teacher'] = data['teacher'][class_mask]
            data['labels'] = data['labels'][class_mask]
            
            print(f"Loaded {len(data['labels'])} samples ({num_classes} classes), mask ratio: {mask_ratio:.1%}")
            
            # Split data (80/20)
            n_samples = len(data['labels'])
            n_train = int(0.8 * n_samples)
            indices = np.random.permutation(n_samples)
            
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
            
            # Prepare datasets
            X_train_student = data['student'][train_idx]
            X_train_teacher = data['teacher'][train_idx]
            y_train = data['labels'][train_idx]
            
            X_test_student = data['student'][test_idx]
            X_test_teacher = data['teacher'][test_idx]
            y_test = data['labels'][test_idx]
            
            print(f"Train: {len(y_train)}, Test: {len(y_test)}")
            
            # Train decoders
            print("\nTraining decoders...")
            
            # Student decoder (masked)
            student_results = train_decoder(X_train_student, y_train, 
                                          X_test_student, y_test)
            
            # Teacher decoder (unmasked)
            teacher_results = train_decoder(X_train_teacher, y_train,
                                          X_test_teacher, y_test)
            
            # Store results
            mask_results = {
                'mask_ratio': mask_ratio,
                'n_samples': n_samples,
                'student': {
                    'train_acc': student_results['train_acc'],
                    'test_acc': student_results['test_acc']
                },
                'teacher': {
                    'train_acc': teacher_results['train_acc'],
                    'test_acc': teacher_results['test_acc']
                },
                'accuracy_gap': teacher_results['test_acc'] - student_results['test_acc']
            }
            
            all_results['mask_results'][f'mask_{mask_idx:02d}'] = mask_results
            
            print(f"\nResults for Mask {mask_idx}:")
            print(f"  Student (masked):   Train: {student_results['train_acc']:.3f}, Test: {student_results['test_acc']:.3f}")
            print(f"  Teacher (unmasked): Train: {teacher_results['train_acc']:.3f}, Test: {teacher_results['test_acc']:.3f}")
            print(f"  Accuracy Gap: {mask_results['accuracy_gap']:.3f}")
            
            # Save predictions for this mask
            np.save(results_dir / f'mask_{mask_idx:02d}_student_pred.npy', 
                    student_results['y_pred'])
            np.save(results_dir / f'mask_{mask_idx:02d}_teacher_pred.npy', 
                    teacher_results['y_pred'])
            np.save(results_dir / f'mask_{mask_idx:02d}_true_labels.npy', y_test)
            
        except Exception as e:
            print(f"Error processing mask {mask_idx}: {e}")
            continue
    
    # Compute summary statistics
    if all_results['mask_results']:
        student_accs = [r['student']['test_acc'] for r in all_results['mask_results'].values()]
        teacher_accs = [r['teacher']['test_acc'] for r in all_results['mask_results'].values()]
        gaps = [r['accuracy_gap'] for r in all_results['mask_results'].values()]
        
        all_results['summary'] = {
            'num_masks_analyzed': len(all_results['mask_results']),
            'student': {
                'mean_test_acc': np.mean(student_accs),
                'std_test_acc': np.std(student_accs),
                'min_test_acc': np.min(student_accs),
                'max_test_acc': np.max(student_accs)
            },
            'teacher': {
                'mean_test_acc': np.mean(teacher_accs),
                'std_test_acc': np.std(teacher_accs),
                'min_test_acc': np.min(teacher_accs),
                'max_test_acc': np.max(teacher_accs)
            },
            'accuracy_gap': {
                'mean': np.mean(gaps),
                'std': np.std(gaps),
                'min': np.min(gaps),
                'max': np.max(gaps)
            }
        }
        
        print("\n" + "="*60)
        print("SUMMARY (averaged over {} masks)".format(len(all_results['mask_results'])))
        print("="*60)
        print(f"\nStudent (masked embeddings):")
        print(f"  Test Accuracy: {all_results['summary']['student']['mean_test_acc']:.3f} ± {all_results['summary']['student']['std_test_acc']:.3f}")
        print(f"  Range: [{all_results['summary']['student']['min_test_acc']:.3f}, {all_results['summary']['student']['max_test_acc']:.3f}]")
        
        print(f"\nTeacher (unmasked embeddings):")
        print(f"  Test Accuracy: {all_results['summary']['teacher']['mean_test_acc']:.3f} ± {all_results['summary']['teacher']['std_test_acc']:.3f}")
        print(f"  Range: [{all_results['summary']['teacher']['min_test_acc']:.3f}, {all_results['summary']['teacher']['max_test_acc']:.3f}]")
        
        print(f"\nAccuracy Gap (Teacher - Student):")
        print(f"  Mean: {all_results['summary']['accuracy_gap']['mean']:.3f} ± {all_results['summary']['accuracy_gap']['std']:.3f}")
        print(f"  Range: [{all_results['summary']['accuracy_gap']['min']:.3f}, {all_results['summary']['accuracy_gap']['max']:.3f}]")
        
        # Create visualization
        create_visualization(all_results, results_dir)
    
    # Save results
    with open(results_dir / 'masked_decoder_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    
    return all_results


def create_visualization(results, output_dir):
    """Create visualization comparing decoder performance across masks."""
    
    # Extract data
    mask_names = list(results['mask_results'].keys())
    student_accs = [results['mask_results'][m]['student']['test_acc'] for m in mask_names]
    teacher_accs = [results['mask_results'][m]['teacher']['test_acc'] for m in mask_names]
    mask_ratios = [results['mask_results'][m]['mask_ratio'] for m in mask_names]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Accuracy comparison
    x = np.arange(len(mask_names))
    width = 0.35
    
    ax1.bar(x - width/2, student_accs, width, label='Student (masked)', alpha=0.8)
    ax1.bar(x + width/2, teacher_accs, width, label='Teacher (unmasked)', alpha=0.8)
    
    ax1.set_xlabel('Mask Sample')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Decoder Performance Across Mask Samples')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('mask_', 'M') for m in mask_names])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add mean lines
    ax1.axhline(y=results['summary']['student']['mean_test_acc'], 
                color='blue', linestyle='--', alpha=0.5, label='Student mean')
    ax1.axhline(y=results['summary']['teacher']['mean_test_acc'], 
                color='orange', linestyle='--', alpha=0.5, label='Teacher mean')
    
    # Plot 2: Mask ratio vs accuracy gap
    gaps = [results['mask_results'][m]['accuracy_gap'] for m in mask_names]
    
    ax2.scatter(mask_ratios, gaps, s=100, alpha=0.7)
    ax2.set_xlabel('Mask Ratio')
    ax2.set_ylabel('Accuracy Gap (Teacher - Student)')
    ax2.set_title('Masking Effect on Decoder Performance')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line if enough points
    if len(mask_ratios) > 2:
        z = np.polyfit(mask_ratios, gaps, 1)
        p = np.poly1d(z)
        ax2.plot(sorted(mask_ratios), p(sorted(mask_ratios)), "r--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'masked_decoder_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_dir / 'masked_decoder_comparison.png'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run decoder analysis on masked embeddings')
    parser.add_argument('--num_masks', type=int, default=3,
                       help='Number of mask samples to analyze')
    parser.add_argument('--num_classes', type=int, default=50,
                       help='Number of classes to use for classification')
    
    args = parser.parse_args()
    
    analyze_multiple_masks(num_masks=args.num_masks, num_classes=args.num_classes)