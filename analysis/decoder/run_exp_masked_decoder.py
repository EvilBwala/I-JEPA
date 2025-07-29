#!/usr/bin/env python3
"""
Run decoder analysis on exp folder masked embeddings across all epochs.
Handles the masked embedding directory structure properly.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def analyze_masked_epoch(epoch_dir: Path, decoder_type: str = 'linear', alpha: float = 0.001, test_size: float = 0.25):
    """Analyze decoder performance for a single epoch with multiple masks."""
    mask_results = {
        'student': [],
        'teacher': []
    }
    
    # Find all mask directories
    mask_dirs = sorted([d for d in epoch_dir.iterdir() if d.is_dir() and d.name.startswith('mask_')])
    
    for mask_dir in mask_dirs[:5]:  # Use all 5 masks
        # Load embeddings
        student_path = mask_dir / 'student_embeddings_masked_pooled.npy'
        teacher_path = mask_dir / 'teacher_embeddings_pooled.npy'
        labels_path = mask_dir / 'labels.npy'
        
        if not all(p.exists() for p in [student_path, teacher_path, labels_path]):
            continue
            
        student_emb = np.load(student_path)
        teacher_emb = np.load(teacher_path)
        labels = np.load(labels_path)
        
        # Train-test split
        X_student_train, X_student_test, y_train, y_test = train_test_split(
            student_emb, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        X_teacher_train, X_teacher_test, _, _ = train_test_split(
            teacher_emb, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Create decoders
        if decoder_type == 'linear':
            student_decoder = LogisticRegression(
                C=1/alpha,  # C is inverse of alpha
                max_iter=1000,
                solver='lbfgs',
                multi_class='multinomial',
                random_state=42,
                n_jobs=-1
            )
            teacher_decoder = LogisticRegression(
                C=1/alpha,
                max_iter=1000,
                solver='lbfgs',
                multi_class='multinomial',
                random_state=42,
                n_jobs=-1
            )
        else:  # neural network
            student_decoder = MLPClassifier(
                hidden_layer_sizes=(128,),
                alpha=alpha,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            )
            teacher_decoder = MLPClassifier(
                hidden_layer_sizes=(128,),
                alpha=alpha,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            )
        
        # Train and evaluate
        student_decoder.fit(X_student_train, y_train)
        student_acc = accuracy_score(y_test, student_decoder.predict(X_student_test))
        mask_results['student'].append(student_acc)
        
        teacher_decoder.fit(X_teacher_train, y_train)
        teacher_acc = accuracy_score(y_test, teacher_decoder.predict(X_teacher_test))
        mask_results['teacher'].append(teacher_acc)
    
    return mask_results


def main():
    """Run decoder analysis across all epochs."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run decoder analysis on exp folder embeddings')
    parser.add_argument('--test_size', type=float, default=0.25, 
                        help='Test set size (default: 0.25)')
    parser.add_argument('--alpha', type=float, default=0.001,
                        help='Regularization parameter (default: 0.001)')
    parser.add_argument('--base_dir', type=str, default='analysis/dim_reduction/embeddings_exp',
                        help='Base directory for embeddings')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    # Storage for results
    all_results = {
        'linear': {'epochs': [], 'student_mean': [], 'student_std': [], 'teacher_mean': [], 'teacher_std': []},
        'nonlinear': {'epochs': [], 'student_mean': [], 'student_std': [], 'teacher_mean': [], 'teacher_std': []}
    }
    
    print("Decoder Analysis on Exp Folder Embeddings")
    print(f"Using alpha = {args.alpha}")
    print(f"Train/Test split: {100*(1-args.test_size):.0f}/{100*args.test_size:.0f}")
    print("="*60)
    
    # Process each epoch
    for epoch in tqdm(range(10), desc="Processing epochs"):
        epoch_dir = base_dir / f"ijepa-64px-epoch={epoch:02d}" / "masked" / "val" / "ijepa_75"
        
        if not epoch_dir.exists():
            print(f"Warning: Directory not found for epoch {epoch}")
            continue
        
        # Run linear decoder
        linear_results = analyze_masked_epoch(epoch_dir, decoder_type='linear', alpha=args.alpha, test_size=args.test_size)
        
        # Run non-linear decoder
        nn_results = analyze_masked_epoch(epoch_dir, decoder_type='nonlinear', alpha=args.alpha, test_size=args.test_size)
        
        # Store results
        for decoder_type, results in [('linear', linear_results), ('nonlinear', nn_results)]:
            all_results[decoder_type]['epochs'].append(epoch)
            all_results[decoder_type]['student_mean'].append(np.mean(results['student']))
            all_results[decoder_type]['student_std'].append(np.std(results['student']))
            all_results[decoder_type]['teacher_mean'].append(np.mean(results['teacher']))
            all_results[decoder_type]['teacher_std'].append(np.std(results['teacher']))
        
        # Print epoch summary
        print(f"\nEpoch {epoch}:")
        print(f"  Linear - Student: {np.mean(linear_results['student']):.4f} ± {np.std(linear_results['student']):.4f}")
        print(f"  Linear - Teacher: {np.mean(linear_results['teacher']):.4f} ± {np.std(linear_results['teacher']):.4f}")
        print(f"  NN - Student: {np.mean(nn_results['student']):.4f} ± {np.std(nn_results['student']):.4f}")
        print(f"  NN - Teacher: {np.mean(nn_results['teacher']):.4f} ± {np.std(nn_results['teacher']):.4f}")
    
    # Save results
    output_dir = Path('analysis/decoder/results_exp')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Include configuration in results
    results_with_config = {
        'config': {
            'alpha': args.alpha,
            'test_size': args.test_size,
            'train_size': 1 - args.test_size
        },
        'results': all_results
    }
    
    with open(output_dir / f'decoder_results_split{int(100*(1-args.test_size))}-{int(100*args.test_size)}.json', 'w') as f:
        json.dump(results_with_config, f, indent=2)
    
    # Create visualization
    create_plots(all_results, output_dir, args.test_size)
    
    print("\nAnalysis complete!")
    print(f"Results saved to {output_dir}")


def create_plots(results, output_dir, test_size=0.25):
    """Create visualization plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Linear decoder plot
    ax = axes[0]
    epochs = results['linear']['epochs']
    
    ax.errorbar(epochs, results['linear']['student_mean'], 
                yerr=results['linear']['student_std'],
                fmt='o-', label='Student (masked)', capsize=5, linewidth=2, markersize=8)
    ax.errorbar(epochs, results['linear']['teacher_mean'],
                yerr=results['linear']['teacher_std'],
                fmt='s-', label='Teacher', capsize=5, linewidth=2, markersize=8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Linear Decoder Performance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(10))
    
    # Non-linear decoder plot
    ax = axes[1]
    
    ax.errorbar(epochs, results['nonlinear']['student_mean'],
                yerr=results['nonlinear']['student_std'],
                fmt='o-', label='Student (masked)', capsize=5, linewidth=2, markersize=8)
    ax.errorbar(epochs, results['nonlinear']['teacher_mean'],
                yerr=results['nonlinear']['teacher_std'],
                fmt='s-', label='Teacher', capsize=5, linewidth=2, markersize=8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Non-linear Decoder Performance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(10))
    
    plt.suptitle(f'Decoder Analysis Across 10 Epochs (α=0.001, {int(100*(1-test_size))}/{int(100*test_size)} split)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_dir / f'decoder_evolution_split{int(100*(1-test_size))}-{int(100*test_size)}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary plot
    create_summary_plot(results, output_dir, test_size)


def create_summary_plot(results, output_dir, test_size=0.25):
    """Create summary comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate averages
    metrics = {
        'Linear\nStudent': np.mean(results['linear']['student_mean']),
        'Linear\nTeacher': np.mean(results['linear']['teacher_mean']),
        'Non-linear\nStudent': np.mean(results['nonlinear']['student_mean']),
        'Non-linear\nTeacher': np.mean(results['nonlinear']['teacher_mean'])
    }
    
    errors = {
        'Linear\nStudent': np.mean(results['linear']['student_std']),
        'Linear\nTeacher': np.mean(results['linear']['teacher_std']),
        'Non-linear\nStudent': np.mean(results['nonlinear']['student_std']),
        'Non-linear\nTeacher': np.mean(results['nonlinear']['teacher_std'])
    }
    
    x = np.arange(len(metrics))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax.bar(x, list(metrics.values()), yerr=list(errors.values()),
                   color=colors, alpha=0.8, capsize=10)
    
    # Add value labels
    for bar, (name, value) in zip(bars, metrics.items()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title(f'Decoder Performance Summary ({int(100*(1-test_size))}/{int(100*test_size)} split, Averaged Across All Epochs)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()))
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'decoder_summary_split{int(100*(1-test_size))}-{int(100*test_size)}.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()