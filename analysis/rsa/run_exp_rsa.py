"""Run RSA analysis on exp folder masked embeddings across all 10 epochs."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from tqdm import tqdm
try:
    from .rsa_analysis import (
        compute_rdm, compare_rdms, compute_rsa_scores,
        plot_rdm, plot_rdm_comparison, compute_rdm_reliability
    )
except ImportError:
    from rsa_analysis import (
        compute_rdm, compare_rdms, compute_rsa_scores,
        plot_rdm, plot_rdm_comparison, compute_rdm_reliability
    )


def load_masked_embeddings(embeddings_dir, epoch, mask_idx=0):
    """Load masked embeddings for a specific epoch."""
    epoch_dir = embeddings_dir / f"ijepa-64px-epoch={epoch:02d}" / "masked" / "val" / "ijepa_75" / f"mask_{mask_idx:02d}"
    
    student_path = epoch_dir / "student_embeddings_masked_pooled.npy"
    teacher_path = epoch_dir / "teacher_embeddings_pooled.npy"
    labels_path = epoch_dir / "labels.npy"
    
    if not all(p.exists() for p in [student_path, teacher_path, labels_path]):
        raise FileNotFoundError(f"Missing files in {epoch_dir}")
    
    student_emb = np.load(student_path)
    teacher_emb = np.load(teacher_path)
    labels = np.load(labels_path)
    
    return student_emb, teacher_emb, labels


def subsample_balanced(embeddings, labels, samples_per_class=10, max_classes=50):
    """Subsample embeddings to have equal samples per class."""
    unique_labels = np.unique(labels)
    
    # Limit number of classes if specified
    if max_classes is not None and len(unique_labels) > max_classes:
        unique_labels = np.sort(unique_labels)[:max_classes]
    
    indices = []
    
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        if len(label_indices) >= samples_per_class:
            selected = np.random.choice(label_indices, samples_per_class, replace=False)
            indices.extend(selected)
    
    indices = np.array(sorted(indices))
    return embeddings[indices], labels[indices], indices


def analyze_exp_embeddings(embeddings_dir, output_dir, epochs, samples_per_class, max_classes, n_masks=5):
    """Analyze RSA for exp folder embeddings across multiple masks."""
    
    print(f"\n{'='*60}")
    print("RSA Analysis on Exp Folder Embeddings")
    print(f"{'='*60}")
    print(f"Epochs: {epochs}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Max classes: {max_classes}")
    print(f"Number of masks: {n_masks}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Storage for results across masks
    all_mask_results = {}
    
    for mask_idx in range(n_masks):
        print(f"\n--- Processing mask {mask_idx} ---")
        
        # Load all embeddings for this mask
        all_embeddings = {}
        common_indices = None
        common_labels = None
        
        for epoch in tqdm(epochs, desc=f"Loading epochs for mask {mask_idx}"):
            try:
                student_emb, teacher_emb, labels = load_masked_embeddings(embeddings_dir, epoch, mask_idx)
                
                # Subsample for efficiency (use same indices across epochs)
                if common_indices is None:
                    student_emb_sub, labels_sub, indices = subsample_balanced(
                        student_emb, labels, samples_per_class, max_classes
                    )
                    teacher_emb_sub, _, _ = subsample_balanced(
                        teacher_emb, labels, samples_per_class, max_classes
                    )
                    common_indices = indices
                    common_labels = labels_sub
                else:
                    # Use same indices for consistency
                    student_emb_sub = student_emb[common_indices]
                    teacher_emb_sub = teacher_emb[common_indices]
                
                all_embeddings[f"student_epoch{epoch:02d}"] = student_emb_sub
                all_embeddings[f"teacher_epoch{epoch:02d}"] = teacher_emb_sub
                
            except Exception as e:
                print(f"  Warning: Failed to load epoch {epoch}: {e}")
                continue
        
        if len(all_embeddings) == 0:
            print(f"  No embeddings loaded for mask {mask_idx}")
            continue
            
        print(f"  Loaded embeddings shape: {list(all_embeddings.values())[0].shape}")
        
        # Compute RSA scores
        rsa_scores, rdms = compute_rsa_scores(all_embeddings)
        all_mask_results[f'mask_{mask_idx}'] = {
            'rsa_scores': rsa_scores,
            'rdms': rdms,
            'n_samples': len(common_labels),
            'n_classes': len(np.unique(common_labels))
        }
    
    # Average results across masks
    print("\n--- Computing average RSA scores across masks ---")
    averaged_scores = compute_average_rsa_scores(all_mask_results, epochs)
    
    # Save results
    save_results(output_dir, all_mask_results, averaged_scores, epochs)
    
    # Create visualizations
    create_rsa_plots(output_dir, averaged_scores, epochs)
    
    print(f"\nResults saved to: {output_dir}")
    
    return averaged_scores


def compute_average_rsa_scores(all_mask_results, epochs):
    """Compute average RSA scores across masks."""
    averaged = {}
    
    # Collect all comparison types
    all_comparisons = set()
    for mask_data in all_mask_results.values():
        all_comparisons.update(mask_data['rsa_scores'].keys())
    
    # Average each comparison
    for comparison in all_comparisons:
        scores = []
        for mask_data in all_mask_results.values():
            if comparison in mask_data['rsa_scores']:
                scores.append(mask_data['rsa_scores'][comparison]['correlation'])
        
        if scores:
            averaged[comparison] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'n_masks': len(scores)
            }
    
    return averaged


def save_results(output_dir, all_mask_results, averaged_scores, epochs):
    """Save RSA results to files."""
    
    # Save raw results
    results_data = {
        'mask_results': {k: v['rsa_scores'] for k, v in all_mask_results.items()},
        'averaged_scores': averaged_scores,
        'config': {
            'epochs': epochs,
            'n_masks': len(all_mask_results)
        }
    }
    
    with open(output_dir / 'rsa_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Print summary
    print("\n=== RSA Results Summary ===")
    
    print("\n1. Student vs Teacher (within epoch):")
    for epoch in epochs:
        key = f"student_epoch{epoch:02d}_vs_teacher_epoch{epoch:02d}"
        if key in averaged_scores:
            mean = averaged_scores[key]['mean']
            std = averaged_scores[key]['std']
            print(f"  Epoch {epoch}: r={mean:.4f} ± {std:.4f}")
    
    print("\n2. Student across adjacent epochs:")
    for i in range(len(epochs)-1):
        key = f"student_epoch{epochs[i]:02d}_vs_student_epoch{epochs[i+1]:02d}"
        if key in averaged_scores:
            mean = averaged_scores[key]['mean']
            std = averaged_scores[key]['std']
            print(f"  Epoch {epochs[i]} → {epochs[i+1]}: r={mean:.4f} ± {std:.4f}")
    
    print("\n3. Teacher across adjacent epochs:")
    for i in range(len(epochs)-1):
        key = f"teacher_epoch{epochs[i]:02d}_vs_teacher_epoch{epochs[i+1]:02d}"
        if key in averaged_scores:
            mean = averaged_scores[key]['mean']
            std = averaged_scores[key]['std']
            print(f"  Epoch {epochs[i]} → {epochs[i+1]}: r={mean:.4f} ± {std:.4f}")


def create_rsa_plots(output_dir, averaged_scores, epochs):
    """Create RSA visualization plots."""
    
    # Evolution plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Student-Teacher similarity across epochs
    st_similarities = []
    st_errors = []
    for epoch in epochs:
        key = f"student_epoch{epoch:02d}_vs_teacher_epoch{epoch:02d}"
        if key in averaged_scores:
            st_similarities.append(averaged_scores[key]['mean'])
            st_errors.append(averaged_scores[key]['std'])
    
    # Adjacent epoch similarities
    student_adjacent = []
    student_adjacent_errors = []
    teacher_adjacent = []
    teacher_adjacent_errors = []
    
    for i in range(len(epochs)-1):
        s_key = f"student_epoch{epochs[i]:02d}_vs_student_epoch{epochs[i+1]:02d}"
        t_key = f"teacher_epoch{epochs[i]:02d}_vs_teacher_epoch{epochs[i+1]:02d}"
        
        if s_key in averaged_scores:
            student_adjacent.append(averaged_scores[s_key]['mean'])
            student_adjacent_errors.append(averaged_scores[s_key]['std'])
        
        if t_key in averaged_scores:
            teacher_adjacent.append(averaged_scores[t_key]['mean'])
            teacher_adjacent_errors.append(averaged_scores[t_key]['std'])
    
    # Plot with error bars
    ax.errorbar(epochs, st_similarities, yerr=st_errors, 
                fmt='o-', label='Student vs Teacher (same epoch)', 
                linewidth=2.5, markersize=10, capsize=5)
    
    ax.errorbar(epochs[:-1], student_adjacent, yerr=student_adjacent_errors,
                fmt='s-', label='Student adjacent epochs',
                linewidth=2.5, markersize=10, capsize=5)
    
    ax.errorbar(epochs[:-1], teacher_adjacent, yerr=teacher_adjacent_errors,
                fmt='^-', label='Teacher adjacent epochs',
                linewidth=2.5, markersize=10, capsize=5)
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('RSA Correlation', fontsize=14)
    ax.set_title('RSA Evolution During Training\n(Averaged across 5 mask samples)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    ax.set_xticks(epochs)
    
    plt.tight_layout()
    fig.savefig(output_dir / "rsa_evolution.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create heatmap of all comparisons
    create_comparison_heatmap(output_dir, averaged_scores, epochs)
    
    print("\nGenerated plots:")
    print("  - rsa_evolution.png")
    print("  - rsa_comparison_heatmap.png")


def create_comparison_heatmap(output_dir, averaged_scores, epochs):
    """Create heatmap showing all pairwise RSA correlations."""
    
    # Create labels for all embeddings
    labels = []
    for epoch in epochs:
        labels.append(f"S{epoch}")  # Student
        labels.append(f"T{epoch}")  # Teacher
    
    n = len(labels)
    correlation_matrix = np.zeros((n, n))
    
    # Fill diagonal with 1s
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Fill matrix with RSA scores
    for i in range(n):
        for j in range(i+1, n):
            # Determine embedding names
            if i % 2 == 0:  # Student
                name1 = f"student_epoch{epochs[i//2]:02d}"
            else:  # Teacher
                name1 = f"teacher_epoch{epochs[i//2]:02d}"
                
            if j % 2 == 0:  # Student
                name2 = f"student_epoch{epochs[j//2]:02d}"
            else:  # Teacher
                name2 = f"teacher_epoch{epochs[j//2]:02d}"
            
            key = f"{name1}_vs_{name2}"
            if key in averaged_scores:
                value = averaged_scores[key]['mean']
            else:
                # Try reverse order
                key = f"{name2}_vs_{name1}"
                if key in averaged_scores:
                    value = averaged_scores[key]['mean']
                else:
                    value = np.nan
            
            correlation_matrix[i, j] = value
            correlation_matrix[j, i] = value
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('RSA Correlation', fontsize=12)
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            if not np.isnan(correlation_matrix[i, j]):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                             ha='center', va='center', 
                             color='white' if correlation_matrix[i, j] < 0.5 else 'black',
                             fontsize=8)
    
    ax.set_title('Pairwise RSA Correlations\n(S=Student, T=Teacher)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig.savefig(output_dir / "rsa_comparison_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Run RSA analysis on exp folder embeddings')
    parser.add_argument('--embeddings_dir', type=str, 
                       default='analysis/dim_reduction/embeddings_exp',
                       help='Directory containing exp folder embeddings')
    parser.add_argument('--output_dir', type=str, 
                       default='analysis/rsa/rsa_results_exp',
                       help='Output directory for RSA results')
    parser.add_argument('--epochs', type=int, nargs='+', 
                       default=list(range(10)),
                       help='Which epochs to analyze (default: all 10)')
    parser.add_argument('--samples_per_class', type=int, default=10,
                       help='Number of samples per class for subsampling')
    parser.add_argument('--max_classes', type=int, default=50,
                       help='Maximum number of classes to use')
    parser.add_argument('--n_masks', type=int, default=5,
                       help='Number of mask samples to analyze')
    
    args = parser.parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    
    # Run analysis
    analyze_exp_embeddings(
        embeddings_dir=embeddings_dir,
        output_dir=output_dir,
        epochs=args.epochs,
        samples_per_class=args.samples_per_class,
        max_classes=args.max_classes,
        n_masks=args.n_masks
    )


if __name__ == "__main__":
    main()