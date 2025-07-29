"""Run RSA analysis on extracted I-JEPA embeddings across epochs."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
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


def load_embeddings(embeddings_dir, epoch, split='val'):
    """Load embeddings for a specific epoch and split."""
    epoch_dir = embeddings_dir / f"ijepa-64px-epoch={epoch:02d}"
    
    # Handle nested directory structure
    if not (epoch_dir / split).exists():
        epoch_dir = epoch_dir / f"ijepa-64px-epoch={epoch:02d}"
    
    student_path = epoch_dir / split / "student_embeddings_pooled.npy"
    teacher_path = epoch_dir / split / "teacher_embeddings_pooled.npy"
    labels_path = epoch_dir / split / "labels.npy"
    
    student_emb = np.load(student_path)
    teacher_emb = np.load(teacher_path)
    labels = np.load(labels_path)
    
    return student_emb, teacher_emb, labels


def subsample_by_class(embeddings, labels, samples_per_class=50):
    """Subsample embeddings to have equal samples per class."""
    unique_labels = np.unique(labels)
    indices = []
    
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        if len(label_indices) >= samples_per_class:
            selected = np.random.choice(label_indices, samples_per_class, replace=False)
            indices.extend(selected)
    
    indices = np.array(indices)
    return embeddings[indices], labels[indices]


def analyze_split(embeddings_dir, output_dir, split, epochs, samples_per_class):
    """Analyze RSA for a specific split."""
    print(f"\n{'='*50}")
    print(f"Analyzing {split.upper()} split")
    print('='*50)
    
    # Create split-specific output directory
    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all embeddings for this split
    all_embeddings = {}
    all_labels = None
    
    for epoch in epochs:
        print(f"Loading epoch {epoch}...")
        student_emb, teacher_emb, labels = load_embeddings(embeddings_dir, epoch, split)
        
        # Subsample for efficiency
        if all_labels is None:
            student_emb, labels = subsample_by_class(student_emb, labels, samples_per_class)
            teacher_emb, _ = subsample_by_class(teacher_emb, labels, samples_per_class)
            all_labels = labels
        else:
            # Use same indices for all epochs
            student_emb = student_emb[:len(all_labels)]
            teacher_emb = teacher_emb[:len(all_labels)]
        
        all_embeddings[f"student_epoch{epoch}"] = student_emb
        all_embeddings[f"teacher_epoch{epoch}"] = teacher_emb
    
    print(f"Loaded embeddings shape: {list(all_embeddings.values())[0].shape}")
    
    # Compute RSA scores between all pairs
    print("\nComputing RSA scores...")
    rsa_scores, rdms = compute_rsa_scores(all_embeddings)
    
    # Save RSA scores for this split
    with open(split_output_dir / "rsa_scores.json", 'w') as f:
        json.dump(rsa_scores, f, indent=2)
    
    # Print key comparisons
    print(f"\n=== RSA Results for {split.upper()} ===")
    print("\n1. Student vs Teacher (within epoch):")
    for epoch in epochs:
        key = f"student_epoch{epoch}_vs_teacher_epoch{epoch}"
        if key in rsa_scores:
            score = rsa_scores[key]['correlation']
            p_val = rsa_scores[key]['p_value']
            print(f"  Epoch {epoch}: r={score:.4f}, p={p_val:.2e}")
    
    print("\n2. Student across epochs:")
    for i in range(len(epochs)-1):
        key = f"student_epoch{epochs[i]}_vs_student_epoch{epochs[i+1]}"
        if key in rsa_scores:
            score = rsa_scores[key]['correlation']
            print(f"  Epoch {epochs[i]} -> {epochs[i+1]}: r={score:.4f}")
    
    print("\n3. Teacher across epochs:")
    for i in range(len(epochs)-1):
        key = f"teacher_epoch{epochs[i]}_vs_teacher_epoch{epochs[i+1]}"
        if key in rsa_scores:
            score = rsa_scores[key]['correlation']
            print(f"  Epoch {epochs[i]} -> {epochs[i+1]}: r={score:.4f}")
    
    # Plot RDMs for first and last epoch
    print("\nPlotting RDMs...")
    
    first_epoch = epochs[0]
    last_epoch = epochs[-1]
    
    # Student evolution
    fig = plot_rdm_comparison(
        rdms[f"student_epoch{first_epoch}"], 
        rdms[f"student_epoch{last_epoch}"],
        title1=f"Student Epoch {first_epoch} ({split})",
        title2=f"Student Epoch {last_epoch} ({split})"
    )
    fig.savefig(split_output_dir / "rdm_student_evolution.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Teacher evolution
    fig = plot_rdm_comparison(
        rdms[f"teacher_epoch{first_epoch}"],
        rdms[f"teacher_epoch{last_epoch}"],
        title1=f"Teacher Epoch {first_epoch} ({split})", 
        title2=f"Teacher Epoch {last_epoch} ({split})"
    )
    fig.savefig(split_output_dir / "rdm_teacher_evolution.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Student vs Teacher comparison (last epoch)
    fig = plot_rdm_comparison(
        rdms[f"student_epoch{last_epoch}"],
        rdms[f"teacher_epoch{last_epoch}"],
        title1=f"Student Epoch {last_epoch} ({split})",
        title2=f"Teacher Epoch {last_epoch} ({split})"
    )
    fig.savefig(split_output_dir / f"rdm_student_vs_teacher_epoch{last_epoch}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create a summary plot of RSA evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Student-Teacher similarity across epochs
    st_similarities = []
    for epoch in epochs:
        key = f"student_epoch{epoch}_vs_teacher_epoch{epoch}"
        if key in rsa_scores:
            st_similarities.append(rsa_scores[key]['correlation'])
    
    # Adjacent epoch similarities
    student_adjacent = []
    teacher_adjacent = []
    for i in range(len(epochs)-1):
        s_key = f"student_epoch{epochs[i]}_vs_student_epoch{epochs[i+1]}"
        t_key = f"teacher_epoch{epochs[i]}_vs_teacher_epoch{epochs[i+1]}"
        if s_key in rsa_scores:
            student_adjacent.append(rsa_scores[s_key]['correlation'])
        if t_key in rsa_scores:
            teacher_adjacent.append(rsa_scores[t_key]['correlation'])
    
    # Plot
    ax.plot(epochs, st_similarities, 'o-', label='Student vs Teacher (same epoch)', linewidth=2)
    ax.plot(epochs[:-1], student_adjacent, 's-', label='Student adjacent epochs', linewidth=2)
    ax.plot(epochs[:-1], teacher_adjacent, '^-', label='Teacher adjacent epochs', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RSA Correlation')
    ax.set_title(f'RSA Evolution During Training ({split.upper()})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    fig.savefig(split_output_dir / "rsa_evolution.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Compute reliability scores
    print(f"\n=== RDM Reliability (split-half) for {split.upper()} ===")
    for epoch in [first_epoch, last_epoch]:
        student_emb = all_embeddings[f"student_epoch{epoch}"]
        teacher_emb = all_embeddings[f"teacher_epoch{epoch}"]
        
        student_rel = compute_rdm_reliability(student_emb, n_splits=5)
        teacher_rel = compute_rdm_reliability(teacher_emb, n_splits=5)
        
        print(f"Epoch {epoch}:")
        print(f"  Student reliability: {student_rel:.4f}")
        print(f"  Teacher reliability: {teacher_rel:.4f}")
    
    print(f"\nResults saved to: {split_output_dir}")
    print("Generated files:")
    print("  - rsa_scores.json")
    print("  - rdm_student_evolution.png") 
    print("  - rdm_teacher_evolution.png")
    print(f"  - rdm_student_vs_teacher_epoch{last_epoch}.png")
    print("  - rsa_evolution.png")
    
    return rsa_scores, rdms


def main():
    # Setup
    parser = argparse.ArgumentParser(description='Run RSA analysis on I-JEPA embeddings')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'both'],
                       help='Which split to analyze (train, val, or both)')
    parser.add_argument('--samples_per_class', type=int, default=100,
                       help='Number of samples per class for subsampling')
    parser.add_argument('--epochs', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                       help='Which epochs to analyze')
    parser.add_argument('--embeddings_dir', type=str, default='analysis/dim_reduction/embeddings',
                       help='Directory containing extracted embeddings')
    parser.add_argument('--output_dir', type=str, default='analysis/rsa/rsa_results',
                       help='Output directory for RSA results')
    args = parser.parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = args.epochs
    samples_per_class = args.samples_per_class
    splits_to_analyze = ['train', 'val'] if args.split == 'both' else [args.split]
    
    print(f"RSA Analysis Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Splits: {splits_to_analyze}")
    print(f"  Samples per class: {samples_per_class}")
    
    # Process each split
    all_results = {}
    for split in splits_to_analyze:
        rsa_scores, rdms = analyze_split(
            embeddings_dir, output_dir, split, epochs, samples_per_class
        )
        all_results[split] = {
            'rsa_scores': rsa_scores,
            'summary': {
                'student_teacher_correlation_final': rsa_scores.get(
                    f"student_epoch{epochs[-1]}_vs_teacher_epoch{epochs[-1]}", {}
                ).get('correlation', None)
            }
        }
    
    # If analyzing both splits, create a comparison plot
    if args.split == 'both':
        print(f"\n{'='*50}")
        print("Creating cross-split comparison")
        print('='*50)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for split_idx, split in enumerate(['train', 'val']):
            ax = ax1 if split_idx == 0 else ax2
            
            # Load RSA scores
            with open(output_dir / split / "rsa_scores.json", 'r') as f:
                rsa_scores = json.load(f)
            
            # Student-Teacher similarity across epochs
            st_similarities = []
            for epoch in epochs:
                key = f"student_epoch{epoch}_vs_teacher_epoch{epoch}"
                if key in rsa_scores:
                    st_similarities.append(rsa_scores[key]['correlation'])
            
            # Adjacent epoch similarities
            student_adjacent = []
            teacher_adjacent = []
            for i in range(len(epochs)-1):
                s_key = f"student_epoch{epochs[i]}_vs_student_epoch{epochs[i+1]}"
                t_key = f"teacher_epoch{epochs[i]}_vs_teacher_epoch{epochs[i+1]}"
                if s_key in rsa_scores:
                    student_adjacent.append(rsa_scores[s_key]['correlation'])
                if t_key in rsa_scores:
                    teacher_adjacent.append(rsa_scores[t_key]['correlation'])
            
            # Plot
            ax.plot(epochs, st_similarities, 'o-', label='Student vs Teacher', linewidth=2)
            ax.plot(epochs[:-1], student_adjacent, 's-', label='Student adjacent', linewidth=2)
            ax.plot(epochs[:-1], teacher_adjacent, '^-', label='Teacher adjacent', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('RSA Correlation')
            ax.set_title(f'{split.upper()} Split')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.suptitle('RSA Evolution Comparison: Train vs Val', fontsize=14)
        plt.tight_layout()
        fig.savefig(output_dir / "rsa_train_vs_val_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Created comparison plot: rsa_train_vs_val_comparison.png")
    
    print("\n" + "="*50)
    print("RSA Analysis Complete!")
    print("="*50)


if __name__ == "__main__":
    main()