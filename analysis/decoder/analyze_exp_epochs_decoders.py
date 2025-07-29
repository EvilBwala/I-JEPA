#!/usr/bin/env python3
"""
Analyze decoder performance (linear and non-linear) across all 10 epochs from exp folder.
Uses optimal alpha=0.001 for regularization.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_epoch_embeddings(base_dir: Path, epoch: int, mask_idx: int = 0):
    """Load embeddings for a specific epoch and mask."""
    epoch_dir = base_dir / f"ijepa-64px-epoch={epoch:02d}" / "masked" / "val" / "ijepa_75" / f"mask_{mask_idx:02d}"
    
    student_emb = np.load(epoch_dir / "student_embeddings_masked_pooled.npy")
    teacher_emb = np.load(epoch_dir / "teacher_embeddings_pooled.npy")
    labels = np.load(epoch_dir / "labels.npy")
    
    return student_emb, teacher_emb, labels


def evaluate_decoder(X_train, X_test, y_train, y_test, decoder_type='linear', alpha=0.001):
    """Evaluate decoder performance."""
    if decoder_type == 'linear':
        # Logistic regression with L2 regularization
        decoder = LogisticRegression(
            C=1/alpha,  # C is inverse of regularization strength
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            random_state=42
        )
    else:  # non-linear
        # MLPClassifier with L2 regularization
        decoder = MLPClassifier(
            hidden_layer_sizes=(128,),
            alpha=alpha,  # L2 penalty parameter
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        )
    
    # Train
    decoder.fit(X_train, y_train)
    
    # Predict
    y_pred = decoder.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, decoder


def analyze_all_epochs():
    """Analyze decoder performance across all epochs."""
    base_dir = Path('analysis/dim_reduction/embeddings_exp')
    output_dir = Path('analysis/decoder/results_exp')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Decoder Analysis of Exp Folder Embeddings (10 Epochs)")
    print("Using optimal alpha=0.001")
    print("="*60)
    
    # Storage for results
    results = {
        'epochs': list(range(10)),
        'linear': {
            'student': {'mean': [], 'std': []},
            'teacher': {'mean': [], 'std': []}
        },
        'nonlinear': {
            'student': {'mean': [], 'std': []},
            'teacher': {'mean': [], 'std': []}
        }
    }
    
    detailed_results = {}
    
    # Process each epoch
    for epoch in tqdm(range(10), desc="Processing epochs"):
        print(f"\n--- Epoch {epoch} ---")
        
        epoch_results = {
            'linear': {'student': [], 'teacher': []},
            'nonlinear': {'student': [], 'teacher': []}
        }
        
        # Analyze multiple mask samples
        for mask_idx in range(5):  # Use all 5 mask samples
            try:
                # Load data
                student_emb, teacher_emb, labels = load_epoch_embeddings(base_dir, epoch, mask_idx)
                
                # Split data
                X_student_train, X_student_test, y_train, y_test = train_test_split(
                    student_emb, labels, test_size=0.3, random_state=42+mask_idx, stratify=labels
                )
                
                X_teacher_train, X_teacher_test, _, _ = train_test_split(
                    teacher_emb, labels, test_size=0.3, random_state=42+mask_idx, stratify=labels
                )
                
                # Evaluate decoders
                for decoder_type in ['linear', 'nonlinear']:
                    # Student decoder
                    student_acc, _ = evaluate_decoder(
                        X_student_train, X_student_test, y_train, y_test, 
                        decoder_type=decoder_type, alpha=0.001
                    )
                    epoch_results[decoder_type]['student'].append(student_acc)
                    
                    # Teacher decoder
                    teacher_acc, _ = evaluate_decoder(
                        X_teacher_train, X_teacher_test, y_train, y_test,
                        decoder_type=decoder_type, alpha=0.001
                    )
                    epoch_results[decoder_type]['teacher'].append(teacher_acc)
                    
            except Exception as e:
                print(f"  Warning: Failed to process mask {mask_idx}: {e}")
                continue
        
        # Calculate statistics for this epoch
        for decoder_type in ['linear', 'nonlinear']:
            for network in ['student', 'teacher']:
                accuracies = epoch_results[decoder_type][network]
                if accuracies:
                    mean_acc = np.mean(accuracies)
                    std_acc = np.std(accuracies)
                    results[decoder_type][network]['mean'].append(mean_acc)
                    results[decoder_type][network]['std'].append(std_acc)
                    
                    print(f"  {decoder_type.capitalize()} {network}: {mean_acc:.4f} ± {std_acc:.4f}")
        
        detailed_results[f'epoch_{epoch}'] = epoch_results
    
    # Save results
    with open(output_dir / 'decoder_analysis_results.json', 'w') as f:
        json.dump({
            'summary': results,
            'detailed': detailed_results,
            'config': {'alpha': 0.001, 'n_mask_samples': 5}
        }, f, indent=2)
    
    # Create visualizations
    create_decoder_plots(results, output_dir)
    
    # Print summary
    print_summary(results)
    
    return results


def create_decoder_plots(results, output_dir):
    """Create visualization plots for decoder analysis."""
    # Set up the plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = results['epochs']
    
    # Plot 1: Linear decoder performance
    ax = axes[0, 0]
    plot_decoder_evolution(ax, results['linear'], epochs, "Linear Decoder Performance")
    
    # Plot 2: Non-linear decoder performance
    ax = axes[0, 1]
    plot_decoder_evolution(ax, results['nonlinear'], epochs, "Non-linear Decoder Performance")
    
    # Plot 3: Student comparison (linear vs non-linear)
    ax = axes[1, 0]
    plot_network_comparison(ax, results, 'student', epochs, "Student Network Decoders")
    
    # Plot 4: Teacher comparison (linear vs non-linear)
    ax = axes[1, 1]
    plot_network_comparison(ax, results, 'teacher', epochs, "Teacher Network Decoders")
    
    plt.suptitle('Decoder Analysis Across 10 Training Epochs\n(α=0.001, 5 mask samples per epoch)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'decoder_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nEvolution plot saved to: {output_path}")
    
    # Create comparison plot
    create_comparison_plot(results, output_dir)


def plot_decoder_evolution(ax, decoder_results, epochs, title):
    """Plot evolution of a specific decoder type."""
    student_mean = decoder_results['student']['mean']
    student_std = decoder_results['student']['std']
    teacher_mean = decoder_results['teacher']['mean']
    teacher_std = decoder_results['teacher']['std']
    
    # Plot with error bands
    ax.errorbar(epochs[:len(student_mean)], student_mean, yerr=student_std,
                fmt='o-', color='#2E86AB', linewidth=2.5, markersize=10,
                label='Student (masked)', capsize=5, alpha=0.8)
    ax.errorbar(epochs[:len(teacher_mean)], teacher_mean, yerr=teacher_std,
                fmt='s-', color='#A23B72', linewidth=2.5, markersize=10,
                label='Teacher', capsize=5, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xticks(epochs)


def plot_network_comparison(ax, results, network, epochs, title):
    """Compare linear vs non-linear decoders for a specific network."""
    linear_mean = results['linear'][network]['mean']
    linear_std = results['linear'][network]['std']
    nonlinear_mean = results['nonlinear'][network]['mean']
    nonlinear_std = results['nonlinear'][network]['std']
    
    # Plot with error bands
    ax.errorbar(epochs[:len(linear_mean)], linear_mean, yerr=linear_std,
                fmt='o-', color='#E63946', linewidth=2.5, markersize=10,
                label='Linear', capsize=5, alpha=0.8)
    ax.errorbar(epochs[:len(nonlinear_mean)], nonlinear_mean, yerr=nonlinear_std,
                fmt='s-', color='#06FFA5', linewidth=2.5, markersize=10,
                label='Non-linear', capsize=5, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xticks(epochs)


def create_comparison_plot(results, output_dir):
    """Create a summary comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate averages across all epochs
    networks = ['Student (masked)', 'Teacher']
    colors = ['#2E86AB', '#A23B72']
    
    # Linear decoder comparison
    linear_means = [
        np.mean(results['linear']['student']['mean']),
        np.mean(results['linear']['teacher']['mean'])
    ]
    linear_stds = [
        np.mean(results['linear']['student']['std']),
        np.mean(results['linear']['teacher']['std'])
    ]
    
    x = np.arange(len(networks))
    width = 0.35
    
    bars1 = ax1.bar(x, linear_means, width, yerr=linear_stds, 
                     color=colors, alpha=0.8, capsize=10)
    ax1.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Linear Decoder Performance\n(Averaged across all epochs)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(networks)
    ax1.set_ylim(0, 1)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars1, linear_means, linear_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Non-linear decoder comparison
    nonlinear_means = [
        np.mean(results['nonlinear']['student']['mean']),
        np.mean(results['nonlinear']['teacher']['mean'])
    ]
    nonlinear_stds = [
        np.mean(results['nonlinear']['student']['std']),
        np.mean(results['nonlinear']['teacher']['std'])
    ]
    
    bars2 = ax2.bar(x, nonlinear_means, width, yerr=nonlinear_stds,
                     color=colors, alpha=0.8, capsize=10)
    ax2.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Non-linear Decoder Performance\n(Averaged across all epochs)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(networks)
    ax2.set_ylim(0, 1)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars2, nonlinear_means, nonlinear_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Decoder Performance Summary (α=0.001)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'decoder_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved to: {output_path}")


def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("Decoder Analysis Summary")
    print("="*60)
    
    # Overall averages
    print("\nAverage Performance Across All Epochs:")
    print("-" * 40)
    
    for decoder_type in ['linear', 'nonlinear']:
        print(f"\n{decoder_type.capitalize()} Decoder:")
        for network in ['student', 'teacher']:
            mean_acc = np.mean(results[decoder_type][network]['mean'])
            std_acc = np.mean(results[decoder_type][network]['std'])
            print(f"  {network.capitalize():8s}: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # Performance improvement over epochs
    print("\nPerformance Change (Epoch 0 → Epoch 9):")
    print("-" * 40)
    
    for decoder_type in ['linear', 'nonlinear']:
        print(f"\n{decoder_type.capitalize()} Decoder:")
        for network in ['student', 'teacher']:
            if len(results[decoder_type][network]['mean']) >= 10:
                start = results[decoder_type][network]['mean'][0]
                end = results[decoder_type][network]['mean'][9]
                change = end - start
                print(f"  {network.capitalize():8s}: {start:.4f} → {end:.4f} ({change:+.4f})")


if __name__ == "__main__":
    analyze_all_epochs()