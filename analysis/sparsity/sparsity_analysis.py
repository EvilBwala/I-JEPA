#!/usr/bin/env python3
"""
Sparsity analysis for I-JEPA embeddings focusing on:
1. Core sparsity metrics
2. Activation patterns analysis
3. Student vs Teacher comparative analysis
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from tqdm import tqdm


class SparsityAnalyzer:
    """Analyze sparsity in I-JEPA student and teacher embeddings."""
    
    def __init__(self, embeddings_dir: str = "../dim_reduction/embeddings",
                 output_dir: str = "results"):
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_embeddings(self, epoch: int, split: str = 'val') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load student and teacher embeddings for a specific epoch."""
        epoch_dir = self.embeddings_dir / f"ijepa-64px-epoch={epoch:02d}"
        
        # Handle nested directory structure
        if not (epoch_dir / split).exists():
            epoch_dir = epoch_dir / f"ijepa-64px-epoch={epoch:02d}"
        
        student_emb = np.load(epoch_dir / split / "student_embeddings_pooled.npy")
        teacher_emb = np.load(epoch_dir / split / "teacher_embeddings_pooled.npy")
        labels = np.load(epoch_dir / split / "labels.npy")
        
        return student_emb, teacher_emb, labels
    
    # 1. CORE SPARSITY METRICS
    
    def compute_l0_sparsity(self, x: np.ndarray, epsilon: float = 1e-6) -> float:
        """Fraction of near-zero elements."""
        return np.mean(np.abs(x) < epsilon)
    
    def compute_l1_sparsity(self, x: np.ndarray) -> float:
        """Hoyer's sparsity measure using L1/L2 ratio."""
        n = x.size
        if n == 1:
            return 0.0
        l1_norm = np.sum(np.abs(x))
        l2_norm = np.sqrt(np.sum(x**2))
        if l2_norm == 0:
            return 1.0
        return (np.sqrt(n) - l1_norm / l2_norm) / (np.sqrt(n) - 1)
    
    def compute_gini_coefficient(self, x: np.ndarray) -> float:
        """Gini coefficient as sparsity measure (0=uniform, 1=sparse)."""
        x = np.abs(x).flatten()
        sorted_x = np.sort(x)
        n = len(x)
        if n == 0 or np.sum(sorted_x) == 0:
            return 0.0
        cumsum = np.cumsum(sorted_x)
        return (2 * np.sum((np.arange(n) + 1) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n
    
    def compute_activation_ratio(self, x: np.ndarray, threshold: float = 0.1) -> float:
        """Ratio of significantly active dimensions."""
        if np.max(np.abs(x)) == 0:
            return 0.0
        x_normalized = np.abs(x) / np.max(np.abs(x))
        return np.mean(x_normalized > threshold)
    
    # 2. ACTIVATION PATTERNS ANALYSIS
    
    def analyze_dead_neurons(self, embeddings: np.ndarray, threshold: float = 0.01) -> Dict:
        """Identify and analyze dead (consistently inactive) neurons."""
        # Compute statistics per neuron across samples
        neuron_means = np.mean(np.abs(embeddings), axis=0)
        neuron_stds = np.std(embeddings, axis=0)
        neuron_max = np.max(np.abs(embeddings), axis=0)
        
        # Dead neurons have very low mean activation
        dead_mask = neuron_means < threshold
        
        # Analyze activation distribution
        activation_percentiles = np.percentile(neuron_means, [25, 50, 75, 90, 95, 99])
        
        return {
            'n_dead': int(np.sum(dead_mask)),
            'n_total': len(neuron_means),
            'dead_ratio': float(np.mean(dead_mask)),
            'dead_indices': dead_mask.nonzero()[0].tolist(),
            'activation_stats': {
                'mean': float(np.mean(neuron_means)),
                'std': float(np.std(neuron_means)),
                'min': float(np.min(neuron_means)),
                'max': float(np.max(neuron_means)),
                'percentiles': {
                    '25': float(activation_percentiles[0]),
                    '50': float(activation_percentiles[1]),
                    '75': float(activation_percentiles[2]),
                    '90': float(activation_percentiles[3]),
                    '95': float(activation_percentiles[4]),
                    '99': float(activation_percentiles[5])
                }
            }
        }
    
    def compute_lifetime_sparsity(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute how sparse each neuron is across all samples."""
        n_features = embeddings.shape[1]
        lifetime_sparsity = np.zeros(n_features)
        
        for i in range(n_features):
            neuron_activations = embeddings[:, i]
            lifetime_sparsity[i] = self.compute_l1_sparsity(neuron_activations)
            
        return lifetime_sparsity
    
    def compute_population_sparsity(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute sparsity of each sample (how many neurons are active)."""
        n_samples = embeddings.shape[0]
        population_sparsity = np.zeros(n_samples)
        
        for i in range(n_samples):
            sample_activations = embeddings[i, :]
            population_sparsity[i] = self.compute_l1_sparsity(sample_activations)
            
        return population_sparsity
    
    # 3. COMPARATIVE ANALYSIS
    
    def compare_student_teacher_sparsity(self, student_emb: np.ndarray, 
                                       teacher_emb: np.ndarray) -> Dict:
        """Compare sparsity characteristics between student and teacher."""
        comparison = {
            'student': self.compute_embedding_metrics(student_emb),
            'teacher': self.compute_embedding_metrics(teacher_emb),
            'differences': {}
        }
        
        # Compute differences
        for metric in comparison['student']:
            if isinstance(comparison['student'][metric], (int, float)):
                diff = comparison['teacher'][metric] - comparison['student'][metric]
                comparison['differences'][metric] = {
                    'absolute': float(diff),
                    'relative': float(diff / (comparison['student'][metric] + 1e-8))
                }
        
        # Additional comparative metrics
        # Correlation of activation patterns
        student_mean_act = np.mean(np.abs(student_emb), axis=0)
        teacher_mean_act = np.mean(np.abs(teacher_emb), axis=0)
        comparison['activation_correlation'] = float(np.corrcoef(student_mean_act, teacher_mean_act)[0, 1])
        
        # Overlap of dead neurons
        student_dead = set(comparison['student']['dead_neurons']['dead_indices'])
        teacher_dead = set(comparison['teacher']['dead_neurons']['dead_indices'])
        
        comparison['dead_neuron_overlap'] = {
            'n_both_dead': len(student_dead & teacher_dead),
            'n_student_only_dead': len(student_dead - teacher_dead),
            'n_teacher_only_dead': len(teacher_dead - student_dead),
            'jaccard_similarity': float(len(student_dead & teacher_dead) / 
                                      (len(student_dead | teacher_dead) + 1e-8))
        }
        
        return comparison
    
    def compute_embedding_metrics(self, embeddings: np.ndarray) -> Dict:
        """Compute all sparsity metrics for an embedding matrix."""
        return {
            'l0_sparsity': float(self.compute_l0_sparsity(embeddings)),
            'l1_sparsity': float(self.compute_l1_sparsity(embeddings)),
            'gini_coefficient': float(self.compute_gini_coefficient(embeddings)),
            'activation_ratio': float(self.compute_activation_ratio(embeddings)),
            'kurtosis': float(stats.kurtosis(embeddings.flatten())),
            'dead_neurons': self.analyze_dead_neurons(embeddings),
            'lifetime_sparsity_stats': {
                'mean': float(np.mean(self.compute_lifetime_sparsity(embeddings))),
                'std': float(np.std(self.compute_lifetime_sparsity(embeddings)))
            },
            'population_sparsity_stats': {
                'mean': float(np.mean(self.compute_population_sparsity(embeddings))),
                'std': float(np.std(self.compute_population_sparsity(embeddings)))
            }
        }
    
    def load_masked_embeddings(self, epoch: int, split: str = 'val', mask_type: str = 'ijepa',
                              mask_ratio: int = 75) -> Dict[str, Dict[str, np.ndarray]]:
        """Load masked embeddings for a specific epoch, returning all mask samples."""
        epoch_dir = self.embeddings_dir / f"ijepa-64px-epoch={epoch:02d}"
        
        mask_dir = epoch_dir / 'masked' / split / f"{mask_type}_{mask_ratio}"
        
        if not mask_dir.exists():
            raise FileNotFoundError(f"No masked embeddings found at {mask_dir}")
        
        # Load all mask samples
        masked_data = {}
        mask_samples = sorted([d for d in mask_dir.iterdir() if d.is_dir() and d.name.startswith('mask_')])
        
        for mask_sample_dir in mask_samples:
            mask_id = mask_sample_dir.name
            masked_data[mask_id] = {
                'student': np.load(mask_sample_dir / 'student_embeddings_masked_pooled.npy'),
                'teacher': np.load(mask_sample_dir / 'teacher_embeddings_pooled.npy'),
                'labels': np.load(mask_sample_dir / 'labels.npy'),
                'mask_pattern': np.load(mask_sample_dir / 'mask_pattern.npy')
            }
            
        return masked_data
    
    def analyze_masked_embeddings(self, masked_data: Dict[str, Dict[str, np.ndarray]]) -> Dict:
        """Analyze sparsity across multiple mask samples and compute average."""
        # Collect metrics for each mask sample
        all_student_metrics = []
        all_teacher_metrics = []
        all_comparisons = []
        
        for mask_id, data in masked_data.items():
            # Compute metrics for this mask sample
            student_metrics = self.compute_embedding_metrics(data['student'])
            teacher_metrics = self.compute_embedding_metrics(data['teacher'])
            
            # Compute comparison
            comparison = self.compare_student_teacher_sparsity(
                data['student'], data['teacher']
            )
            
            all_student_metrics.append(student_metrics)
            all_teacher_metrics.append(teacher_metrics)
            all_comparisons.append(comparison)
        
        # Average metrics across mask samples
        avg_metrics = {
            'n_mask_samples': len(masked_data),
            'student': self._average_metrics(all_student_metrics),
            'teacher': self._average_metrics(all_teacher_metrics),
            'differences': {},
            'mask_variance': {}
        }
        
        # Compute average differences
        for metric in ['l0_sparsity', 'l1_sparsity', 'gini_coefficient', 'activation_ratio']:
            student_values = [m[metric] for m in all_student_metrics]
            teacher_values = [m[metric] for m in all_teacher_metrics]
            
            avg_diff = np.mean(teacher_values) - np.mean(student_values)
            avg_metrics['differences'][metric] = {
                'absolute': float(avg_diff),
                'relative': float(avg_diff / (np.mean(student_values) + 1e-8))
            }
            
            # Compute variance across masks
            avg_metrics['mask_variance'][metric] = {
                'student_std': float(np.std(student_values)),
                'teacher_std': float(np.std(teacher_values))
            }
        
        # Average activation correlation
        correlations = [comp['activation_correlation'] for comp in all_comparisons]
        avg_metrics['activation_correlation'] = {
            'mean': float(np.mean(correlations)),
            'std': float(np.std(correlations))
        }
        
        return avg_metrics
    
    def _average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Average metrics across multiple samples."""
        avg_metrics = {}
        
        # Simple numeric metrics
        for key in ['l0_sparsity', 'l1_sparsity', 'gini_coefficient', 'activation_ratio', 'kurtosis']:
            values = [m[key] for m in metrics_list]
            avg_metrics[key] = float(np.mean(values))
        
        # Dead neurons
        dead_counts = [m['dead_neurons']['n_dead'] for m in metrics_list]
        total_counts = [m['dead_neurons']['n_total'] for m in metrics_list]
        avg_metrics['dead_neurons'] = {
            'n_dead': float(np.mean(dead_counts)),
            'n_total': int(total_counts[0]),  # Should be same for all
            'dead_ratio': float(np.mean(dead_counts) / total_counts[0]),
            'activation_stats': {
                'mean': float(np.mean([m['dead_neurons']['activation_stats']['mean'] for m in metrics_list])),
                'std': float(np.mean([m['dead_neurons']['activation_stats']['std'] for m in metrics_list]))
            }
        }
        
        # Lifetime and population sparsity
        avg_metrics['lifetime_sparsity_stats'] = {
            'mean': float(np.mean([m['lifetime_sparsity_stats']['mean'] for m in metrics_list])),
            'std': float(np.mean([m['lifetime_sparsity_stats']['std'] for m in metrics_list]))
        }
        avg_metrics['population_sparsity_stats'] = {
            'mean': float(np.mean([m['population_sparsity_stats']['mean'] for m in metrics_list])),
            'std': float(np.mean([m['population_sparsity_stats']['std'] for m in metrics_list]))
        }
        
        return avg_metrics

    def analyze_epoch(self, epoch: int, split: str = 'val', analyze_masked: bool = False) -> Dict:
        """Analyze sparsity for a single epoch, optionally including masked embeddings."""
        print(f"\nAnalyzing epoch {epoch} ({split} split)...")
        
        # Load regular embeddings
        student_emb, teacher_emb, labels = self.load_embeddings(epoch, split)
        
        # Compute comparative analysis
        results = self.compare_student_teacher_sparsity(student_emb, teacher_emb)
        results['epoch'] = epoch
        results['split'] = split
        results['n_samples'] = student_emb.shape[0]
        results['n_features'] = student_emb.shape[1]
        
        # Analyze masked embeddings if requested
        if analyze_masked:
            try:
                masked_data = self.load_masked_embeddings(epoch, split)
                results['masked_analysis'] = self.analyze_masked_embeddings(masked_data)
                print(f"  Analyzed {len(masked_data)} mask samples")
            except FileNotFoundError as e:
                print(f"  Warning: {e}")
                results['masked_analysis'] = None
        
        return results
    
    def run_analysis(self, epochs: List[int] = [0, 1, 2, 3, 4], 
                    split: str = 'val') -> Dict:
        """Run complete sparsity analysis across epochs."""
        print("="*60)
        print("Sparsity Analysis for I-JEPA Embeddings")
        print("="*60)
        print(f"Analyzing epochs: {epochs}")
        print(f"Split: {split}")
        
        all_results = {}
        epoch_evolution = {
            'epochs': epochs,
            'student': {metric: [] for metric in ['l0_sparsity', 'l1_sparsity', 
                                                 'gini_coefficient', 'activation_ratio',
                                                 'dead_ratio']},
            'teacher': {metric: [] for metric in ['l0_sparsity', 'l1_sparsity', 
                                                 'gini_coefficient', 'activation_ratio',
                                                 'dead_ratio']}
        }
        
        # Analyze each epoch
        for epoch in tqdm(epochs, desc="Processing epochs"):
            results = self.analyze_epoch(epoch, split)
            all_results[f'epoch_{epoch}'] = results
            
            # Track evolution
            for network in ['student', 'teacher']:
                for metric in epoch_evolution[network]:
                    if metric == 'dead_ratio':
                        value = results[network]['dead_neurons']['dead_ratio']
                    else:
                        value = results[network][metric]
                    epoch_evolution[network][metric].append(value)
        
        # Save results
        output_file = self.output_dir / f'sparsity_analysis_{split}.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        # Create visualizations
        self.create_visualizations(epoch_evolution, all_results, split)
        
        # Print summary
        self.print_summary(all_results, epochs)
        
        return all_results
    
    def create_visualizations(self, evolution: Dict, all_results: Dict, split: str):
        """Create visualization plots for sparsity analysis."""
        print("\nCreating visualizations...")
        
        # 1. Sparsity evolution plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Sparsity Evolution Across Epochs ({split} split)', fontsize=16)
        
        metrics = ['l0_sparsity', 'l1_sparsity', 'gini_coefficient', 
                  'activation_ratio', 'dead_ratio']
        titles = ['L0 Sparsity', 'L1 Sparsity', 'Gini Coefficient', 
                 'Activation Ratio', 'Dead Neuron Ratio']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 3, idx % 3]
            
            ax.plot(evolution['epochs'], evolution['student'][metric], 
                   'o-', label='Student', linewidth=2, markersize=8)
            ax.plot(evolution['epochs'], evolution['teacher'][metric], 
                   's-', label='Teacher', linewidth=2, markersize=8)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'sparsity_evolution_{split}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Dead neuron analysis plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Dead neuron counts
        student_dead = [all_results[f'epoch_{e}']['student']['dead_neurons']['n_dead'] 
                       for e in evolution['epochs']]
        teacher_dead = [all_results[f'epoch_{e}']['teacher']['dead_neurons']['n_dead'] 
                       for e in evolution['epochs']]
        
        ax1.bar([e - 0.2 for e in evolution['epochs']], student_dead, 
               0.4, label='Student', alpha=0.7)
        ax1.bar([e + 0.2 for e in evolution['epochs']], teacher_dead, 
               0.4, label='Teacher', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Number of Dead Neurons')
        ax1.set_title('Dead Neurons Across Epochs')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Activation correlation
        correlations = [all_results[f'epoch_{e}']['activation_correlation'] 
                       for e in evolution['epochs']]
        ax2.plot(evolution['epochs'], correlations, 'o-', linewidth=2, markersize=8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Correlation')
        ax2.set_title('Student-Teacher Activation Pattern Correlation')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.suptitle(f'Activation Pattern Analysis ({split} split)', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'activation_analysis_{split}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: sparsity_evolution_{split}.png")
        print(f"  Saved: activation_analysis_{split}.png")
    
    def print_summary(self, all_results: Dict, epochs: List[int]):
        """Print analysis summary."""
        print("\n" + "="*60)
        print("SPARSITY ANALYSIS SUMMARY")
        print("="*60)
        
        print("\n1. Sparsity Trends:")
        print("   " + "-"*40)
        print("   Metric         | Network  | First→Last | Change")
        print("   " + "-"*40)
        
        metrics = [('L0 Sparsity', 'l0_sparsity'), 
                  ('L1 Sparsity', 'l1_sparsity'),
                  ('Gini Coeff', 'gini_coefficient'),
                  ('Dead Ratio', 'dead_neurons', 'dead_ratio')]
        
        for metric_name, *metric_path in metrics:
            for network in ['student', 'teacher']:
                if len(metric_path) == 2:
                    first = all_results[f'epoch_{epochs[0]}'][network][metric_path[0]][metric_path[1]]
                    last = all_results[f'epoch_{epochs[-1]}'][network][metric_path[0]][metric_path[1]]
                else:
                    first = all_results[f'epoch_{epochs[0]}'][network][metric_path[0]]
                    last = all_results[f'epoch_{epochs[-1]}'][network][metric_path[0]]
                
                change = last - first
                print(f"   {metric_name:14s} | {network:8s} | {first:.3f}→{last:.3f} | {change:+.3f}")
        
        print("\n2. Student vs Teacher Comparison (Final Epoch):")
        print("   " + "-"*40)
        final_epoch = f'epoch_{epochs[-1]}'
        final_results = all_results[final_epoch]
        
        print(f"   Activation Correlation: {final_results['activation_correlation']:.3f}")
        print(f"   Dead Neuron Overlap: {final_results['dead_neuron_overlap']['jaccard_similarity']:.3f}")
        print(f"   Both Dead: {final_results['dead_neuron_overlap']['n_both_dead']}")
        print(f"   Student Only Dead: {final_results['dead_neuron_overlap']['n_student_only_dead']}")
        print(f"   Teacher Only Dead: {final_results['dead_neuron_overlap']['n_teacher_only_dead']}")
        
        # If masked analysis was performed, add summary
        if analyze_masked and any(r.get('masked_analysis') for r in all_results.values()):
            self.print_masked_summary(all_results, epochs)
        
        print("\n" + "="*60)
    
    def print_masked_summary(self, all_results: Dict, epochs: List[int]):
        """Print summary of masked embedding analysis."""
        print("\n3. Masked Embedding Analysis Summary:")
        print("   " + "-"*40)
        
        # Find epochs with masked analysis
        masked_epochs = [(e, all_results[f'epoch_{e}']['masked_analysis']) 
                        for e in epochs 
                        if all_results[f'epoch_{e}'].get('masked_analysis')]
        
        if not masked_epochs:
            print("   No masked analysis found.")
            return
            
        print("   Metric         | Network  | Unmasked → Masked | Change")
        print("   " + "-"*40)
        
        for epoch, masked_analysis in masked_epochs:
            print(f"\n   Epoch {epoch} ({masked_analysis['n_mask_samples']} mask samples):")
            
            metrics = ['l1_sparsity', 'gini_coefficient', 'activation_ratio']
            
            for metric in metrics:
                for network in ['student', 'teacher']:
                    unmasked = all_results[f'epoch_{epoch}'][network][metric]
                    masked = masked_analysis[network][metric]
                    change = masked - unmasked
                    
                    metric_name = metric.replace('_', ' ').title()
                    print(f"   {metric_name:14s} | {network:8s} | {unmasked:.3f} → {masked:.3f} | {change:+.3f}")
            
            # Show variance across masks
            print(f"\n   Variance across masks (std):")
            for metric in metrics:
                student_std = masked_analysis['mask_variance'][metric]['student_std']
                teacher_std = masked_analysis['mask_variance'][metric]['teacher_std']
                metric_name = metric.replace('_', ' ').title()
                print(f"   {metric_name:14s} | Student: {student_std:.4f} | Teacher: {teacher_std:.4f}")
            
            # Activation correlation
            corr_mean = masked_analysis['activation_correlation']['mean']
            corr_std = masked_analysis['activation_correlation']['std']
            print(f"\n   Activation Correlation: {corr_mean:.3f} ± {corr_std:.3f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze sparsity in I-JEPA embeddings')
    parser.add_argument('--embeddings_dir', type=str, 
                       default='analysis/dim_reduction/embeddings',
                       help='Directory containing embeddings')
    parser.add_argument('--output_dir', type=str,
                       default='analysis/sparsity/results',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, nargs='+',
                       default=[0, 1, 2, 3, 4],
                       help='Epochs to analyze')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val'],
                       help='Dataset split to analyze')
    parser.add_argument('--analyze_masked', action='store_true',
                       help='Include analysis of masked embeddings')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = SparsityAnalyzer(args.embeddings_dir, args.output_dir)
    results = analyzer.run_analysis(args.epochs, args.split, analyze_masked=args.analyze_masked)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()