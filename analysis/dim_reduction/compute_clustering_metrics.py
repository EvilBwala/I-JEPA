#!/usr/bin/env python3
"""
Compute clustering metrics for dimensionality-reduced embeddings.
Compares actual class clustering to random baseline.
"""

import numpy as np
from pathlib import Path
import json
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score, v_measure_score
)
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse


class ClusteringMetrics:
    """Compute various clustering quality metrics."""
    
    def __init__(self, output_dir: str = "clustering_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_internal_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """Compute internal clustering metrics that don't require ground truth."""
        metrics = {}
        
        # Silhouette coefficient (-1 to 1, higher is better)
        # Measures how similar an object is to its own cluster compared to other clusters
        try:
            metrics['silhouette'] = float(silhouette_score(embeddings, labels))
        except:
            metrics['silhouette'] = np.nan
        
        # Davies-Bouldin index (lower is better)
        # Average similarity ratio of each cluster with its most similar cluster
        try:
            metrics['davies_bouldin'] = float(davies_bouldin_score(embeddings, labels))
        except:
            metrics['davies_bouldin'] = np.nan
        
        # Calinski-Harabasz index (higher is better)
        # Ratio of between-cluster to within-cluster dispersion
        try:
            metrics['calinski_harabasz'] = float(calinski_harabasz_score(embeddings, labels))
        except:
            metrics['calinski_harabasz'] = np.nan
        
        return metrics
    
    def compute_external_metrics(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
        """Compute external clustering metrics using ground truth."""
        metrics = {}
        
        # Adjusted Rand Index (-1 to 1, higher is better)
        metrics['adjusted_rand'] = float(adjusted_rand_score(true_labels, pred_labels))
        
        # Normalized Mutual Information (0 to 1, higher is better)
        metrics['normalized_mutual_info'] = float(normalized_mutual_info_score(true_labels, pred_labels))
        
        # V-measure (0 to 1, higher is better)
        # Harmonic mean of homogeneity and completeness
        metrics['v_measure'] = float(v_measure_score(true_labels, pred_labels))
        
        return metrics
    
    def compute_class_separation(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """Compute metrics for class separation quality."""
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # Compute class centroids
        centroids = np.zeros((n_classes, embeddings.shape[1]))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            centroids[i] = embeddings[mask].mean(axis=0)
        
        # Intra-class distances (compactness)
        intra_distances = []
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_points = embeddings[mask]
            if len(class_points) > 1:
                distances = cdist(class_points, [centroids[i]], metric='euclidean').flatten()
                intra_distances.extend(distances)
        
        # Inter-class distances (separation)
        inter_distances = []
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                inter_distances.append(dist)
        
        metrics = {
            'mean_intra_distance': float(np.mean(intra_distances)),
            'std_intra_distance': float(np.std(intra_distances)),
            'mean_inter_distance': float(np.mean(inter_distances)),
            'std_inter_distance': float(np.std(inter_distances)),
            'separation_ratio': float(np.mean(inter_distances) / (np.mean(intra_distances) + 1e-8))
        }
        
        return metrics
    
    def create_random_baseline(self, embeddings: np.ndarray, labels: np.ndarray, 
                             n_random_trials: int = 100) -> dict:
        """Create random baseline by shuffling labels."""
        n_samples = len(labels)
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        random_metrics = {
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': [],
            'separation_ratio': []
        }
        
        for _ in range(n_random_trials):
            # Random labels preserving class distribution
            random_labels = np.random.choice(unique_labels, size=n_samples)
            
            # Compute metrics
            internal = self.compute_internal_metrics(embeddings, random_labels)
            separation = self.compute_class_separation(embeddings, random_labels)
            
            for key in ['silhouette', 'davies_bouldin', 'calinski_harabasz']:
                if not np.isnan(internal[key]):
                    random_metrics[key].append(internal[key])
            random_metrics['separation_ratio'].append(separation['separation_ratio'])
        
        # Compute statistics
        baseline_stats = {}
        for key, values in random_metrics.items():
            if values:
                baseline_stats[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'percentile_5': float(np.percentile(values, 5)),
                    'percentile_95': float(np.percentile(values, 95))
                }
        
        return baseline_stats
    
    def analyze_embeddings(self, embeddings: np.ndarray, labels: np.ndarray, 
                         method: str, epoch: int, network: str) -> dict:
        """Analyze embeddings and compare to random baseline."""
        print(f"  Analyzing {network} {method} embeddings...")
        
        # Compute actual metrics
        internal_metrics = self.compute_internal_metrics(embeddings, labels)
        separation_metrics = self.compute_class_separation(embeddings, labels)
        
        # Perform K-means clustering
        n_classes = len(np.unique(labels))
        kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(embeddings)
        
        # External metrics comparing to true labels
        external_metrics = self.compute_external_metrics(labels, pred_labels)
        
        # Random baseline
        print(f"    Computing random baseline...")
        random_baseline = self.create_random_baseline(embeddings, labels)
        
        # Compute z-scores relative to random baseline
        z_scores = {}
        for metric in ['silhouette', 'calinski_harabasz', 'separation_ratio']:
            if metric in random_baseline and metric in {**internal_metrics, **separation_metrics}:
                actual_value = internal_metrics.get(metric, separation_metrics.get(metric))
                baseline_mean = random_baseline[metric]['mean']
                baseline_std = random_baseline[metric]['std']
                if baseline_std > 0:
                    z_scores[f'{metric}_z_score'] = float((actual_value - baseline_mean) / baseline_std)
        
        # For Davies-Bouldin, lower is better, so invert z-score
        if 'davies_bouldin' in random_baseline and 'davies_bouldin' in internal_metrics:
            actual_value = internal_metrics['davies_bouldin']
            baseline_mean = random_baseline['davies_bouldin']['mean']
            baseline_std = random_baseline['davies_bouldin']['std']
            if baseline_std > 0:
                z_scores['davies_bouldin_z_score'] = float((baseline_mean - actual_value) / baseline_std)
        
        results = {
            'method': method,
            'epoch': epoch,
            'network': network,
            'n_samples': len(labels),
            'n_classes': n_classes,
            'internal_metrics': internal_metrics,
            'separation_metrics': separation_metrics,
            'external_metrics': external_metrics,
            'random_baseline': random_baseline,
            'z_scores': z_scores
        }
        
        return results


def analyze_exp_embeddings(reduced_dir: Path, output_dir: Path, epochs: list, methods: list):
    """Analyze clustering metrics for exp folder embeddings."""
    
    metrics_analyzer = ClusteringMetrics(str(output_dir))
    all_results = {}
    
    print("Computing clustering metrics for dimensionality-reduced embeddings...")
    print(f"Epochs: {epochs}")
    print(f"Methods: {methods}")
    
    for epoch in epochs:
        print(f"\n--- Epoch {epoch} ---")
        epoch_dir = reduced_dir / f"epoch_{epoch:02d}"
        
        if not epoch_dir.exists():
            print(f"  Warning: Directory not found for epoch {epoch}")
            continue
        
        # Load labels
        labels = np.load(epoch_dir / "labels.npy")
        
        # Use averaged results across masks
        avg_dir = epoch_dir / "averaged"
        
        epoch_results = {}
        
        for method in methods:
            method_results = {}
            
            for network in ['student', 'teacher']:
                embedding_file = avg_dir / f"{network}_{method}.npy"
                
                if embedding_file.exists():
                    embeddings = np.load(embedding_file)
                    
                    # Analyze embeddings
                    results = metrics_analyzer.analyze_embeddings(
                        embeddings, labels, method, epoch, network
                    )
                    
                    method_results[network] = results
                else:
                    print(f"  Warning: {embedding_file} not found")
            
            epoch_results[method] = method_results
        
        all_results[f'epoch_{epoch:02d}'] = epoch_results
    
    # Save all results
    with open(output_dir / 'clustering_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create visualizations
    create_clustering_plots(all_results, output_dir, epochs, methods)
    
    # Print summary
    print_clustering_summary(all_results, epochs, methods)
    
    return all_results


def create_clustering_plots(results: dict, output_dir: Path, epochs: list, methods: list):
    """Create visualization plots for clustering metrics."""
    
    # Evolution plots for each metric
    metrics_to_plot = [
        ('silhouette', 'Silhouette Score', True),
        ('davies_bouldin', 'Davies-Bouldin Index', False),
        ('calinski_harabasz', 'Calinski-Harabasz Index', True),
        ('separation_ratio', 'Class Separation Ratio', True),
        ('v_measure', 'V-Measure Score', True)
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric, title, higher_better) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        for method in methods:
            for network in ['student', 'teacher']:
                values = []
                
                for epoch in epochs:
                    epoch_key = f'epoch_{epoch:02d}'
                    if epoch_key in results and method in results[epoch_key]:
                        if network in results[epoch_key][method]:
                            result = results[epoch_key][method][network]
                            
                            # Get metric value
                            if metric in result['internal_metrics']:
                                value = result['internal_metrics'][metric]
                            elif metric in result['separation_metrics']:
                                value = result['separation_metrics'][metric]
                            elif metric in result['external_metrics']:
                                value = result['external_metrics'][metric]
                            else:
                                value = np.nan
                            
                            values.append(value)
                        else:
                            values.append(np.nan)
                
                # Plot
                label = f'{method.upper()} {network.capitalize()}'
                style = '-' if network == 'student' else '--'
                ax.plot(epochs[:len(values)], values, style, marker='o', label=label, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} Evolution', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)
        
        # Add arrow to indicate better direction
        if higher_better:
            ax.annotate('Better →', xy=(0.95, 0.95), xycoords='axes fraction',
                       ha='right', va='top', fontsize=10, color='green',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))
        else:
            ax.annotate('← Better', xy=(0.05, 0.95), xycoords='axes fraction',
                       ha='left', va='top', fontsize=10, color='green',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))
    
    # Z-score comparison plot
    ax = axes[5]
    
    # Collect z-scores for last epoch
    last_epoch = f'epoch_{epochs[-1]:02d}'
    z_score_data = []
    
    if last_epoch in results:
        for method in methods:
            if method in results[last_epoch]:
                for network in ['student', 'teacher']:
                    if network in results[last_epoch][method]:
                        z_scores = results[last_epoch][method][network]['z_scores']
                        for metric, z_score in z_scores.items():
                            z_score_data.append({
                                'Method': method.upper(),
                                'Network': network.capitalize(),
                                'Metric': metric.replace('_z_score', ''),
                                'Z-Score': z_score
                            })
    
    if z_score_data:
        import pandas as pd
        df = pd.DataFrame(z_score_data)
        
        # Create grouped bar plot
        metric_names = df['Metric'].unique()
        x = np.arange(len(metric_names))
        width = 0.15
        
        for i, (method, network) in enumerate([('pca', 'student'), ('pca', 'teacher'), 
                                               ('umap', 'student'), ('umap', 'teacher')]):
            mask = (df['Method'] == method.upper()) & (df['Network'] == network.capitalize())
            values = []
            for metric in metric_names:
                metric_mask = mask & (df['Metric'] == metric)
                if metric_mask.any():
                    values.append(df[metric_mask]['Z-Score'].values[0])
                else:
                    values.append(0)
            
            offset = (i - 1.5) * width
            label = f'{method.upper()} {network.capitalize()}'
            ax.bar(x + offset, values, width, label=label, alpha=0.8)
        
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Z-Score vs Random', fontsize=12)
        ax.set_title(f'Clustering Quality vs Random Baseline (Epoch {epochs[-1]})', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names], rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Significant (z>2)')
        ax.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('Clustering Metrics Analysis for Dimensionality-Reduced Embeddings', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'clustering_metrics_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nCreated visualization: clustering_metrics_evolution.png")


def print_clustering_summary(results: dict, epochs: list, methods: list):
    """Print summary of clustering metrics."""
    
    print("\n" + "="*60)
    print("CLUSTERING METRICS SUMMARY")
    print("="*60)
    
    # Summary for last epoch
    last_epoch = f'epoch_{epochs[-1]:02d}'
    
    if last_epoch in results:
        print(f"\nFinal Epoch ({epochs[-1]}) Results:")
        print("-"*40)
        
        for method in methods:
            print(f"\n{method.upper()} Method:")
            
            if method in results[last_epoch]:
                for network in ['student', 'teacher']:
                    if network in results[last_epoch][method]:
                        result = results[last_epoch][method][network]
                        
                        print(f"\n  {network.capitalize()} Network:")
                        
                        # Internal metrics
                        internal = result['internal_metrics']
                        print(f"    Silhouette Score: {internal['silhouette']:.4f}")
                        print(f"    Davies-Bouldin Index: {internal['davies_bouldin']:.4f}")
                        print(f"    Calinski-Harabasz Index: {internal['calinski_harabasz']:.1f}")
                        
                        # External metrics
                        external = result['external_metrics']
                        print(f"    V-Measure: {external['v_measure']:.4f}")
                        print(f"    Adjusted Rand Index: {external['adjusted_rand']:.4f}")
                        
                        # Z-scores
                        z_scores = result['z_scores']
                        print(f"    Z-scores vs Random:")
                        for metric, z_score in z_scores.items():
                            print(f"      {metric}: {z_score:.2f}")
    
    # Best performing configuration
    print("\n" + "="*60)
    print("BEST PERFORMING CONFIGURATIONS")
    print("="*60)
    
    best_configs = {
        'silhouette': {'value': -1, 'config': ''},
        'v_measure': {'value': 0, 'config': ''},
        'separation_ratio': {'value': 0, 'config': ''}
    }
    
    for epoch_key in results:
        epoch = int(epoch_key.split('_')[1])
        for method in methods:
            if method in results[epoch_key]:
                for network in ['student', 'teacher']:
                    if network in results[epoch_key][method]:
                        result = results[epoch_key][method][network]
                        config = f"Epoch {epoch}, {method.upper()} {network}"
                        
                        # Check silhouette
                        if result['internal_metrics']['silhouette'] > best_configs['silhouette']['value']:
                            best_configs['silhouette']['value'] = result['internal_metrics']['silhouette']
                            best_configs['silhouette']['config'] = config
                        
                        # Check v_measure
                        if result['external_metrics']['v_measure'] > best_configs['v_measure']['value']:
                            best_configs['v_measure']['value'] = result['external_metrics']['v_measure']
                            best_configs['v_measure']['config'] = config
                        
                        # Check separation ratio
                        if result['separation_metrics']['separation_ratio'] > best_configs['separation_ratio']['value']:
                            best_configs['separation_ratio']['value'] = result['separation_metrics']['separation_ratio']
                            best_configs['separation_ratio']['config'] = config
    
    for metric, info in best_configs.items():
        print(f"\nBest {metric}: {info['value']:.4f}")
        print(f"  Configuration: {info['config']}")


def main():
    parser = argparse.ArgumentParser(description='Compute clustering metrics for reduced embeddings')
    parser.add_argument('--reduced_dir', type=str, 
                       default='reduced_embeddings_exp',
                       help='Directory containing reduced embeddings')
    parser.add_argument('--output_dir', type=str, 
                       default='clustering_results',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, nargs='+', 
                       default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                       help='Epochs to analyze')
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['pca', 'umap'],
                       help='Reduction methods to analyze')
    
    args = parser.parse_args()
    
    reduced_dir = Path(args.reduced_dir)
    output_dir = Path(args.output_dir)
    
    analyze_exp_embeddings(reduced_dir, output_dir, args.epochs, args.methods)


if __name__ == "__main__":
    main()