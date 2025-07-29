#!/usr/bin/env python3
"""
Analyze label-aware metrics for dimensionality-reduced embeddings.
Focuses on metrics that directly measure class preservation.
"""

import numpy as np
from pathlib import Path
import json
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, v_measure_score,
    homogeneity_score, completeness_score, fowlkes_mallows_score,
    adjusted_mutual_info_score
)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse


class LabelAwareMetrics:
    """Compute label-aware clustering and classification metrics."""
    
    def __init__(self, output_dir: str = "label_aware_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_clustering_agreement(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
        """Compute various clustering agreement metrics."""
        metrics = {}
        
        # Adjusted Rand Index - chance-corrected measure of agreement
        metrics['adjusted_rand_index'] = float(adjusted_rand_score(true_labels, pred_labels))
        
        # Normalized Mutual Information - information-theoretic measure
        metrics['normalized_mutual_info'] = float(normalized_mutual_info_score(true_labels, pred_labels))
        
        # Adjusted Mutual Information - AMI with chance correction
        metrics['adjusted_mutual_info'] = float(adjusted_mutual_info_score(true_labels, pred_labels))
        
        # V-measure - harmonic mean of homogeneity and completeness
        metrics['v_measure'] = float(v_measure_score(true_labels, pred_labels))
        
        # Homogeneity - each cluster contains only members of a single class
        metrics['homogeneity'] = float(homogeneity_score(true_labels, pred_labels))
        
        # Completeness - all members of a class are assigned to the same cluster
        metrics['completeness'] = float(completeness_score(true_labels, pred_labels))
        
        # Fowlkes-Mallows Index - geometric mean of precision and recall
        metrics['fowlkes_mallows'] = float(fowlkes_mallows_score(true_labels, pred_labels))
        
        return metrics
    
    def compute_knn_accuracy(self, embeddings: np.ndarray, labels: np.ndarray, 
                           k_values: list = [1, 5, 10, 20]) -> dict:
        """Compute k-NN classification accuracy."""
        results = {}
        
        for k in k_values:
            if k < len(np.unique(labels)):
                knn = KNeighborsClassifier(n_neighbors=k)
                # 5-fold cross-validation
                scores = cross_val_score(knn, embeddings, labels, cv=5, scoring='accuracy')
                results[f'knn_{k}_accuracy'] = float(np.mean(scores))
                results[f'knn_{k}_std'] = float(np.std(scores))
        
        return results
    
    def compute_class_purity(self, embeddings: np.ndarray, labels: np.ndarray, 
                           n_neighbors: int = 10) -> dict:
        """Compute class purity in local neighborhoods."""
        n_samples = len(labels)
        
        # Find k nearest neighbors for each point
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 to include self
        nbrs.fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        # Compute purity for each point
        purities = []
        for i in range(n_samples):
            neighbor_labels = labels[indices[i, 1:]]  # Exclude self
            same_class = np.sum(neighbor_labels == labels[i])
            purity = same_class / n_neighbors
            purities.append(purity)
        
        results = {
            'mean_local_purity': float(np.mean(purities)),
            'std_local_purity': float(np.std(purities)),
            'min_local_purity': float(np.min(purities)),
            'max_local_purity': float(np.max(purities))
        }
        
        return results
    
    def compute_cluster_entropy(self, true_labels: np.ndarray, cluster_labels: np.ndarray) -> dict:
        """Compute entropy of true labels within each cluster."""
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        cluster_entropies = []
        cluster_sizes = []
        
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_true_labels = true_labels[mask]
            cluster_size = len(cluster_true_labels)
            
            if cluster_size > 0:
                # Count occurrences of each true label
                _, counts = np.unique(cluster_true_labels, return_counts=True)
                probabilities = counts / cluster_size
                
                # Compute entropy
                cluster_entropy = entropy(probabilities)
                cluster_entropies.append(cluster_entropy)
                cluster_sizes.append(cluster_size)
        
        # Weighted average entropy
        cluster_entropies = np.array(cluster_entropies)
        cluster_sizes = np.array(cluster_sizes)
        weighted_entropy = np.sum(cluster_entropies * cluster_sizes) / np.sum(cluster_sizes)
        
        results = {
            'mean_cluster_entropy': float(np.mean(cluster_entropies)),
            'weighted_cluster_entropy': float(weighted_entropy),
            'max_cluster_entropy': float(np.max(cluster_entropies)),
            'n_clusters': n_clusters
        }
        
        return results
    
    def analyze_multiple_clusterings(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """Try multiple clustering algorithms and compare to true labels."""
        n_classes = len(np.unique(labels))
        all_metrics = {}
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(embeddings)
        all_metrics['kmeans'] = self.compute_clustering_agreement(labels, kmeans_labels)
        all_metrics['kmeans'].update(self.compute_cluster_entropy(labels, kmeans_labels))
        
        # Agglomerative clustering
        agglo = AgglomerativeClustering(n_clusters=n_classes)
        agglo_labels = agglo.fit_predict(embeddings)
        all_metrics['agglomerative'] = self.compute_clustering_agreement(labels, agglo_labels)
        all_metrics['agglomerative'].update(self.compute_cluster_entropy(labels, agglo_labels))
        
        # DBSCAN (density-based)
        # Estimate eps using k-distance graph
        from sklearn.neighbors import NearestNeighbors
        k = min(5, len(embeddings) - 1)
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        k_distances = distances[:, -1]
        eps = np.percentile(k_distances, 90)  # Use 90th percentile as eps
        
        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan_labels = dbscan.fit_predict(embeddings)
        
        # Only compute metrics if DBSCAN found clusters
        if len(np.unique(dbscan_labels)) > 1:
            # Filter out noise points (-1)
            mask = dbscan_labels != -1
            if np.sum(mask) > 0:
                all_metrics['dbscan'] = self.compute_clustering_agreement(
                    labels[mask], dbscan_labels[mask]
                )
                all_metrics['dbscan']['noise_ratio'] = float(np.mean(dbscan_labels == -1))
        
        return all_metrics
    
    def compute_class_separability(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """Compute metrics for class separability."""
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # Compute class centroids
        centroids = np.zeros((n_classes, embeddings.shape[1]))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            centroids[i] = embeddings[mask].mean(axis=0)
        
        # Within-class scatter
        within_class_scatter = 0
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_points = embeddings[mask]
            if len(class_points) > 1:
                # Sum of squared distances to centroid
                scatter = np.sum(np.linalg.norm(class_points - centroids[i], axis=1) ** 2)
                within_class_scatter += scatter
        
        # Between-class scatter
        global_centroid = embeddings.mean(axis=0)
        between_class_scatter = 0
        for i, label in enumerate(unique_labels):
            n_samples = np.sum(labels == label)
            between_class_scatter += n_samples * np.linalg.norm(centroids[i] - global_centroid) ** 2
        
        # Fisher's discriminant ratio
        fisher_ratio = between_class_scatter / (within_class_scatter + 1e-8)
        
        # Average pairwise centroid distance
        from scipy.spatial.distance import pdist
        centroid_distances = pdist(centroids, metric='euclidean')
        
        results = {
            'fisher_ratio': float(fisher_ratio),
            'within_class_scatter': float(within_class_scatter),
            'between_class_scatter': float(between_class_scatter),
            'mean_centroid_distance': float(np.mean(centroid_distances)),
            'min_centroid_distance': float(np.min(centroid_distances)),
            'max_centroid_distance': float(np.max(centroid_distances))
        }
        
        return results


def analyze_epoch_embeddings(reduced_dir: Path, output_dir: Path, epochs: list, methods: list):
    """Analyze label-aware metrics for all epochs and methods."""
    
    analyzer = LabelAwareMetrics(str(output_dir))
    all_results = {}
    
    print("Computing label-aware metrics for dimensionality-reduced embeddings...")
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
            print(f"  Analyzing {method}...")
            method_results = {}
            
            for network in ['student', 'teacher']:
                embedding_file = avg_dir / f"{network}_{method}.npy"
                
                if embedding_file.exists():
                    embeddings = np.load(embedding_file)
                    
                    print(f"    {network} network...")
                    
                    # Compute all metrics
                    results = {}
                    
                    # k-NN accuracy
                    results.update(analyzer.compute_knn_accuracy(embeddings, labels))
                    
                    # Local purity
                    results.update(analyzer.compute_class_purity(embeddings, labels))
                    
                    # Multiple clustering algorithms
                    clustering_results = analyzer.analyze_multiple_clusterings(embeddings, labels)
                    results['clustering'] = clustering_results
                    
                    # Class separability
                    results.update(analyzer.compute_class_separability(embeddings, labels))
                    
                    method_results[network] = results
                else:
                    print(f"    Warning: {embedding_file} not found")
            
            epoch_results[method] = method_results
        
        all_results[f'epoch_{epoch:02d}'] = epoch_results
    
    # Save results
    with open(output_dir / 'label_aware_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create visualizations
    create_label_aware_plots(all_results, output_dir, epochs, methods)
    
    # Print summary
    print_label_aware_summary(all_results, epochs, methods)
    
    return all_results


def create_label_aware_plots(results: dict, output_dir: Path, epochs: list, methods: list):
    """Create visualization plots for label-aware metrics."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. k-NN accuracy evolution
    ax = axes[0, 0]
    for method in methods:
        for network in ['student', 'teacher']:
            for k in [1, 5, 10]:
                values = []
                for epoch in epochs:
                    epoch_key = f'epoch_{epoch:02d}'
                    if (epoch_key in results and method in results[epoch_key] and 
                        network in results[epoch_key][method]):
                        value = results[epoch_key][method][network].get(f'knn_{k}_accuracy', np.nan)
                        values.append(value)
                    else:
                        values.append(np.nan)
                
                label = f'{method.upper()} {network} (k={k})'
                style = '-' if k == 5 else ('--' if k == 1 else ':')
                if len(values) > 0:
                    ax.plot(epochs[:len(values)], values, style, 
                           label=label, alpha=0.8, linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('k-NN Accuracy')
    ax.set_title('k-NN Classification Accuracy')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    
    # 2. Local purity
    ax = axes[0, 1]
    for method in methods:
        for network in ['student', 'teacher']:
            values = []
            for epoch in epochs:
                epoch_key = f'epoch_{epoch:02d}'
                if (epoch_key in results and method in results[epoch_key] and 
                    network in results[epoch_key][method]):
                    value = results[epoch_key][method][network].get('mean_local_purity', np.nan)
                    values.append(value)
                else:
                    values.append(np.nan)
            
            label = f'{method.upper()} {network}'
            style = '-' if network == 'student' else '--'
            marker = 'o' if method == 'pca' else 's'
            ax.plot(epochs[:len(values)], values, style, marker=marker,
                   label=label, linewidth=2, markersize=8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Local Purity')
    ax.set_title('Class Purity in Local Neighborhoods (k=10)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    
    # 3. V-measure comparison
    ax = axes[0, 2]
    clustering_methods = ['kmeans', 'agglomerative']
    
    for method in methods:
        for network in ['student', 'teacher']:
            for cluster_method in clustering_methods:
                values = []
                for epoch in epochs:
                    epoch_key = f'epoch_{epoch:02d}'
                    if (epoch_key in results and method in results[epoch_key] and 
                        network in results[epoch_key][method] and
                        'clustering' in results[epoch_key][method][network]):
                        clustering = results[epoch_key][method][network]['clustering']
                        if cluster_method in clustering:
                            value = clustering[cluster_method].get('v_measure', np.nan)
                            values.append(value)
                        else:
                            values.append(np.nan)
                    else:
                        values.append(np.nan)
                
                if any(not np.isnan(v) for v in values):
                    label = f'{method.upper()} {network} ({cluster_method})'
                    style = '-' if cluster_method == 'kmeans' else '--'
                    ax.plot(epochs[:len(values)], values, style, 
                           label=label, alpha=0.8, linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('V-Measure Score')
    ax.set_title('Clustering Agreement with True Labels')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    
    # 4. Fisher ratio
    ax = axes[1, 0]
    for method in methods:
        for network in ['student', 'teacher']:
            values = []
            for epoch in epochs:
                epoch_key = f'epoch_{epoch:02d}'
                if (epoch_key in results and method in results[epoch_key] and 
                    network in results[epoch_key][method]):
                    value = results[epoch_key][method][network].get('fisher_ratio', np.nan)
                    values.append(value)
                else:
                    values.append(np.nan)
            
            label = f'{method.upper()} {network}'
            style = '-' if network == 'student' else '--'
            marker = 'o' if method == 'pca' else 's'
            ax.plot(epochs[:len(values)], values, style, marker=marker,
                   label=label, linewidth=2, markersize=8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel("Fisher's Discriminant Ratio")
    ax.set_title('Class Separability (Between/Within Scatter)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    
    # 5. Cluster entropy
    ax = axes[1, 1]
    for method in methods:
        for network in ['student', 'teacher']:
            values = []
            for epoch in epochs:
                epoch_key = f'epoch_{epoch:02d}'
                if (epoch_key in results and method in results[epoch_key] and 
                    network in results[epoch_key][method] and
                    'clustering' in results[epoch_key][method][network]):
                    clustering = results[epoch_key][method][network]['clustering']
                    if 'kmeans' in clustering:
                        value = clustering['kmeans'].get('weighted_cluster_entropy', np.nan)
                        values.append(value)
                    else:
                        values.append(np.nan)
                else:
                    values.append(np.nan)
            
            if any(not np.isnan(v) for v in values):
                label = f'{method.upper()} {network}'
                style = '-' if network == 'student' else '--'
                marker = 'o' if method == 'pca' else 's'
                ax.plot(epochs[:len(values)], values, style, marker=marker,
                       label=label, linewidth=2, markersize=8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weighted Cluster Entropy')
    ax.set_title('Label Entropy within Clusters (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    
    # 6. Summary heatmap
    ax = axes[1, 2]
    
    # Create summary matrix for final epoch
    final_epoch = f'epoch_{epochs[-1]:02d}'
    if final_epoch in results:
        summary_data = []
        row_labels = []
        
        metrics_to_show = ['knn_5_accuracy', 'mean_local_purity', 'fisher_ratio']
        metric_names = ['5-NN Accuracy', 'Local Purity', 'Fisher Ratio']
        
        for method in methods:
            for network in ['student', 'teacher']:
                if method in results[final_epoch] and network in results[final_epoch][method]:
                    row_data = []
                    for metric in metrics_to_show:
                        value = results[final_epoch][method][network].get(metric, 0)
                        row_data.append(value)
                    
                    # Add best clustering v-measure
                    clustering = results[final_epoch][method][network].get('clustering', {})
                    best_v_measure = 0
                    for cluster_method in ['kmeans', 'agglomerative']:
                        if cluster_method in clustering:
                            v_measure = clustering[cluster_method].get('v_measure', 0)
                            best_v_measure = max(best_v_measure, v_measure)
                    row_data.append(best_v_measure)
                    
                    summary_data.append(row_data)
                    row_labels.append(f'{method.upper()}\n{network}')
        
        summary_data = np.array(summary_data).T
        
        # Normalize each metric to 0-1 for better visualization
        for i in range(summary_data.shape[0]):
            row_min = summary_data[i].min()
            row_max = summary_data[i].max()
            if row_max > row_min:
                summary_data[i] = (summary_data[i] - row_min) / (row_max - row_min)
        
        im = ax.imshow(summary_data, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(row_labels)))
        ax.set_xticklabels(row_labels, rotation=45, ha='right')
        ax.set_yticks(range(len(metric_names) + 1))
        ax.set_yticklabels(metric_names + ['Best V-Measure'])
        ax.set_title(f'Normalized Metrics Summary (Epoch {epochs[-1]})')
        
        # Add text annotations
        for i in range(summary_data.shape[0]):
            for j in range(summary_data.shape[1]):
                text = ax.text(j, i, f'{summary_data[i, j]:.2f}',
                             ha='center', va='center',
                             color='white' if summary_data[i, j] < 0.5 else 'black')
        
        plt.colorbar(im, ax=ax, label='Normalized Score')
    
    plt.suptitle('Label-Aware Metrics Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'label_aware_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nCreated visualization: label_aware_metrics.png")


def print_label_aware_summary(results: dict, epochs: list, methods: list):
    """Print summary of label-aware metrics."""
    
    print("\n" + "="*60)
    print("LABEL-AWARE METRICS SUMMARY")
    print("="*60)
    
    # Final epoch summary
    final_epoch = f'epoch_{epochs[-1]:02d}'
    
    if final_epoch in results:
        print(f"\nFinal Epoch ({epochs[-1]}) Results:")
        print("-"*40)
        
        for method in methods:
            print(f"\n{method.upper()} Method:")
            
            if method in results[final_epoch]:
                for network in ['student', 'teacher']:
                    if network in results[final_epoch][method]:
                        result = results[final_epoch][method][network]
                        
                        print(f"\n  {network.capitalize()} Network:")
                        
                        # k-NN accuracy
                        print(f"    k-NN Accuracy (k=5): {result.get('knn_5_accuracy', 0):.3f}")
                        
                        # Local purity
                        print(f"    Mean Local Purity: {result.get('mean_local_purity', 0):.3f}")
                        
                        # Fisher ratio
                        print(f"    Fisher Ratio: {result.get('fisher_ratio', 0):.3f}")
                        
                        # Best clustering performance
                        if 'clustering' in result:
                            best_v_measure = 0
                            best_method = ''
                            for cluster_method, metrics in result['clustering'].items():
                                if 'v_measure' in metrics and metrics['v_measure'] > best_v_measure:
                                    best_v_measure = metrics['v_measure']
                                    best_method = cluster_method
                            
                            print(f"    Best V-Measure: {best_v_measure:.3f} ({best_method})")
    
    # Best configurations across all epochs
    print("\n" + "="*60)
    print("BEST CONFIGURATIONS")
    print("="*60)
    
    best_configs = {
        'knn_accuracy': {'value': 0, 'config': ''},
        'local_purity': {'value': 0, 'config': ''},
        'v_measure': {'value': 0, 'config': ''},
        'fisher_ratio': {'value': 0, 'config': ''}
    }
    
    for epoch_key in results:
        epoch = int(epoch_key.split('_')[1])
        for method in methods:
            if method in results[epoch_key]:
                for network in ['student', 'teacher']:
                    if network in results[epoch_key][method]:
                        result = results[epoch_key][method][network]
                        config = f"Epoch {epoch}, {method.upper()} {network}"
                        
                        # Check k-NN accuracy
                        knn_acc = result.get('knn_5_accuracy', 0)
                        if knn_acc > best_configs['knn_accuracy']['value']:
                            best_configs['knn_accuracy']['value'] = knn_acc
                            best_configs['knn_accuracy']['config'] = config
                        
                        # Check local purity
                        purity = result.get('mean_local_purity', 0)
                        if purity > best_configs['local_purity']['value']:
                            best_configs['local_purity']['value'] = purity
                            best_configs['local_purity']['config'] = config
                        
                        # Check Fisher ratio
                        fisher = result.get('fisher_ratio', 0)
                        if fisher > best_configs['fisher_ratio']['value']:
                            best_configs['fisher_ratio']['value'] = fisher
                            best_configs['fisher_ratio']['config'] = config
                        
                        # Check V-measure
                        if 'clustering' in result:
                            for cluster_method, metrics in result['clustering'].items():
                                if 'v_measure' in metrics:
                                    v_measure = metrics['v_measure']
                                    if v_measure > best_configs['v_measure']['value']:
                                        best_configs['v_measure']['value'] = v_measure
                                        best_configs['v_measure']['config'] = f"{config} ({cluster_method})"
    
    for metric, info in best_configs.items():
        metric_name = metric.replace('_', ' ').title()
        print(f"\nBest {metric_name}: {info['value']:.3f}")
        print(f"  Configuration: {info['config']}")


def main():
    parser = argparse.ArgumentParser(description='Compute label-aware metrics for reduced embeddings')
    parser.add_argument('--reduced_dir', type=str, 
                       default='reduced_embeddings_exp',
                       help='Directory containing reduced embeddings')
    parser.add_argument('--output_dir', type=str, 
                       default='label_aware_results',
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
    
    analyze_epoch_embeddings(reduced_dir, output_dir, args.epochs, args.methods)


if __name__ == "__main__":
    main()