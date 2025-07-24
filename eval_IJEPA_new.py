import os
import torch
import numpy as np
import argparse
from model_new import IJEPA, TinyImageNetDataModule
import pytorch_lightning as pl
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def extract_features(model, dataloader, device):
    """
    Extract features from a pretrained model
    """
    features = []
    labels = []
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # Handle both labeled and unlabeled data
            if isinstance(batch, list) and len(batch) == 2:
                x, label = batch
                labels.extend(label.cpu().numpy())
            else:
                x = batch
            
            # Set model to test mode to get embeddings
            model.model.mode = "test"
            
            # Forward pass
            x = x.to(device)
            feat = model.model(x)
            
            # Average pooling over patches
            feat = feat.mean(dim=1)
            features.append(feat.cpu().numpy())
    
    features = np.vstack(features)
    return features, np.array(labels) if labels else None

def visualize_features(features, labels=None, n_components=2, method='pca'):
    """
    Visualize features using dimensionality reduction
    """
    # Reduce dimensionality
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced_features = reducer.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    if labels is not None and len(np.unique(labels)) <= 20:
        # If we have labels and not too many classes, color by class
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                reduced_features[mask, 0],
                reduced_features[mask, 1],
                label=f"Class {label}",
                alpha=0.6
            )
        plt.legend()
    else:
        # Otherwise just plot all points
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.6)
    
    plt.title(f"Feature visualization using {method.upper()}")
    plt.savefig(f"feature_visualization_{method}.png")
    plt.close()
    
    return reduced_features

def evaluate_clustering(features, true_labels=None, n_clusters=200):
    """
    Evaluate clustering performance
    """
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    results = {
        "inertia": kmeans.inertia_,
    }
    
    # If we have true labels, compute clustering metrics
    if true_labels is not None:
        results["adjusted_rand_index"] = adjusted_rand_score(true_labels, cluster_labels)
        results["normalized_mutual_info"] = normalized_mutual_info_score(true_labels, cluster_labels)
    
    return results, cluster_labels

def main(args):
    """
    Main evaluation function
    """
    # Set random seeds for reproducibility
    pl.seed_everything(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # Create data module
    data_module = TinyImageNetDataModule(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )
    
    # Setup data module
    data_module.setup()
    
    # Load model from checkpoint
    print(f"Loading model from {args.checkpoint}")
    model = IJEPA.load_from_checkpoint(args.checkpoint)
    
    # Extract features
    print("Extracting features from validation set...")
    val_loader = data_module.val_dataloader()
    features, labels = extract_features(model, val_loader, device)
    
    print(f"Extracted features shape: {features.shape}")
    
    # Visualize features
    if args.visualize:
        print("Visualizing features...")
        visualize_features(features, labels, method='pca')
        visualize_features(features, labels, method='tsne')
    
    # Evaluate clustering
    if args.evaluate_clustering:
        print("Evaluating clustering...")
        n_clusters = len(np.unique(labels)) if labels is not None else args.n_clusters
        results, cluster_labels = evaluate_clustering(features, labels, n_clusters=n_clusters)
        
        print("Clustering results:")
        for k, v in results.items():
            print(f"  {k}: {v}")
    
    return features, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pretrained I-JEPA model")
    
    # Model and dataset parameters
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--dataset_path", type=str, default="tiny-imagenet-200",
                        help="Path to the TinyImageNet dataset")
    parser.add_argument("--img_size", type=int, default=64,
                        help="Image size")
    
    # Evaluation parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize features using PCA and t-SNE")
    parser.add_argument("--evaluate_clustering", action="store_true",
                        help="Evaluate clustering performance")
    parser.add_argument("--n_clusters", type=int, default=200,
                        help="Number of clusters for K-means (default: 200 for TinyImageNet)")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU instead of GPU")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    main(args) 