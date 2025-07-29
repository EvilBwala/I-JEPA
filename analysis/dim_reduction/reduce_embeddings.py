#!/usr/bin/env python3
"""
Dimensionality reduction script for I-JEPA embeddings using driada library.
Applies PCA, UMAP, and Laplacian Eigenmaps to student and teacher embeddings across all epochs.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from driada.dim_reduction import MVData
from driada.utils.data import rescale

class EmbeddingReducer:
    """Performs dimensionality reduction on I-JEPA embeddings using driada."""
    
    def __init__(
        self,
        embeddings_dir: str = "analysis/dim_reduction/embeddings",
        output_dir: str = "analysis/dim_reduction/reduced_embeddings",
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        normalize: bool = True,
        random_state: int = 42
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.normalize = normalize
        self.random_state = random_state
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_embeddings(self, epoch: int, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load student and teacher embeddings for a given epoch and split."""
        # Check if we're processing masked embeddings
        if 'masked' in str(self.embeddings_dir):
            # For masked embeddings, structure is different
            # Assume path like: .../masked/val/ijepa_75/mask_00
            split_dir = self.embeddings_dir
            
            # Load masked embeddings
            student_file = "student_embeddings_masked_pooled.npy"
            teacher_file = "teacher_embeddings_pooled.npy"
        else:
            # Original structure for unmasked embeddings
            epoch_dir = self.embeddings_dir / f"ijepa-64px-epoch={epoch:02d}"
            
            # Handle the nested directory structure
            if epoch < 4:
                epoch_dir = epoch_dir / f"ijepa-64px-epoch={epoch:02d}"
            
            split_dir = epoch_dir / split
            
            # Load embeddings and labels
            student_file = "student_embeddings_pooled.npy"
            teacher_file = "teacher_embeddings_pooled.npy"
        
        student_emb = np.load(split_dir / student_file)
        teacher_emb = np.load(split_dir / teacher_file)
        labels = np.load(split_dir / "labels.npy")
        
        return student_emb, teacher_emb, labels
    
    def reduce_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        method: str,
        fit_mvdata: Optional['MVData'] = None
    ) -> Tuple[np.ndarray, 'MVData']:
        """Apply dimensionality reduction to embeddings.
        
        Returns:
            Tuple of (reduced_embeddings, mvdata_object)
        """
        # driada expects data in shape (n_features, n_samples)
        # Our embeddings are (n_samples, n_features), so transpose
        embeddings_T = embeddings.T
        
        # Create MVData object
        mvdata = MVData(embeddings_T, labels=labels, rescale_rows=self.normalize)
        
        # Apply reduction method
        if method == 'pca':
            embedding = mvdata.get_embedding(
                method='pca',
                dim=self.n_components
            )
        elif method == 'umap':
            # For UMAP with fit/transform capability
            if fit_mvdata is not None:
                # Use the same UMAP parameters as the fit data
                # This is a limitation of driada - no direct fit/transform
                # So we'll use the same parameters and hope for consistency
                embedding = mvdata.get_embedding(
                    method='umap',
                    dim=self.n_components,
                    n_neighbors=self.n_neighbors,
                    min_dist=self.min_dist,
                    metric=self.metric,
                    random_state=self.random_state
                )
            else:
                embedding = mvdata.get_embedding(
                    method='umap',
                    dim=self.n_components,
                    n_neighbors=self.n_neighbors,
                    min_dist=self.min_dist,
                    metric=self.metric,
                    random_state=self.random_state
                )
        elif method == 'laplacian':
            embedding = mvdata.get_embedding(
                method='le',  # 'le' is Laplacian Eigenmaps in driada
                dim=self.n_components,
                n_neighbors=self.n_neighbors,
                metric=self.metric
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Get coordinates and transpose back to (n_samples, n_components)
        reduced = embedding.coords.T
        
        return reduced, mvdata
    
    def process_epoch(
        self,
        epoch: int,
        splits: List[str] = ['train', 'val'],
        methods: List[str] = ['pca', 'umap', 'laplacian']
    ) -> Dict:
        """Process all embeddings for a single epoch."""
        results = {}
        
        for split in splits:
            print(f"  Processing {split} split...")
            
            # Load embeddings
            student_emb, teacher_emb, labels = self.load_embeddings(epoch, split)
            
            # Filter by number of classes if specified
            if hasattr(self, 'num_classes') and self.num_classes is not None:
                class_mask = labels < self.num_classes
                student_emb = student_emb[class_mask]
                teacher_emb = teacher_emb[class_mask]
                labels = labels[class_mask]
                print(f"    Filtered to {len(labels)} samples from {self.num_classes} classes")
            
            results[split] = {
                'labels': labels,
                'student': {},
                'teacher': {}
            }
            
            # Apply each reduction method
            for method in methods:
                print(f"    Applying {method}...")
                
                try:
                    # Reduce student embeddings
                    student_reduced, _ = self.reduce_embeddings(student_emb, labels, method)
                    results[split]['student'][method] = student_reduced
                    
                    # Reduce teacher embeddings
                    teacher_reduced, _ = self.reduce_embeddings(teacher_emb, labels, method)
                    results[split]['teacher'][method] = teacher_reduced
                except Exception as e:
                    print(f"      Warning: {method} failed for {split} split: {e}")
                    # Store NaN array as placeholder
                    results[split]['student'][method] = np.full((len(labels), self.n_components), np.nan)
                    results[split]['teacher'][method] = np.full((len(labels), self.n_components), np.nan)
        
        return results
    
    def save_results(self, results: Dict, epoch: int):
        """Save reduced embeddings to disk."""
        epoch_dir = self.output_dir / f"epoch_{epoch:02d}"
        epoch_dir.mkdir(exist_ok=True)
        
        for split, split_data in results.items():
            split_dir = epoch_dir / split
            split_dir.mkdir(exist_ok=True)
            
            # Save labels
            np.save(split_dir / "labels.npy", split_data['labels'])
            
            # Save reduced embeddings
            for network in ['student', 'teacher']:
                for method, reduced_emb in split_data[network].items():
                    filename = f"{network}_{method}.npy"
                    np.save(split_dir / filename, reduced_emb)
        
        # Save metadata
        # Get methods from any available split
        methods_used = []
        for split_data in results.values():
            if 'student' in split_data and split_data['student']:
                methods_used = list(split_data['student'].keys())
                break
        
        metadata = {
            'n_components': self.n_components,
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'metric': self.metric,
            'normalized': self.normalize,
            'methods': methods_used,
            'splits': list(results.keys())
        }
        
        with open(epoch_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def compute_cross_epoch_reductions(
        self,
        epochs: List[int],
        split: str = 'val',
        method: str = 'umap',
        network: str = 'student'
    ):
        """Compute reductions with consistent initialization across epochs.
        
        Note: driada doesn't support explicit fit/transform, so we'll save
        all epochs separately and ensure consistency through random seed.
        """
        print(f"\nComputing cross-epoch {method} reductions for {network} network ({split} split)...")
        
        cross_epoch_dir = self.output_dir / "cross_epoch" / method
        cross_epoch_dir.mkdir(parents=True, exist_ok=True)
        
        all_embeddings = []
        all_labels = []
        epoch_indices = []
        
        # Collect all embeddings across epochs
        for epoch in epochs:
            print(f"  Loading epoch {epoch}...")
            
            # Load embeddings
            if network == 'student':
                emb, _, labels = self.load_embeddings(epoch, split)
            else:
                _, emb, labels = self.load_embeddings(epoch, split)
            
            all_embeddings.append(emb)
            all_labels.append(labels)
            epoch_indices.extend([epoch] * len(labels))
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.concatenate(all_labels)
        epoch_indices = np.array(epoch_indices)
        
        print(f"  Reducing combined data of shape {all_embeddings.shape}...")
        
        # Apply reduction to all data at once
        reduced_all, _ = self.reduce_embeddings(
            all_embeddings, 
            epoch_indices,  # Use epoch as label for visualization
            method
        )
        
        # Split back by epoch and save
        for epoch in epochs:
            mask = epoch_indices == epoch
            epoch_reduced = reduced_all[mask]
            epoch_labels = all_labels[mask]
            
            # Save results
            epoch_file = cross_epoch_dir / f"{network}_epoch_{epoch:02d}_{split}.npy"
            np.save(epoch_file, epoch_reduced)
            
            # Save labels for first epoch only
            if epoch == epochs[0]:
                np.save(cross_epoch_dir / f"labels_{split}.npy", epoch_labels)
        
        print(f"  Cross-epoch results saved to {cross_epoch_dir}")
    
    def run(
        self,
        epochs: Optional[List[int]] = None,
        splits: List[str] = ['train', 'val'],
        methods: List[str] = ['pca', 'umap', 'laplacian'],
        compute_cross_epoch: bool = True
    ):
        """Run dimensionality reduction on all specified epochs."""
        if epochs is None:
            epochs = [0, 1, 2, 3, 4]
        
        print(f"Starting dimensionality reduction...")
        print(f"  Epochs: {epochs}")
        print(f"  Splits: {splits}")
        print(f"  Methods: {methods}")
        print(f"  Output directory: {self.output_dir}")
        
        # Process each epoch independently
        for epoch in tqdm(epochs, desc="Processing epochs"):
            print(f"\nEpoch {epoch}:")
            results = self.process_epoch(epoch, splits, methods)
            self.save_results(results, epoch)
        
        # Compute cross-epoch reductions for visualization consistency
        if compute_cross_epoch and 'umap' in methods:
            for network in ['student', 'teacher']:
                self.compute_cross_epoch_reductions(
                    epochs, split='val', method='umap', network=network
                )
        
        print("\nDimensionality reduction complete!")
        print(f"Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Reduce dimensionality of I-JEPA embeddings using driada"
    )
    parser.add_argument(
        '--embeddings_dir', type=str, default='analysis/dim_reduction/embeddings',
        help='Directory containing extracted embeddings'
    )
    parser.add_argument(
        '--output_dir', type=str, default='analysis/dim_reduction/reduced_embeddings',
        help='Output directory for reduced embeddings'
    )
    parser.add_argument(
        '--n_components', type=int, default=2,
        help='Number of components for reduction'
    )
    parser.add_argument(
        '--n_neighbors', type=int, default=15,
        help='Number of neighbors for UMAP and Laplacian Eigenmaps'
    )
    parser.add_argument(
        '--min_dist', type=float, default=0.1,
        help='Minimum distance for UMAP'
    )
    parser.add_argument(
        '--metric', type=str, default='euclidean',
        help='Distance metric to use'
    )
    parser.add_argument(
        '--epochs', type=int, nargs='+', default=[0, 1, 2, 3, 4],
        help='Which epochs to process'
    )
    parser.add_argument(
        '--splits', type=str, nargs='+', default=['train', 'val'],
        help='Which splits to process'
    )
    parser.add_argument(
        '--methods', type=str, nargs='+', 
        default=['pca', 'umap', 'laplacian'],
        help='Which reduction methods to apply'
    )
    parser.add_argument(
        '--no_normalize', action='store_true',
        help='Do not normalize embeddings before reduction'
    )
    parser.add_argument(
        '--no_cross_epoch', action='store_true',
        help='Do not compute cross-epoch reductions'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--num_classes', type=int, default=None,
        help='Number of classes to use (default: all)'
    )
    
    args = parser.parse_args()
    
    # Initialize reducer
    reducer = EmbeddingReducer(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        normalize=not args.no_normalize,
        random_state=args.seed
    )
    
    # Set num_classes if specified
    if args.num_classes is not None:
        reducer.num_classes = args.num_classes
    
    # Run reduction
    reducer.run(
        epochs=args.epochs,
        splits=args.splits,
        methods=args.methods,
        compute_cross_epoch=not args.no_cross_epoch
    )


if __name__ == "__main__":
    main()