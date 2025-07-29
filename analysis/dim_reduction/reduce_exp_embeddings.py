#!/usr/bin/env python3
"""
Dimensionality reduction for exp folder masked embeddings.
Applies PCA and UMAP to student and teacher embeddings across all epochs.
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


class ExpEmbeddingReducer:
    """Performs dimensionality reduction on exp folder masked embeddings."""
    
    def __init__(
        self,
        embeddings_dir: str = "analysis/dim_reduction/embeddings_exp",
        output_dir: str = "analysis/dim_reduction/reduced_embeddings_exp",
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        normalize: bool = True,
        random_state: int = 42,
        num_classes: int = 50
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.normalize = normalize
        self.random_state = random_state
        self.num_classes = num_classes
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_masked_embeddings(self, epoch: int, mask_idx: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load masked embeddings for a specific epoch and mask."""
        epoch_dir = self.embeddings_dir / f"ijepa-64px-epoch={epoch:02d}" / "masked" / "val" / "ijepa_75" / f"mask_{mask_idx:02d}"
        
        student_emb = np.load(epoch_dir / "student_embeddings_masked_pooled.npy")
        teacher_emb = np.load(epoch_dir / "teacher_embeddings_pooled.npy")
        labels = np.load(epoch_dir / "labels.npy")
        
        # Filter by number of classes
        if self.num_classes is not None and self.num_classes < 200:
            class_mask = labels < self.num_classes
            student_emb = student_emb[class_mask]
            teacher_emb = teacher_emb[class_mask]
            labels = labels[class_mask]
            
        return student_emb, teacher_emb, labels
    
    def reduce_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        method: str
    ) -> np.ndarray:
        """Apply dimensionality reduction to embeddings."""
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
            embedding = mvdata.get_embedding(
                method='umap',
                dim=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Get coordinates and transpose back to (n_samples, n_components)
        reduced = embedding.coords.T
        
        return reduced
    
    def process_epoch(
        self,
        epoch: int,
        methods: List[str] = ['pca', 'umap'],
        n_masks: int = 5
    ) -> Dict:
        """Process all embeddings for a single epoch."""
        print(f"\nProcessing epoch {epoch}...")
        
        results = {
            'epoch': epoch,
            'num_classes': self.num_classes,
            'masks': {}
        }
        
        # Process multiple masks
        for mask_idx in range(n_masks):
            try:
                # Load embeddings
                student_emb, teacher_emb, labels = self.load_masked_embeddings(epoch, mask_idx)
                
                mask_results = {
                    'labels': labels,
                    'n_samples': len(labels),
                    'student': {},
                    'teacher': {}
                }
                
                # Apply each reduction method
                for method in methods:
                    print(f"  Mask {mask_idx}, applying {method}...")
                    
                    try:
                        # Reduce student embeddings
                        student_reduced = self.reduce_embeddings(student_emb, labels, method)
                        mask_results['student'][method] = student_reduced
                        
                        # Reduce teacher embeddings
                        teacher_reduced = self.reduce_embeddings(teacher_emb, labels, method)
                        mask_results['teacher'][method] = teacher_reduced
                        
                    except Exception as e:
                        print(f"    Warning: {method} failed: {e}")
                        mask_results['student'][method] = np.full((len(labels), self.n_components), np.nan)
                        mask_results['teacher'][method] = np.full((len(labels), self.n_components), np.nan)
                
                results['masks'][f'mask_{mask_idx:02d}'] = mask_results
                
            except Exception as e:
                print(f"  Warning: Failed to process mask {mask_idx}: {e}")
                continue
        
        return results
    
    def save_results(self, results: Dict, epoch: int):
        """Save reduced embeddings to disk."""
        epoch_dir = self.output_dir / f"epoch_{epoch:02d}"
        epoch_dir.mkdir(exist_ok=True)
        
        # Average results across masks for visualization
        all_masks_data = list(results['masks'].values())
        if len(all_masks_data) > 0:
            # Use first mask's labels (should be same for all)
            labels = all_masks_data[0]['labels']
            np.save(epoch_dir / "labels.npy", labels)
            
            # Save individual mask results
            for mask_name, mask_data in results['masks'].items():
                mask_dir = epoch_dir / mask_name
                mask_dir.mkdir(exist_ok=True)
                
                for network in ['student', 'teacher']:
                    for method, reduced_emb in mask_data[network].items():
                        filename = f"{network}_{method}.npy"
                        np.save(mask_dir / filename, reduced_emb)
            
            # Also save averaged results for easier visualization
            avg_dir = epoch_dir / "averaged"
            avg_dir.mkdir(exist_ok=True)
            
            for network in ['student', 'teacher']:
                for method in ['pca', 'umap']:
                    # Collect all mask results
                    all_reduced = []
                    for mask_data in all_masks_data:
                        if method in mask_data[network]:
                            all_reduced.append(mask_data[network][method])
                    
                    if all_reduced:
                        # Average across masks
                        avg_reduced = np.mean(all_reduced, axis=0)
                        np.save(avg_dir / f"{network}_{method}.npy", avg_reduced)
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'n_components': self.n_components,
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'metric': self.metric,
            'normalized': self.normalize,
            'num_classes': self.num_classes,
            'n_masks': len(results['masks']),
            'methods': list(all_masks_data[0]['student'].keys()) if all_masks_data else []
        }
        
        with open(epoch_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def run(
        self,
        epochs: Optional[List[int]] = None,
        methods: List[str] = ['pca', 'umap'],
        n_masks: int = 5
    ):
        """Run dimensionality reduction on all specified epochs."""
        if epochs is None:
            epochs = list(range(10))
        
        print(f"Starting dimensionality reduction...")
        print(f"  Epochs: {epochs}")
        print(f"  Methods: {methods}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Masks per epoch: {n_masks}")
        print(f"  Output directory: {self.output_dir}")
        
        # Process each epoch
        for epoch in tqdm(epochs, desc="Processing epochs"):
            results = self.process_epoch(epoch, methods, n_masks)
            self.save_results(results, epoch)
        
        # Create evolution visualization
        self.create_evolution_plot(epochs, methods)
        
        print("\nDimensionality reduction complete!")
        print(f"Results saved to: {self.output_dir}")
    
    def create_evolution_plot(self, epochs: List[int], methods: List[str]):
        """Create a plot showing the evolution of representations."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # For UMAP, create a plot showing first and last epoch
        if 'umap' in methods and len(epochs) >= 2:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            first_epoch = epochs[0]
            last_epoch = epochs[-1]
            
            for i, epoch in enumerate([first_epoch, last_epoch]):
                epoch_dir = self.output_dir / f"epoch_{epoch:02d}" / "averaged"
                
                if epoch_dir.exists():
                    # Load data
                    labels = np.load(self.output_dir / f"epoch_{epoch:02d}" / "labels.npy")
                    
                    # Student UMAP
                    student_umap = np.load(epoch_dir / "student_umap.npy")
                    ax = axes[i, 0]
                    scatter = ax.scatter(student_umap[:, 0], student_umap[:, 1], 
                                       c=labels, cmap='tab20', s=30, alpha=0.7)
                    ax.set_title(f'Student UMAP - Epoch {epoch}')
                    ax.set_xlabel('UMAP 1')
                    ax.set_ylabel('UMAP 2')
                    
                    # Teacher UMAP
                    teacher_umap = np.load(epoch_dir / "teacher_umap.npy")
                    ax = axes[i, 1]
                    scatter = ax.scatter(teacher_umap[:, 0], teacher_umap[:, 1],
                                       c=labels, cmap='tab20', s=30, alpha=0.7)
                    ax.set_title(f'Teacher UMAP - Epoch {epoch}')
                    ax.set_xlabel('UMAP 1')
                    ax.set_ylabel('UMAP 2')
            
            plt.suptitle(f'UMAP Evolution: Epoch {first_epoch} vs Epoch {last_epoch}\n({self.num_classes} classes)', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'umap_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("\nCreated evolution plot: umap_evolution.png")


def main():
    parser = argparse.ArgumentParser(
        description="Reduce dimensionality of exp folder masked embeddings"
    )
    parser.add_argument(
        '--embeddings_dir', type=str, default='embeddings_exp',
        help='Directory containing exp folder embeddings'
    )
    parser.add_argument(
        '--output_dir', type=str, default='reduced_embeddings_exp',
        help='Output directory for reduced embeddings'
    )
    parser.add_argument(
        '--n_components', type=int, default=2,
        help='Number of components for reduction'
    )
    parser.add_argument(
        '--n_neighbors', type=int, default=15,
        help='Number of neighbors for UMAP'
    )
    parser.add_argument(
        '--min_dist', type=float, default=0.1,
        help='Minimum distance for UMAP'
    )
    parser.add_argument(
        '--epochs', type=int, nargs='+', default=list(range(10)),
        help='Which epochs to process'
    )
    parser.add_argument(
        '--methods', type=str, nargs='+', default=['pca', 'umap'],
        help='Which reduction methods to apply'
    )
    parser.add_argument(
        '--num_classes', type=int, default=50,
        help='Number of classes to use'
    )
    parser.add_argument(
        '--n_masks', type=int, default=5,
        help='Number of masks to process per epoch'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Initialize reducer
    reducer = ExpEmbeddingReducer(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        normalize=True,
        random_state=args.seed,
        num_classes=args.num_classes
    )
    
    # Run reduction
    reducer.run(
        epochs=args.epochs,
        methods=args.methods,
        n_masks=args.n_masks
    )


if __name__ == "__main__":
    main()