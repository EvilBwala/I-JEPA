#!/usr/bin/env python3
"""
Visualization script for reduced I-JEPA embeddings.
Creates comprehensive plots showing evolution of representations across epochs.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EmbeddingVisualizer:
    """Visualizes reduced I-JEPA embeddings."""
    
    def __init__(
        self,
        reduced_dir: str = "analysis/dim_reduction/reduced_embeddings",
        output_dir: str = "analysis/dim_reduction/embedding_plots",
        figsize: Tuple[int, int] = (12, 10),
        dpi: int = 150,
        alpha: float = 0.7,
        point_size: int = 30
    ):
        self.reduced_dir = Path(reduced_dir)
        self.output_dir = Path(output_dir)
        self.figsize = figsize
        self.dpi = dpi
        self.alpha = alpha
        self.point_size = point_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color palette for classes
        self.colors = None
    
    def load_reduced_embeddings(
        self, 
        epoch: int, 
        split: str, 
        method: str, 
        network: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load reduced embeddings and labels."""
        epoch_dir = self.reduced_dir / f"epoch_{epoch:02d}" / split
        
        embeddings = np.load(epoch_dir / f"{network}_{method}.npy")
        labels = np.load(epoch_dir / "labels.npy")
        
        return embeddings, labels
    
    def load_cross_epoch_embeddings(
        self,
        epochs: List[int],
        split: str,
        method: str,
        network: str
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Load cross-epoch embeddings."""
        cross_epoch_dir = self.reduced_dir / "cross_epoch" / method
        
        results = {}
        for epoch in epochs:
            emb_file = cross_epoch_dir / f"{network}_epoch_{epoch:02d}_{split}.npy"
            if emb_file.exists():
                embeddings = np.load(emb_file)
                labels = np.load(cross_epoch_dir / f"labels_{split}.npy")
                results[epoch] = (embeddings, labels)
        
        return results
    
    def setup_colors(self, n_classes: int):
        """Setup color palette for classes."""
        if self.colors is None:
            self.colors = cm.get_cmap('tab20' if n_classes <= 20 else 'hsv')(
                np.linspace(0, 1, n_classes)
            )
    
    def plot_single_embedding(
        self, 
        ax: plt.Axes,
        embeddings: np.ndarray,
        labels: np.ndarray,
        title: str,
        show_legend: bool = True
    ):
        """Plot a single 2D embedding."""
        unique_labels = np.unique(labels)
        self.setup_colors(len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[self.colors[i]],
                label=f'Class {int(label)}',
                alpha=self.alpha,
                s=self.point_size,
                edgecolors='white',
                linewidth=0.5
            )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        
        if show_legend and len(unique_labels) <= 20:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                     borderaxespad=0., frameon=True, fontsize=10)
    
    def plot_epoch_comparison(
        self,
        epoch: int,
        split: str = 'val',
        methods: List[str] = ['pca', 'umap', 'laplacian']
    ):
        """Compare different DR methods for a single epoch."""
        n_methods = len(methods)
        fig, axes = plt.subplots(2, n_methods, figsize=(n_methods*6, 12))
        
        fig.suptitle(f'Dimensionality Reduction Comparison - Epoch {epoch} ({split} split)', 
                    fontsize=16, fontweight='bold')
        
        for i, method in enumerate(methods):
            # Student embeddings
            try:
                student_emb, labels = self.load_reduced_embeddings(
                    epoch, split, method, 'student'
                )
                self.plot_single_embedding(
                    axes[0, i] if n_methods > 1 else axes[0],
                    student_emb,
                    labels,
                    f'Student - {method.upper()}',
                    show_legend=(i == n_methods - 1)
                )
            except Exception as e:
                print(f"Warning: Could not load student {method} for epoch {epoch}: {e}")
            
            # Teacher embeddings
            try:
                teacher_emb, labels = self.load_reduced_embeddings(
                    epoch, split, method, 'teacher'
                )
                self.plot_single_embedding(
                    axes[1, i] if n_methods > 1 else axes[1],
                    teacher_emb,
                    labels,
                    f'Teacher - {method.upper()}',
                    show_legend=(i == n_methods - 1)
                )
            except Exception as e:
                print(f"Warning: Could not load teacher {method} for epoch {epoch}: {e}")
        
        plt.tight_layout()
        output_file = self.output_dir / f"epoch_{epoch:02d}_{split}_comparison.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")
    
    def plot_evolution(
        self,
        epochs: List[int],
        split: str = 'val',
        method: str = 'umap'
    ):
        """Plot evolution of embeddings across epochs."""
        n_epochs = len(epochs)
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, n_epochs, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'{method.upper()} Evolution Across Epochs ({split} split)', 
                    fontsize=18, fontweight='bold')
        
        # Try to load cross-epoch embeddings first
        use_cross_epoch = False
        if (self.reduced_dir / "cross_epoch" / method).exists():
            student_cross = self.load_cross_epoch_embeddings(epochs, split, method, 'student')
            teacher_cross = self.load_cross_epoch_embeddings(epochs, split, method, 'teacher')
            if len(student_cross) == n_epochs and len(teacher_cross) == n_epochs:
                use_cross_epoch = True
                print(f"  Using cross-epoch embeddings for consistent visualization")
        
        for i, epoch in enumerate(epochs):
            # Student
            ax_student = fig.add_subplot(gs[0, i])
            if use_cross_epoch:
                student_emb, labels = student_cross[epoch]
            else:
                student_emb, labels = self.load_reduced_embeddings(
                    epoch, split, method, 'student'
                )
            self.plot_single_embedding(
                ax_student,
                student_emb,
                labels,
                f'Student - Epoch {epoch}',
                show_legend=(i == n_epochs - 1)
            )
            
            # Teacher
            ax_teacher = fig.add_subplot(gs[1, i])
            if use_cross_epoch:
                teacher_emb, labels = teacher_cross[epoch]
            else:
                teacher_emb, labels = self.load_reduced_embeddings(
                    epoch, split, method, 'teacher'
                )
            self.plot_single_embedding(
                ax_teacher,
                teacher_emb,
                labels,
                f'Teacher - Epoch {epoch}',
                show_legend=(i == n_epochs - 1)
            )
        
        output_file = self.output_dir / f"{method}_evolution_{split}.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")
    
    def plot_student_teacher_overlay(
        self,
        epoch: int,
        split: str = 'val',
        method: str = 'umap'
    ):
        """Plot student and teacher embeddings overlaid."""
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Load embeddings
        student_emb, labels = self.load_reduced_embeddings(epoch, split, method, 'student')
        teacher_emb, _ = self.load_reduced_embeddings(epoch, split, method, 'teacher')
        
        unique_labels = np.unique(labels)
        self.setup_colors(len(unique_labels))
        
        # Plot with different markers for student/teacher
        for i, label in enumerate(unique_labels):
            mask = labels == label
            
            # Student - circles
            ax.scatter(
                student_emb[mask, 0],
                student_emb[mask, 1],
                c=[self.colors[i]],
                marker='o',
                alpha=self.alpha,
                s=self.point_size,
                edgecolors='white',
                linewidth=0.5,
                label=f'Student - Class {int(label)}'
            )
            
            # Teacher - squares
            ax.scatter(
                teacher_emb[mask, 0],
                teacher_emb[mask, 1],
                c=[self.colors[i]],
                marker='s',
                alpha=self.alpha,
                s=self.point_size,
                edgecolors='black',
                linewidth=0.5,
                label=f'Teacher - Class {int(label)}'
            )
        
        ax.set_title(f'Student vs Teacher Overlay - {method.upper()} - Epoch {epoch} ({split})',
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        
        if len(unique_labels) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                     borderaxespad=0., frameon=True, fontsize=10)
        
        plt.tight_layout()
        output_file = self.output_dir / f"overlay_epoch_{epoch:02d}_{split}_{method}.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")
    
    def plot_class_centroids_evolution(
        self,
        epochs: List[int],
        split: str = 'val',
        method: str = 'umap'
    ):
        """Plot evolution of class centroids across epochs."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Collect centroids
        student_centroids = {epoch: {} for epoch in epochs}
        teacher_centroids = {epoch: {} for epoch in epochs}
        
        for epoch in epochs:
            student_emb, labels = self.load_reduced_embeddings(epoch, split, method, 'student')
            teacher_emb, _ = self.load_reduced_embeddings(epoch, split, method, 'teacher')
            
            for label in np.unique(labels):
                mask = labels == label
                student_centroids[epoch][label] = np.mean(student_emb[mask], axis=0)
                teacher_centroids[epoch][label] = np.mean(teacher_emb[mask], axis=0)
        
        # Setup colors
        unique_labels = sorted(set().union(*[set(c.keys()) for c in student_centroids.values()]))
        self.setup_colors(len(unique_labels))
        
        # Plot trajectories
        for i, label in enumerate(unique_labels):
            # Student trajectories
            student_points = np.array([
                student_centroids[epoch].get(label, [np.nan, np.nan]) 
                for epoch in epochs
            ])
            valid = ~np.isnan(student_points).any(axis=1)
            
            ax1.plot(student_points[valid, 0], student_points[valid, 1], 
                    'o-', color=self.colors[i], label=f'Class {int(label)}',
                    markersize=8, linewidth=2, alpha=0.8)
            
            # Add epoch labels
            for j, (epoch, valid_flag) in enumerate(zip(epochs, valid)):
                if valid_flag:
                    ax1.annotate(str(epoch), 
                               (student_points[j, 0], student_points[j, 1]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8)
            
            # Teacher trajectories
            teacher_points = np.array([
                teacher_centroids[epoch].get(label, [np.nan, np.nan]) 
                for epoch in epochs
            ])
            valid = ~np.isnan(teacher_points).any(axis=1)
            
            ax2.plot(teacher_points[valid, 0], teacher_points[valid, 1], 
                    's-', color=self.colors[i], label=f'Class {int(label)}',
                    markersize=8, linewidth=2, alpha=0.8)
            
            # Add epoch labels
            for j, (epoch, valid_flag) in enumerate(zip(epochs, valid)):
                if valid_flag:
                    ax2.annotate(str(epoch), 
                               (teacher_points[j, 0], teacher_points[j, 1]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8)
        
        ax1.set_title(f'Student Centroid Evolution - {method.upper()}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Component 1', fontsize=12)
        ax1.set_ylabel('Component 2', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax2.set_title(f'Teacher Centroid Evolution - {method.upper()}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Component 1', fontsize=12)
        ax2.set_ylabel('Component 2', fontsize=12)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.suptitle(f'Class Centroid Evolution Across Epochs ({split} split)', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        output_file = self.output_dir / f"centroid_evolution_{split}_{method}.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")
    
    def create_summary_grid(
        self,
        epochs: List[int],
        split: str = 'val'
    ):
        """Create a summary grid showing all methods and epochs."""
        methods = ['pca', 'umap', 'laplacian']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(len(methods)*2, len(epochs), figure=fig, hspace=0.4, wspace=0.3)
        
        fig.suptitle(f'Complete Dimensionality Reduction Summary ({split} split)', 
                    fontsize=20, fontweight='bold')
        
        for i, method in enumerate(methods):
            for j, epoch in enumerate(epochs):
                # Student
                ax_student = fig.add_subplot(gs[i*2, j])
                try:
                    student_emb, labels = self.load_reduced_embeddings(
                        epoch, split, method, 'student'
                    )
                    self.plot_single_embedding(
                        ax_student,
                        student_emb,
                        labels,
                        f'S-{method[:3].upper()}-E{epoch}',
                        show_legend=False
                    )
                except Exception as e:
                    ax_student.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax_student.set_title(f'S-{method[:3].upper()}-E{epoch}')
                
                # Teacher
                ax_teacher = fig.add_subplot(gs[i*2+1, j])
                try:
                    teacher_emb, labels = self.load_reduced_embeddings(
                        epoch, split, method, 'teacher'
                    )
                    self.plot_single_embedding(
                        ax_teacher,
                        teacher_emb,
                        labels,
                        f'T-{method[:3].upper()}-E{epoch}',
                        show_legend=False
                    )
                except Exception as e:
                    ax_teacher.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax_teacher.set_title(f'T-{method[:3].upper()}-E{epoch}')
        
        output_file = self.output_dir / f"summary_grid_{split}.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_file}")
    
    def run(
        self,
        epochs: Optional[List[int]] = None,
        splits: List[str] = ['val'],
        methods: List[str] = ['pca', 'umap', 'laplacian'],
        plot_types: List[str] = ['comparison', 'evolution', 'overlay', 'centroids', 'summary']
    ):
        """Generate all visualization plots."""
        if epochs is None:
            epochs = [0, 1, 2, 3, 4]
        
        print(f"Starting visualization...")
        print(f"  Epochs: {epochs}")
        print(f"  Splits: {splits}")
        print(f"  Methods: {methods}")
        print(f"  Plot types: {plot_types}")
        print(f"  Output directory: {self.output_dir}")
        
        for split in splits:
            print(f"\nProcessing {split} split:")
            
            # Single epoch comparisons
            if 'comparison' in plot_types:
                print("  Creating method comparison plots...")
                for epoch in epochs:
                    self.plot_epoch_comparison(epoch, split, methods)
            
            # Evolution plots
            if 'evolution' in plot_types:
                print("  Creating evolution plots...")
                for method in methods:
                    try:
                        self.plot_evolution(epochs, split, method)
                    except Exception as e:
                        print(f"    Warning: Could not create evolution plot for {method}: {e}")
            
            # Overlay plots
            if 'overlay' in plot_types:
                print("  Creating overlay plots...")
                for epoch in epochs:
                    for method in methods:
                        try:
                            self.plot_student_teacher_overlay(epoch, split, method)
                        except Exception as e:
                            print(f"    Warning: Could not create overlay for {method} epoch {epoch}: {e}")
            
            # Centroid evolution
            if 'centroids' in plot_types:
                print("  Creating centroid evolution plots...")
                for method in methods:
                    try:
                        self.plot_class_centroids_evolution(epochs, split, method)
                    except Exception as e:
                        print(f"    Warning: Could not create centroid plot for {method}: {e}")
            
            # Summary grid
            if 'summary' in plot_types:
                print("  Creating summary grid...")
                try:
                    self.create_summary_grid(epochs, split)
                except Exception as e:
                    print(f"    Warning: Could not create summary grid: {e}")
        
        print(f"\nVisualization complete! Plots saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize reduced I-JEPA embeddings"
    )
    parser.add_argument(
        '--reduced_dir', type=str, default='analysis/dim_reduction/reduced_embeddings',
        help='Directory containing reduced embeddings'
    )
    parser.add_argument(
        '--output_dir', type=str, default='analysis/dim_reduction/embedding_plots',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--epochs', type=int, nargs='+', default=[0, 1, 2, 3, 4],
        help='Which epochs to visualize'
    )
    parser.add_argument(
        '--splits', type=str, nargs='+', default=['val'],
        help='Which splits to visualize'
    )
    parser.add_argument(
        '--methods', type=str, nargs='+', 
        default=['pca', 'umap', 'laplacian'],
        help='Which reduction methods to visualize'
    )
    parser.add_argument(
        '--plot_types', type=str, nargs='+',
        default=['comparison', 'evolution', 'overlay', 'centroids', 'summary'],
        help='Types of plots to create'
    )
    parser.add_argument(
        '--figsize', type=int, nargs=2, default=[12, 10],
        help='Figure size (width height)'
    )
    parser.add_argument(
        '--dpi', type=int, default=150,
        help='DPI for saved figures'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.7,
        help='Transparency for scatter plots'
    )
    parser.add_argument(
        '--point_size', type=int, default=30,
        help='Size of scatter plot points'
    )
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = EmbeddingVisualizer(
        reduced_dir=args.reduced_dir,
        output_dir=args.output_dir,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        alpha=args.alpha,
        point_size=args.point_size
    )
    
    # Run visualization
    visualizer.run(
        epochs=args.epochs,
        splits=args.splits,
        methods=args.methods,
        plot_types=args.plot_types
    )


if __name__ == "__main__":
    main()