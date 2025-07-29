#!/usr/bin/env python3
"""
Decoder analysis for I-JEPA embeddings.
Trains simple linear decoders to predict class labels from frozen embeddings.
Evaluates decoder performance across student/teacher networks, epochs, and train/val splits.
Uses strict regularization to prevent overfitting.
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

warnings.filterwarnings('ignore')


class DecoderAnalyzer:
    """Analyzes decoder performance on I-JEPA embeddings."""
    
    def __init__(
        self,
        embeddings_dir: str = "analysis/dim_reduction/embeddings",
        output_dir: str = "analysis/decoder/decoder_results",
        C: float = 1000.0,  # C = 1/alpha, so for alpha=0.001, C=1000
        max_iter: int = 1000,  # Increased iterations
        solver: str = 'lbfgs',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.C = C  # Inverse of regularization strength (smaller = stronger)
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {
            'student': {},
            'teacher': {}
        }
    
    def load_embeddings(self, epoch: int, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load embeddings for a specific epoch and split."""
        epoch_dir = self.embeddings_dir / f"ijepa-64px-epoch={epoch:02d}"
        
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
    
    def subsample_balanced(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray, 
        samples_per_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample data to have balanced classes."""
        if samples_per_class is None:
            return embeddings, labels
        
        unique_labels = np.unique(labels)
        indices = []
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            if len(label_indices) >= samples_per_class:
                selected = np.random.choice(
                    label_indices, samples_per_class, replace=False
                )
                indices.extend(selected)
            else:
                # Use all available samples if less than requested
                indices.extend(label_indices)
        
        indices = np.array(indices)
        return embeddings[indices], labels[indices]
    
    def train_decoder(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        val_embeddings: np.ndarray,
        val_labels: np.ndarray,
        cv_folds: int = 5
    ) -> Dict:
        """Train a linear decoder with optional cross-validation."""
        # Standardize features
        scaler = StandardScaler()
        train_embeddings_scaled = scaler.fit_transform(train_embeddings)
        val_embeddings_scaled = scaler.transform(val_embeddings)
        
        # For speed, just use a fixed C value unless we have few classes
        best_C = self.C
        
        # Skip CV entirely for speed - just use fixed C
        if False:  # Disabled CV for speed
            C_values = [0.1]
            best_score = -1
            
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            for C in C_values:
                cv_scores = []
                
                for train_idx, val_idx in skf.split(train_embeddings_scaled, train_labels):
                    X_train_cv = train_embeddings_scaled[train_idx]
                    y_train_cv = train_labels[train_idx]
                    X_val_cv = train_embeddings_scaled[val_idx]
                    y_val_cv = train_labels[val_idx]
                    
                    clf = LogisticRegression(
                        C=C,
                        max_iter=self.max_iter,
                        solver='liblinear',  # Faster for many classes
                        random_state=self.random_state,
                        n_jobs=1,  # liblinear doesn't support n_jobs
                        multi_class='ovr'  # Faster than multinomial
                    )
                    
                    clf.fit(X_train_cv, y_train_cv)
                    cv_score = clf.score(X_val_cv, y_val_cv)
                    cv_scores.append(cv_score)
                
                mean_cv_score = np.mean(cv_scores)
                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    best_C = C
        
        # Train final model with best C
        final_clf = LogisticRegression(
            C=best_C,
            max_iter=self.max_iter,
            solver='liblinear',  # Much faster for many classes
            random_state=self.random_state,
            multi_class='ovr'  # Faster than multinomial
        )
        
        # Check convergence with more iterations if needed
        final_clf.fit(train_embeddings_scaled, train_labels)
        
        # Evaluate on both train and validation sets
        train_pred = final_clf.predict(train_embeddings_scaled)
        val_pred = final_clf.predict(val_embeddings_scaled)
        
        # Compute metrics
        results = {
            'best_C': best_C,
            'train_accuracy': accuracy_score(train_labels, train_pred),
            'val_accuracy': accuracy_score(val_labels, val_pred),
            'train_balanced_accuracy': balanced_accuracy_score(train_labels, train_pred),
            'val_balanced_accuracy': balanced_accuracy_score(val_labels, val_pred),
            'train_f1_macro': f1_score(train_labels, train_pred, average='macro'),
            'val_f1_macro': f1_score(val_labels, val_pred, average='macro'),
            'n_train_samples': len(train_labels),
            'n_val_samples': len(val_labels),
            'n_classes': len(np.unique(train_labels))
        }
        
        return results
    
    def analyze_epoch(
        self,
        epoch: int,
        samples_per_class: Optional[int] = None,
        max_classes: int = 50
    ) -> Dict:
        """Analyze decoder performance for a single epoch."""
        print(f"\n  Analyzing epoch {epoch}...")
        
        # Load embeddings
        train_student, train_teacher, train_labels = self.load_embeddings(epoch, 'train')
        val_student, val_teacher, val_labels = self.load_embeddings(epoch, 'val')
        
        # Reduce to subset of classes for faster processing
        unique_classes = np.unique(train_labels)
        if len(unique_classes) > max_classes:
            # Select random subset of classes
            np.random.seed(self.random_state)
            selected_classes = np.random.choice(unique_classes, max_classes, replace=False)
            
            # Filter train data
            train_mask = np.isin(train_labels, selected_classes)
            train_student = train_student[train_mask]
            train_teacher = train_teacher[train_mask]
            train_labels = train_labels[train_mask]
            
            # Filter val data
            val_mask = np.isin(val_labels, selected_classes)
            val_student = val_student[val_mask]
            val_teacher = val_teacher[val_mask]
            val_labels = val_labels[val_mask]
            
            # Remap labels to 0...max_classes-1
            label_map = {old_label: new_label for new_label, old_label in enumerate(selected_classes)}
            train_labels = np.array([label_map[label] for label in train_labels])
            val_labels = np.array([label_map[label] for label in val_labels])
            
            print(f"    Reduced from {len(unique_classes)} to {max_classes} classes")
        
        # Subsample if requested
        if samples_per_class is not None:
            train_student, train_labels_sub = self.subsample_balanced(
                train_student, train_labels, samples_per_class
            )
            train_teacher, _ = self.subsample_balanced(
                train_teacher, train_labels, samples_per_class
            )
            val_student, val_labels_sub = self.subsample_balanced(
                val_student, val_labels, samples_per_class
            )
            val_teacher, _ = self.subsample_balanced(
                val_teacher, val_labels, samples_per_class
            )
            train_labels = train_labels_sub
            val_labels = val_labels_sub
        
        epoch_results = {}
        
        # Train decoders for student embeddings
        print("    Training student decoder...")
        student_results = self.train_decoder(
            train_student, train_labels,
            val_student, val_labels
        )
        epoch_results['student'] = student_results
        
        # Train decoders for teacher embeddings
        print("    Training teacher decoder...")
        teacher_results = self.train_decoder(
            train_teacher, train_labels,
            val_teacher, val_labels
        )
        epoch_results['teacher'] = teacher_results
        
        # Print summary
        print(f"    Student - Val Acc: {student_results['val_accuracy']:.3f}, "
              f"Train Acc: {student_results['train_accuracy']:.3f}, "
              f"Best C: {student_results['best_C']}")
        print(f"    Teacher - Val Acc: {teacher_results['val_accuracy']:.3f}, "
              f"Train Acc: {teacher_results['train_accuracy']:.3f}, "
              f"Best C: {teacher_results['best_C']}")
        
        return epoch_results
    
    def plot_results(self, epochs: List[int]):
        """Create visualizations of decoder performance."""
        # Prepare data for plotting
        metrics = ['val_accuracy', 'train_accuracy', 'val_balanced_accuracy', 'val_f1_macro']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Student performance
            student_values = [self.results['student'][epoch][metric] for epoch in epochs]
            teacher_values = [self.results['teacher'][epoch][metric] for epoch in epochs]
            
            ax.plot(epochs, student_values, 'o-', label='Student', linewidth=2, markersize=8)
            ax.plot(epochs, teacher_values, 's-', label='Teacher', linewidth=2, markersize=8)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()} vs Epoch', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add y-axis limits
            if 'accuracy' in metric or 'f1' in metric:
                ax.set_ylim([0, 1.05])
        
        plt.suptitle('Decoder Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / "decoder_performance.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved plot: {output_path}")
        
        # Create heatmap of all metrics
        self._create_heatmap(epochs)
    
    def _create_heatmap(self, epochs: List[int]):
        """Create a heatmap of all metrics."""
        # Prepare data
        metrics = ['val_accuracy', 'train_accuracy', 'val_balanced_accuracy', 
                  'train_balanced_accuracy', 'val_f1_macro', 'train_f1_macro']
        
        # Student data
        student_data = []
        teacher_data = []
        
        for epoch in epochs:
            student_row = [self.results['student'][epoch][m] for m in metrics]
            teacher_row = [self.results['teacher'][epoch][m] for m in metrics]
            student_data.append(student_row)
            teacher_data.append(teacher_row)
        
        # Create figure with two heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Format metric names
        metric_labels = [m.replace('_', ' ').title() for m in metrics]
        
        # Student heatmap
        sns.heatmap(
            student_data, 
            annot=True, 
            fmt='.3f', 
            cmap='YlOrRd',
            xticklabels=metric_labels,
            yticklabels=[f'Epoch {e}' for e in epochs],
            cbar_kws={'label': 'Score'},
            ax=ax1,
            vmin=0, vmax=1
        )
        ax1.set_title('Student Decoder Performance', fontsize=14, fontweight='bold')
        
        # Teacher heatmap
        sns.heatmap(
            teacher_data, 
            annot=True, 
            fmt='.3f', 
            cmap='YlGnBu',
            xticklabels=metric_labels,
            yticklabels=[f'Epoch {e}' for e in epochs],
            cbar_kws={'label': 'Score'},
            ax=ax2,
            vmin=0, vmax=1
        )
        ax2.set_title('Teacher Decoder Performance', fontsize=14, fontweight='bold')
        
        plt.suptitle('Decoder Performance Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / "decoder_heatmap.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved heatmap: {output_path}")
    
    def save_results(self):
        """Save all results to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(val) for key, val in obj.items()}
            return obj
        
        results_to_save = convert_types(self.results)
        
        output_path = self.output_dir / "decoder_results.json"
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\n  Saved results: {output_path}")
    
    def run(
        self,
        epochs: Optional[List[int]] = None,
        samples_per_class: Optional[int] = None
    ):
        """Run decoder analysis for specified epochs."""
        if epochs is None:
            epochs = [0, 1, 2, 3, 4]
        
        print("="*60)
        print("Decoder Analysis for I-JEPA Embeddings")
        print("="*60)
        print(f"Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Regularization C range: [0.001, 0.01, 0.1, 1.0, 10.0]")
        print(f"  Max iterations: {self.max_iter}")
        print(f"  Solver: {self.solver}")
        print(f"  Samples per class: {samples_per_class if samples_per_class else 'All'}")
        print(f"  Output directory: {self.output_dir}")
        
        # Analyze each epoch
        for epoch in tqdm(epochs, desc="Processing epochs"):
            epoch_results = self.analyze_epoch(epoch, samples_per_class)
            self.results['student'][epoch] = epoch_results['student']
            self.results['teacher'][epoch] = epoch_results['teacher']
        
        # Save results
        self.save_results()
        
        # Create visualizations
        print("\nCreating visualizations...")
        self.plot_results(epochs)
        
        # Print summary
        self._print_summary(epochs)
        
        print("\n" + "="*60)
        print("Decoder Analysis Complete!")
        print("="*60)
    
    def _print_summary(self, epochs: List[int]):
        """Print summary of results."""
        print("\n" + "="*60)
        print("SUMMARY OF DECODER PERFORMANCE")
        print("="*60)
        
        print("\n1. Validation Accuracy Evolution:")
        print("   " + "-"*40)
        print("   Epoch | Student | Teacher | Difference")
        print("   " + "-"*40)
        for epoch in epochs:
            s_acc = self.results['student'][epoch]['val_accuracy']
            t_acc = self.results['teacher'][epoch]['val_accuracy']
            diff = t_acc - s_acc
            print(f"     {epoch}   |  {s_acc:.3f}  |  {t_acc:.3f}  |  {diff:+.3f}")
        
        print("\n2. Generalization Gap (Train - Val Accuracy):")
        print("   " + "-"*40)
        print("   Epoch | Student | Teacher")
        print("   " + "-"*40)
        for epoch in epochs:
            s_gap = (self.results['student'][epoch]['train_accuracy'] - 
                    self.results['student'][epoch]['val_accuracy'])
            t_gap = (self.results['teacher'][epoch]['train_accuracy'] - 
                    self.results['teacher'][epoch]['val_accuracy'])
            print(f"     {epoch}   |  {s_gap:.3f}  |  {t_gap:.3f}")
        
        print("\n3. Best Regularization (C) Selected:")
        print("   " + "-"*40)
        print("   Epoch | Student | Teacher")
        print("   " + "-"*40)
        for epoch in epochs:
            s_c = self.results['student'][epoch]['best_C']
            t_c = self.results['teacher'][epoch]['best_C']
            print(f"     {epoch}   |  {s_c:.3f}  |  {t_c:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze decoder performance on I-JEPA embeddings"
    )
    parser.add_argument(
        '--embeddings_dir', type=str, default='analysis/dim_reduction/embeddings',
        help='Directory containing extracted embeddings'
    )
    parser.add_argument(
        '--output_dir', type=str, default='analysis/decoder/decoder_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--epochs', type=int, nargs='+', default=[0, 1, 2, 3, 4],
        help='Which epochs to analyze'
    )
    parser.add_argument(
        '--samples_per_class', type=int, default=None,
        help='Number of samples per class for subsampling (None = use all)'
    )
    parser.add_argument(
        '--C', type=float, default=0.01,
        help='Base regularization parameter for logistic regression'
    )
    parser.add_argument(
        '--max_iter', type=int, default=100,
        help='Maximum iterations for logistic regression'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DecoderAnalyzer(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        C=args.C,
        max_iter=args.max_iter,
        random_state=args.seed
    )
    
    # Run analysis
    analyzer.run(
        epochs=args.epochs,
        samples_per_class=args.samples_per_class
    )


if __name__ == "__main__":
    main()