#!/usr/bin/env python3
"""
Fast SGD-based decoder analysis for I-JEPA embeddings using full dataset.
Uses SGDClassifier for efficient training on large datasets.
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

warnings.filterwarnings('ignore')


class SGDDecoderAnalyzer:
    """Fast decoder analysis using SGD."""
    
    def __init__(
        self,
        embeddings_dir: str = "analysis/dim_reduction/embeddings",
        output_dir: str = "analysis/decoder/sgd_decoder_results",
        max_classes: int = 50,
        alpha: float = 0.01,  # Regularization
        max_iter: int = 100,
        random_state: int = 42
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.max_classes = max_classes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {'student': {}, 'teacher': {}}
        
        # Fixed class selection
        np.random.seed(random_state)
        self.selected_classes = None
    
    def load_and_filter(self, epoch: int, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load embeddings and filter to selected classes."""
        epoch_dir = self.embeddings_dir / f"ijepa-64px-epoch={epoch:02d}"
        
        # Handle nested directory structure
        if not (epoch_dir / split).exists():
            epoch_dir = epoch_dir / f"ijepa-64px-epoch={epoch:02d}"
        
        # Load data
        student_emb = np.load(epoch_dir / split / "student_embeddings_pooled.npy")
        teacher_emb = np.load(epoch_dir / split / "teacher_embeddings_pooled.npy")
        labels = np.load(epoch_dir / split / "labels.npy")
        
        # Select classes (only once)
        if self.selected_classes is None:
            unique_classes = np.unique(labels)
            self.selected_classes = np.random.choice(
                unique_classes, 
                min(self.max_classes, len(unique_classes)), 
                replace=False
            )
            print(f"    Selected {len(self.selected_classes)} classes from {len(unique_classes)}")
        
        # Filter to selected classes
        class_mask = np.isin(labels, self.selected_classes)
        student_emb = student_emb[class_mask]
        teacher_emb = teacher_emb[class_mask]
        labels = labels[class_mask]
        
        # Remap labels
        label_map = {old: new for new, old in enumerate(self.selected_classes)}
        labels = np.array([label_map[label] for label in labels])
        
        return student_emb, teacher_emb, labels
    
    def train_sgd_decoder(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        val_embeddings: np.ndarray,
        val_labels: np.ndarray
    ) -> Dict:
        """Train SGD decoder."""
        # Standardize
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_embeddings)
        val_scaled = scaler.transform(val_embeddings)
        
        # Train classifier
        clf = SGDClassifier(
            loss='log_loss',  # Logistic regression
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=1e-3,
            early_stopping=True,  # Stop when validation score stops improving
            validation_fraction=0.1,  # Use 10% for early stopping
            n_iter_no_change=10,  # Stop after 10 epochs without improvement
            random_state=self.random_state,
            n_jobs=-1
        )
        
        start_time = time.time()
        clf.fit(train_scaled, train_labels)
        train_time = time.time() - start_time
        
        # Evaluate
        train_pred = clf.predict(train_scaled)
        val_pred = clf.predict(val_scaled)
        
        return {
            'train_accuracy': accuracy_score(train_labels, train_pred),
            'val_accuracy': accuracy_score(val_labels, val_pred),
            'train_balanced_accuracy': balanced_accuracy_score(train_labels, train_pred),
            'val_balanced_accuracy': balanced_accuracy_score(val_labels, val_pred),
            'n_train': len(train_labels),
            'n_val': len(val_labels),
            'train_time': train_time,
            'n_iter': clf.n_iter_
        }
    
    def analyze_epoch(self, epoch: int) -> Dict:
        """Analyze single epoch."""
        print(f"\n  Epoch {epoch}:")
        
        # Load data
        train_student, train_teacher, train_labels = self.load_and_filter(epoch, 'train')
        val_student, val_teacher, val_labels = self.load_and_filter(epoch, 'val')
        
        print(f"    Train samples: {len(train_labels)}, Val samples: {len(val_labels)}")
        
        # Train and evaluate
        print("    Training student decoder...")
        student_results = self.train_sgd_decoder(
            train_student, train_labels, val_student, val_labels
        )
        
        print("    Training teacher decoder...")
        teacher_results = self.train_sgd_decoder(
            train_teacher, train_labels, val_teacher, val_labels
        )
        
        print(f"    Student - Val: {student_results['val_accuracy']:.3f}, "
              f"Train: {student_results['train_accuracy']:.3f} "
              f"(time: {student_results['train_time']:.1f}s)")
        print(f"    Teacher - Val: {teacher_results['val_accuracy']:.3f}, "
              f"Train: {teacher_results['train_accuracy']:.3f} "
              f"(time: {teacher_results['train_time']:.1f}s)")
        
        return {'student': student_results, 'teacher': teacher_results}
    
    def plot_results(self, epochs: List[int]):
        """Create visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Validation accuracy
        student_val = [self.results['student'][e]['val_accuracy'] for e in epochs]
        teacher_val = [self.results['teacher'][e]['val_accuracy'] for e in epochs]
        student_train = [self.results['student'][e]['train_accuracy'] for e in epochs]
        teacher_train = [self.results['teacher'][e]['train_accuracy'] for e in epochs]
        
        # Validation accuracy plot
        ax1.plot(epochs, student_val, 'o-', label='Student Val', linewidth=2, markersize=8)
        ax1.plot(epochs, teacher_val, 's-', label='Teacher Val', linewidth=2, markersize=8)
        ax1.plot(epochs, student_train, 'o--', label='Student Train', linewidth=2, markersize=8, alpha=0.5)
        ax1.plot(epochs, teacher_train, 's--', label='Teacher Train', linewidth=2, markersize=8, alpha=0.5)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('SGD Decoder Performance', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # Training time
        student_time = [self.results['student'][e]['train_time'] for e in epochs]
        teacher_time = [self.results['teacher'][e]['train_time'] for e in epochs]
        
        ax2.bar([e - 0.2 for e in epochs], student_time, 0.4, label='Student', alpha=0.7)
        ax2.bar([e + 0.2 for e in epochs], teacher_time, 0.4, label='Teacher', alpha=0.7)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Training Time (seconds)', fontsize=12)
        ax2.set_title('Training Time per Epoch', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'SGD Decoder Analysis (Full Dataset, {self.max_classes} classes)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / "sgd_decoder_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved plot: {output_path}")
    
    def run(self, epochs: Optional[List[int]] = None):
        """Run fast analysis."""
        if epochs is None:
            epochs = [0, 1, 2, 3, 4]
        
        print("="*60)
        print("SGD Decoder Analysis (Full Dataset)")
        print("="*60)
        print(f"  Max classes: {self.max_classes}")
        print(f"  Regularization alpha: {self.alpha}")
        print(f"  Max iterations: {self.max_iter}")
        print(f"  Epochs: {epochs}")
        
        total_start = time.time()
        
        # Analyze each epoch
        for epoch in tqdm(epochs, desc="Processing"):
            epoch_results = self.analyze_epoch(epoch)
            self.results['student'][epoch] = epoch_results['student']
            self.results['teacher'][epoch] = epoch_results['teacher']
        
        total_time = time.time() - total_start
        
        # Save results
        output_path = self.output_dir / "sgd_decoder_results.json"
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n  Saved results: {output_path}")
        
        # Plot
        self.plot_results(epochs)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("Epoch | Student Val | Teacher Val | S-Train | T-Train")
        print("-"*60)
        for epoch in epochs:
            sv = self.results['student'][epoch]['val_accuracy']
            tv = self.results['teacher'][epoch]['val_accuracy']
            st = self.results['student'][epoch]['train_accuracy']
            tt = self.results['teacher'][epoch]['train_accuracy']
            print(f"  {epoch}   |    {sv:.3f}    |    {tv:.3f}    |  {st:.3f}  |  {tt:.3f}")
        
        print(f"\nTotal analysis time: {total_time:.1f} seconds")
        print("="*60)
        print("SGD Decoder Analysis Complete!")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Fast SGD decoder analysis")
    parser.add_argument('--embeddings_dir', type=str, default='analysis/dim_reduction/embeddings')
    parser.add_argument('--output_dir', type=str, default='analysis/decoder/sgd_decoder_results')
    parser.add_argument('--epochs', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--max_classes', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    analyzer = SGDDecoderAnalyzer(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        max_classes=args.max_classes,
        alpha=args.alpha,
        max_iter=args.max_iter,
        random_state=args.seed
    )
    
    analyzer.run(epochs=args.epochs)


if __name__ == "__main__":
    main()