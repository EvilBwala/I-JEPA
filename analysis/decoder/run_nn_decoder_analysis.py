#!/usr/bin/env python3
"""
Neural network decoder analysis for I-JEPA embeddings.
Trains shallow MLPs with strong regularization to predict class labels.
Includes early stopping, dropout, and weight decay to prevent overfitting.
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class LinearDecoder(nn.Module):
    """Simple linear decoder with optional hidden layer."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout_rate: float = 0.5,
        use_bn: bool = True
    ):
        super().__init__()
        
        if hidden_dim is None:
            # Pure linear model
            self.model = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Dropout(dropout_rate),
                nn.Linear(input_dim, num_classes)
            )
        else:
            # Single hidden layer
            layers = []
            
            # Input normalization
            layers.append(nn.LayerNorm(input_dim))
            layers.append(nn.Dropout(dropout_rate))
            
            # Hidden layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, num_classes))
            
            self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class NeuralDecoderAnalyzer:
    """Analyzes neural decoder performance on I-JEPA embeddings."""
    
    def __init__(
        self,
        embeddings_dir: str = "analysis/dim_reduction/embeddings",
        output_dir: str = "analysis/decoder/nn_decoder_results",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,  # Strong L2 regularization
        dropout_rate: float = 0.5,   # High dropout
        max_epochs: int = 50,        # Limited epochs
        patience: int = 5,           # Early stopping patience
        random_state: int = 42
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {
            'linear': {'student': {}, 'teacher': {}},
            'mlp': {'student': {}, 'teacher': {}}
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
    
    def create_dataloaders(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        val_embeddings: np.ndarray,
        val_labels: np.ndarray
    ) -> Tuple[DataLoader, DataLoader, StandardScaler]:
        """Create PyTorch dataloaders with standardization."""
        # Standardize features
        scaler = StandardScaler()
        train_embeddings_scaled = scaler.fit_transform(train_embeddings)
        val_embeddings_scaled = scaler.transform(val_embeddings)
        
        # Convert to tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(train_embeddings_scaled),
            torch.LongTensor(train_labels)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_embeddings_scaled),
            torch.LongTensor(val_labels)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader, scaler
    
    def train_decoder(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_type: str
    ) -> Dict:
        """Train decoder with early stopping."""
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Training history
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        # Early stopping
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.max_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)
        
        # Final evaluation
        model.eval()
        all_train_preds = []
        all_train_labels = []
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                all_train_preds.extend(predicted.cpu().numpy())
                all_train_labels.extend(labels.numpy())
            
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.numpy())
        
        # Compute final metrics
        results = {
            'model_type': model_type,
            'best_epoch': best_epoch,
            'total_epochs': len(train_losses),
            'train_accuracy': accuracy_score(all_train_labels, all_train_preds),
            'val_accuracy': accuracy_score(all_val_labels, all_val_preds),
            'train_balanced_accuracy': balanced_accuracy_score(all_train_labels, all_train_preds),
            'val_balanced_accuracy': balanced_accuracy_score(all_val_labels, all_val_preds),
            'train_f1_macro': f1_score(all_train_labels, all_train_preds, average='macro'),
            'val_f1_macro': f1_score(all_val_labels, all_val_preds, average='macro'),
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'n_train_samples': len(all_train_labels),
            'n_val_samples': len(all_val_labels),
            'n_classes': len(np.unique(all_train_labels))
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
        
        # Reduce to subset of classes for consistency with linear decoder
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
        
        # Subsample if needed
        if samples_per_class is not None:
            # Implement balanced subsampling
            train_student, train_labels = self._subsample_balanced(
                train_student, train_labels, samples_per_class
            )
            train_teacher, _ = self._subsample_balanced(
                train_teacher, train_labels, samples_per_class
            )
            val_student, val_labels = self._subsample_balanced(
                val_student, val_labels, samples_per_class
            )
            val_teacher, _ = self._subsample_balanced(
                val_teacher, val_labels, samples_per_class
            )
        
        input_dim = train_student.shape[1]
        num_classes = len(np.unique(train_labels))
        
        epoch_results = {}
        
        # Test both student and teacher embeddings
        for emb_type, train_emb, val_emb in [
            ('student', train_student, val_student),
            ('teacher', train_teacher, val_teacher)
        ]:
            print(f"    Training {emb_type} decoders...")
            
            # Create dataloaders
            train_loader, val_loader, scaler = self.create_dataloaders(
                train_emb, train_labels, val_emb, val_labels
            )
            
            # Train linear decoder
            linear_model = LinearDecoder(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=None,
                dropout_rate=self.dropout_rate
            )
            linear_results = self.train_decoder(
                linear_model, train_loader, val_loader, 'linear'
            )
            
            # Train MLP decoder (with small hidden layer)
            mlp_model = LinearDecoder(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=min(128, input_dim // 2),  # Small hidden layer
                dropout_rate=self.dropout_rate
            )
            mlp_results = self.train_decoder(
                mlp_model, train_loader, val_loader, 'mlp'
            )
            
            epoch_results[emb_type] = {
                'linear': linear_results,
                'mlp': mlp_results
            }
            
            # Print summary
            print(f"      Linear - Val Acc: {linear_results['val_accuracy']:.3f}, "
                  f"Train Acc: {linear_results['train_accuracy']:.3f}, "
                  f"Stopped at epoch {linear_results['best_epoch']}")
            print(f"      MLP    - Val Acc: {mlp_results['val_accuracy']:.3f}, "
                  f"Train Acc: {mlp_results['train_accuracy']:.3f}, "
                  f"Stopped at epoch {mlp_results['best_epoch']}")
        
        return epoch_results
    
    def _subsample_balanced(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        samples_per_class: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample data to have balanced classes."""
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
                indices.extend(label_indices)
        
        indices = np.array(indices)
        return embeddings[indices], labels[indices]
    
    def plot_results(self, epochs: List[int]):
        """Create comprehensive visualizations."""
        # Comparison plot: Linear vs MLP
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['val_accuracy', 'train_accuracy']
        networks = ['student', 'teacher']
        
        for i, network in enumerate(networks):
            for j, metric in enumerate(metrics):
                ax = axes[i, j]
                
                # Get values for both decoder types
                linear_values = []
                mlp_values = []
                for e in epochs:
                    if e in self.results['linear'][network]:
                        linear_values.append(self.results['linear'][network][e]['linear'][metric])
                    if e in self.results['mlp'][network]:
                        mlp_values.append(self.results['mlp'][network][e]['mlp'][metric])
                
                ax.plot(epochs, linear_values, 'o-', label='Linear', 
                       linewidth=2, markersize=8)
                ax.plot(epochs, mlp_values, 's-', label='MLP', 
                       linewidth=2, markersize=8)
                
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
                ax.set_title(f'{network.capitalize()} - {metric.replace("_", " ").title()}',
                           fontsize=14)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.05])
        
        plt.suptitle('Neural Decoder Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / "nn_decoder_performance.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved plot: {output_path}")
        
        # Create generalization gap plot
        self._plot_generalization_gap(epochs)
    
    def _plot_generalization_gap(self, epochs: List[int]):
        """Plot generalization gap (train - val accuracy)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        for ax, network in zip([ax1, ax2], ['student', 'teacher']):
            # Calculate gaps
            linear_gaps = []
            mlp_gaps = []
            
            for epoch in epochs:
                linear_gap = (
                    self.results['linear'][network][epoch]['linear']['train_accuracy'] -
                    self.results['linear'][network][epoch]['linear']['val_accuracy']
                )
                mlp_gap = (
                    self.results['mlp'][network][epoch]['mlp']['train_accuracy'] -
                    self.results['mlp'][network][epoch]['mlp']['val_accuracy']
                )
                linear_gaps.append(linear_gap)
                mlp_gaps.append(mlp_gap)
            
            ax.plot(epochs, linear_gaps, 'o-', label='Linear', 
                   linewidth=2, markersize=8)
            ax.plot(epochs, mlp_gaps, 's-', label='MLP', 
                   linewidth=2, markersize=8)
            
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Generalization Gap', fontsize=12)
            ax.set_title(f'{network.capitalize()} - Generalization Gap', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Generalization Gap Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / "generalization_gap.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: {output_path}")
    
    def save_results(self):
        """Save all results to JSON file."""
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(val) for key, val in obj.items()}
            return obj
        
        # Combine results
        combined_results = {
            'linear': convert_types(self.results['linear']),
            'mlp': convert_types(self.results['mlp']),
            'config': {
                'device': str(self.device),
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'dropout_rate': self.dropout_rate,
                'max_epochs': self.max_epochs,
                'patience': self.patience
            }
        }
        
        output_path = self.output_dir / "nn_decoder_results.json"
        with open(output_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        print(f"\n  Saved results: {output_path}")
    
    def run(
        self,
        epochs: Optional[List[int]] = None,
        samples_per_class: Optional[int] = None
    ):
        """Run neural decoder analysis."""
        if epochs is None:
            epochs = [0, 1, 2, 3, 4]
        
        print("="*60)
        print("Neural Network Decoder Analysis for I-JEPA Embeddings")
        print("="*60)
        print(f"Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Epochs to analyze: {epochs}")
        print(f"  Weight decay: {self.weight_decay}")
        print(f"  Dropout rate: {self.dropout_rate}")
        print(f"  Max training epochs: {self.max_epochs}")
        print(f"  Early stopping patience: {self.patience}")
        print(f"  Samples per class: {samples_per_class if samples_per_class else 'All'}")
        print(f"  Output directory: {self.output_dir}")
        
        # Analyze each epoch
        for epoch in tqdm(epochs, desc="Processing epochs"):
            epoch_results = self.analyze_epoch(epoch, samples_per_class)
            
            # Store results
            for network in ['student', 'teacher']:
                self.results['linear'][network][epoch] = epoch_results[network]['linear']
                self.results['mlp'][network][epoch] = epoch_results[network]['mlp']
        
        # Save results
        self.save_results()
        
        # Create visualizations
        print("\nCreating visualizations...")
        self.plot_results(epochs)
        
        # Print summary
        self._print_summary(epochs)
        
        print("\n" + "="*60)
        print("Neural Decoder Analysis Complete!")
        print("="*60)
    
    def _print_summary(self, epochs: List[int]):
        """Print summary of results."""
        print("\n" + "="*60)
        print("SUMMARY OF NEURAL DECODER PERFORMANCE")
        print("="*60)
        
        print("\n1. Best Validation Accuracy:")
        print("   " + "-"*55)
        print("   Epoch | Student Linear | Student MLP | Teacher Linear | Teacher MLP")
        print("   " + "-"*55)
        for epoch in epochs:
            sl = self.results['linear']['student'][epoch]['linear']['val_accuracy']
            sm = self.results['mlp']['student'][epoch]['mlp']['val_accuracy']
            tl = self.results['linear']['teacher'][epoch]['linear']['val_accuracy']
            tm = self.results['mlp']['teacher'][epoch]['mlp']['val_accuracy']
            print(f"     {epoch}   |     {sl:.3f}      |    {sm:.3f}    |     {tl:.3f}      |    {tm:.3f}")
        
        print("\n2. Early Stopping Epochs:")
        print("   " + "-"*55)
        print("   Epoch | Student Linear | Student MLP | Teacher Linear | Teacher MLP")
        print("   " + "-"*55)
        for epoch in epochs:
            sl = self.results['linear']['student'][epoch]['linear']['best_epoch']
            sm = self.results['mlp']['student'][epoch]['mlp']['best_epoch']
            tl = self.results['linear']['teacher'][epoch]['linear']['best_epoch']
            tm = self.results['mlp']['teacher'][epoch]['mlp']['best_epoch']
            print(f"     {epoch}   |       {sl:2d}       |     {sm:2d}      |       {tl:2d}       |     {tm:2d}")


def main():
    parser = argparse.ArgumentParser(
        description="Neural network decoder analysis for I-JEPA embeddings"
    )
    parser.add_argument(
        '--embeddings_dir', type=str, default='analysis/dim_reduction/embeddings',
        help='Directory containing extracted embeddings'
    )
    parser.add_argument(
        '--output_dir', type=str, default='analysis/decoder/nn_decoder_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--epochs', type=int, nargs='+', default=[0, 1, 2, 3, 4],
        help='Which epochs to analyze'
    )
    parser.add_argument(
        '--samples_per_class', type=int, default=None,
        help='Number of samples per class for subsampling'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-2,
        help='Weight decay (L2 regularization)'
    )
    parser.add_argument(
        '--dropout_rate', type=float, default=0.5,
        help='Dropout rate'
    )
    parser.add_argument(
        '--max_epochs', type=int, default=50,
        help='Maximum training epochs'
    )
    parser.add_argument(
        '--patience', type=int, default=5,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Initialize analyzer
    analyzer = NeuralDecoderAnalyzer(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        device=device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        random_state=args.seed
    )
    
    # Run analysis
    analyzer.run(
        epochs=args.epochs,
        samples_per_class=args.samples_per_class
    )


if __name__ == "__main__":
    main()