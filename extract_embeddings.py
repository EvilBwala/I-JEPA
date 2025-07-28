#!/usr/bin/env python3
"""
Extract embeddings from trained I-JEPA model for train/val/test sets.
Computes embeddings for unmasked images from both student and teacher encoders.
Final version with proper label handling.
"""

import argparse
import os
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import copy
from PIL import Image

# Import model components
from model_new import IJEPA, IJEPA_base, TinyImageNetDataModule


class TinyImageNetDatasetWithLabels(Dataset):
    """TinyImageNet dataset that returns both images and labels."""
    
    def __init__(self, dataset_path, stage='train', img_size=64):
        self.dataset_path = dataset_path
        self.stage = stage
        self.img_size = img_size
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset based on stage."""
        if self.stage == 'train':
            train_dir = os.path.join(self.dataset_path, 'train')
            class_dirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
            
            for class_idx, class_dir in enumerate(class_dirs):
                self.class_to_idx[class_dir] = class_idx
                class_path = os.path.join(train_dir, class_dir)
                
                # Check for images in 'images' subfolder
                images_path = os.path.join(class_path, 'images')
                if os.path.exists(images_path):
                    image_files = [f for f in os.listdir(images_path) if f.endswith(('.JPEG', '.jpeg', '.jpg', '.png'))]
                    for img_file in image_files:
                        self.images.append(os.path.join(images_path, img_file))
                        self.labels.append(class_idx)
                else:
                    # Look for images directly in class folder
                    image_files = [f for f in os.listdir(class_path) if f.endswith(('.JPEG', '.jpeg', '.jpg', '.png'))]
                    for img_file in image_files:
                        self.images.append(os.path.join(class_path, img_file))
                        self.labels.append(class_idx)
        
        elif self.stage == 'val':
            val_dir = os.path.join(self.dataset_path, 'val_parsed')
            if os.path.exists(val_dir):
                class_dirs = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
                
                # Use same class mapping as training
                train_dir = os.path.join(self.dataset_path, 'train')
                train_class_dirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
                for class_idx, class_dir in enumerate(train_class_dirs):
                    self.class_to_idx[class_dir] = class_idx
                
                for class_dir in class_dirs:
                    if class_dir in self.class_to_idx:
                        class_idx = self.class_to_idx[class_dir]
                        class_path = os.path.join(val_dir, class_dir)
                        image_files = [f for f in os.listdir(class_path) if f.endswith(('.JPEG', '.jpeg', '.jpg', '.png'))]
                        for img_file in image_files:
                            self.images.append(os.path.join(class_path, img_file))
                            self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} images with labels for {self.stage} stage")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return placeholder image and label in case of error
            return torch.zeros((3, self.img_size, self.img_size)), label


class IJEPA_base_fixed(IJEPA_base):
    """Fixed version of IJEPA_base that works with MPS/CPU."""
    
    def __init__(self, *args, **kwargs):
        # Call parent init but skip the problematic line
        nn.Module.__init__(self)
        
        # Copy initialization from parent
        self.img_size = kwargs.get('img_size', 224)
        self.patch_size = kwargs.get('patch_size', 16)
        self.in_chans = kwargs.get('in_chans', 3)
        self.embed_dim = kwargs.get('embed_dim', 768)
        self.enc_depth = kwargs.get('enc_depth', 12)
        self.enc_heads = kwargs.get('enc_heads', 12)
        self.pred_depth = kwargs.get('pred_depth', 12)
        self.layer_dropout = kwargs.get('layer_dropout', 0.)
        self.mode = kwargs.get('mode', 'train')
        self.fuzzy = kwargs.get('fuzzy', 0)
        
        # Initialize components
        from x_transformers import Encoder
        
        self.patch_embed = nn.Conv2d(self.in_chans, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        num_patches = (self.img_size // self.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        
        self.teacher_encoder = Encoder(
            dim=self.embed_dim,
            heads=self.enc_heads, 
            depth=self.enc_depth, 
            layer_dropout=self.layer_dropout,
        )
        
        # Create student encoder without forcing CUDA
        self.student_encoder = copy.deepcopy(self.teacher_encoder)
        
        # Initialize predictor
        from model_new import Predictor
        self.predictor = Predictor(self.embed_dim, self.enc_heads, self.pred_depth)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)


class EmbeddingExtractor:
    """Extract embeddings from I-JEPA model using both encoders."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() or device == 'mps' else 'cpu')
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.base_model = None
        
    def load_model(self):
        """Load model from checkpoint with fixed initialization."""
        print(f"Loading model from {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Extract hyperparameters
        hparams = checkpoint['hyper_parameters']
        
        # Create fixed base model
        self.base_model = IJEPA_base_fixed(
            img_size=hparams['img_size'],
            patch_size=hparams['patch_size'],
            in_chans=hparams.get('in_chans', 3),
            embed_dim=hparams['embed_dim'],
            enc_depth=hparams['enc_depth'],
            enc_heads=hparams['enc_heads'],
            pred_depth=hparams.get('decoder_depth', 2),  # Use decoder_depth from checkpoint
            fuzzy=hparams.get('fuzzy', 0)
        )
        
        # Load state dict for base model
        state_dict = checkpoint['state_dict']
        base_model_state = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        self.base_model.load_state_dict(base_model_state, strict=False)
        
        self.base_model.eval()
        self.base_model.to(self.device)
        
        # Extract model info
        self.embed_dim = self.base_model.embed_dim
        self.patch_size = self.base_model.patch_size
        self.img_size = self.base_model.img_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        print(f"Model loaded: embed_dim={self.embed_dim}, patch_size={self.patch_size}, img_size={self.img_size}")
        
    def extract_both_embeddings(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract embeddings from both encoders."""
        with torch.no_grad():
            # Convert images to patches
            x = self.base_model.patch_embed(images)  # [B, embed_dim, H, W]
            x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
            
            # Add positional embeddings
            x = x + self.base_model.pos_embed
            
            # Pass through both encoders
            student_embeddings = self.base_model.student_encoder(x)  # [B, num_patches, embed_dim]
            teacher_embeddings = self.base_model.teacher_encoder(x)  # [B, num_patches, embed_dim]
            
        return {
            'student': student_embeddings,
            'teacher': teacher_embeddings
        }


def process_split(extractor: EmbeddingExtractor, 
                 dataloader: DataLoader, 
                 split_name: str,
                 output_dir: Path,
                 pool_patches: bool = True) -> Dict:
    """Process a data split and extract embeddings from both encoders."""
    print(f"\nProcessing {split_name} split...")
    
    # Create output directory
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Storage lists
    all_student_embeddings = []
    all_teacher_embeddings = []
    all_labels = []
    
    # Statistics
    total_samples = 0
    
    # Process batches
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Extracting {split_name}")):
        images = images.to(extractor.device)
        batch_size = images.shape[0]
        total_samples += batch_size
        
        # Extract embeddings from both encoders
        embeddings = extractor.extract_both_embeddings(images)
        
        student_emb = embeddings['student']  # [B, num_patches, embed_dim]
        teacher_emb = embeddings['teacher']  # [B, num_patches, embed_dim]
        
        # Pool patches if requested
        if pool_patches:
            student_emb = student_emb.mean(dim=1)  # [B, embed_dim]
            teacher_emb = teacher_emb.mean(dim=1)  # [B, embed_dim]
            
        all_student_embeddings.append(student_emb.cpu().numpy())
        all_teacher_embeddings.append(teacher_emb.cpu().numpy())
        all_labels.append(labels.numpy())
        
    # Concatenate results
    all_labels = np.concatenate(all_labels)
    
    # Save embeddings
    if pool_patches:
        student_array = np.vstack(all_student_embeddings)  # [N, embed_dim]
        teacher_array = np.vstack(all_teacher_embeddings)  # [N, embed_dim]
        np.save(split_dir / 'student_embeddings_pooled.npy', student_array)
        np.save(split_dir / 'teacher_embeddings_pooled.npy', teacher_array)
    else:
        student_array = np.vstack(all_student_embeddings)  # [N, num_patches, embed_dim]
        teacher_array = np.vstack(all_teacher_embeddings)  # [N, num_patches, embed_dim]
        np.save(split_dir / 'student_embeddings_patches.npy', student_array)
        np.save(split_dir / 'teacher_embeddings_patches.npy', teacher_array)
        
    # Save labels
    np.save(split_dir / 'labels.npy', all_labels)
    
    # Compute embedding statistics
    if pool_patches:
        # Compute similarity between student and teacher embeddings
        # Normalize embeddings
        student_norm = student_array / (np.linalg.norm(student_array, axis=1, keepdims=True) + 1e-8)
        teacher_norm = teacher_array / (np.linalg.norm(teacher_array, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        similarities = np.sum(student_norm * teacher_norm, axis=1)
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
    else:
        avg_similarity = 0
        std_similarity = 0
    
    # Statistics
    stats = {
        'split': split_name,
        'total_samples': total_samples,
        'embed_dim': extractor.embed_dim,
        'num_patches': extractor.num_patches,
        'pooled': pool_patches,
        'avg_cosine_similarity': float(avg_similarity),
        'std_cosine_similarity': float(std_similarity)
    }
    
    return stats


def analyze_encoders(output_dir: Path, splits: List[str] = ['train', 'val']):
    """Analyze differences between student and teacher encoders."""
    print("\n=== Encoder Analysis ===")
    
    for split in splits:
        split_dir = output_dir / split
        if not split_dir.exists():
            continue
            
        print(f"\n{split.upper()} split:")
        
        # Load pooled embeddings
        student_path = split_dir / 'student_embeddings_pooled.npy'
        teacher_path = split_dir / 'teacher_embeddings_pooled.npy'
        
        if student_path.exists() and teacher_path.exists():
            student_emb = np.load(student_path)
            teacher_emb = np.load(teacher_path)
            
            # Compute statistics
            print(f"  Student embedding stats:")
            print(f"    - Mean norm: {np.mean(np.linalg.norm(student_emb, axis=1)):.4f}")
            print(f"    - Std norm: {np.std(np.linalg.norm(student_emb, axis=1)):.4f}")
            
            print(f"  Teacher embedding stats:")
            print(f"    - Mean norm: {np.mean(np.linalg.norm(teacher_emb, axis=1)):.4f}")
            print(f"    - Std norm: {np.std(np.linalg.norm(teacher_emb, axis=1)):.4f}")
            
            # Compute differences
            diff = student_emb - teacher_emb
            print(f"  Difference stats:")
            print(f"    - Mean L2 distance: {np.mean(np.linalg.norm(diff, axis=1)):.4f}")
            print(f"    - Max L2 distance: {np.max(np.linalg.norm(diff, axis=1)):.4f}")
            print(f"    - Min L2 distance: {np.min(np.linalg.norm(diff, axis=1)):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from I-JEPA model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='embeddings',
                       help='Output directory for embeddings')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--no_pool', action='store_true',
                       help='Save patch-level embeddings without pooling')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu/mps)')
    parser.add_argument('--analyze', action='store_true',
                       help='Perform analysis of encoder differences')
    
    args = parser.parse_args()
    
    # Setup
    checkpoint_name = Path(args.checkpoint).stem
    output_dir = Path(args.output_dir) / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting embeddings from: {args.checkpoint}")
    print(f"Output directory: {output_dir}")
    
    # Load model
    extractor = EmbeddingExtractor(args.checkpoint, device=args.device)
    extractor.load_model()
    
    # Create datasets with labels
    train_dataset = TinyImageNetDatasetWithLabels(
        dataset_path=args.data_path,
        stage='train',
        img_size=extractor.img_size
    )
    
    val_dataset = TinyImageNetDatasetWithLabels(
        dataset_path=args.data_path,
        stage='val',
        img_size=extractor.img_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda')
    )
    
    # Process each split
    all_stats = []
    
    for split_name, dataloader in [('train', train_loader), ('val', val_loader)]:
        stats = process_split(
            extractor=extractor,
            dataloader=dataloader,
            split_name=split_name,
            output_dir=output_dir,
            pool_patches=not args.no_pool
        )
        all_stats.append(stats)
        print(f"Completed {split_name}: {stats['total_samples']} samples")
        print(f"  - Avg cosine similarity (student vs teacher): {stats['avg_cosine_similarity']:.4f}")
        
    # Save metadata
    metadata = {
        'checkpoint': args.checkpoint,
        'extraction_date': datetime.now().isoformat(),
        'model_info': {
            'embed_dim': extractor.embed_dim,
            'patch_size': extractor.patch_size,
            'img_size': extractor.img_size,
            'num_patches': extractor.num_patches
        },
        'extraction_config': {
            'batch_size': args.batch_size,
            'pool_patches': not args.no_pool,
            'device': args.device
        },
        'statistics': all_stats
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nExtraction complete! Results saved to: {output_dir}")
    print("\nExtraction summary:")
    for stats in all_stats:
        print(f"  {stats['split']}: {stats['total_samples']} samples")
        
    # Perform analysis if requested
    if args.analyze:
        analyze_encoders(output_dir)
        
    print("\n=== Files created ===")
    for split in ['train', 'val']:
        split_dir = output_dir / split
        if split_dir.exists():
            print(f"\n{split}/")
            for file in sorted(split_dir.glob('*.npy')):
                print(f"  - {file.name}")


if __name__ == '__main__':
    main()