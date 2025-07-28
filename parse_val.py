#!/usr/bin/env python3
"""Parse TinyImageNet validation set into class folders."""

import os
import shutil
from pathlib import Path

def parse_val_dataset(data_path='tiny-imagenet-200'):
    """Parse validation set into class folders like train set."""
    
    val_dir = Path(data_path) / 'val'
    val_parsed_dir = Path(data_path) / 'val_parsed'
    
    # Read validation annotations
    val_annotations_file = val_dir / 'val_annotations.txt'
    
    if not val_annotations_file.exists():
        print(f"Error: {val_annotations_file} not found!")
        return
        
    # Create parsed directory
    val_parsed_dir.mkdir(exist_ok=True)
    
    # Read annotations
    with open(val_annotations_file, 'r') as f:
        lines = f.readlines()
    
    # Parse each line
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            img_name = parts[0]
            class_id = parts[1]
            
            # Create class directory
            class_dir = val_parsed_dir / class_id
            class_dir.mkdir(exist_ok=True)
            
            # Copy image
            src = val_dir / 'images' / img_name
            dst = class_dir / img_name
            
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
    
    # Count results
    num_classes = len(list(val_parsed_dir.glob('*')))
    num_images = len(list(val_parsed_dir.glob('*/*.JPEG')))
    
    print(f"Parsed validation set:")
    print(f"  - Classes: {num_classes}")
    print(f"  - Images: {num_images}")
    print(f"  - Output: {val_parsed_dir}")

if __name__ == '__main__':
    parse_val_dataset()