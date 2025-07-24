import torch
import pytorch_lightning as pl
from pretrain_IJEPA import IJEPA
import os

def load_and_show_model_properties():
    # Path to checkpoint
    checkpoint_path = "lightning_logs/version_0/checkpoints/epoch=9-step=70.ckpt"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    # Load the model from checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    model = IJEPA.load_from_checkpoint(checkpoint_path)
    
    # Print model hyperparameters
    print("\nModel Hyperparameters:")
    for key, value in model.hparams.items():
        print(f"{key}: {value}")
    
    # Print model architecture summary
    print("\nModel Architecture:")
    print(f"Student Encoder: {model.model.student_encoder.__class__.__name__}")
    print(f"Teacher Encoder: {model.model.teacher_encoder.__class__.__name__}")
    print(f"Predictor: {model.model.predictor.__class__.__name__}")
    
    # Print model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print additional model info
    print("\nAdditional Model Info:")
    print(f"Number of target blocks (M): {model.M}")
    print(f"Embedding dimension: {model.embed_dim}")
    print(f"Patch size: {model.patch_size}")
    print(f"Number of tokens: {model.num_tokens}")
    
    return model

if __name__ == "__main__":
    model = load_and_show_model_properties() 