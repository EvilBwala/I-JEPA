import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
import argparse
from model_new import IJEPA, TinyImageNetDataModule

def main(args):
    """
    Main function to set up and run the pretraining process
    """
    # Set random seeds for reproducibility
    pl.seed_everything(args.seed)
    
    # Create data module
    data_module = TinyImageNetDataModule(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        data_fraction=args.data_fraction,
    )
    
    # Create model
    model = IJEPA(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=args.embed_dim,
        enc_heads=args.enc_heads,
        enc_depth=args.enc_depth,
        decoder_depth=args.decoder_depth,
        lr=args.lr,
        weight_decay=args.weight_decay,
        target_aspect_ratio=(args.target_aspect_ratio_min, args.target_aspect_ratio_max),
        target_scale=(args.target_scale_min, args.target_scale_max),
        context_aspect_ratio=args.context_aspect_ratio,
        context_scale=(args.context_scale_min, args.context_scale_max),
        M=args.M,
        m=args.m,
        m_start_end=(args.m_start, args.m_end),
        fuzzy=args.fuzzy
    )
    
    # Define callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelSummary(max_depth=2),
    ]
    
    # Add checkpoint callback
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f"ijepa-{args.img_size}px-{{epoch:02d}}",
            save_top_k=-1,  # Save all checkpoints
            every_n_epochs=1,  # Save every epoch
            monitor=None,  # Don't monitor any metric
        )
        callbacks.append(checkpoint_callback)
    
    # Set up logger
    logger = None
    if args.use_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name or f"ijepa-{args.img_size}px-{args.embed_dim}d",
            save_dir=args.wandb_save_dir,
        )
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        logger=logger,
        log_every_n_steps=1,  # Log every step
        check_val_every_n_epoch=1,  # Run validation every epoch
        val_check_interval=None,    # Don't use fractional validation
        limit_val_batches=args.limit_val_batches,
        enable_progress_bar=False,  # Disable progress bar
        enable_model_summary=True,  # Enable model summary
        enable_checkpointing=True,  # Enable checkpointing
        num_sanity_val_steps=0,     # Skip sanity validation steps to avoid cluttering output
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Save final model if specified
    if args.save_final_model:
        final_path = os.path.join(args.checkpoint_dir, f"ijepa-{args.img_size}px-final.ckpt")
        trainer.save_checkpoint(final_path)
        print(f"Final model saved to {final_path}")
    
    return model, trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain I-JEPA model on TinyImageNet")
    
    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, default="tiny-imagenet-200",
                        help="Path to the TinyImageNet dataset")
    parser.add_argument("--img_size", type=int, default=64,
                        help="Image size (TinyImageNet is 64x64)")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of the dataset to use (0.0-1.0)")
    
    # Model parameters
    parser.add_argument("--patch_size", type=int, default=8,
                        help="Patch size for the transformer")
    parser.add_argument("--embed_dim", type=int, default=192,
                        help="Embedding dimension")
    parser.add_argument("--enc_heads", type=int, default=8,
                        help="Number of attention heads in the encoder")
    parser.add_argument("--enc_depth", type=int, default=12,
                        help="Depth of the encoder")
    parser.add_argument("--decoder_depth", type=int, default=4,
                        help="Depth of the decoder")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=2,
                        help="Maximum number of epochs")
    parser.add_argument("--gradient_clip_val", type=float, default=0.1,
                        help="Gradient clipping value")
    
    # I-JEPA specific parameters
    parser.add_argument("--target_aspect_ratio_min", type=float, default=0.75,
                        help="Minimum target aspect ratio")
    parser.add_argument("--target_aspect_ratio_max", type=float, default=1.5,
                        help="Maximum target aspect ratio")
    parser.add_argument("--target_scale_min", type=float, default=0.15,
                        help="Minimum target scale")
    parser.add_argument("--target_scale_max", type=float, default=0.2,
                        help="Maximum target scale")
    parser.add_argument("--context_aspect_ratio", type=float, default=1.0,
                        help="Context aspect ratio")
    parser.add_argument("--context_scale_min", type=float, default=0.85,
                        help="Minimum context scale")
    parser.add_argument("--context_scale_max", type=float, default=1.0,
                        help="Maximum context scale")
    parser.add_argument("--M", type=int, default=4,
                        help="Number of different target blocks")
    parser.add_argument("--m", type=float, default=0.996,
                        help="Initial momentum value")
    parser.add_argument("--m_start", type=float, default=0.996,
                        help="Starting momentum value")
    parser.add_argument("--m_end", type=float, default=1.0,
                        help="Ending momentum value")
    parser.add_argument("--fuzzy", type=int, default=0,
                        help="Use fuzzy target blocks")
    
    # Hardware parameters
    parser.add_argument("--accelerator", type=str, default="gpu",
                        help="Accelerator type (gpu, cpu, tpu)")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of devices to use")
    parser.add_argument("--precision", type=int, default=16,
                        help="Precision for training (16, 32)")
    
    # Logging and checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_final_model", action="store_true",
                        help="Save the final model after training")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="ijepa-tiny-imagenet",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--wandb_save_dir", type=str, default="./wandb",
                        help="Directory to save W&B files")
    parser.add_argument("--val_check_interval", type=float, default=0.5,
                        help="Validation check interval (fraction of epoch or int steps)")
    parser.add_argument("--limit_val_batches", type=float, default=0.25,
                        help="Limit validation batches (fraction or int)")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Run training
    model, trainer = main(args) 