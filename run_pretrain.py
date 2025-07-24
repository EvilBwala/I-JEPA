import os
import subprocess
import argparse

def run_pretraining(config_name):
    """
    Run pretraining with a predefined configuration
    """
    configs = {
        "small": [
            "--img_size", "64",
            "--patch_size", "8",
            "--embed_dim", "128",
            "--enc_heads", "4",
            "--enc_depth", "6",
            "--decoder_depth", "2",
            "--batch_size", "64",
            "--max_epochs", "1",
            "--checkpoint_dir", "checkpoints/small"
        ],
        "verbose": [
            "--img_size", "64",
            "--patch_size", "8",
            "--embed_dim", "96",
            "--enc_heads", "4",
            "--enc_depth", "4",
            "--decoder_depth", "2",
            "--batch_size", "8",
            "--max_epochs", "2",
            "--checkpoint_dir", "checkpoints/verbose",
            "--limit_val_batches", "1.0"  # Use all validation batches
        ],
        "fast": [
            "--img_size", "64",
            "--patch_size", "8",
            "--embed_dim", "96",
            "--enc_heads", "4",
            "--enc_depth", "4",
            "--decoder_depth", "2",
            "--batch_size", "16",
            "--max_epochs", "5",
            "--checkpoint_dir", "checkpoints/fast",
            "--data_fraction", "0.1"  # Use only 10% of the data
        ],
        "medium": [
            "--img_size", "64",
            "--patch_size", "8",
            "--embed_dim", "192",
            "--enc_heads", "8",
            "--enc_depth", "12",
            "--decoder_depth", "4",
            "--batch_size", "32",
            "--max_epochs", "50",
            "--checkpoint_dir", "checkpoints/medium"
        ],
        "large": [
            "--img_size", "64",
            "--patch_size", "8",
            "--embed_dim", "256",
            "--enc_heads", "12",
            "--enc_depth", "16",
            "--decoder_depth", "6",
            "--batch_size", "16",
            "--max_epochs", "100",
            "--checkpoint_dir", "checkpoints/large"
        ],
        "cpu": [
            "--img_size", "64",
            "--patch_size", "8",
            "--embed_dim", "96",
            "--enc_heads", "4",
            "--enc_depth", "4",
            "--decoder_depth", "2",
            "--batch_size", "8",
            "--max_epochs", "5",
            "--accelerator", "cpu",
            "--precision", "32",
            "--checkpoint_dir", "checkpoints/cpu"
        ]
    }
    
    if config_name not in configs:
        print(f"Configuration '{config_name}' not found. Available configs: {list(configs.keys())}")
        return
    
    # Build command
    cmd = ["python", "pretrain_IJEPA_new.py"] + configs[config_name]
    
    # Add any additional arguments
    if args.additional_args:
        cmd.extend(args.additional_args)
    
    # Print command
    print("Running command:")
    print(" ".join(cmd))
    
    # Run command
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run I-JEPA pretraining with predefined configurations")
    parser.add_argument("config", choices=["small", "medium", "large", "cpu", "verbose", "fast"],
                        help="Configuration to use (small, medium, large, cpu, verbose, or fast)")
    parser.add_argument("additional_args", nargs="*",
                        help="Additional arguments to pass to pretrain_IJEPA_new.py")
    
    args = parser.parse_args()
    
    run_pretraining(args.config) 