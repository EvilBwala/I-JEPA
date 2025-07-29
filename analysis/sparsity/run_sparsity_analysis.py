#!/usr/bin/env python3
"""Run sparsity analysis on I-JEPA embeddings."""

import argparse
from pathlib import Path
from sparsity_analysis import SparsityAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Analyze sparsity in I-JEPA embeddings')
    parser.add_argument('--embeddings_dir', type=str, 
                       default='analysis/dim_reduction/embeddings',
                       help='Directory containing extracted embeddings')
    parser.add_argument('--output_dir', type=str, 
                       default='analysis/sparsity/results',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                       help='Which epochs to analyze')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                       help='Which split to analyze')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SparsityAnalyzer(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir
    )
    
    # Run analysis
    results = analyzer.run_analysis(
        epochs=args.epochs,
        split=args.split
    )
    
    print("\nSparsity analysis complete!")
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()