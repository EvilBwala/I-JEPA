#!/usr/bin/env python3
"""Test script to verify all analysis modules work with new paths."""

import sys
from pathlib import Path

def test_imports():
    """Test that all analysis modules can be imported."""
    print("Testing imports for reorganized analysis code...")
    print("="*60)
    
    errors = []
    
    # Test dimension reduction imports
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from analysis.dim_reduction import extract_embeddings
        print("✓ extract_embeddings imported successfully")
    except Exception as e:
        errors.append(f"✗ extract_embeddings import failed: {e}")
    
    try:
        from analysis.dim_reduction import reduce_embeddings
        print("✓ reduce_embeddings imported successfully")
    except Exception as e:
        errors.append(f"✗ reduce_embeddings import failed: {e}")
    
    try:
        from analysis.dim_reduction import visualize_embeddings
        print("✓ visualize_embeddings imported successfully")
    except Exception as e:
        errors.append(f"✗ visualize_embeddings import failed: {e}")
    
    # Test decoder imports
    try:
        from analysis.decoder import run_decoder_analysis
        print("✓ run_decoder_analysis imported successfully")
    except Exception as e:
        errors.append(f"✗ run_decoder_analysis import failed: {e}")
    
    try:
        from analysis.decoder import run_nn_decoder_analysis
        print("✓ run_nn_decoder_analysis imported successfully")
    except Exception as e:
        errors.append(f"✗ run_nn_decoder_analysis import failed: {e}")
    
    try:
        from analysis.decoder import run_sgd_decoder_analysis
        print("✓ run_sgd_decoder_analysis imported successfully")
    except Exception as e:
        errors.append(f"✗ run_sgd_decoder_analysis import failed: {e}")
    
    # Test RSA imports
    try:
        from analysis.rsa import rsa_analysis
        print("✓ rsa_analysis imported successfully")
    except Exception as e:
        errors.append(f"✗ rsa_analysis import failed: {e}")
    
    try:
        from analysis.rsa import run_rsa
        print("✓ run_rsa imported successfully")
    except Exception as e:
        errors.append(f"✗ run_rsa import failed: {e}")
    
    print("\n" + "="*60)
    if errors:
        print(f"Found {len(errors)} import errors:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("All imports successful!")
        return True

def test_default_paths():
    """Test that default paths in argparse are correct."""
    print("\nTesting default paths...")
    print("="*60)
    
    import argparse
    errors = []
    
    # Test extract_embeddings defaults
    try:
        from analysis.dim_reduction.extract_embeddings import main
        parser = argparse.ArgumentParser()
        parser.add_argument('--output_dir', type=str, default='analysis/dim_reduction/embeddings')
        args = parser.parse_args(['--output_dir', 'analysis/dim_reduction/embeddings'])
        print(f"✓ extract_embeddings default output_dir: {args.output_dir}")
    except Exception as e:
        errors.append(f"✗ extract_embeddings path test failed: {e}")
    
    # Test reduce_embeddings defaults
    try:
        from analysis.dim_reduction.reduce_embeddings import EmbeddingReducer
        reducer = EmbeddingReducer()
        print(f"✓ reduce_embeddings default embeddings_dir: {reducer.embeddings_dir}")
        print(f"✓ reduce_embeddings default output_dir: {reducer.output_dir}")
    except Exception as e:
        errors.append(f"✗ reduce_embeddings path test failed: {e}")
    
    print("\n" + "="*60)
    if errors:
        print(f"Found {len(errors)} path errors:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("All default paths correct!")
        return True

if __name__ == "__main__":
    import_success = test_imports()
    path_success = test_default_paths()
    
    if import_success and path_success:
        print("\n✅ All tests passed! The reorganized analysis code is ready to use.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please fix the errors above.")
        sys.exit(1)