#!/bin/bash
# Run dimensionality reduction on exp folder embeddings with 50 classes

echo "Running dimensionality reduction on exp folder embeddings"
echo "Using 50 classes for all epochs"
echo "============================================================"

# Process each epoch
for epoch in {0..9}; do
    echo ""
    echo "Processing epoch $epoch..."
    
    # For masked embeddings, we need to process each mask directory
    # Using the first mask (mask_00) as representative
    python reduce_embeddings.py \
        --embeddings_dir "embeddings_exp/ijepa-64px-epoch=$(printf "%02d" $epoch)/masked/val/ijepa_75/mask_00" \
        --output_dir "reduced_embeddings_exp/epoch_$(printf "%02d" $epoch)" \
        --num_classes 50 \
        --epochs $epoch \
        --splits val \
        --methods pca umap \
        --no_cross_epoch
        
    if [ $? -eq 0 ]; then
        echo "✓ Completed dimensionality reduction for epoch $epoch"
    else
        echo "✗ Failed to process epoch $epoch"
    fi
done

echo ""
echo "============================================================"
echo "Dimensionality reduction complete!"
echo "Results saved to: reduced_embeddings_exp/"
echo "============================================================"