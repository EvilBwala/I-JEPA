#!/bin/bash
# Extract embeddings from all checkpoints in exp folder with I-JEPA masking
# 5 mask realizations per checkpoint

OUTPUT_BASE="embeddings_exp"

echo "Extracting embeddings from exp folder checkpoints with I-JEPA masking..."
echo "Output directory: $OUTPUT_BASE"

# Process each checkpoint
for ckpt in ../../exp/ijepa-64px-epoch=*.ckpt; do
    epoch=$(basename "$ckpt" | sed 's/ijepa-64px-epoch=\([0-9]*\)\.ckpt/\1/')
    echo ""
    echo "=========================================="
    echo "Processing checkpoint: $ckpt (epoch $epoch)"
    echo "=========================================="
    
    python extract_embeddings.py \
        --checkpoint "$ckpt" \
        --data_path ../../tiny-imagenet-200 \
        --output_dir "$OUTPUT_BASE" \
        --with_masks \
        --num_mask_samples 5 \
        --samples_per_mask 500 \
        --batch_size 50 \
        --num_workers 2
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed epoch $epoch"
    else
        echo "✗ Failed to process epoch $epoch"
    fi
done

echo ""
echo "=========================================="
echo "Embedding extraction complete!"
echo "Results saved to: $OUTPUT_BASE"
echo "=========================================="