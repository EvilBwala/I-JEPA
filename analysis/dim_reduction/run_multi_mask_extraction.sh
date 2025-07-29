#!/bin/bash
# Extract embeddings with 10 different random mask patterns using I-JEPA strategy

echo "Extracting embeddings with 10 different mask patterns..."
echo "This will apply the same I-JEPA masking strategy (M=4 blocks) with different random positions"

python extract_embeddings.py \
    --checkpoint ../../checkpoints/ijepa-64px-epoch=04.ckpt \
    --data_path ../../datasets/tiny-imagenet-200 \
    --output_dir embeddings \
    --with_masks \
    --num_mask_samples 10 \
    --samples_per_mask 500 \
    --batch_size 50 \
    --num_workers 2

echo "Done! Now run the sparsity analysis:"
echo "python ../sparsity/analyze_masked_only.py"