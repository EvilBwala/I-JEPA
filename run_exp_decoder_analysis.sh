#!/bin/bash
# Run decoder analysis on all exp folder embeddings

echo "Running decoder analysis on exp folder embeddings (10 epochs)"
echo "Using alpha=0.001 (C=1000)"
echo "============================================================"

# For each epoch, run decoder analysis on masked embeddings
for epoch in {0..9}; do
    echo ""
    echo "Processing epoch $epoch..."
    
    # Run linear decoder analysis
    python analysis/decoder/run_decoder_analysis.py \
        --embeddings_dir "analysis/dim_reduction/embeddings_exp/ijepa-64px-epoch=$(printf "%02d" $epoch)/masked" \
        --output_dir "analysis/decoder/results_exp/epoch_$epoch" \
        --epochs $epoch \
        --C 1000.0 \
        --max_iter 1000 \
        --train_split val \
        --eval_split val
        
    echo "✓ Completed linear decoder for epoch $epoch"
done

echo ""
echo "Now running non-linear decoder analysis..."
echo ""

# Run non-linear decoder analysis for each epoch
for epoch in {0..9}; do
    echo "Processing epoch $epoch with neural network decoder..."
    
    python analysis/decoder/run_nn_decoder_analysis.py \
        --embeddings_dir "analysis/dim_reduction/embeddings_exp/ijepa-64px-epoch=$(printf "%02d" $epoch)/masked" \
        --output_dir "analysis/decoder/nn_results_exp/epoch_$epoch" \
        --epochs $epoch \
        --alpha 0.001 \
        --train_split val \
        --eval_split val
        
    echo "✓ Completed non-linear decoder for epoch $epoch"
done

echo ""
echo "============================================================"
echo "Decoder analysis complete!"
echo "Results saved to:"
echo "  - Linear: analysis/decoder/results_exp/"
echo "  - Non-linear: analysis/decoder/nn_results_exp/"
echo "============================================================"