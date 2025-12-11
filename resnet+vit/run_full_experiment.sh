#!/bin/bash

echo "========================================"
echo "FULL EXPERIMENT PIPELINE"
echo "========================================"
echo ""
echo "Step 1: ResNet-50 Block Pruning (L2, MLP, Encoder)"
echo "Step 2: Generalization Test on Transfer Datasets"
echo ""

# Step 1: Run ResNet-50 block pruning
echo "========================================"
echo "Running 5_resnet50_block.py"
echo "========================================"
python 5_resnet50_block.py

if [ $? -ne 0 ]; then
    echo "Error: 5_resnet50_block.py failed"
    exit 1
fi

echo ""
echo "5_resnet50_block.py completed!"
echo ""

# Step 2: Run overfitting experiment
echo "========================================"
echo "Running 6_overfit_exp.py"
echo "========================================"
python 6_overfit_exp.py

if [ $? -ne 0 ]; then
    echo "Error: 6_overfit_exp.py failed"
    exit 1
fi

echo ""
echo "========================================"
echo "FULL PIPELINE COMPLETED!"
echo "========================================"
