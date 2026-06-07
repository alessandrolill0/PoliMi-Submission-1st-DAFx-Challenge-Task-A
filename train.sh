#!/bin/bash

set -e

echo "Inizio l'elaborazione del dataset (da 0001 a 0016)..."

for i in {0001..0016}; do
    TARGET_PATH="target/2026-DATASET-STRIPPED/random_IR_${i}.npz"
    
    echo "=========================================================="
    echo "Esecuzione in corso per: $TARGET_PATH"
    echo "=========================================================="
    
    python Model/ParamsEstimation.py --target "$TARGET_PATH"
    
done

