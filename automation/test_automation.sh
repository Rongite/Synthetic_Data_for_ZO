#!/bin/bash
# Automation tool test script

echo "========================================="
echo "Testing Automation Pipeline"
echo "========================================="

# Test stage1 generator
echo -e "\n[Test 1] Stage 1: Synthetic Data Generator"
python3 automation/stage1_generation/generator.py \
        automation/configs/examples/stage1_example_copa_mezo.yaml 2>&1 | head -20

if [ -d "Data_v2/synthetic/Copa_mezo_gpt4o_v1" ]; then
    echo "✓ Stage 1 generator works correctly"
    echo "  Generated scripts:"
    ls Data_v2/synthetic/Copa_mezo_gpt4o_v1/scripts/
else
    echo "✗ Stage 1 generator failed"
fi

# Test stage2 trainer (dry run)
echo -e "\n[Test 2] Stage 2: Training Pipeline (Dry Run)"
python3 automation/stage2_training/trainer.py \
        automation/configs/examples/stage2_example_training.yaml \
        --dry-run 2>&1 | head -30

if [ $? -eq 0 ]; then
    echo "✓ Stage 2 trainer works correctly"
else
    echo "✗ Stage 2 trainer failed"
fi

# Display directory structure
echo -e "\n[Test 3] Verify directory structure"
echo "Data_v2:"
ls -d Data_v2/*/ 2>/dev/null
echo -e "\nResults_v2:"
ls -d Results_v2/*/ 2>/dev/null | head -5
echo -e "\nPending:"
ls -d Pending_Manual_Classification/*/ 2>/dev/null

echo -e "\n========================================="
echo "Testing complete!"
echo "========================================="
