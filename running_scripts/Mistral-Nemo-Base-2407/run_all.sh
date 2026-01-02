#!/bin/bash

# Run all scripts in the current directory sequentially
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Sort by filename and execute sequentially
for script in "$SCRIPT_DIR"/*.sh; do
    # Skip itself
    if [ "$(basename "$script")" == "run_all.sh" ]; then
        continue
    fi

    echo "=========================================="
    echo "Running script: $(basename "$script")"
    echo "=========================================="

    bash "$script"

    if [ $? -ne 0 ]; then
        echo "Error: $(basename "$script") execution failed"
        exit 1
    fi

    echo ""
done

echo "All scripts executed successfully"
