# Synthetic Data Automation Pipeline

**Version**: 2.1
**Last Updated**: 2026-01-01
**Status**: ✅ All P0 Bugs Fixed

Complete automation tools for synthetic data generation and training

---

## 📚 Core Documentation

### **Recommended Reading Order for New Users**

1. **[SETUP_GUIDE.md](SETUP_GUIDE.md)** 🔧 - Environment Setup Guide (Must-read for first-time users)
2. **[USER_GUIDE.md](USER_GUIDE.md)** 📖 - User Manual (Complete guide)
3. **[COMPLETE_PIPELINE_SIMULATION.md](COMPLETE_PIPELINE_SIMULATION.md)** 📋 - Complete Pipeline Example
4. **[BATCH_GUIDE.md](BATCH_GUIDE.md)** 📦 - Batch Solution 3++ Complete Guide
5. **[TOOLS_REFERENCE.md](TOOLS_REFERENCE.md)** 🛠️ - All Tools Reference Manual

### **Other Documentation**

- **[BUG_FIXES_SUMMARY.md](BUG_FIXES_SUMMARY.md)** - Bug Fixes Summary (v2.1)
- **[stage2_training/RESULTS_MANAGEMENT.md](stage2_training/RESULTS_MANAGEMENT.md)** - Training Results Management System

## Directory Structure

```
automation/
├── stage1_generation/              # Stage 1: Synthetic Data Generation
│   ├── generator.py                # Data generator (using Batch Solution 3++)
│   ├── experiment_manager_batch.py # Batch experiment manager (core)
│   ├── batch_tools/                # Batch management tools
│   │   ├── list_batches.py         # List all batches
│   │   ├── list_batch_experiments.py  # View experiments in batch
│   │   ├── list_shared_experiments.py # View physical data
│   │   ├── compare_experiments.py     # Compare experiment parameters
│   │   ├── list_data_paths.py ⭐   # Find data paths (new)
│   │   └── resolve_data_path.py ⭐ # Path conversion (new)
│   ├── tools/                      # Manual review tools
│   │   ├── review_top20.py
│   │   ├── annotate_samples.py
│   │   └── publish_dataset.py (optional)
│   └── WORKFLOW.md                 # Complete workflow documentation
│
├── stage2_training/                # Stage 2: Training
│   ├── trainer.py                  # Training automation (LoRA bugs fixed)
│   ├── list_results.py             # View training results
│   └── RESULTS_MANAGEMENT.md       # Results_v2 system documentation
│
├── configs/                        # Configuration files
│   ├── examples/                   # Configuration examples
│   │   ├── stage1_full_example_copa.yaml
│   │   └── stage2_example_training.yaml
│   ├── stage1/                     # Stage 1 configs (user)
│   └── stage2/                     # Stage 2 configs (user)
│
├── config.py                       # Unified path configuration (new)
├── fix_hardcoded_paths.py          # Path fixing script (for maintenance)
│
├── SETUP_GUIDE.md                  # Environment setup guide (new)
├── USER_GUIDE.md                   # User manual
├── BATCH_GUIDE.md                  # Batch Solution 3++ guide
├── TOOLS_REFERENCE.md              # All tools reference (new)
├── COMPLETE_PIPELINE_SIMULATION.md # Complete pipeline example
├── BUG_FIXES_SUMMARY.md            # Bug fixes summary (new)
└── README.md                       # This document
```

## Quick Start

### Stage 1: Generate Synthetic Data (Using Batch Solution 3++)

#### Step 1: Create Configuration File

```bash
cp configs/examples/stage1_full_example_copa.yaml configs/stage1/my_copa_config.yaml
```

Edit the configuration file, key modifications:

```yaml
# Experiment Management (Batch Solution 3++)
experiment:
  batch_id: "batch_20241229_temperature"  # Batch ID (optional)
  purpose: "temperature_study"            # Experiment purpose
  description: "Study the impact of temperature parameter on synthetic data quality"

# Dataset Configuration
dataset:
  task_name: "copa"
  input_path: "Data/original/Copa/copa_train.jsonl"

# Generation Configuration
generation:
  model: "gpt-4o"
  temperature: 0.7           # These parameters will calculate fingerprint
  top_p: 1.0                 # For automatic deduplication
  rephrase_prompt: |
    Your rephrasing prompt...

# Validation Configuration
validation:
  model: "gpt-4o"
  temperature: 0.0
  validation_prompt: |
    Your validation prompt...
  few_shot_examples:
    - ...
```

#### Step 2: Generate Data

```bash
cd automation/stage1_generation

# Run generator
python generator.py --config ../configs/stage1/my_copa_config.yaml

# The system will automatically:
# 1. Calculate parameter fingerprint (based on model, temperature, top_p, etc.)
# 2. Check if data with the same fingerprint exists in _shared/Copa/
# 3. If exists → Reuse physical data, create batch symbolic link
# 4. If not → Generate new data to _shared/, create batch symbolic link
```

Output directory structure:

```
Data_v2/synthetic/
├── _shared/                                # Physical data storage
│   └── Copa/
│       └── temp07_topp10_gpt4o/           # Actual data files
│           ├── copa_train.jsonl
│           ├── copa_validation.jsonl
│           ├── copa_test.jsonl
│           ├── .fingerprint               # Parameter fingerprint
│           └── experiment_metadata.json   # Complete metadata
│
└── batch_20241229_temperature/            # Batch view (symbolic links)
    └── Copa/
        └── temp07_topp10_gpt4o -> ../../_shared/Copa/temp07_topp10_gpt4o/
```

#### Step 3: View and Manage Experiments

```bash
# List all batches
python batch_tools/list_batches.py --verbose

# View experiments in batch
python batch_tools/list_batch_experiments.py batch_20241229_temperature --verbose

# View all experiments in physical storage (check if parameter configuration already generated)
python batch_tools/list_shared_experiments.py --dataset Copa --verbose

# Compare parameters of two experiments
python batch_tools/compare_experiments.py \
    --shared Copa/temp07_topp10_gpt4o \
    --shared Copa/temp09_topp10_gpt4o
```

### Stage 2: Training

#### Step 1: Create Training Configuration

```bash
cp configs/examples/stage2_example_training.yaml configs/stage2/my_training.yaml
```

Edit configuration file:

```yaml
experiment:
  purpose: "temperature_study"  # Training experiment purpose (independent from data generation purpose)
  description: "Evaluate training effectiveness of data generated with different temperature parameters"

model: "meta-llama/Llama-3.2-1B"
task: "Copa"
method: "zo"  # zo, fo_full, fo_lora

data:
  # Using data generated with Batch solution
  path: "Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o"

hyperparameters:
  learning_rate: [1e-6, 5e-7]
  batch_size: 16
  steps: 20000
  seed: 0

cuda_devices: "0"
```

#### Step 2: Execute Training

```bash
cd automation/stage2_training

# Preview (without execution)
python trainer.py --config ../configs/stage2/my_training.yaml --dry-run

# Execute training
python trainer.py --config ../configs/stage2/my_training.yaml
```

Results saved in:

```
Results_v2/
└── temperature_study/              # Organized by training experiment purpose
    └── Llama-3.2-1B/
        └── Copa_zo_temp07_topp10_1e-6/
            └── 20241229_120000/
                ├── train.out
                ├── train.err
                └── experiment_config.yaml
```

---

## 🆕 v2.1 New Features

### 1. Unified Path Configuration System

```python
# automation/config.py - Automatically detect project root
from config import PROJECT_ROOT, DATA_V2_DIR, RESULTS_V2_DIR
```

**Advantages**:
- ✅ Support running project from any location
- ✅ Automatic path detection
- ✅ Environment variable override support

### 2. Data Path Finding Tool

```bash
# Quickly find training data paths
python stage1_generation/batch_tools/list_data_paths.py --dataset Copa --format yaml
```

**Purpose**: Quickly find data paths when writing training configurations

### 3. Path Conversion Tool

```bash
# Batch path ↔ Shared path conversion
python stage1_generation/batch_tools/resolve_data_path.py "Data_v2/..."
```

**Purpose**: Understand the relationship between batch and shared paths

### 4. Bug Fixes

✅ **Fixed 5 P0-Level Bugs**:
1. LoRA training script name error
2. LoRA environment variable error
3. Hardcoded absolute paths
4. batch_tools path hierarchy error
5. list_data_paths API error

See details: [BUG_FIXES_SUMMARY.md](BUG_FIXES_SUMMARY.md)

---

## Core Advantages of Batch Solution 3++

### 1. Automatic Parameter Deduplication

```bash
# Experiment 1: batch_20241229_temperature
generation:
  temperature: 0.7
  top_p: 1.0
# → Generated to _shared/Copa/temp07_topp10_gpt4o/

# Experiment 2: batch_20241230_topp (reusing same parameters)
generation:
  temperature: 0.7  # Same
  top_p: 1.0        # Same
# → Detected same fingerprint, directly reuse _shared/Copa/temp07_topp10_gpt4o/
# → Create symbolic link to batch_20241230_topp/Copa/temp07_topp10_gpt4o/
```

### 2. Flexible Experiment Organization

```
Data_v2/synthetic/
├── _shared/                        # Physical storage (shared by all experiments)
│   └── Copa/
│       ├── temp05_topp10_gpt4o/
│       ├── temp07_topp10_gpt4o/
│       └── temp09_topp10_gpt4o/
│
├── batch_20241229_temperature/     # Batch 1: Temperature experiment
│   └── Copa/
│       ├── temp05_topp10_gpt4o -> ../../_shared/Copa/temp05_topp10_gpt4o/
│       ├── temp07_topp10_gpt4o -> ../../_shared/Copa/temp07_topp10_gpt4o/
│       └── temp09_topp10_gpt4o -> ../../_shared/Copa/temp09_topp10_gpt4o/
│
└── batch_20241230_model/           # Batch 2: Model comparison
    └── Copa/
        └── temp07_topp10_gpt4o -> ../../_shared/Copa/temp07_topp10_gpt4o/  (Reused!)
```

### 3. Complete Metadata Tracking

Each physical experiment directory contains:
- `.fingerprint`: Parameter fingerprint (12-char MD5)
- `experiment_metadata.json`: Complete configuration, creation time, original batch, etc.

## Common Scenarios

### Scenario 1: Temperature Parameter Experiment

```bash
# Create 3 config files, only modifying temperature
for temp in 0.5 0.7 0.9; do
  sed "s/temperature:.*/temperature: $temp/" base_config.yaml > copa_temp${temp/.}.yaml
  python generator.py --config copa_temp${temp/.}.yaml
done

# System automatically creates 3 independent physical experiments
# _shared/Copa/temp05_topp10_gpt4o/
# _shared/Copa/temp07_topp10_gpt4o/
# _shared/Copa/temp09_topp10_gpt4o/
```

### Scenario 2: Cross-batch Data Reuse

```bash
# Batch1 already generated temp07 data
# Batch2 needs same parameters → Automatically reuse, no regeneration

python batch_tools/list_batch_experiments.py batch_20241230_model --verbose
# Output:
# ⚡ Data reuse: Yes (original batch: batch_20241229_temperature)
```

## Compatibility with Training Scripts

### ✅ Direct Use of Data_v2 Paths (Recommended)

trainer.py can directly use `Data_v2/` paths, **no publish needed**:

```yaml
# Training configuration
data:
  # Recommended: Use batch path (more intuitive)
  path: "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"

  # Or use shared path (also works)
  # path: "Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa"
```

### Optional: Publish to Data/ (For Legacy Script Compatibility Only)

```bash
python stage1_generation/tools/publish_dataset.py \
    --source Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o \
    --dataset Copa \
    --target Data/rejection_sampling/0_data
```

**Note**: Only for compatibility with legacy training script structure, not needed for new projects.

## Migrating Old Data

To migrate synthetic data and training results from old projects to the new system, see: **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)**

Information needed:
- Dataset name and location
- Generation model and method (recommended)
- Training experiment purpose
- Training configuration and hyperparameters

## FAQ

### Q1: Will the Batch solution waste storage space?

No. Data with the same parameter configuration is stored only once in `_shared/`, different batches reference it through symbolic links.

### Q2: How to check if data for a certain parameter configuration has been generated?

```bash
python batch_tools/list_shared_experiments.py --dataset Copa --verbose
```

### Q3: Can I delete old batches?

Yes. Deleting batch directories won't delete physical data (in `_shared/`). Only manually delete physical data when it's not referenced by any batch.

### Q4: What's the difference between training experiment purpose and data generation purpose?

- **Data generation purpose** (`experiment.purpose`): Why generate this data (e.g., `temperature_study`)
- **Training experiment purpose**: Why train with this data (e.g., `baseline_comparison`)

They are independent, training results are organized by training experiment purpose in `Results_v2/`.

## Dependencies

```bash
# Python dependencies
pip install pyyaml openai tqdm

# Environment variables
export OPENAI_API_KEY="your-api-key"
```

## Contributing

Issues and Pull Requests are welcome!

## License

MIT
