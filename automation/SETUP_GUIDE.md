# Environment Setup and Installation Guide

**Version**: 2.1
**Last Updated**: 2026-01-01

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Environment Configuration](#environment-configuration)
4. [Verify Installation](#verify-installation)
5. [Fixed Issues](#fixed-issues)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware Requirements

- **Storage Space**: At least 50GB available space (for models and data)
- **Memory**: Recommended 16GB+ (required for training large models)
- **GPU**: CUDA-compatible GPU (for model training)

### Software Requirements

```bash
# Python version
Python 3.8+

# Required system packages
git
curl
```

---

## Installation Steps

### Step 1: Clone Project

```bash
# Clone project to any directory
git clone <repository-url> /path/to/Synthetic_Data_for_ZO
cd /path/to/Synthetic_Data_for_ZO
```

### Step 2: Install Python Dependencies

```bash
# Install core dependencies
pip install pyyaml openai tqdm datasets transformers torch

# If using requirements.txt
pip install -r requirements.txt
```

### Step 3: Configure API Keys

```bash
# Set OpenAI API key (required for Stage 1 data generation)
export OPENAI_API_KEY="your-api-key-here"

# Optional: Use custom API service
export OPENAI_API_BASE="https://api.custom.com/v1"
```

**Permanent Configuration** (Recommended):

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

---

## Environment Configuration

### Project Path Configuration (Automatic)

This project uses a unified path configuration system, **no manual setup required**.

#### Method 1: Automatic Detection (Recommended)

The system will automatically detect the project root directory:

```python
# automation/config.py automatically locates project root
PROJECT_ROOT = /path/to/Synthetic_Data_for_ZO  # Auto-detected
```

#### Method 2: Environment Variable Override (Optional)

If automatic detection fails, you can manually specify:

```bash
# Set project root directory
export SYNTHETIC_DATA_PROJECT_ROOT="/your/custom/path/Synthetic_Data_for_ZO"
```

### Verify Path Configuration

```bash
# Verify configuration is correct
python automation/config.py
```

**Expected Output**:
```
================================================================================
🔧 Project Configuration
================================================================================
PROJECT_ROOT:         /path/to/Synthetic_Data_for_ZO
AUTOMATION_DIR:       /path/to/Synthetic_Data_for_ZO/automation
DATA_V2_DIR:          /path/to/Synthetic_Data_for_ZO/Data_v2
RESULTS_V2_DIR:       /path/to/Synthetic_Data_for_ZO/Results_v2
...
================================================================================
✅ All critical paths verified
================================================================================
```

---

## Verify Installation

### Quick Test

```bash
# 1. Verify path configuration
python automation/config.py

# 2. List available tools
python automation/stage1_generation/batch_tools/list_batches.py
python automation/stage2_training/list_results.py

# 3. View available data
python automation/stage1_generation/batch_tools/list_data_paths.py
```

### Complete Functionality Test

#### Test Stage 1 (Data Generation)

```bash
# Generate scripts using example config (without actually calling API)
python automation/stage1_generation/generator.py \
    automation/configs/examples/stage1_full_example_copa.yaml
```

#### Test Stage 2 (Model Training)

```bash
# Preview training process (without actual training)
python automation/stage2_training/trainer.py \
    automation/configs/examples/stage2_example_training.yaml \
    --dry-run
```

---

## Fixed Issues

This version (v2.1) has fixed the following important issues:

### ✅ Bug #1: LoRA Training Script Name Error

**Issue**: trainer.py cannot find LoRA training script
**Symptom**: `FileNotFoundError: lora_finetune_original.sh`
**Fix**: Corrected to the correct script name
**Impact**: LoRA training now works properly

### ✅ Bug #2: LoRA Environment Variable Error

**Issue**: `lora_rank` parameter is ignored, always uses default value
**Symptom**: `lora_rank` in config file doesn't take effect
**Fix**: Use correct environment variable name `RANK`
**Impact**: LoRA rank parameter can now be passed correctly

### ✅ Bug #3: Hardcoded Absolute Paths

**Issue**: 12 files contain hardcoded absolute paths
**Symptom**: Cannot run on other machines
**Fix**: Created unified `config.py` configuration system
**Impact**: Project can now run from any location

### ✅ Bug #4: batch_tools Path Hierarchy Error

**Issue**: Tools under batch_tools/ fail to import config
**Symptom**: `ModuleNotFoundError: No module named 'config'`
**Fix**: Corrected path hierarchy
**Impact**: All batch management tools work properly

### ✅ Bug #5: list_data_paths API Error

**Issue**: Tool uses wrong API call
**Symptom**: `AttributeError: 'list' object has no attribute 'items'`
**Fix**: Corrected API usage
**Impact**: Data path finding tool works properly

See detailed bug fix descriptions: [BUG_FIXES_SUMMARY.md](BUG_FIXES_SUMMARY.md)

---

## New Tools

### 1. Data Path Finding Tool

**Purpose**: Quickly find training data paths

```bash
# List all available data
python automation/stage1_generation/batch_tools/list_data_paths.py

# View only specific dataset
python automation/stage1_generation/batch_tools/list_data_paths.py --dataset Copa

# Output YAML format (can be copied to config file)
python automation/stage1_generation/batch_tools/list_data_paths.py --format yaml
```

### 2. Path Resolution Tool

**Purpose**: Convert between batch paths and shared paths

```bash
# Batch path → Shared path
python automation/stage1_generation/batch_tools/resolve_data_path.py \
    "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"

# Shared path → All referenced batches
python automation/stage1_generation/batch_tools/resolve_data_path.py \
    "Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa"
```

### 3. Training Results Management Tool

**Purpose**: View all training results

```bash
# View results summary
python automation/stage2_training/list_results.py

# View detailed results
python automation/stage2_training/list_results.py --detail

# Filter by specific experiment purpose
python automation/stage2_training/list_results.py --purpose hyperparameter_tuning
```

---

## Troubleshooting

### Issue 1: "Cannot find config module"

**Error Message**:
```
ModuleNotFoundError: No module named 'config'
```

**Solution**:
```bash
# Make sure to run from project root directory
cd /path/to/Synthetic_Data_for_ZO
python automation/config.py

# If still failing, set environment variable
export SYNTHETIC_DATA_PROJECT_ROOT="/path/to/Synthetic_Data_for_ZO"
```

### Issue 2: "LoRA training script not found"

**Error Message**:
```
FileNotFoundError: lora_finetune_original.sh
```

**Cause**: Using old version of code

**Solution**:
```bash
# Verify trainer.py is updated
grep "fo_lora_finetune" automation/stage2_training/trainer.py

# Should see:
# 'fo_lora': ('fo_lora_finetune_original.sh', 'fo_lora_finetune_synthetic.sh')
```

### Issue 3: "lora_rank parameter not taking effect"

**Symptom**: Always uses rank=16 regardless of config file settings

**Cause**: Using old version of trainer.py

**Solution**:
```bash
# Verify environment variable name is corrected
grep 'env_vars\["RANK"\]' automation/stage2_training/trainer.py

# Should see:
# env_vars["RANK"] = str(self.config['hyperparameters'].get('lora_rank', 8))
```

### Issue 4: "Path does not exist"

**Error Message**:
```
FileNotFoundError: [Errno 2] No such file or directory: '/home/ubuntu/...'
```

**Cause**: Path configuration issue

**Solution**:
```bash
# 1. Verify path configuration
python automation/config.py

# 2. If path is wrong, set environment variable
export SYNTHETIC_DATA_PROJECT_ROOT="$(pwd)"
python automation/config.py
```

### Issue 5: "API key not set"

**Error Message**:
```
Error: API key not found
```

**Solution**:
```bash
# Set API key
export OPENAI_API_KEY="your-key-here"

# Or set in config file (not recommended, may leak)
# generation:
#   api_key: "your-key-here"
```

---

## Next Steps

After installation, recommended reading order:

1. **[USER_GUIDE.md](USER_GUIDE.md)** - User Manual
2. **[BATCH_GUIDE.md](BATCH_GUIDE.md)** - Batch Data Management System
3. **[COMPLETE_PIPELINE_SIMULATION.md](COMPLETE_PIPELINE_SIMULATION.md)** - Complete Pipeline Example
4. **[TOOLS_REFERENCE.md](TOOLS_REFERENCE.md)** - All Tools Reference Manual

---

## Getting Help

- **Bug Reports**: See [BUG_FIXES_SUMMARY.md](BUG_FIXES_SUMMARY.md)
- **Complete Documentation**: See [README.md](../README.md)
- **Tool Reference**: See [TOOLS_REFERENCE.md](TOOLS_REFERENCE.md)

---

**Environment setup complete! Start your first experiment!** 🚀
