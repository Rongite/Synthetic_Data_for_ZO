# Synthetic Data Migration Guide

## Overview

This guide explains how to migrate synthetic data from the old project (Backup) to the new project, ensuring compatibility with the training scripts and MeZO data loader.

## Key Requirements

The migrated data must satisfy three critical requirements:

1. **Directory Structure**: Match the path pattern expected by training scripts
2. **File Naming**: Follow exact naming conventions required by `tasks.py`
3. **Location**: Data must be placed in the correct directory hierarchy

---

## Current Situation

### Old Project Data Location
```
/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/
├── Copa/
│   ├── copa_train.jsonl
│   ├── copa_validation.jsonl
│   ├── copa_test.jsonl
│   └── mezo_gpt/
│       ├── version_1/
│       ├── version_2/
│       └── ...
├── BOOLQ/
│   ├── boolq_train.jsonl
│   └── boolq_validation.jsonl
├── CB/
│   ├── cb_train.jsonl
│   ├── cb_validation.jsonl
│   └── cb_test.jsonl
├── RTE/
│   ├── rte_train.jsonl
│   ├── rte_validation.jsonl
│   └── rte_test.jsonl
└── ArcC_Cloze/
    ├── ARC-Challenge_train.jsonl
    ├── ARC-Challenge_validation.jsonl
    └── ARC-Challenge_test.jsonl
```

### New Project Current Structure
```
/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/
└── original/
    ├── Copa/
    ├── BOOLQ/
    ├── CB/
    ├── RTE/
    ├── ArcC_Cloze/
    └── ArcC_MC/
```

**Missing**: `Data/rejection_sampling/0_data/` directory (needs to be created)

---

## Required File Naming Conventions

Based on `tasks.py` (lines 200-962), each dataset requires specific file names:

| Dataset | Task Name in tasks.py | Required Files |
|---------|----------------------|----------------|
| **Copa** | `copa` | `copa_train.jsonl`<br>`copa_validation.jsonl` |
| **BOOLQ** | `boolq` | `boolq_train.jsonl`<br>`boolq_validation.jsonl` |
| **CB** | `cb` | `cb_train.jsonl`<br>`cb_validation.jsonl` |
| **RTE** | `rte` | `rte_train.jsonl`<br>`rte_validation.jsonl` |
| **ArcC_Cloze** | `arcc_cloze` | `ARC-Challenge_train.jsonl`<br>`ARC-Challenge_validation.jsonl` |
| **ArcC_MC** | `arcc_mc` | `ARC-Challenge_train.jsonl`<br>`ARC-Challenge_validation.jsonl` |

### How tasks.py Loads Data

From `tasks.py` analysis:

```python
# Example: CopaDataset (lines 272-299)
def load_dataset(self, path=None):
    if path and os.path.exists(path):
        print(f"Loading COPA dataset from local JSONL files at: {path}")
        train_file = os.path.join(path, "copa_train.jsonl")  # ← Expects this exact name
        valid_file = os.path.join(path, "copa_validation.jsonl")  # ← Expects this exact name
        # Load files...
```

**Key Insight**: The TASK variable in training scripts points to a directory, and `tasks.py` looks for specific filenames inside that directory.

---

## Training Script Path Expectations

From `running_scripts/Llama-3.2-1B/1_0_mezo_syn_copa.sh` (line 8):

```bash
TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/Copa
```

Pattern for all synthetic data training scripts:
```bash
TASK={PROJECT_ROOT}/Data/rejection_sampling/0_data/{DATASET_NAME}
```

---

## Migration Instructions

### Step 1: Create Required Directory Structure

```bash
# Navigate to new project Data directory
cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data

# Create the rejection_sampling directory structure
mkdir -p rejection_sampling/0_data/{Copa,BOOLQ,CB,RTE,ArcC_Cloze,ArcC_MC}
```

### Step 2: Copy Synthetic Data Files

Choose ONE of the following methods:

#### Option A: Copy Files (Creates Independent Copy)

```bash
# Set source and destination base paths
OLD_DATA="/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data"
NEW_DATA="/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data"

# Copy Copa dataset
cp ${OLD_DATA}/Copa/copa_train.jsonl ${NEW_DATA}/Copa/
cp ${OLD_DATA}/Copa/copa_validation.jsonl ${NEW_DATA}/Copa/
cp ${OLD_DATA}/Copa/copa_test.jsonl ${NEW_DATA}/Copa/

# Copy BOOLQ dataset
cp ${OLD_DATA}/BOOLQ/boolq_train.jsonl ${NEW_DATA}/BOOLQ/
cp ${OLD_DATA}/BOOLQ/boolq_validation.jsonl ${NEW_DATA}/BOOLQ/

# Copy CB dataset
cp ${OLD_DATA}/CB/cb_train.jsonl ${NEW_DATA}/CB/
cp ${OLD_DATA}/CB/cb_validation.jsonl ${NEW_DATA}/CB/
cp ${OLD_DATA}/CB/cb_test.jsonl ${NEW_DATA}/CB/

# Copy RTE dataset
cp ${OLD_DATA}/RTE/rte_train.jsonl ${NEW_DATA}/RTE/
cp ${OLD_DATA}/RTE/rte_validation.jsonl ${NEW_DATA}/RTE/
cp ${OLD_DATA}/RTE/rte_test.jsonl ${NEW_DATA}/RTE/

# Copy ArcC_Cloze dataset
cp ${OLD_DATA}/ArcC_Cloze/ARC-Challenge_train.jsonl ${NEW_DATA}/ArcC_Cloze/
cp ${OLD_DATA}/ArcC_Cloze/ARC-Challenge_validation.jsonl ${NEW_DATA}/ArcC_Cloze/
cp ${OLD_DATA}/ArcC_Cloze/ARC-Challenge_test.jsonl ${NEW_DATA}/ArcC_Cloze/

# Copy ArcC_MC dataset (if exists in old project)
# Note: Check if ArcC_MC exists in old project first
if [ -d "${OLD_DATA}/ArcC_MC" ]; then
    cp ${OLD_DATA}/ArcC_MC/ARC-Challenge_train.jsonl ${NEW_DATA}/ArcC_MC/
    cp ${OLD_DATA}/ArcC_MC/ARC-Challenge_validation.jsonl ${NEW_DATA}/ArcC_MC/
    cp ${OLD_DATA}/ArcC_MC/ARC-Challenge_test.jsonl ${NEW_DATA}/ArcC_MC/
fi
```

#### Option B: Create Symbolic Links (Saves Disk Space)

```bash
# Set source and destination base paths
OLD_DATA="/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data"
NEW_DATA="/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data"

# Link Copa dataset
ln -s ${OLD_DATA}/Copa/copa_train.jsonl ${NEW_DATA}/Copa/
ln -s ${OLD_DATA}/Copa/copa_validation.jsonl ${NEW_DATA}/Copa/
ln -s ${OLD_DATA}/Copa/copa_test.jsonl ${NEW_DATA}/Copa/

# Link BOOLQ dataset
ln -s ${OLD_DATA}/BOOLQ/boolq_train.jsonl ${NEW_DATA}/BOOLQ/
ln -s ${OLD_DATA}/BOOLQ/boolq_validation.jsonl ${NEW_DATA}/BOOLQ/

# Link CB dataset
ln -s ${OLD_DATA}/CB/cb_train.jsonl ${NEW_DATA}/CB/
ln -s ${OLD_DATA}/CB/cb_validation.jsonl ${NEW_DATA}/CB/
ln -s ${OLD_DATA}/CB/cb_test.jsonl ${NEW_DATA}/CB/

# Link RTE dataset
ln -s ${OLD_DATA}/RTE/rte_train.jsonl ${NEW_DATA}/RTE/
ln -s ${OLD_DATA}/RTE/rte_validation.jsonl ${NEW_DATA}/RTE/
ln -s ${OLD_DATA}/RTE/rte_test.jsonl ${NEW_DATA}/RTE/

# Link ArcC_Cloze dataset
ln -s ${OLD_DATA}/ArcC_Cloze/ARC-Challenge_train.jsonl ${NEW_DATA}/ArcC_Cloze/
ln -s ${OLD_DATA}/ArcC_Cloze/ARC-Challenge_validation.jsonl ${NEW_DATA}/ArcC_Cloze/
ln -s ${OLD_DATA}/ArcC_Cloze/ARC-Challenge_test.jsonl ${NEW_DATA}/ArcC_Cloze/

# Link ArcC_MC dataset (if exists)
if [ -d "${OLD_DATA}/ArcC_MC" ]; then
    ln -s ${OLD_DATA}/ArcC_MC/ARC-Challenge_train.jsonl ${NEW_DATA}/ArcC_MC/
    ln -s ${OLD_DATA}/ArcC_MC/ARC-Challenge_validation.jsonl ${NEW_DATA}/ArcC_MC/
    ln -s ${OLD_DATA}/ArcC_MC/ARC-Challenge_test.jsonl ${NEW_DATA}/ArcC_MC/
fi
```

### Step 3: Verify Migration

After copying or linking, verify the structure:

```bash
cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data

# Check directory structure
tree -L 2

# Expected output:
# .
# ├── ArcC_Cloze
# │   ├── ARC-Challenge_test.jsonl
# │   ├── ARC-Challenge_train.jsonl
# │   └── ARC-Challenge_validation.jsonl
# ├── ArcC_MC
# │   ├── ARC-Challenge_test.jsonl
# │   ├── ARC-Challenge_train.jsonl
# │   └── ARC-Challenge_validation.jsonl
# ├── BOOLQ
# │   ├── boolq_train.jsonl
# │   └── boolq_validation.jsonl
# ├── CB
# │   ├── cb_test.jsonl
# │   ├── cb_train.jsonl
# │   └── cb_validation.jsonl
# ├── Copa
# │   ├── copa_test.jsonl
# │   ├── copa_train.jsonl
# │   └── copa_validation.jsonl
# └── RTE
#     ├── rte_test.jsonl
#     ├── rte_train.jsonl
#     └── rte_validation.jsonl

# Verify files exist and are readable
for dataset in Copa BOOLQ CB RTE ArcC_Cloze; do
    echo "Checking ${dataset}..."
    ls -lh ${dataset}/
done
```

### Step 4: Test with Training Scripts

Test that the migrated data works with existing training scripts:

```bash
# Navigate to MeZO directory
cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models

# Test Copa dataset loading (dry run - check if files are found)
TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/Copa

# The training script should print:
# "Loading COPA dataset from local JSONL files at: {TASK}"
# "Loaded X samples from {TASK}/copa_train.jsonl"
# "Loaded Y samples from {TASK}/copa_validation.jsonl"
```

---

## Complete Directory Structure After Migration

```
/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/
├── original/
│   ├── Copa/
│   │   ├── copa_train.jsonl
│   │   ├── copa_validation.jsonl
│   │   └── copa_test.jsonl
│   ├── BOOLQ/
│   ├── CB/
│   ├── RTE/
│   ├── ArcC_Cloze/
│   └── ArcC_MC/
└── rejection_sampling/
    └── 0_data/
        ├── Copa/
        │   ├── copa_train.jsonl          ← Required
        │   ├── copa_validation.jsonl     ← Required
        │   └── copa_test.jsonl           ← Optional
        ├── BOOLQ/
        │   ├── boolq_train.jsonl         ← Required
        │   └── boolq_validation.jsonl    ← Required
        ├── CB/
        │   ├── cb_train.jsonl            ← Required
        │   ├── cb_validation.jsonl       ← Required
        │   └── cb_test.jsonl             ← Optional
        ├── RTE/
        │   ├── rte_train.jsonl           ← Required
        │   ├── rte_validation.jsonl      ← Required
        │   └── rte_test.jsonl            ← Optional
        ├── ArcC_Cloze/
        │   ├── ARC-Challenge_train.jsonl      ← Required
        │   ├── ARC-Challenge_validation.jsonl ← Required
        │   └── ARC-Challenge_test.jsonl       ← Optional
        └── ArcC_MC/
            ├── ARC-Challenge_train.jsonl      ← Required
            ├── ARC-Challenge_validation.jsonl ← Required
            └── ARC-Challenge_test.jsonl       ← Optional
```

---

## Handling Multiple Versions of Synthetic Data

If the old project has multiple versions (e.g., `version_1`, `version_2`, etc.) under subdirectories like `Copa/mezo_gpt/version_X/`, you need to decide which version to use:

### Option 1: Use Latest Version

```bash
# Example: Copa has multiple versions, use the latest one
OLD_VERSION="${OLD_DATA}/Copa/mezo_gpt/version_13-2"
cp ${OLD_VERSION}/copa_train.jsonl ${NEW_DATA}/Copa/

# Keep validation from the base directory (usually shared)
cp ${OLD_DATA}/Copa/copa_validation.jsonl ${NEW_DATA}/Copa/
```

### Option 2: Test Multiple Versions

Create separate directories for each version:

```bash
# Create version-specific directories
mkdir -p ${NEW_DATA}/Copa/version_1
mkdir -p ${NEW_DATA}/Copa/version_2

# Copy each version
cp ${OLD_DATA}/Copa/mezo_gpt/version_1/copa_train.jsonl ${NEW_DATA}/Copa/version_1/
cp ${OLD_DATA}/Copa/copa_validation.jsonl ${NEW_DATA}/Copa/version_1/

# When training, specify the version in TASK path:
# TASK=${NEW_DATA}/Copa/version_1
```

---

## Compatibility Matrix

| Component | Path Pattern | File Naming | Status |
|-----------|-------------|-------------|--------|
| **Training Scripts** | `Data/rejection_sampling/0_data/{Dataset}/` | Dataset-specific | ✓ Required |
| **tasks.py** | Any valid path | Dataset-specific | ✓ Required |
| **Automation Pipeline** | Configurable via YAML | Follows tasks.py | ✓ Compatible |

---

## Troubleshooting

### Issue 1: "Train or validation JSONL file is missing"

**Cause**: File names don't match expected patterns in `tasks.py`

**Solution**: Check file naming exactly matches the table in "Required File Naming Conventions" section

```bash
# Verify file names (case-sensitive!)
ls -l ${NEW_DATA}/Copa/
# Should see: copa_train.jsonl, copa_validation.jsonl (lowercase!)
```

### Issue 2: "Unable to load JSONL file"

**Cause**: File permissions or symbolic link broken

**Solution**:
```bash
# Check file permissions
chmod 644 ${NEW_DATA}/Copa/*.jsonl

# If using symlinks, verify they're valid
file ${NEW_DATA}/Copa/copa_train.jsonl
# Should show: symbolic link to ...
```

### Issue 3: "Dataset is empty after loading"

**Cause**: JSONL format issues or encoding problems

**Solution**:
```bash
# Verify JSONL format
head -n 5 ${NEW_DATA}/Copa/copa_train.jsonl | python3 -m json.tool

# Check encoding
file ${NEW_DATA}/Copa/copa_train.jsonl
# Should show: UTF-8 Unicode text
```

---

## Migration Checklist

- [ ] Create `Data/rejection_sampling/0_data/` directory structure
- [ ] Create subdirectories for each dataset: Copa, BOOLQ, CB, RTE, ArcC_Cloze, ArcC_MC
- [ ] Copy or link `*_train.jsonl` files for all datasets
- [ ] Copy or link `*_validation.jsonl` files for all datasets
- [ ] Copy or link `*_test.jsonl` files (optional)
- [ ] Verify file naming matches `tasks.py` requirements exactly
- [ ] Test file permissions (should be readable: 644)
- [ ] Run `tree -L 2` to verify structure
- [ ] Test with one training script to confirm data loads correctly
- [ ] Document which version of synthetic data was migrated (if applicable)

---

## References

- **Training Scripts**: `/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/running_scripts/`
- **Data Loader**: `/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/PromptZO/MeZO/large_models/tasks.py`
- **Old Data Location**: `/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/`
- **New Data Target**: `/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/`

---

## Summary

To successfully migrate old synthetic data to the new project:

1. **Create the directory**: `Data/rejection_sampling/0_data/{Dataset}/`
2. **Use exact file names**: Follow the table in "Required File Naming Conventions"
3. **Choose migration method**: Copy files OR create symbolic links
4. **Verify structure**: Use `tree` command to check
5. **Test loading**: Run a training script to confirm data is accessible

The migration is complete when training scripts can successfully load data from:
```
TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/{Dataset}
```
