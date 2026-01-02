# Batch Management Tools

These tools are used to manage and view experiment data under Batch Scheme 3++.

---

## 📁 Directory Structure Overview

```
Data_v2/synthetic/
├── _shared/                              # Physical data storage
│   ├── Copa/
│   │   ├── temp05_topp10_gpt4o/         # Experiment data (physical storage)
│   │   ├── temp07_topp09_gpt4o/
│   │   └── temp09_topp10_gpt4o/
│   └── CB/
│       └── temp07_topp10_gpt4o/
│
├── batch_20241229_temperature/           # Batch 1: Temperature experiment
│   ├── Copa/
│   │   ├── temp05_topp10_gpt4o -> ../../_shared/Copa/temp05_topp10_gpt4o/
│   │   ├── temp07_topp10_gpt4o -> ../../_shared/Copa/temp07_topp10_gpt4o/
│   │   └── temp09_topp10_gpt4o -> ../../_shared/Copa/temp09_topp10_gpt4o/
│   └── CB/
│       └── temp07_topp10_gpt4o -> ../../_shared/CB/temp07_topp10_gpt4o/
│
└── batch_20241230_topp/                  # Batch 2: top_p experiment
    └── Copa/
        ├── temp07_topp08_gpt4o -> ../../_shared/Copa/temp07_topp08_gpt4o/
        └── temp07_topp09_gpt4o -> ../../_shared/Copa/temp07_topp09_gpt4o/  (Reused!)
```

---

## 🔧 Tool List

### 1. list_batches.py
**Purpose**: List all batch groups

### 5. list_data_paths.py ⭐ New
**Purpose**: List all available training data paths

```bash
# List all data
python list_data_paths.py

# View specific batch only
python list_data_paths.py --batch batch_20241229_temperature

# View specific dataset only
python list_data_paths.py --dataset Copa

# Output YAML format (can be copied to config file)
python list_data_paths.py --format yaml
```

**Output Example**:
```
📁 Copa / temp07_topp10_gpt4o
   ✅ Batch path (recommended):
      Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa

   📦 Shared path (physical storage):
      Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa
```

**Use Cases**:
- Find data paths when writing training configurations
- Quickly browse all available data
- Generate configuration file templates

---

### 6. resolve_data_path.py ⭐ New
**Purpose**: Convert between batch paths and shared paths

```bash
# Batch path → Shared path
python resolve_data_path.py "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"

# Shared path → All referencing batches
python resolve_data_path.py "Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa"
```

**Use Cases**:
- Find physical data location
- Understand data reuse situation
- Verify symbolic links are correct

---

### 1. list_batches.py
**Purpose**: List all batch groups

```bash
# Basic usage
python list_batches.py

# Show detailed information (including experiment count in each batch)
python list_batches.py --verbose
```

**Output Example**:
```
================================================================================
Found 2 batches
================================================================================

📦 batch_20241229_temperature
   Experiments: 4
   Copa: 3 experiments
      - temp05_topp10_gpt4o
      - temp07_topp10_gpt4o
      - temp09_topp10_gpt4o
   CB: 1 experiment
      - temp07_topp10_gpt4o

📦 batch_20241230_topp
   Experiments: 2
   Copa: 2 experiments
      - temp07_topp08_gpt4o
      - temp07_topp09_gpt4o
```

---

### 2. list_batch_experiments.py
**Purpose**: List all experiments in specified batch

```bash
# List all experiments in batch
python list_batch_experiments.py batch_20241229_temperature

# List only Copa dataset experiments
python list_batch_experiments.py batch_20241229_temperature --dataset Copa

# Show detailed information (including metadata, physical paths, etc.)
python list_batch_experiments.py batch_20241229_temperature --verbose
```

**Output Example**:
```
================================================================================
Batch: batch_20241229_temperature
Experiments: 4
================================================================================

📊 Copa (3 experiments)
--------------------------------------------------------------------------------

  🔧 temp05_topp10_gpt4o
     Batch view: batch_20241229_temperature/Copa/temp05_topp10_gpt4o
     Physical storage: _shared/Copa/temp05_topp10_gpt4o
     Created: 2024-12-29 10:30:00
     Parameter fingerprint: f8e3d2a1b9c0
     ⚡ Data reuse: No (newly generated)
     Generation model: gpt-4o
     Temperature: 0.5
     Top_p: 1.0

  🔧 temp07_topp09_gpt4o
     Batch view: batch_20241229_temperature/Copa/temp07_topp09_gpt4o
     ⚡ Data reuse: Yes (original batch: batch_20241228_pilot)
```

---

### 3. list_shared_experiments.py
**Purpose**: List all physical experiment data in _shared/

```bash
# List all physical experiments
python list_shared_experiments.py

# List only Copa dataset experiments
python list_shared_experiments.py --dataset Copa

# Show detailed information (including metadata, parameter configurations, etc.)
python list_shared_experiments.py --verbose
```

**Output Example**:
```
================================================================================
Physical experiment data in _shared/
Experiments: 5
================================================================================

📊 Copa (4 experiments)
--------------------------------------------------------------------------------

  📦 temp05_topp10_gpt4o
     Physical path: _shared/Copa/temp05_topp10_gpt4o
     Parameter fingerprint: f8e3d2a1b9c0...
     Created: 2024-12-29 10:30:00
     Original batch: batch_20241229_temperature

     Generation config:
       Model: gpt-4o
       Temperature: 0.5
       Top_p: 1.0
       Max tokens: 256

     Validation config:
       Model: gpt-4o
       Temperature: 0.0

     Experiment purpose: temperature_study
     Experiment description: Study the impact of temperature parameter on synthetic data quality
```

**Use Cases**:
- See which parameter configurations have already generated data
- Avoid duplicate generation of data with same parameters
- Understand physical storage usage

---

### 4. compare_experiments.py
**Purpose**: Compare parameter configurations of different experiments

```bash
# Compare two specific experiments in _shared/
python compare_experiments.py \
    --shared Copa/temp07_topp10_gpt4o \
    --shared Copa/temp09_topp10_gpt4o

# Compare experiments in two batches (simplified version)
python compare_experiments.py \
    --batch1 batch_20241229_temperature \
    --dataset1 Copa \
    --batch2 batch_20241230_topp \
    --dataset2 Copa
```

**Output Example**:
```
================================================================================
Experiment Parameter Comparison
================================================================================

Experiment 1: Copa/temp07_topp10_gpt4o
Experiment 2: Copa/temp09_topp10_gpt4o

✅ Same parameters:
  generation.model: gpt-4o
  generation.top_p: 1.0
  generation.max_tokens: 256
  validation.model: gpt-4o
  validation.temperature: 0.0

⚠️  Different parameters:
  generation.temperature:
    Experiment 1: 0.7
    Experiment 2: 0.9

🔧 Note: Parameters differ, need to generate data separately
  Experiment 1 fingerprint: a1b2c3d4e5f6
  Experiment 2 fingerprint: f6e5d4c3b2a1
```

**Use Cases**:
- Verify experiment configuration is correct
- Understand parameter differences between experiments
- Determine if two experiments should share data (same fingerprint)

---

## 🎯 Common Usage Scenarios

### Scenario 1: View current experiment batches

```bash
python list_batches.py --verbose
```

### Scenario 2: View experiment details in specific batch

```bash
# View all Copa dataset experiments in batch
python list_batch_experiments.py batch_20241229_temperature --dataset Copa --verbose
```

### Scenario 3: Check if a parameter configuration has already been generated

```bash
# View all Copa dataset experiments in _shared/
python list_shared_experiments.py --dataset Copa --verbose

# Search for temp07 experiments
python list_shared_experiments.py --dataset Copa | grep temp07
```

### Scenario 4: Verify data reuse is effective

```bash
# View experiments in batch, check "data reuse" status
python list_batch_experiments.py batch_20241230_topp --verbose

# Should see output similar to:
# ⚡ Data reuse: Yes (original batch: batch_20241229_temperature)
```

### Scenario 5: Compare parameter differences between two experiments

```bash
# Compare two temperature experiments
python compare_experiments.py \
    --shared Copa/temp07_topp10_gpt4o \
    --shared Copa/temp09_topp10_gpt4o
```

---

## 📝 Tool Output Explanation

### Symbol Meanings

- 📦 Batch group
- 📊 Dataset
- 🔧 Experiment
- ⚡ Data reuse status
- ✅ Same/Success
- ⚠️ Different/Warning
- 💾 Tip/Info

### Data Reuse Indicators

- **Data reuse: No (newly generated)** - This is the first generation of this parameter configuration, physical data is in _shared/
- **Data reuse: Yes (original batch: batch_XXX)** - This experiment reuses data generated by a previous batch

### Parameter Fingerprint

- Parameter fingerprint is an MD5 hash calculated based on all key parameters affecting data generation (model, temperature, top_p, prompts, etc.)
- Same fingerprint = Same parameter configuration = Can reuse physical data
- Different fingerprint = Different parameter configuration = Need to generate data separately

---

## 🔍 Troubleshooting

### Issue 1: Cannot find batch

```bash
# Check if batch exists
python list_batches.py

# If expected batch is not visible, check directory structure
ls -la Data_v2/synthetic/
```

### Issue 2: Experiment shows as empty batch

```bash
# Check batch directory contents
ls -la Data_v2/synthetic/batch_20241229_temperature/

# Check if symbolic links are correct
ls -laR Data_v2/synthetic/batch_20241229_temperature/
```

### Issue 3: Metadata read failure

```bash
# Check if metadata file exists
ls -la Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/experiment_metadata.json

# Manually view metadata
cat Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/experiment_metadata.json | jq .
```

---

## 💡 Best Practices

1. **Regularly check batch list**: Use `list_batches.py --verbose` to understand experiment organization
2. **Check before generation**: Use `list_shared_experiments.py` to check if there's already an experiment with same parameters
3. **Verify reuse**: After generation, use `list_batch_experiments.py --verbose` to confirm data reuse is effective
4. **Compare parameters**: Use `compare_experiments.py` to verify experiment configuration differences

---

## Developer Information

- **Created**: 2024-12-29
- **Version**: 1.0
- **Maintained by**: Synthetic Data Generation Team
