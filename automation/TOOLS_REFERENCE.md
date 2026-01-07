# Tools Reference Manual

**Version**: 2.1
**Last Updated**: 2026-01-01

This document provides complete reference for all automation tools.

---

## 📋 Table of Contents

1. [Stage 1 Tools (Data Generation)](#stage-1-tools)
2. [Stage 2 Tools (Model Training)](#stage-2-tools)
3. [Batch Management Tools](#batch-management-tools)
4. [Data Path Tools (New)](#data-path-tools)
5. [Configuration and Diagnostic Tools](#configuration-and-diagnostic-tools)

---

## Stage 1 Tools

### generator.py

**Purpose**: Generate data synthesis scripts

**Location**: `automation/stage1_generation/generator.py`

**Usage**:
```bash
python automation/stage1_generation/generator.py <config.yaml>
```

**Parameters**:
- `config.yaml`: Stage 1 configuration file path (required)

**Output**:
- Creates generation scripts in `Data_v2/synthetic/_shared/{Dataset}/{exp_name}/scripts/`
- If batch_id specified, creates symbolic links in `Data_v2/synthetic/{batch_id}/`

**Example**:
```bash
python automation/stage1_generation/generator.py \
    automation/configs/examples/stage1_full_example_copa.yaml
```

**Features**:
- ✅ Automatic parameter deduplication (Batch Solution 3++)
- ✅ Automatically detect and reuse existing data
- ✅ Multi-dataset support
- ✅ Fully configuration-driven

---

### review_top20.py

**Purpose**: Interactive review first 20 samples

**location**: `automation/stage1_generation/tools/review_top20.py`

**Usage**:
```bash
# copy to experimentdirectory
cp automation/stage1_generation/tools/review_top20.py <exp_dir>/scripts/

# run（ in generate first 20 samples back ）
cd <exp_dir>/scripts/
python review_top20.py
```

**Interactive workflow**:
```
sample 1/20:
Original: The man broke his toe because...
Rephrased: The individual fractured his toe due to...

Is this a good rephrase? (y/n): y

sample 2/20:
...
```

**output**:
- `../copa_train_top20_annotated.jsonl` - Annotate first 20 samples
- High-quality samplesAutomatically extract forfew-shot

---

### annotate_samples.py

**Purpose**: Annotatesample21-40YesNoandOriginalsame

**location**: `automation/stage1_generation/tools/annotate_samples.py`

**Usage**:
```bash
cd <exp_dir>/scripts/
python annotate_samples.py
```

**Interaction**:
```
Please annotate whether samples are same as original (yes/no):
sample 21: same/not the same? same
sample 22: same/not the same? not the same
...
```

**output**:
- `../copa_train_samples21to40_annotated.jsonl`

---

### extract_samples.py

**Purpose**:  Extract high-quality samples from annotated data

**location**: `automation/stage1_generation/tools/extract_samples.py`

**Usage**:
```bash
python automation/stage1_generation/tools/extract_samples.py \
    <annotated.jsonl> \
    --output <output.jsonl> \
    --num-samples 5
```

---

### publish_dataset.py

**Purpose**: Release data to training directory (optional)

**location**: `automation/stage1_generation/tools/publish_dataset.py`

**Usage**:
```bash
python automation/stage1_generation/tools/publish_dataset.py \
    --source Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa \
    --dataset Copa \
    --target Data/rejection_sampling/0_data
```

**note**:
- ⚠️ Training scripts can directly use `Data_v2/` path, this tool is optional
- Only forCompatible with oldtrainingscriptstructure

---

## stage2tool

### trainer.py

**Purpose**: Automated model training

**location**: `automation/stage2_training/trainer.py`

**Usage**:
```bash
# executetraining
python automation/stage2_training/trainer.py <config.yaml>

# Preview (no actual training)
python automation/stage2_training/trainer.py <config.yaml> --dry-run
```

**supporttrainingmethod**:
- `zo` - MeZO (Zero-order optimization)
- `fo_full` - Full parameter fine-tuning
- `fo_lora` - LoRAFine-tuning

**configureexample**:
```yaml
experiment:
  purpose: "hyperparameter_tuning"
  description: "Test different learning rates"

model: "meta-llama/Llama-3.2-1B"
task: "Copa"
method: "fo_lora"

data:
  path: "Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa"

hyperparameters:
  learning_rate: [1e-6, 5e-7]  # Grid search
  batch_size: 16
  steps: 20000
  seed: 0
  lora_rank: 8  # LoRASpecific use 

cuda_devices: "0"
```

**output**:
```
Results_v2/
└── hyperparameter_tuning/  # Organized by experiment purpose
    └── Llama-3.2-1B/
        └── Copa_fo_lora_temp07_1e-6/
            └── 20260101_120000/
                ├── train.out
                ├── train.err
                └── experiment_config.yaml
```

---

### list_results.py

**Purpose**: View training results summary 

**location**: `automation/stage2_training/list_results.py`

**Usage**:
```bash
# View summary 
python automation/stage2_training/list_results.py

# viewDetailedresult
python automation/stage2_training/list_results.py --detail

# Filter specific experiment purpose
python automation/stage2_training/list_results.py --purpose hyperparameter_tuning
```

**outputexample**:
```
================================================================================
Training results summary  - Results_v2
================================================================================

📁 experimentpurpose: hyperparameter_tuning
   experimentcount: 5
   └─ meta-llama: 5 experiment

📁 experimentpurpose: baseline_comparison
   experimentcount: 3
   └─ meta-llama: 3 experiment

total: 2 experimentpurpose, 8 trainingexperiment
```

---

## Batchmanagetool

### list_batches.py

**Purpose**: Listallbatch

**location**: `automation/stage1_generation/batch_tools/list_batches.py`

**Usage**:
```bash
# Listallbatch
python automation/stage1_generation/batch_tools/list_batches.py

# ShowDetailedinformation
python automation/stage1_generation/batch_tools/list_batches.py --verbose
```

**outputexample**:
```
================================================================================
Found 2 batch
================================================================================

📦 batch_20241229_temperature
   Number of experiments: 3
   Copa: 3 experiment

📦 batch_20241230_topp
   Number of experiments: 2
   Copa: 2 experiment
```

---

### list_batch_experiments.py

**Purpose**: viewbatch in experiment

**location**: `automation/stage1_generation/batch_tools/list_batch_experiments.py`

**Usage**:
```bash
# Listbatch in experiment
python automation/stage1_generation/batch_tools/list_batch_experiments.py <batch_id>

# Only view specificdataset
python automation/stage1_generation/batch_tools/list_batch_experiments.py <batch_id> --dataset Copa

# ShowDetailedinformation
python automation/stage1_generation/batch_tools/list_batch_experiments.py <batch_id> --verbose
```

**outputexample**:
```
📊 Copa (3 experiment)

  🔧 temp05_topp10_gpt4o
     ⚡ data reuse: No (Newly generated)

  🔧 temp07_topp10_gpt4o
     ⚡ data reuse: Yes (Originalbatch: batch_20241228_pilot)
```

---

### list_shared_experiments.py

**Purpose**: viewphysical storageallexperiment

**location**: `automation/stage1_generation/batch_tools/list_shared_experiments.py`

**Usage**:
```bash
# List all physical experiments
python automation/stage1_generation/batch_tools/list_shared_experiments.py

# Only view specificdataset
python automation/stage1_generation/batch_tools/list_shared_experiments.py --dataset Copa

# ShowDetailedinformation
python automation/stage1_generation/batch_tools/list_shared_experiments.py --verbose
```

**Purpose**:
- View which parameter configurations have already generated data
- Avoid duplicate generation of same parameter data
- Understand physical storage usage

---

### compare_experiments.py

**Purpose**: compareexperimentparameter

**location**: `automation/stage1_generation/batch_tools/compare_experiments.py`

**Usage**:
```bash
# compareTwo physicalexperiment
python automation/stage1_generation/batch_tools/compare_experiments.py \
    --shared Copa/temp07_topp10_gpt4o \
    --shared Copa/temp09_topp10_gpt4o

# comparebatch in experiment
python automation/stage1_generation/batch_tools/compare_experiments.py \
    --batch1 batch_20241229_temperature \
    --batch2 batch_20241230_topp \
    --dataset1 Copa \
    --dataset2 Copa
```

**outputexample**:
```
✅ sameparameter:
  generation.model: gpt-4o
  generation.top_p: 1.0

⚠️  Differentparameter:
  generation.temperature:
    experiment1: 0.7
    experiment2: 0.9
```

---

## datapathtool

### list_data_paths.py ⭐ New

**Purpose**: Listall can  use trainingdatapath

**location**: `automation/stage1_generation/batch_tools/list_data_paths.py`

**Usage**:
```bash
# Listalldata
python automation/stage1_generation/batch_tools/list_data_paths.py

# Only view certainbatch
python automation/stage1_generation/batch_tools/list_data_paths.py \
    --batch batch_20241229_temperature

# Only view certaindataset
python automation/stage1_generation/batch_tools/list_data_paths.py --dataset Copa

# Output YAML format (can be directly copied to configuration file)
python automation/stage1_generation/batch_tools/list_data_paths.py --format yaml
```

**Output example (Concise mode)**:
```
====================================================================================================
📊  can  use trainingdatapath
====================================================================================================

====================================================================================================
🗂️  Batch: batch_20241229_temperature
====================================================================================================

📁 Copa / temp07_topp10_gpt4o
   📝 description: Study temperature parameter impact on synthetic data quality

   ✅ Batch path (recommended - Organized by experiment purpose):
      Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa

   📦 Sharedpath（physical storage）:
      Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa
```

**Output example (YAML mode)**:
```yaml
# Can be directly copied to training configuration file
data:
  # Batch path (recommended)
  path: "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"
  # oruse Sharedpath
  # path: "Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa"
```

**Use Cases**:
- Find data path when writing training configuration
- Quick browseall can  use data
- Generate configuration file template

---

### resolve_data_path.py ⭐ New

**Purpose**: convertbatchpathandsharedpath

**location**: `automation/stage1_generation/batch_tools/resolve_data_path.py`

**Usage**:
```bash
# Batchpath → Sharedpath
python automation/stage1_generation/batch_tools/resolve_data_path.py \
    "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"

# Shared path → all batches reference
python automation/stage1_generation/batch_tools/resolve_data_path.py \
    "Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa"
```

**outputexample**:
```
====================================================================================================
🔍 pathparse：Batch → Shared
====================================================================================================

inputpath: batch_20241229_temperature/Copa/temp07_topp10_gpt4o
Physical path: _shared/Copa/temp07_topp10_gpt4o

✅ This data is referenced by the following batches:
  • batch_20241229_temperature
  • batch_20241230_model_comparison
```

**Use Cases**:
- Find physical data location
- Understand data reuse situation
- validatesymbolic linkYesNocorrect

---

## configureandDiagnostictool

### config.py

**Purpose**: validateprojectpathconfigure

**location**: `automation/config.py`

**Usage**:
```bash
# validateconfigure
python automation/config.py
```

**output**:
```
================================================================================
🔧 projectconfigure
================================================================================
PROJECT_ROOT:         /path/to/Synthetic_Data_for_ZO
AUTOMATION_DIR:       /path/to/Synthetic_Data_for_ZO/automation
DATA_V2_DIR:          /path/to/Synthetic_Data_for_ZO/Data_v2
RESULTS_V2_DIR:       /path/to/Synthetic_Data_for_ZO/Results_v2
================================================================================
✅ All critical paths validated successfully
================================================================================
```

**Environment variable override**:
```bash
export SYNTHETIC_DATA_PROJECT_ROOT="/your/custom/path"
python automation/config.py
```

---

### fix_hardcoded_paths.py

**Purpose**: Batch fix hardcoded paths (maintenance tool)

**location**: `automation/fix_hardcoded_paths.py`

**Usage**:
```bash
cd automation
python fix_hardcoded_paths.py
```

**Description**:
- Already in v2.1Version in execute
- Fix 10 files with hardcoded path issues
- General users do not need to use this tool

---

## tooluseworkflow

### Completedatagenerateworkflow

```bash
# 1. generatescript
python automation/stage1_generation/generator.py config.yaml

# 2. viewgeneratebatch
python automation/stage1_generation/batch_tools/list_batches.py --verbose

# 3. Execute data generation (see USER_GUIDE.md)
cd Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/scripts/
python rephrase_top20.py
python review_top20.py
...

# 4. Finddatapath（ fortrainingconfigure）
python automation/stage1_generation/batch_tools/list_data_paths.py --dataset Copa --format yaml
```

### Completetrainingworkflow

```bash
# 1. Write training configuration (use list_data_paths to get path)
vim automation/configs/stage2/my_training.yaml

# 2. Previewtraining
python automation/stage2_training/trainer.py my_training.yaml --dry-run

# 3. executetraining
python automation/stage2_training/trainer.py my_training.yaml

# 4. viewresult
python automation/stage2_training/list_results.py --detail
```

---

## FAQ

### Q: How to quickly find data path for training?

A: use `list_data_paths.py`:
```bash
python automation/stage1_generation/batch_tools/list_data_paths.py \
    --dataset Copa --format yaml
```

### Q: How to check if a parameter configuration has already generated data?

A: use `list_shared_experiments.py`:
```bash
python automation/stage1_generation/batch_tools/list_shared_experiments.py \
    --dataset Copa --verbose
```

### Q: Should training configuration use batch path or shared path?

A: **Both work!** Recommend using batch path (more intuitive)：
```yaml
data:
  path: "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"
```

### Q: Still need to use publish_dataset.py?

A: **Not needed!** trainer.py can directly use `Data_v2/` path. `publish_dataset.py` only for compatibility with old code.

---

## UpdateLog

### v2.1 (2026-01-01)
- ✅ New `list_data_paths.py` - data path finder tool
- ✅ New `resolve_data_path.py` - path conversion tool
- ✅ fixLoRAtrainingscriptNameerror
- ✅ fixLoRAenvironment variableerror
- ✅ Remove all hardcoded paths, use unified config.py
- ✅ fixbatch_toolspathimportissue

### v2.0 (2024-12-30)
- ✅ Implementation of Batch Solution 3++
- ✅ FullyconfigureDrivensystem
- ✅ Multipledatasetsupport

---

**Complete tool ecosystem! Use these tools to improve efficiency!** 🚀
