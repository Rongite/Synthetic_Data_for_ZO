# Synthetic Data for Zero-Order Optimization

**Complete Experiment Management and Automation System - MeZO Optimization**

Research on the impact of synthetic data on MeZO (Memory-Efficient Zero-Order Optimizer) training effectiveness.

**System Status**: ✅ Full Functionality Implemented | v2.1 (2026-01-01) - All P0-level Bugs Fixed

---

## 📋 Table of Contents

- [Core Features](#core-features)
- [Quick Start](#quick-start)
- [Complete Documentation](#complete-documentation)
- [Directory Structure](#directory-structure)

---

## Core Features

### ✅ **Experiment Management System**
- **Classification by experiment purpose**: `prompt_engineering`, `temperature_study`, `model_comparison`, etc.
- **Smart parameter fingerprinting**: Automatically identifies identical parameter configurations to avoid accidental overwrites
- **Overwrite strategies**: Same parameters with optional overwrite/auto-overwrite/never overwrite
- **Complete metadata traceability**: All experiment parameters automatically recorded

### ✅ **Dataset Generation (Stage 1)**
- **Configuration-driven**: All parameters set through YAML config, no manual script editing required
- **GPT-4o generation**: Automatically calls API to generate synthetic data
- **Rejection Sampling**: Automatic data quality validation
- **MeZO compatible**: Automatically generates dataset structure expected by MeZO
- **Manual checkpoints**: Support for manual review, annotation, and prompt testing

### ✅ **Model Training (Stage 2)**
- **Hyperparameter grid search**: Automatically runs all parameter combinations
- **Multiple optimization methods**: MeZO (zo) / Full Fine-tuning (fo_full) / LoRA (fo_lora)
- **Automatic result management**: Training results organized by experiment purpose
- **Timestamp isolation**: Each training run automatically creates a timestamped directory

---

## Quick Start

### **Step 0: Environment Setup (First-time Use)**

```bash
# 1. Install dependencies
pip install pyyaml openai tqdm datasets transformers torch

# 2. Configure API key
export OPENAI_API_KEY="your-key-here"

# 3. Verify configuration
python automation/config.py

# For detailed configuration instructions: automation/SETUP_GUIDE.md
```

### **Option 1: View Complete Guide**

```bash
# Complete functionality summary and usage guide
cat COMPLETE_SYSTEM_SUMMARY.md
```

### **Option 2: Test Immediately**

```bash
# 1. Generate scripts from test configuration
python automation/stage1_generation/generator.py \
       automation/configs/examples/stage1_full_example_copa.yaml

# 2. View generated directory structure
tree Data_v2/synthetic/prompt_engineering/copa_mezo_v1/

# 3. Generate data (requires API key)
export OPENAI_API_KEY="your-key"
cd Data_v2/synthetic/prompt_engineering/copa_mezo_v1/scripts/
python rephrase_all.py
python validate.py

# 4. Use directly for MeZO training
cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/
python PromptZO/MeZO/large_models/run.py \
    --task Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa \
    --model meta-llama/Llama-3.2-1B
```

---

## Complete Documentation

### **📌 Main Documentation**

| Document | Description |
|------|------|
| **`automation/SETUP_GUIDE.md`** | 🔧 **Environment Setup Guide** (Required reading for first-time users) |
| **`automation/USER_GUIDE.md`** | 📖 **User Manual** (Recommended starting point) |
| **`automation/COMPLETE_PIPELINE_SIMULATION.md`** | 📋 **Complete Pipeline Examples** (Detailed steps) |
| `automation/BATCH_GUIDE.md` | Batch Solution 3++ Complete Guide |
| `automation/TOOLS_REFERENCE.md` | All Tools Reference Manual |
| `automation/stage2_training/RESULTS_MANAGEMENT.md` | Stage 2: Training Results Management |
| `automation/BUG_FIXES_SUMMARY.md` | Bug Fix Summary (v2.1) |

### **Configuration Examples**

| File | Description |
|------|------|
| `automation/configs/examples/stage1_full_example_copa.yaml` | Complete Stage 1 configuration example |
| `automation/configs/examples/stage2_example_training.yaml` | Stage 2 training configuration example |

---

## Directory Structure

### **Core Directories**

```
Synthetic_Data_for_ZO/
├── README.md                           # 📌 This file
├── COMPLETE_SYSTEM_SUMMARY.md          # 📌 Complete usage guide
│
├── automation/                         # Automation tools
│   ├── stage1_generation/              # Stage 1: Data generation
│   │   ├── generator.py                # Script generator
│   │   ├── experiment_manager.py       # Experiment management
│   │   ├── create_experiment.py        # Parameter tuning experiment tool
│   │   ├── archive_validated_config.py # Configuration archival tool
│   │   ├── list_experiments.py         # Experiment listing tool
│   │   └── tools/                      # Manual review tools
│   │       └── review_top20.py
│   │
│   ├── stage2_training/                # Stage 2: Model training
│   │   └── trainer.py
│   │
│   └── configs/                        # Configuration files
│       ├── examples/                   # Configuration examples
│       └── stage1/
│           ├── templates/              # Validated prompt templates
│           ├── experiments/            # Parameter tuning experiment configs
│           └── drafts/                 # Pending validation configs
│
├── Data_v2/                            # 🆕 New data management
│   ├── original/                       # Original datasets
│   └── synthetic/                      # Synthetic data
│       └── {experiment_purpose}/       # Experiment purpose (top-level classification)
│           └── {experiment_id}/        # Experiment ID (parameter isolation)
│               ├── {DatasetName}/      # MeZO dataset directory
│               │   ├── {task}_train.jsonl
│               │   ├── {task}_validation.jsonl
│               │   └── {task}_test.jsonl
│               ├── scripts/
│               ├── generation_config.yaml
│               ├── experiment_metadata.json
│               └── README.md
│
├── Results_v2/                         # 🆕 New results directory
│   └── {experiment_purpose}/
│       └── {Model}/
│           └── {Task}_{Method}_{DataType}_{LR}/
│               └── {Timestamp}/
│
├── PromptZO/MeZO/                      # MeZO training code
│
├── Data/                               # Old data directory (preserved)
├── results/                            # Old results directory (preserved)
└── Pending_Manual_Classification/      # Data pending manual classification
```

### **Three-Tier Directory Design**

```
Data_v2/synthetic/
├── prompt_engineering/          # 📁 Tier 1: Experiment purpose classification
│   ├── copa_mezo_v1/            # 📁 Tier 2: Experiment ID (parameter isolation)
│   │   └── Copa/                # 📁 Tier 3: MeZO dataset directory
│   │       ├── copa_train.jsonl
│   │       ├── copa_validation.jsonl
│   │       └── copa_test.jsonl
│   └── copa_mezo_v2/
│
├── temperature_study/           # 📁 Different experiment purpose
│   ├── copa_mezo_temp05/
│   ├── copa_mezo_temp07/
│   └── copa_mezo_temp09/
│
└── model_comparison/            # 📁 Different experiment purpose
    ├── copa_mezo_gpt4o/
    └── copa_mezo_gpt4omini/
```

---

## Key Features

### **1. Experiment Purpose Classification**
Top-level directories isolate different experiment types to avoid data confusion:
```yaml
experiment:
  purpose: "temperature_study"  # Experiment purpose
```

### **2. Parameter Fingerprint Recognition**
Automatically calculates parameter hash; same parameters can be overwritten, different parameters automatically isolated:
- Same experiment purpose + Same parameter fingerprint → Prompt whether to overwrite
- Different experiment purpose OR Different parameters → Automatically create new directory

### **3. Full MeZO Compatibility**
Automatically generates dataset structure expected by MeZO training scripts:
```
{experiment_id}/
└── Copa/                      # Directory name expected by MeZO
    ├── copa_train.jsonl       # Filename expected by MeZO
    ├── copa_validation.jsonl
    └── copa_test.jsonl
```

MeZO training command:
```bash
python PromptZO/MeZO/large_models/run.py \
    --task Data_v2/synthetic/{purpose}/{exp_id}/Copa
```

### **4. Automated File Management**
- Training set: Synthetic data + rejection sampling validation
- Validation set: Automatically copied from original data
- Test set: Automatically copied from original data

### **5. Prompt Version Management**
- `templates/` - Validated prompts for reuse
- `experiments/` - Parameter tuning experiment configs
- One-click creation of parameter tuning experiments, no repeated manual review needed

### **6. Training Results Management (Results_v2)**
- **🔴 Independent experiment purpose**: Training purpose is independent from data generation purpose (same data can be used for multiple different training experiments)
- **Experiment purpose classification**: Results organized by training experiment purpose
- **Complete traceability**: Trace back to dataset through experiment_config.yaml
- **Management tools**: list_results.py for quick viewing of all training results

**Important**: Stage 1 and Stage 2 experiment purposes are independent!
```
Data: Data_v2/synthetic/prompt_engineering/...  ← Data generation purpose
Training: Results_v2/hyperparameter_tuning/...      ← Training purpose (can be different)
```

---

## Common Commands

### **Stage 1: Generate Synthetic Data**

```bash
# 1. Generate scripts
python automation/stage1_generation/generator.py <config.yaml>

# 2. View all batches
python automation/stage1_generation/batch_tools/list_batches.py --verbose

# 3. Find data paths (for training config) ⭐ New
python automation/stage1_generation/batch_tools/list_data_paths.py --dataset Copa --format yaml
```

### **Stage 2: Model Training**

```bash
# Automatic training (results classified by experiment purpose)
python automation/stage2_training/trainer.py <config.yaml>

# Preview (without actual execution)
python automation/stage2_training/trainer.py <config.yaml> --dry-run

# View training results summary
python automation/stage2_training/list_results.py

# View detailed results for specific experiment purpose
python automation/stage2_training/list_results.py --detail --purpose prompt_engineering
```

---

## Supported Datasets

- Copa (SuperGLUE)
- BOOLQ
- CB (CommitmentBank)
- RTE (Recognizing Textual Entailment)
- ArcC_Cloze
- ArcC_MC (Multiple Choice)

---

## Supported Optimization Methods

- **MeZO (zo)**: Memory-Efficient Zero-Order Optimizer
- **Full Fine-tuning (fo_full)**: First-order full parameter fine-tuning
- **LoRA (fo_lora)**: Low-Rank Adaptation fine-tuning

---

## Environment Requirements

```bash
# Python 3.8+
pip install pyyaml openai tqdm datasets transformers torch

# API Key (required for Stage 1)
export OPENAI_API_KEY="your-api-key"

# Project path (auto-detected, can also be set manually)
export SYNTHETIC_DATA_PROJECT_ROOT="/path/to/project"  # Optional
```

**Detailed configuration instructions**: See `automation/SETUP_GUIDE.md`

---

## Frequently Asked Questions

### Q: How to avoid new data overwriting old data?
A: The Batch solution automatically recognizes identical configurations through parameter fingerprinting, automatically reusing data instead of overwriting.

### Q: How to quickly find data paths for training?
A: Use the newly added `list_data_paths.py` tool:
```bash
python automation/stage1_generation/batch_tools/list_data_paths.py --dataset Copa --format yaml
```

### Q: Can generated datasets be used directly for training?
A: Yes! trainer.py can directly use `Data_v2/` paths without needing to publish:
```bash
# Training configuration
data:
  path: "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"
```

### Q: Has LoRA training been fixed?
A: ✅ Yes! v2.1 has fixed all LoRA-related bugs (script names, environment variables). See `automation/BUG_FIXES_SUMMARY.md` for details

---

## Contact and Support

View complete documentation: `COMPLETE_SYSTEM_SUMMARY.md`

View old system migration instructions: `Pending_Manual_Classification/README.md`

---

**Start your first experiment!** 🚀
