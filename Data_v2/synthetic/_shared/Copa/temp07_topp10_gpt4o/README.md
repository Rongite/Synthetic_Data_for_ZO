# Copa Synthetic Data Generation

**Generation Time**: 2025-12-30 19:06:38

## Experiment Information

- **Experiment Purpose**: temperature_comparison
- **ExperimentID**: N/A
- **Experiment Description**: comparetemperature=0.5/0.7/0.9 for CopaSynthetic dataqualityImpact
- **Parameter Fingerprint**: a5df2df31852

## Configuration Information

- **Generation Strategy**: direct_all
- **Task**: Copa
- **Training Method**: mezo
- **Dataset**: copa
- **Generation Model**: gpt-4o
- **Temperature**: 0.7
- **Version**: v1

## Directory Structure

```
temp07_topp10_gpt4o/
├── Copa/     # 🆕 DatasetDirectory（MeZOCan be directly used）
│   ├── copa_train.jsonl              # synthetic+Validation back Training set
│   ├── copa_validation.jsonl         # Validation set（Copied from original）
│   └── copa_test.jsonl               # Test set（Copied from original）
├── scripts/
│   ├── rephrase_all.py      # Rephrase all data
│   ├── rephrase_top20.py    # Rephrase top 20 difficult samples
│   ├── rephrase_rest.py     # Rephrase remaining samples
│   └── validate.py          # Validation script（Rejection sampling+Datasetfinalation）
├── generation_config.yaml   # ConfigurationFilecopy
├── experiment_metadata.json # Experiment metadata
└── README.md               # This file
```

## Usage

### 1. Set environment variables

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"  # Optional
```

### 2. Generate synthetic data

```bash
# direct_all Mode：directlyGenerateAllData
python scripts/rephrase_all.py

```

### 3. useDatasetTrainingModel

```bash
# useMeZOTraining
python PromptZO/MeZO/large_models/run.py \
    --task Copa \
    --model meta-llama/Llama-3.2-1B \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4
```

## finalDatasetstructure

```
Copa/
├── copa_train.jsonl       # Synthetic data
├── copa_validation.jsonl  # Validation set（fromoriginalData）
└── copa_test.jsonl        # Test set（fromoriginalData）
```

thisDirectorycandirectlypass to MeZOTrainingScriptuse。

## Prompt information

### Rephrase Prompt

```
You are tasked with rephrasing the given premise while preserving its original meaning. Your goal is to create rephrased data optimized for enhancing gradient estimation in training with a memory-effi...
```

See details `generation_config.yaml`
