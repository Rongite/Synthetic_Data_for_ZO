# Copa Synthetic Data Generation

**Generation Time**: 2025-12-30 18:43:11

## Experiment Information

- **Experiment Purpose**: prompt_engineering
- **ExperimentID**: copa_mezo_v1
- **Experiment Description**: CopaTaskpromptoptimizeExperiment
- **Parameter Fingerprint**: b90eb4799593

## Configuration Information

- **Task**: Copa
- **Training Method**: mezo
- **Dataset**: copa
- **Generation Model**: gpt-4o
- **Temperature**: 0.9
- **ValidationModel**: gpt-4o
- **Version**: v1

## Directory Structure

```
temp09_topp10_gpt4o/
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
# method1: Rephrase all data
python scripts/rephrase_all.py

# method2: classifycategoryRephrasedifficultSample and RemainingSample
python scripts/rephrase_top20.py
python scripts/rephrase_rest.py
```

### 3. ValidationDataqualityandfinalationDataset

```bash
python scripts/validate.py
```

thisScript will ：
1. userejection samplingValidationSynthetic dataquality
2.  will ValidationpassDataRename as officialTraining set
3.  from originalDatasetCopyvalidation and testFile
4. GenerateCompleteMeZO can  use Dataset

### 4. useDatasetTrainingModel

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
├── copa_train.jsonl       # synthetic+Validation back Training set
├── copa_validation.jsonl  # Validation set（fromoriginalData）
└── copa_test.jsonl        # Test set（fromoriginalData）
```

thisDirectorycandirectlypass to MeZOTrainingScriptuse。

## Prompt information

### Rephrase Prompt

```
You are tasked with rephrasing the given premise while preserving its original meaning. Your goal is to create rephrased data optimized for enhancing gradient estimation in training with a memory-effi...
```

### Validation Prompt

```
Task: Verify if the rephrased premise maintains consistency with the correct answer choice.

{{VALIDATION_FEWSHOT}}

Original premise: "{original_premise}"
Rephrased premise: "{rephrased_premise}"
Cho...
```

See details `generation_config.yaml`
