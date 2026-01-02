# Stage 1 Generation Tools

This directory contains manual checkpoint tools required for Stage 1 synthetic data generation.

---

## ⚠️ Important Note: Only Synthesizing Train Data

**The pipeline only synthesizes/rephrases training data (`{dataset}_train.jsonl`), validation and test data are directly copied from the original dataset**:

- ✅ **{dataset}_train.jsonl** → Goes through rephrase → validation → rejection sampling (synthetic data)
- 📋 **{dataset}_validation.jsonl** → Directly copied from `Data/original/{Dataset}/` (original data)
- 📋 **{dataset}_test.jsonl** → Directly copied from `Data/original/{Dataset}/` (original data)

**Reasons**:
1. **Validation and test data need to remain standardized** for fair evaluation of model performance
2. **Synthetic data is only used for training** to improve model learning during training phase
3. **Reuse original evaluation data** to ensure comparability of results across different experiments

**Reflected in the pipeline**:
- `rephrase_top20.py` / `rephrase_rest.py` / `rephrase_all.py` → Only process train data
- `validate.py` → Only validate train data, and automatically copy validation/test files from the original dataset at the end

---

## Tool List

### 1. review_top20.py
**Purpose**: Checkpoint 1 - Manual review of the first 20 samples

Review the quality of the first 20 synthetic samples. Approved samples will be made into few-shot examples for generating remaining data.

```bash
# Usage (run in the generated script directory)
cd Data_v2/synthetic/{purpose}/{experiment_id}/scripts/
python review_top20.py
```

**Output**: `validation_checkpoints/top20_review.json`

---

### 2. extract_samples.py
**Purpose**: Extract specified range of samples from synthetic data

Extract samples in a specific range for subsequent manual annotation (typically 21-80).

```bash
# Extract samples 21-80
python extract_samples.py --range 21-80 --input Copa/copa_train.jsonl

# Extract samples 21-40
python extract_samples.py --range 21-40 --input Copa/copa_train.jsonl

# Extract samples 41-80
python extract_samples.py --range 41-80 --input Copa/copa_train.jsonl

# Specify output path
python extract_samples.py \
    --range 21-80 \
    --input Copa/copa_train.jsonl \
    --output validation_checkpoints/custom_samples.jsonl
```

**Output**: `validation_checkpoints/samples_{range}.jsonl`

---

### 3. annotate_samples.py
**Purpose**: Checkpoint 2 - Manual annotation of sample semantic consistency

Provides an interactive interface for users to annotate whether the rephrased premise of each sample is semantically consistent with the original premise.

```bash
# Annotate extracted samples
python annotate_samples.py validation_checkpoints/samples_21_80.jsonl

# Custom output file
python annotate_samples.py \
    validation_checkpoints/samples_21_80.jsonl \
    --output validation_checkpoints/custom_annotated.json

# Restart annotation (don't continue from previous progress)
python annotate_samples.py \
    validation_checkpoints/samples_21_80.jsonl \
    --no-resume
```

**Interactive Interface**:
```
Sample 1/60 (21st in original data):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Original premise:
  "The tenant misplaced his keys."

Choice 1: His landlord unlocked the door.
Choice 2: His landlord repaired the door.
Question: effect
Correct answer: Choice 1

Is the semantic meaning consistent?
  [s] same - Semantically consistent, rephrase is valid
  [n] not the same - Semantic changed, rephrase is invalid
  [k] skip - Skip this sample
  [q] quit - Save and exit

Your judgment: s
```

**Output**: `validation_checkpoints/samples_21_80_annotated.json`

**Annotation Result Format**:
```json
{
  "metadata": {
    "total_samples": 60,
    "annotated_count": 60,
    "same_count": 57,
    "not_same_count": 3,
    "last_updated": "2024-12-26T10:30:00"
  },
  "annotations": [
    {
      "index": 20,
      "original_premise": "The tenant misplaced his keys.",
      "rephrased_premise": "The tenant lost his apartment keys.",
      "choice1": "His landlord unlocked the door.",
      "choice2": "His landlord repaired the door.",
      "question_type": "effect",
      "label": 0,
      "correct_answer": "Choice 1",
      "judgment": "same",
      "note": ""
    }
  ]
}
```

---

### 4. generate_validation_test.py
**Purpose**: Generate judger prompt test script

Automatically generates a script for testing judger prompt accuracy based on manual annotation results.

```bash
# Use default paths
python generate_validation_test.py

# Specify all parameters
python generate_validation_test.py \
    --annotations validation_checkpoints/samples_21_80_annotated.json \
    --fewshot-range 21-40 \
    --test-range 41-80 \
    --output scripts/validate_prompt_test.py \
    --api-key sk-your-api-key \
    --base-url https://api.openai.com/v1
```

**Functions**:
1. Read annotation results
2. Extract samples marked as "same" from 21-40 → few-shot examples
3. Extract all samples from 41-80 → test_set (including ground truth)
4. Generate complete test script `validate_prompt_test.py`

**Output**: `scripts/validate_prompt_test.py`

**Usage of Generated Test Script**:
```bash
# Run test (Checkpoint 3)
cd Data_v2/synthetic/{purpose}/{experiment_id}/scripts/
python validate_prompt_test.py

# If accuracy ≥ 95%, automatically create pass marker
# validation_checkpoints/prompt_test_passed.flag
```

---

### 5. publish_dataset.py
**Purpose**: Publish validated data from Data_v2 to the path expected by training scripts

Publishes data generated by the pipeline from `Data_v2/synthetic/{purpose}/{experiment_id}/{Dataset}/` to a path that training scripts can directly use.

```bash
# Basic usage: Publish to Data/synthetic/{Dataset}/
python publish_dataset.py \
    --source Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa \
    --dataset Copa

# Publish to rejection_sampling path (compatible with old training scripts)
python publish_dataset.py \
    --source Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa \
    --dataset Copa \
    --target Data/rejection_sampling/0_data

# Archive version to subdirectory simultaneously
python publish_dataset.py \
    --source Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa \
    --dataset Copa \
    --target Data/rejection_sampling/0_data \
    --archive mezo_gpt/version_1

# Use symbolic links (save disk space)
python publish_dataset.py \
    --source Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa \
    --dataset Copa \
    --symlink
```

**Functions**:
- Copy or symlink data files to target directory
- Optional archiving to subdirectory (e.g., mezo_gpt/version_1/)
- Create publish metadata record (.publish_metadata.json)
- Automatically check if necessary files exist

**Output**: `Data/synthetic/{Dataset}/` or `Data/rejection_sampling/0_data/{Dataset}/`
```
Copa/
├── copa_train.jsonl
├── copa_validation.jsonl
├── copa_test.jsonl
├── .publish_metadata.json
└── mezo_gpt/              # Optional archive
    └── version_1/
        └── copa_train.jsonl
```

**Training Script Usage**:
```bash
# Set TASK path in training script
TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/synthetic/Copa
```

---

## Complete Pipeline Examples

### Scenario A: First-time Generation (Including All Manual Checkpoints)

```bash
# Step 1: Create draft configuration
vim automation/configs/stage1/drafts/copa_mezo_v1_draft.yaml

# Step 2: Generate scripts
python automation/stage1_generation/generator.py \
       automation/configs/stage1/drafts/copa_mezo_v1_draft.yaml

# Step 3: Generate first 20 samples
cd Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa/scripts/
python rephrase_top20.py

# 🔴 Checkpoint 1: Review first 20 samples
python ../../../tools/review_top20.py
# → Output: validation_checkpoints/top20_review.json

# Step 4: Generate remaining 380 samples (using approved few-shot)
python rephrase_rest.py

# Step 5: Extract samples 21-80 for annotation
cd ..  # Return to dataset directory
python ../../../automation/stage1_generation/tools/extract_samples.py \
       --range 21-80 \
       --input Copa/copa_train.jsonl

# 🔴 Checkpoint 2: Manual annotation of 21-80
python ../../../automation/stage1_generation/tools/annotate_samples.py \
       validation_checkpoints/samples_21_80.jsonl
# → Output: validation_checkpoints/samples_21_80_annotated.json

# Step 6: Generate judger test script
python ../../../automation/stage1_generation/tools/generate_validation_test.py
# → Output: scripts/validate_prompt_test.py

# 🔴 Checkpoint 3: Test judger prompt
cd scripts/
python validate_prompt_test.py

# If accuracy < 95%:
#   1. Modify automation/configs/stage1/drafts/copa_mezo_v1_draft.yaml
#   2. Re-run generator.py
#   3. Re-run validate_prompt_test.py

# Step 7: Batch validate all 400 samples (using validated judger prompt)
python validate.py
# → Output: Copa/copa_train_validated.jsonl

# Step 8: Publish data to path expected by training scripts
cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/
python automation/stage1_generation/tools/publish_dataset.py \
       --source Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa \
       --dataset Copa \
       --target Data/rejection_sampling/0_data \
       --archive mezo_gpt/copa_mezo_v1
# → Output: Data/rejection_sampling/0_data/Copa/copa_train.jsonl (etc.)

# Step 9: Archive as validated template
python automation/stage1_generation/archive_validated_config.py \
       --source automation/configs/stage1/drafts/copa_mezo_v1_draft.yaml \
       --data-dir Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa/
# → Output: automation/configs/stage1/templates/copa_mezo_validated.yaml
```

### Scenario B: Parameter Tuning Experiment (No Manual Checkpoints)

```bash
# Step 1: Create experiment configuration based on validated template
python automation/stage1_generation/create_experiment.py \
       --template automation/configs/stage1/templates/copa_mezo_validated.yaml \
       --version v2 \
       --param generation.temperature=0.7

# Step 2: Generate scripts
python automation/stage1_generation/generator.py \
       automation/configs/stage1/experiments/copa_mezo_v2_temperature07.yaml

# Step 3: Run directly (no manual review needed)
cd Data_v2/synthetic/prompt_engineering/copa_mezo_v2/Copa/scripts/
python rephrase_all.py   # Directly generate 400, includes validated few-shot

# Step 4: Batch validate (using validated judger prompt)
python validate.py

# Step 5: Publish data (if training is needed)
cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/
python automation/stage1_generation/tools/publish_dataset.py \
       --source Data_v2/synthetic/prompt_engineering/copa_mezo_v2/Copa \
       --dataset Copa \
       --target Data/rejection_sampling/0_data \
       --archive mezo_gpt/copa_mezo_v2_temp07
```

---

## Important Notes

1. **Sample Indexing**: All tools use 1-based indexing (1st sample), but internally convert to 0-based
2. **Checkpoint Order**: Must complete checkpoints in order: Checkpoint 1 → Checkpoint 2 → Checkpoint 3
3. **Data Leakage**: validate.py has been fixed and will automatically exclude samples 21-40 (used as judger few-shot)
4. **Accuracy Requirement**: Checkpoint 3's judger prompt must achieve ≥95% accuracy to pass
5. **Save Progress**: annotate_samples.py supports resuming after interruption, progress is automatically saved
6. **Data Publishing**: After completing validation, **must** use `publish_dataset.py` to publish data to the path expected by training scripts
7. **Training Script Paths**:
   - Original data: `Data/original/{Dataset}/`
   - Synthetic data: `Data/rejection_sampling/0_data/{Dataset}/` (recommended) or `Data/synthetic/{Dataset}/`
   - Training scripts specify data path via `TASK` environment variable

---

## Troubleshooting

### Sample Extraction Failed
```bash
# Check if input file path is correct
ls -la Copa/copa_train.jsonl

# Use absolute path
python extract_samples.py \
    --range 21-80 \
    --input /absolute/path/to/Copa/copa_train.jsonl
```

### Annotation Tool Cannot Run
```bash
# Ensure samples have been extracted
ls validation_checkpoints/samples_21_80.jsonl

# Check file format
head validation_checkpoints/samples_21_80.jsonl
```

### Low Judger Test Accuracy
1. Check number of few-shot examples (recommend ≥5 "same" samples)
2. Check if few-shot examples include edge cases
3. Adjust validation_prompt description in config file
4. Add more explicit judgment criteria

---

## Developer Information

- **Created**: 2024-12-29
- **Version**: 1.0
- **Maintained by**: Synthetic Data Generation Team
