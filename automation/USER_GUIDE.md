# Synthetic Data Generation System - User Manual

**Version**: 2.0
**Last Updated**: 2025-12-30

---

## 📚 Table of Contents

1. [System Overview](#system-overview)
2. [Quick Start](#quick-start)
3. [API Configuration](#api-configuration)
4. [Multi-Dataset Support](#multi-dataset-support)
5. [Complete Pipeline Process](#complete-pipeline-process)
6. [Common Scenarios](#common-scenarios)
7. [Tool Usage](#tool-usage)
8. [Troubleshooting](#troubleshooting)

---

## System Overview

### Features

This system provides a **fully automated synthetic data generation pipeline**, supporting:

- ✅ **Multi-Dataset Support** - Copa, BOOLQ, CB, RTE, ArcC, etc., zero code modification
- ✅ **Unified API Configuration** - Centralized management of OpenAI API or compatible services
- ✅ **Quality Assurance Mechanism** - 3 manual review breakpoints + automatic validation
- ✅ **Automatic Few-shot Generation** - Automatically extract and inject from high-quality samples
- ✅ **Rejection Sampling** - Automatically replace unqualified samples with original data
- ✅ **Parameter Deduplication** - Batch solution automatically detects same parameter configurations and reuses data

### Core Features

#### 1. Fully Configuration-Driven

```yaml
# Just modify the configuration file, no code changes needed
dataset:
  fields: ["premise", "choice1", "choice2", "question", "label"]

generation:
  field_to_rephrase: "premise"
  rephrase_prompt: |
    {{REPHRASE_FEWSHOT}}  # Automatically inject few-shot
    Original premise: "{premise}"
    ...
```

#### 2. Automatic Field Adaptation

```python
# System automatically generates function signatures for different datasets
Copa:   generate_prompt(premise, choice1, choice2, question, label)
BOOLQ:  generate_prompt(question, answer, passage)
CB:     generate_prompt(premise, hypothesis, label)
```

#### 3. 100% Universal Tools

All tool scripts automatically recognize dataset fields, no modification needed:
- `review_top20.py` - Review top 20 samples
- `annotate_samples.py` - Annotate samples 21-80
- `generate_validation_test.py` - Test validation accuracy

---

## Quick Start

### First-Time Usage (Using Copa as Example)

#### 1. Configure API

API configuration is set separately in generation and validation sections (no environment variables needed):

```yaml
generation:
  # Rewriter API configuration
  api_key: "sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c"
  base_url: "https://api2.aigcbest.top/v1"
  timeout: 120

  model: "gpt-4o"
  temperature: 0.9
  # ... other generation configuration ...

validation:
  # Judge API configuration
  api_key: "sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c"
  base_url: "https://api2.aigcbest.top/v1"
  timeout: 120

  model: "gpt-4o"
  temperature: 0.0
  # ... other validation configuration ...
```

All example configuration files (`configs/examples/`) already include default API configuration, you can use directly or modify to your own configuration.

#### 2. Generate Scripts

```bash
cd automation/stage1_generation

python generator.py \
  --config ../configs/examples/stage1_full_example_copa.yaml \
  --auto-resolve
```

**Output**:
```
Generating rephrasing scripts...
  ✓ rephrase_top20.py   # Generate first 20 samples
  ✓ rephrase_rest.py    # Generate remaining samples
  ✓ validate.py         # Validation script

Output directory: Data_v2/synthetic/_shared/Copa/temp09_topp10_gpt4o/
```

#### 3. Execute Pipeline

```bash
cd Data_v2/synthetic/_shared/Copa/temp09_topp10_gpt4o/scripts

# Breakpoint 1: Generate and review first 20 samples
python rephrase_top20.py
cp ../../../../automation/stage1_generation/tools/review_top20.py .
python review_top20.py
# Input: y, y, n, y, ... (mark high-quality samples)

# Breakpoint 2A: Samples 21-40
python rephrase_rest.py  # Pause at sample 40
cp ../../../../automation/stage1_generation/tools/annotate_samples.py .
python annotate_samples.py
# Input: same, same, not the same, ...

# Breakpoint 2B: Samples 41-80 testing
python rephrase_rest.py  # Pause at sample 80
cp ../../../../automation/stage1_generation/tools/generate_validation_test.py .
python generate_validation_test.py
# System tests validation accuracy, needs ≥95%

# Breakpoint 3: Automated validation of remaining data
python rephrase_rest.py  # Complete all data
python validate.py  # Automatic validation
```

#### 4. Obtain Final Dataset

```
Copa/
├── copa_train.jsonl          # 400 training samples (95%+ rephrased)
├── copa_validation.jsonl     # Validation set
└── copa_test.jsonl          # Test set
```

---

## API Configuration

### Configuration File Location

`automation/api_config.yaml`

### Basic Configuration

```yaml
# Default configuration (used by all tasks)
default:
  provider: "custom"
  base_url: "https://api2.aigcbest.top/v1"
  api_key_env: "OPENAI_API_KEY"
  model: "gpt-4o"
  timeout: 120

# Data generation specific configuration
generation:
  base_url: "https://api2.aigcbest.top/v1"
  model: "gpt-4o"

# Data validation specific configuration
validation:
  base_url: "https://api2.aigcbest.top/v1"
  model: "gpt-4o"
```

### Environment Variable Configuration

```bash
# Required: Set API Key
export OPENAI_API_KEY="sk-your-api-key"

# Optional: Override default API Base URL
export OPENAI_API_BASE="https://api.openai.com/v1"
```

**Configuration Priority**: Environment Variables > Configuration File > Code Defaults

### Switching API Providers

#### Method 1: Temporary Switch (Environment Variable)

```bash
# Use official OpenAI API (temporary)
export OPENAI_API_BASE="https://api.openai.com/v1"
python generator.py --config config.yaml
```

#### Method 2: Permanent Switch (Modify Configuration File)

Edit `automation/api_config.yaml`:

```yaml
generation:
  base_url: "https://api.openai.com/v1"  # Switch to official API

validation:
  base_url: "https://api2.aigcbest.top/v1"  # Keep third-party (cost saving)
```

#### Method 3: Hybrid Strategy

```yaml
# Use official API for high-quality tasks
generation:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o"

# Use third-party for low-cost tasks
validation:
  base_url: "https://api2.aigcbest.top/v1"
  model: "gpt-4o"
```

---

## Multi-Dataset Support

### Supported Dataset Types

| Dataset | Task Type | Field Structure | Rephrase Target |
|--------|----------|----------|----------|
| Copa | Causal Reasoning | premise, choice1, choice2, question, label | premise |
| BOOLQ | Yes/No Question Answering | question, answer, passage | passage |
| CB | Textual Entailment (3-class) | premise, hypothesis, label | premise |
| RTE | Textual Entailment (2-class) | premise, hypothesis, label | premise |
| ArcC | Multiple Choice | id, question, choices, answerKey | question |

### Adding New Datasets

#### Step 1: Create Configuration File

Copy example configuration and modify:

```bash
cp automation/configs/examples/stage1_full_example_copa.yaml \
   automation/configs/stage1_my_new_dataset.yaml
```

#### Step 2: Modify Configuration

```yaml
task_name: "MyDataset"

dataset:
  task_name: "mydataset"
  dataset_name: "MyDataset"
  input_path: "Data/original/MyDataset/mydataset_train.jsonl"

  # ⭐ Declare your fields
  fields:
    - "field1"
    - "field2"
    - "field3"

generation:
  # ⭐ Specify field to rephrase
  field_to_rephrase: "field1"

  # ⭐ Write rephrase prompt (using field references)
  rephrase_prompt: |
    {{REPHRASE_FEWSHOT}}  # Must include this placeholder

    Original field1: "{field1}"
    Context field2: "{field2}"
    Context field3: "{field3}"

    Rephrase the field1:

validation:
  # ⭐ Write validation prompt
  validation_prompt: |
    {{VALIDATION_FEWSHOT}}  # Must include this placeholder

    Original: "{original_field1}"
    Rephrased: "{rephrased_field1}"

    Are they the same? [same/not the same]:
```

#### Step 3: Generate Scripts

```bash
python automation/stage1_generation/generator.py \
  --config automation/configs/stage1_my_new_dataset.yaml
```

System will automatically:
- ✅ Generate `generate_prompt(field1, field2, field3)` function
- ✅ Adapt all tool scripts
- ✅ Use correct API configuration

#### Step 4: Execute Same Pipeline

```bash
cd Data_v2/synthetic/_shared/MyDataset/.../scripts
python rephrase_top20.py
python review_top20.py  # Tools automatically adapt to new fields!
# ... Exactly the same workflow
```

### Configuration File Examples

Complete examples in `automation/configs/examples/`:
- `stage1_example_copa.yaml` - Copa dataset
- `stage1_example_boolq.yaml` - BOOLQ dataset
- `stage1_example_cb.yaml` - CB dataset
- `stage1_example_rte.yaml` - RTE dataset

---

## Complete Pipeline Process

### Process Overview

```
Original Data
   ↓
Breakpoint 1 (Samples 1-20)
   ├─ Generate 20 samples
   ├─ Manual quality review
   ├─ Rejection sampling (unqualified → original)
   └─ Extract few-shot (17 high-quality samples)
   ↓
Breakpoint 2A (Samples 21-40)
   ├─ Generate 20 samples using few-shot
   ├─ Manual quality review
   ├─ Rejection sampling (unqualified → original)
   └─ Generate validation few-shot (18 samples)
   ↓
Breakpoint 2B (Samples 41-80)
   ├─ Generate 40 test samples
   ├─ Manual quality review
   ├─ Rejection sampling (unqualified → original)
   ├─ Generate test set (test AI judge)
   └─ Test accuracy needs ≥95% to continue
   ↓
Breakpoint 3 (Samples 81-N)
   ├─ Automatically generate remaining data
   ├─ Automatic quality validation
   └─ Rejection sampling
   ↓
Final Dataset
```

### Breakpoint 1: Top20 Sample Review

**Purpose**: Generate high-quality few-shot examples for subsequent data generation

```bash
# 1. Generate first 20 samples
python rephrase_top20.py

# 2. Manual review (batch input mode)
cp ../../../../automation/stage1_generation/tools/review_top20.py .
python review_top20.py
```

**Interactive Example** (batch input of unqualified sample numbers):
```
================================================================================
Comparison of First 20 Samples - Please carefully review original and rephrased data
================================================================================

【Sample 1】
  Original premise:  My body cast a shadow over the grass.
  Rephrased premise:  A shadow appeared on the grass beside me.
  Choice1: The sun was rising.
  Choice2: The grass was cut.
  Question: cause

【Sample 2】
  Original premise:  The woman tolerated her friend's difficult behavior.
  Rephrased premise:  The woman was patient with her friend's challenging attitude.
  Choice1: The woman knew her friend was going through a hard time.
  Choice2: The woman felt that her friend took advantage of her kindness.
  Question: cause

... (Displaying all 20 samples)...

【Sample 20】
  Original premise:  It got dark outside.
  Rephrased premise:  As night approached, the sky lost its light.
  Choice1: Snowflakes began to fall from the sky.
  Choice2: The moon became visible in the sky.
  Question: effect

================================================================================

Enter the numbers of unqualified samples (1-20), separated by commas
Example: 3,7,15  means samples 3, 7, 15 are unqualified
If all qualified, press Enter directly

Unqualified sample numbers: 3,7,12  ← User enters all unqualified sample numbers at once

Statistics:
  Qualified samples: 17
  Unqualified samples: 3

Executing rejection sampling...
✓ Rejection sampling completed: 17 rephrased + 3 original data

Generating few-shot examples...
✓ Generated 17 few-shot examples

✓ Few-shot injection successful!
   Injected into: rephrase_rest.py
```

### Breakpoint 2A: Processing Samples 21-40

**Purpose**: Generate validation few-shot examples for automatic quality validation

```bash
# 1. Generate samples 21-40 (using 17 rephrase few-shot examples)
python rephrase_rest.py  # After generating complete data, process samples 21-40

# 2. Manual review (batch input mode)
cp ../../../../automation/stage1_generation/tools/annotate_samples.py .
python annotate_samples.py --range 21-40
```

**Interactive Example** (batch input of unqualified sample numbers):
```
================================================================================
Comparison of Samples 21-40 - Please carefully review original and rephrased data
================================================================================

【Sample 21】
  Original premise:  The girl received a trophy.
  Rephrased premise:  The young lady was awarded a trophy for her achievement.
  choice1: She won a spelling bee.
  choice2: She made a new friend.
  question: cause
  label: 0

【Sample 22】
  Original premise:  The man broke his toe.
  Rephrased premise:  The gentleman fractured his toe.
  choice1: He got a hole in his sock.
  choice2: He dropped a hammer on his foot.
  question: cause
  label: 1

... (Displaying all 20 samples)...

【Sample 40】
  Original premise:  The man went to the bank.
  Rephrased premise:  The gentleman visited the financial institution.
  choice1: He wanted to cash a check.
  choice2: He wanted to buy groceries.
  question: cause
  label: 0

================================================================================

Enter the numbers of unqualified samples (21-40), separated by commas
Example: 23,27,35  means these samples are unqualified
If all qualified, press Enter directly

Unqualified sample numbers: 23,35  ← User enters all unqualified sample numbers at once

Statistics:
  Qualified samples: 18
  Unqualified samples: 2

【Breakpoint 2A: Processing Samples 21-40】

Executing rejection sampling...
✓ Rejection sampling completed: 18 rephrased + 2 original data

Generating validation few-shot examples...
✓ Generated 18 validation few-shot examples
✓ Validation few-shot saved: validation_checkpoints/validation_fewshot.json

✅ Breakpoint 2A completion summary:
  1. Rejection sampling: 18/20 samples kept rephrased
  2. Validation few-shot: generated 18 examples
```

### Breakpoint 2B: Testing Samples 41-80

**Purpose**: Test the accuracy of validation prompt

```bash
# 1. Manual annotation (batch input mode)
python annotate_samples.py --range 41-80
```

**Interactive Example** (batch input of unqualified sample numbers):
```
================================================================================
Comparison of Samples 41-80 - Please carefully review original and rephrased data
================================================================================

【Sample 41】
  Original premise:  The man went away for the weekend.
  Rephrased premise:  The gentleman departed for a weekend getaway.
  choice1: He wanted to spend time with family.
  choice2: He was bored.
  question: cause

【Sample 42】
  Original premise:  The girl laughed at the joke.
  Rephrased premise:  The young lady found the joke amusing.
  choice1: She found it funny.
  choice2: She was polite.
  question: cause

... (Displaying all 40 samples)...

【Sample 80】
  Original premise:  The child ate the cookie.
  Rephrased premise:  The youngster consumed the cookie.
  choice1: It looked delicious.
  choice2: It fell on the floor.
  question: cause

================================================================================

Enter the numbers of unqualified samples (41-80), separated by commas
Example: 43,47,55,72  means these samples are unqualified
If all qualified, press Enter directly

Unqualified sample numbers: 43,47,72  ← User enters all unqualified sample numbers at once

Statistics:
  Qualified samples: 37
  Unqualified samples: 3

【Breakpoint 2B: Processing Samples 41-80】

Executing rejection sampling...
✓ Rejection sampling completed: 37 rephrased + 3 original data

Generating test_set...
✓ Generated 40 test samples
  Ground Truth annotation statistics:
  - same (qualified): 37
  - not the same (unqualified): 3
✓ Test set saved: validation_checkpoints/validation_test_set.json
  Purpose: Test AI judge validation prompt accuracy

✅ Breakpoint 2B completion summary:
  1. Rejection sampling: 37/40 samples kept rephrased
  2. Test set: generated 40 annotated samples
  3. Ground Truth: same=37, not the same=3

Next step:
  Use test_set to test validation prompt accuracy
  Run: python generate_validation_test.py
```

### Breakpoint 3: Automated Validation

**Purpose**: Automatically generate and validate all remaining data

```bash
# 1. Generate remaining data (samples 81-N)
python rephrase_rest.py  # Complete all data

# 2. Automatic validation
python validate.py
```

**Validation Process**:
```
Total samples: 380
Validating: 100%|████████████████████| 380/380 [30:52<00:00]

✅ Validation complete!
   Passed validation: 364 (95.8%)
   Failed validation: 16 (4.2%)

Rejection sampling results:
   ✓ 364 rephrased data (quality qualified)
   ✗ 16 original data (replaced unqualified samples)

✅ Dataset completed!
```

---

## Common Scenarios

### Scenario 1: Processing New Dataset

```bash
# 1. Create configuration file
cp automation/configs/examples/stage1_example_copa.yaml \
   automation/configs/stage1_my_dataset.yaml

# 2. Modify fields and prompt
vim automation/configs/stage1_my_dataset.yaml

# 3. Generate scripts
python automation/stage1_generation/generator.py \
  --config automation/configs/stage1_my_dataset.yaml

# 4. Execute pipeline (exact same workflow)
cd Data_v2/synthetic/_shared/MyDataset/.../scripts
python rephrase_top20.py
python review_top20.py
# ... Other steps
```

### Scenario 2: Parameter Tuning Experiments (Exploratory)

Use **two_stage mode** (default) for parameter comparison, each parameter configuration needs to go through the complete two-stage process:

```bash
# Experiment A: temperature=0.7
# configs/copa_temp07.yaml
generation:
  strategy: "two_stage"  # Or omit (default)
  temperature: 0.7

python generator.py --config copa_temp07.yaml

# Experiment B: temperature=0.9
# configs/copa_temp09.yaml
generation:
  strategy: "two_stage"
  temperature: 0.9

python generator.py --config copa_temp09.yaml

# Batch system will automatically:
# 1. Detect different parameters
# 2. Create 2 independent datasets
# _shared/Copa/temp07_topp10_gpt4o/
# _shared/Copa/temp09_topp10_gpt4o/
```

### Scenario 3: Parameter Research (Prompt Already Determined)

🔥 **Use Case**: When you have already obtained usable prompts (including few-shot examples) through the first two-stage generation, and want to quickly explore the impact of different parameters (temperature, top_p, etc.) on synthetic data quality.

Use **direct_all mode** to skip the two-stage process and directly generate all data:

```bash
# 1. Prepare configuration file (reference configs/examples/stage1_direct_all_copa.yaml)
cp automation/configs/examples/stage1_direct_all_copa.yaml \
   automation/configs/stage1/temperature_study.yaml

# 2. Edit configuration file
vim automation/configs/stage1/temperature_study.yaml
```

**Key Configuration**:

```yaml
experiment:
  batch_id: "batch_20241230_temperature_study"
  purpose: "temperature_comparison"
  description: "Compare the impact of temperature parameter on synthetic data quality"

generation:
  strategy: "direct_all"  # 🔥 Key: Skip two-stage
  model: "gpt-4o"
  temperature: 0.7  # 🔬 Parameter to study
  top_p: 1.0        # 🔬 Parameter to study

  # ⚠️ Important: prompt must include complete few-shot examples
  # Do not use {{REPHRASE_FEWSHOT}} placeholder
  rephrase_prompt: |
    You are tasked with rephrasing...

    ### Few-shot Examples:
    Original premise: "My body cast a shadow over the grass."
    Rephrased premise: "A shadow appeared on the grass beside me."

    Original premise: "The woman tolerated her friend's difficult behavior."
    Rephrased premise: "The woman was patient with her friend's challenging attitude."

    # ... More few-shot examples (obtained from first two_stage generation)

    ### Your Task:
    Rephrase the following premise...
    **Original premise**: "{premise}"
    ...
```

**Generation and Execution**:

```bash
# 3. Generate scripts (only generates rephrase_all.py)
cd automation/stage1_generation
python generator.py --config ../configs/stage1/temperature_study.yaml

# Output example:
# Generation strategy: direct_all
# Experiment purpose: temperature_comparison
# Task: Copa
# Generation model: gpt-4o
# ================================================================================
# Generating rephrasing scripts...
#   ✓ rephrase_all.py
#   (direct_all mode: skipping top20 and rest scripts)
#
# Skipping validation script generation (direct_all mode)

# 4. Run generation directly (no breakpoint review needed)
cd Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/scripts
python rephrase_all.py

# Generate all 400 data samples at once
# Data saved in: Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa/copa_train.jsonl
```

**Quickly Compare Different Parameters**:

```bash
# Experiment 1: temperature=0.5
sed 's/temperature: .*/temperature: 0.5/' temperature_study.yaml > temp05.yaml
python generator.py --config ../configs/stage1/temp05.yaml
cd Data_v2/synthetic/_shared/Copa/temp05_topp10_gpt4o/scripts && python rephrase_all.py

# Experiment 2: temperature=0.7
sed 's/temperature: .*/temperature: 0.7/' temperature_study.yaml > temp07.yaml
python generator.py --config ../configs/stage1/temp07.yaml
cd Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/scripts && python rephrase_all.py

# Experiment 3: temperature=0.9
sed 's/temperature: .*/temperature: 0.9/' temperature_study.yaml > temp09.yaml
python generator.py --config ../configs/stage1/temp09.yaml
cd Data_v2/synthetic/_shared/Copa/temp09_topp10_gpt4o/scripts && python rephrase_all.py

# Automatically creates 3 independent datasets, all under the same batch:
# batch_20241230_temperature_study/Copa/temp05_topp10_gpt4o/
# batch_20241230_temperature_study/Copa/temp07_topp10_gpt4o/
# batch_20241230_temperature_study/Copa/temp09_topp10_gpt4o/
```

**Comparison of two_stage vs direct_all**:

| Feature | two_stage (Scenario 2) | direct_all (Scenario 3) |
|------|-------------------|-------------------|
| **Use Case** | Exploratory experiments, prompt undetermined | Prompt determined, parameter research |
| **Generation Process** | top20 → review → rest | Generate all directly |
| **Breakpoints** | 4 | 0 |
| **Manual Involvement** | Required (sample review) | Not required |
| **Generation Time** | Long (multiple reviews needed) | Short (one-time generation) |
| **Few-shot** | Extracted from review | Provided in configuration |
| **Validation** | Required | Optional (usually skipped) |

### Scenario 4: Parallel Processing of Multiple Datasets

Open multiple terminal windows:

```bash
# Window 1: Copa
cd Data_v2/synthetic/_shared/Copa/.../scripts
python rephrase_top20.py

# Window 2: BOOLQ
cd Data_v2/synthetic/_shared/BOOLQ/.../scripts
python rephrase_top20.py

# Window 3: CB
cd Data_v2/synthetic/_shared/CB/.../scripts
python rephrase_top20.py
```

All datasets use the same:
- ✅ API configuration
- ✅ Pipeline workflow
- ✅ Tool scripts

---

## Tool Usage

### review_top20.py - Review First 20 Samples

**Features**:
- Display original vs rephrased comparison
- Accept user y/n input
- Automatically execute rejection sampling
- Extract few-shot and inject into rephrase_rest.py

**Usage**:
```bash
cp ../../../../automation/stage1_generation/tools/review_top20.py .
python review_top20.py
```

**Input**:
- `y` - Accept this rephrase
- `n` - Reject this rephrase (will use original data)
- `quit` - Exit review

**Auto-adaptation**: Tool automatically recognizes dataset field structure and displays correctly

### annotate_samples.py - Annotate Samples 21-40

**Features**:
- Display comparison of samples 21-40
- Accept user same/not the same input
- Generate validation few-shot examples
- Inject into validate.py

**Usage**:
```bash
cp ../../../../automation/stage1_generation/tools/annotate_samples.py .
python annotate_samples.py
```

**Input**:
- `same` - Rephrase maintains same semantics
- `not the same` - Rephrase changes semantics
- `quit` - Exit annotation

### generate_validation_test.py - Test Validation Accuracy

**Features**:
- Use samples 41-80 as test set
- Compare system judgment vs manual judgment
- Calculate accuracy (needs ≥95%)
- Save test set

**Usage**:
```bash
cp ../../../../automation/stage1_generation/tools/generate_validation_test.py .
python generate_validation_test.py
```

**Input**:
- `same` / `not` - Confirm your judgment

---

## Troubleshooting

### Common Issues

#### Q1: API Call Failed

**Error**:
```
Error: Authentication failed
```

**Solution**:
```bash
# Check if API key is set
echo $OPENAI_API_KEY

# Reset
export OPENAI_API_KEY="your-api-key"

# Check API base URL
echo $OPENAI_API_BASE
```

#### Q2: Generated Script Can't Find Few-shot

**Error**:
```
NameError: name 'FEWSHOT_EXAMPLES' is not defined
```

**Cause**: Did not execute review_top20.py for few-shot injection

**Solution**:
```bash
# Must first run review_top20.py
python review_top20.py

# Then can run rephrase_rest.py
python rephrase_rest.py
```

#### Q3: Tool Script Shows Field Error

**Error**:
```
KeyError: 'premise'
```

**Cause**: Tool script did not correctly read configuration file

**Solution**:
```bash
# Ensure running in scripts directory
pwd  # Should be .../scripts/

# Ensure configuration file exists
ls ../generation_config.yaml

# If configuration file doesn't exist, regenerate scripts
```

#### Q4: Validation Accuracy < 95%

**Error**:
```
Accuracy: 87.5% < 95%
```

**Cause**: Validation prompt quality not good enough

**Solution**:
1. Check if samples 21-40 annotation is accurate
2. Adjust validation prompt
3. Re-run annotate_samples.py
4. Re-test

#### Q5: Parameter Fingerprint Conflict

**Hint**:
```
⚠️ Detected same parameter configuration, will reuse existing data
```

**Explanation**: This is normal! Batch system automatically deduplicates

**If you really want to generate new data**:
- Modify any key parameter (temperature, top_p, model, etc.)
- Or manually specify different semantic name

### Error Log Location

Error logs during generation:
```
Data_v2/synthetic/_shared/<Dataset>/<exp_name>/
├── scripts/
│   ├── rephrase_top20.py
│   └── ...
└── logs/  # If there are errors, check this directory
```

---

## Best Practices

### 1. Configuration File Management

```bash
automation/configs/
├── examples/          # Example configurations (do not modify)
└── stage1/           # Your configurations
    ├── copa_v1.yaml
    ├── copa_v2.yaml  # Version control
    └── boolq.yaml
```

### 2. API Configuration Strategy

```yaml
# Recommended: Hybrid strategy
generation:
  base_url: "https://api.openai.com/v1"  # High quality
  model: "gpt-4o"

validation:
  base_url: "https://api2.aigcbest.top/v1"  # Low cost
  model: "gpt-4o"
```

### 3. Breakpoint Review Recommendations

- **Breakpoint 1**: Strict review, acceptance rate recommended 80-90%
- **Breakpoint 2A**: Accurate annotation, this determines validation quality
- **Breakpoint 2B**: Patient testing, accuracy must be ≥95%

### 4. Data Quality Checks

Final dataset should satisfy:
- ✅ Rephrase rate ≥ 90%
- ✅ Validation pass rate ≥ 95%
- ✅ Manual spot check 10% samples quality qualified

---

## Related Documentation

- **[BATCH_GUIDE.md](BATCH_GUIDE.md)** - Batch Solution 3++ Detailed Guide
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Old Data Migration
- **[README.md](README.md)** - Project Overview

---

**Version History**:
- v2.0 (2025-12-30): Integrated API configuration, multi-dataset support documentation
- v1.0 (2024-12-29): Initial version
