# Complete Pipeline Interaction Flow Simulation

This document simulates the complete interaction process between users and the system in the entire synthetic data generation pipeline, covering **all 5 Datasets** and **two generation strategies**.

## 📚 Table of Contents

### Part 1: Two-Stage Mode (Detailed Complete Workflow)
- [Scenario 1: Copa Dataset (Two-Stage Mode)](#scenario-1-copa-Dataset-two-stage-Mode)
- [Scenario 2: BOOLQ Dataset (Two-Stage Mode)](#scenario-2-boolq-Dataset-two-stage-Mode)
- [Scenario 3: CB Dataset (Two-Stage Mode)](#scenario-3-cb-Dataset-two-stage-Mode)
- [Scenario 4: RTE Dataset (Two-Stage Mode)](#scenario-4-rte-Dataset-two-stage-Mode)
- [Scenario 5: ArcC Dataset (Two-Stage Mode)](#scenario-5-arcc-Dataset-two-stage-Mode)

### Part 2: Direct-All Mode (Parameter Study)
- [Scenario 6: Copa Dataset (Direct-All Mode)](#scenario-6-copa-Dataset-direct-all-Mode)
- [Scenario 7: BOOLQ Dataset (Direct-All Mode)](#scenario-7-boolq-Dataset-direct-all-Mode)
- [Scenario 8: CB Dataset (Direct-All Mode)](#scenario-8-cb-Dataset-direct-all-Mode)
- [Scenario 9: RTE Dataset (Direct-All Mode)](#scenario-9-rte-Dataset-direct-all-Mode)
- [Scenario 10: ArcC Dataset (Direct-All Mode)](#scenario-10-arcc-Dataset-direct-all-Mode)

### Appendix
- [Dataset Comparison Table](#Dataset-comparison-table)
- [Key Points Summary](#key-points-summary)

---

# Part 1: Two-Stage Mode (Detailed Complete Workflow)

## Scenario 1: Copa Dataset (Two-Stage Mode)

### Step 0: Prepare Configuration File

```bash
$ cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/automation

# Copy configuration template
$ cp configs/examples/stage1_full_example_copa.yaml configs/stage1/my_copa.yaml

# Edit Configuration file
$ vim configs/stage1/my_copa.yaml
```

**Configuration Content**:
```yaml
experiment:
  batch_id: "batch_20241230_copa_baseline"
  purpose: "copa_baseline"
  description: "Copa Dataset baseline experiment"

task_name: "Copa"
training_method: "mezo"

Dataset:
  task_name: "copa"
  Dataset_name: "Copa"
  input_path: "Data/original/Copa/copa_train.jsonl"
  original_dir: "Data/original/Copa"
  fields:
    - "premise"
    - "choice1"
    - "choice2"
    - "question"
    - "label"

generation:
  # Rewriter API configuration
  api_key: "sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c"
  base_url: "https://api2.aigcbest.top/v1"
  timeout: 120

  strategy: "two_stage"  # Default, exploratory experiment
  Model: "gpt-4o"
  temperature: 0.9
  top_p: 1.0
  field_to_rephrase: "premise"

  rephrase_prompt: |
    You are tasked with rephrasing the given premise...
    {{REPHRASE_FEWSHOT}}
    ... (complete prompt) ...

validation:
  # Judge API configuration
  api_key: "sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c"
  base_url: "https://api2.aigcbest.top/v1"
  timeout: 120

  Model: "gpt-4o"
  temperature: 0.0

  validation_prompt: |
    Task: Verify if the rephrased premise maintains consistency...
    {{VALIDATION_FEWSHOT}}
    ... (complete prompt) ...

  few_shot_examples: []  # Initially empty, auto-Generate after checkpoint 2A
```

### Step 1: Generate Scripts

```bash
$ cd automation/stage1_generation
$ python generator.py ../configs/stage1/my_copa.yaml
```

**System Output**:
```
================================================================================
Synthetic Data Generation Script Auto-Generator
================================================================================
Generation Strategy: two_stage
Experiment Purpose: copa_baseline
Experiment Description: Copa Dataset baseline experiment
Task: Copa
Training Method: mezo
Generation Model: gpt-4o
Validation Model: gpt-4o
================================================================================

================================================================================
🔧 Batch Experiment Management
================================================================================
Batch ID: batch_20241230_copa_baseline
Dataset: Copa
Parameter Fingerprint: a7b3c2d91f45
Semantic Name: temp09_topp10_gpt4o
================================================================================

🔍 Searching for fingerprint a7b3c2d91f45 in _shared/Copa/...
✓ No matching experiment found, will create new experiment

📂 Creating New Experiment
   Physical Storage: _shared/Copa/temp09_topp10_gpt4o
   Batch View: batch_20241230_copa_baseline/Copa/temp09_topp10_gpt4o

Output Directory: Data_v2/synthetic/_shared/Copa/temp09_topp10_gpt4o
Parameter Fingerprint: a7b3c2d91f45
Dataset Directory: Data_v2/synthetic/_shared/Copa/temp09_topp10_gpt4o/Copa

Generating rephrase scripts...
  ✓ rephrase_all.py
  ✓ rephrase_top20.py
  ✓ rephrase_rest.py

Generating validation scripts...
  ✓ validate.py

Saving configuration...
✓ Config Copy: generation_config.yaml
✓ Experiment Metadata: experiment_metadata.json
✓ README: README.md

================================================================================
Generation Complete!
================================================================================

Script Location: Data_v2/synthetic/_shared/Copa/temp09_topp10_gpt4o/scripts

Usage (two_stage Mode):
  1. Run generation: python .../rephrase_top20.py
  2. Run validation: python .../validate.py

Note: API configuration loaded from config file, no need to set environment variables
```

### Step 2: Checkpoint 1 - Generate First 20 Samples

```bash
$ cd Data_v2/synthetic/_shared/Copa/temp09_topp10_gpt4o/scripts
$ python rephrase_top20.py
```

**System Output**:
```
Loaded 400 original data samples
Output file: ../Copa/copa_train_top20.jsonl

Processing data: 100%|████████████████████| 20/20 [02:15<00:00]

Complete! Output: ../Copa/copa_train_top20.jsonl
```

### Step 3: Checkpoint 1 - Manual Review of First 20 Samples

```bash
$ cp ../../../../automation/stage1_generation/tools/review_top20.py .
$ python review_top20.py
```

**System Output**:
```
Loading data...
  Original: /home/ubuntu/.../Data/original/Copa/copa_train.jsonl
  Rephrased: ../Copa/copa_train_top20.jsonl

================================================================================
First 20 Samples Comparison - Please carefully review original vs rephrased data
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

... (Samples 3-20) ...

================================================================================

Enter the sample numbers (1-20) that are unqualified, separated by commas
Example: 3,7,15  means samples 3, 7, 15 are unqualified
If all are qualified, press Enter directly

Unqualified sample numbers:
```

**User Input**:
```
3,7,8
```

**System Continues Output**:
```
Statistics:
  Qualified samples: 17
  Unqualified samples: 3

Performing rejection sampling...
  Sample 3: Using original data (unqualified)
  Sample 7: Using original data (unqualified)
  Sample 8: Using original data (unqualified)

✓ Rejection sampling complete
  - 17 rephrased data samples (good quality)
  - 3 original data samples (replaced rejected samples)

Saving results...
✓ Saved: ../Copa/copa_train_top20.jsonl

Generating few-shot examples...
✓ Generated 17 few-shot examples

Injecting few-shot into rephrase_rest.py...
✓ Few-shot examples injected to: rephrase_rest.py
  Backup saved: rephrase_rest.py.backup

================================================================================
✅ Checkpoint 1 Complete!
================================================================================

Next step:
  Run: python rephrase_rest.py
```

### Step 4: Generate Remaining Data

```bash
$ python rephrase_rest.py
```

**System Output**:
```
Loaded 400 original data samples
Output file: ../Copa/copa_train_rest.jsonl

Processing data: 100%|████████████████████| 380/380 [45:30<00:00]

Complete! Output: ../Copa/copa_train_rest.jsonl
```

### Step 5: Merge Data

```bash
$ cat ../Copa/copa_train_top20.jsonl ../Copa/copa_train_rest.jsonl > ../Copa/copa_train.jsonl
$ wc -l ../Copa/copa_train.jsonl
```

**System Output**:
```
400 ../Copa/copa_train.jsonl
```

### Step 6: Checkpoint 2A - Process Samples 21-40

```bash
$ cp ../../../../automation/stage1_generation/tools/annotate_samples.py .
$ python annotate_samples.py --range 21-40
```

**System Output**：
```
Loading data...
  Original: /home/ubuntu/.../Data/original/Copa/copa_train.jsonl
  Rephrased: ../Copa/copa_train.jsonl

================================================================================
Samples 21-40 Comparison - Please carefully review original and rephrased data
================================================================================

【Sample 21】
  Original premise:  The girl received a trophy.
  Rephrased premise:  The young lady was awarded a trophy for her achievement.
  choice1: She won a spelling bee.
  choice2: She made a new friend.
  question: cause
  label: 0

... (Samples 22-40) ...

================================================================================

Please enter unqualified sample numbers (21-40), separate multiple numbers with commas
Example: 23,27,35  indicates these samples are unqualified
If all are qualified, press Enter directly

Unqualified sample numbers:
```

**User Input**：
```
23,35
```

**System Continues Output**：
```
Statistics:
  Qualified samples: 18
  Unqualified samples: 2

【Checkpoint 2A: Processing samples 21-40】

Performing rejection sampling...
  Sample23: Using original data (unqualified)
  Sample35: Using original data (unqualified)

✓ Rejection sampling complete: ../validation_checkpoints/samples_21_40_validated.jsonl

Generating validation few-shot examples...
✓ Generated 18 validation few-shot examples
✓ Validation few-shot saved: ../validation_checkpoints/validation_fewshot.json

✓ Annotation records saved: ../validation_checkpoints/samples_21_40_annotation.json

================================================================================
✅ Samples 21-40 processing complete!
================================================================================

✅ Checkpoint 2A Complete Summary:
  1. Rejection sampling: 18/20 samples kept rephrased version
  2. Validation few-shot: Generated 18 examples

Next step:
  Run: python annotate_samples.py --range 41-80
```

### Step7: Checkpoint 2B - Processing samples 41-80

```bash
$ python annotate_samples.py --range 41-80
```

**System Output** (After displaying samples 41-80):
```
Please enter unqualified sample numbers (41-80), separate multiple numbers with commas
Example: 43,47,55,72  indicates these samples are unqualified
If all are qualified, press Enter directly

Unqualified sample numbers:
```

**User Input**：
```
43,47,72
```

**System Continues Output**：
```
Statistics:
  Qualified samples: 37
  Unqualified samples: 3

【Checkpoint 2B: Processing samples 41-80】

Performing rejection sampling...
  Sample43: Using original data (unqualified)
  Sample47: Using original data (unqualified)
  Sample72: Using original data (unqualified)

✓ Rejection sampling complete: ../validation_checkpoints/samples_41_80_validated.jsonl

Generating test_set...
✓ Generated 40 test samples
  Ground Truth annotation Statistics:
  - same (qualified): 37
  - not the same (unqualified): 3
✓ Test set saved: ../validation_checkpoints/validation_test_set.json
  Purpose: Test accuracy of AI judge validation prompt

✓ Annotation records saved: ../validation_checkpoints/samples_41_80_annotation.json

================================================================================
✅ Samples 41-80 processing complete!
================================================================================

✅ Checkpoint 2B Complete Summary:
  1. Rejection sampling: 37/40 samples kept rephrased version
  2. Test set: Generated 40 annotated samples
  3. Ground Truth: same=37, not the same=3

Next step:
  Use test_set to test validation prompt accuracy
  Run: python generate_validation_test.py
```

### Step8: Test AI Judge Accuracy

```bash
$ cp ../../../../automation/stage1_generation/tools/generate_validation_test.py .
$ python generate_validation_test.py
```

**System Output**：
```
loadingtest set...
  File: ../validation_checkpoints/validation_test_set.json
  Samplecount: 40

loadingvalidationconfiguration...
  Model: gpt-4o
  Temperature: 0.0

Starting testAI judge...

test samples 1/40: 100%|████████████████████| 40/40 [05:30<00:00]

================================================================================
📊 test results
================================================================================
Totaltest samples: 40
AIjudged as same: 38
AIjudged as not the same: 2

andGround Truthcomparison:
  ✓ correctly judged: 39
  ✗ incorrectly judged: 1

accuracy: 97.5%

================================================================================
✅ test passed！accuracy ≥ 95%
================================================================================

Can proceed to Checkpoint 3（automatically validate remaining data）
```

### Step9: Checkpoint 3 - automatically validate remaining data

```bash
$ python validate.py
```

**System Output**：
```
loading training data...
  File: ../Copa/copa_train.jsonl
  Totalsamplescount: 400

processedsamples（Checkpoint 1and2）: 80
to be validatedsamples: 320 (samples81-400)

loadingvalidationconfiguration...
  Model: gpt-4o
  Temperature: 0.0
  Few-shot examples: 18

Starting automatic validation...

Validation progress: 100%|████████████████████| 320/320 [38:45<00:00]

================================================================================
📊 Validation results statistics
================================================================================
total validatedsamples: 320
judged as same: 307 (95.9%)
judged as not the same: 13 (4.1%)

Performing rejection sampling...
  ✓ kept rephrased version: 307 
  ✗ replaced with original: 13 

Saving final data...
✓ Saved: ../Copa/copa_train_final.jsonl

Copyingvalidationandtest set...
✓ copied: ../Copa/copa_validation.jsonl
✓ copied: ../Copa/copa_test.jsonl

================================================================================
✅ Dataset generation complete！
================================================================================

Final Dataset:
  training set: ../Copa/copa_train_final.jsonl (400)
    - rephrased data: 359 (89.8%)
    - original data: 41 (10.2%)
  validation set: ../Copa/copa_validation.jsonl
  test set: ../Copa/copa_test.jsonl

Dataset path: Data_v2/synthetic/_shared/Copa/temp09_topp10_gpt4o/Copa/
Can be directly used for MeZO training！
```

---

## Scenario2: BOOLQ Dataset (Two-Stage Mode)

### Dataset Characteristics

- **Task Type**: Boolean question answering（Yes/No）
- **Field to Rephrase**: `passage`（passage）
- **Other Fields**: `question`（question）、`label`（0=No, 1=Yes）

### Key Configuration Modifications

```yaml
experiment:
  purpose: "boolq_baseline"

task_name: "BOOLQ"
training_method: "mezo"

Dataset:
  task_name: "boolq"
  Dataset_name: "BOOLQ"
  input_path: "Data/original/BOOLQ/boolq_train.jsonl"
  original_dir: "Data/original/BOOLQ"
  fields:
    - "passage"
    - "question"
    - "label"

generation:
  field_to_rephrase: "passage"  # BOOLQrephrasedpassagefield

  rephrase_prompt: |
    You are tasked with rephrasing the given passage...
    {{REPHRASE_FEWSHOT}}

    **Original passage**: "{passage}"
    **Question**: "{question}"
    **Answer**: {"Yes" if label == 1 else "No"}

    **Directly output only one rephrased passage**:
```

### The Workflow is the same as Copa

executeStep1-10andCopasame，only the field name changed from`premise`to`passage`。

### Sample Comparison Examples (BOOLQ specific)

```
【Sample 1】
  Original passage:  The Supreme Court of the United States is the highest federal court...
  Rephrased passage:  As the highest federal court in the United States, the Supreme Court...
  question: is the supreme court the highest court in the united states
  label: 1 (Yes)

【Sample 2】
  Original passage:  A mule is the offspring of a male donkey and a female horse...
  Rephrased passage:  The hybrid animal known as a mule results from breeding a male donkey...
  question: can a mule reproduce
  label: 0 (No)
```

---

## Scenario3: CB Dataset (Two-Stage Mode)

### Dataset Characteristics

- **Task Type**: Natural language inference（NLI）
- **Field to Rephrase**: `hypothesis`（hypothesis）
- **Other Fields**: `premise`（premise）、`label`（0=entailment, 1=contradiction, 2=neutral）

### Key Configuration Modifications

```yaml
experiment:
  purpose: "cb_baseline"

task_name: "CB"
training_method: "mezo"

Dataset:
  task_name: "cb"
  Dataset_name: "CB"
  input_path: "Data/original/CB/cb_train.jsonl"
  original_dir: "Data/original/CB"
  fields:
    - "premise"
    - "hypothesis"
    - "label"

generation:
  field_to_rephrase: "hypothesis"  # CBrephrasedhypothesisfield

  rephrase_prompt: |
    You are tasked with rephrasing the given hypothesis...
    {{REPHRASE_FEWSHOT}}

    **Premise**: "{premise}"
    **Original hypothesis**: "{hypothesis}"
    **Label**: {["entailment", "contradiction", "neutral"][label]}

    **Directly output only one rephrased hypothesis**:
```

### Sample Comparison Examples (CB specific)

```
【Sample 1】
  Original premise:  It was a complex language. Not written down but handed down.
  Original hypothesis:  the language was written down
  Rephrased hypothesis:  the language existed in written form
  label: 1 (contradiction)

【Sample 2】
  Original premise:  Valence the void great quietness is there.
  Original hypothesis:  Great quietness is in the void.
  Rephrased hypothesis:  The void contains significant quietness.
  label: 0 (entailment)
```

---

## Scenario4: RTE Dataset (Two-Stage Mode)

### Dataset Characteristics

- **Task Type**: Natural language inference（Recognizing Textual Entailment）
- **Field to Rephrase**: `premise`（premise）
- **Other Fields**: `hypothesis`（hypothesis）、`label`（0=entailment, 1=not_entailment）
- **Data Example**:
  ```json
  {"premise": "No Weapons of Mass Destruction Found in Iraq Yet.",
   "hypothesis": "Weapons of Mass Destruction Found in Iraq.",
   "label": 1}
  ```

### Key Configuration Modifications

```yaml
experiment:
  batch_id: "batch_20241230_rte_baseline"
  purpose: "rte_baseline"
  description: "RTEDatasetbaseline experiment"

task_name: "RTE"
training_method: "mezo"

Dataset:
  task_name: "rte"
  Dataset_name: "RTE"
  input_path: "Data/original/RTE/rte_train.jsonl"
  original_dir: "Data/original/RTE"
  fields:
    - "premise"
    - "hypothesis"
    - "label"

generation:
  strategy: "two_stage"
  Model: "gpt-4o"
  temperature: 0.9
  top_p: 1.0
  field_to_rephrase: "premise"  # RTErephrasedpremisefield

  rephrase_prompt: |
    You are tasked with rephrasing the given premise for a textual entailment task.
    {{REPHRASE_FEWSHOT}}

    **Original premise**: "{premise}"
    **Hypothesis**: "{hypothesis}"
    **Label**: {["entailment", "not_entailment"][label]}

    **Directly output only one rephrased premise**:

validation:
  Model: "gpt-4o"
  temperature: 0.0

  validation_prompt: |
    Task: Verify if the rephrased premise maintains semantic consistency...
    {{VALIDATION_FEWSHOT}}

    **Original premise**: "{original_premise}"
    **Rephrased premise**: "{rephrased_premise}"
    **Hypothesis**: "{hypothesis}"

    Is the rephrased premise semantically equivalent? Answer "same" or "not the same":

  few_shot_examples: []
```

### Sample Comparison Examples (RTE specific)

```
【Sample 1】
  Original premise:  No Weapons of Mass Destruction Found in Iraq Yet.
  Rephrased premise:  Weapons of mass destruction have not been discovered in Iraq so far.
  hypothesis: Weapons of Mass Destruction Found in Iraq.
  label: 1 (not_entailment)

【Sample 2】
  Original premise:  The European Union says the Greek Cypriot community will be admitted to the EU.
  Rephrased premise:  According to the European Union, Greek Cypriots will join the EU.
  hypothesis: Cyprus was divided into two parts in 1974.
  label: 1 (not_entailment)

【Sample 3】
  Original premise:  Russia's Mikhail Khodorkovsky, the former head of oil giant Yukos, was convicted.
  Rephrased premise:  Mikhail Khodorkovsky, who previously led the major oil company Yukos, was found guilty.
  hypothesis: Mikhail Khodorkovsky was Russia's richest man.
  label: 1 (not_entailment)
```

### Complete Execution Workflow

Workflow andCopacompletely the same（Step1-10），only need to：
1. PrepareRTEConfiguration file
2. execute `python generator.py ../configs/stage1/my_rte.yaml`
3. according toCheckpoint 1→Checkpoint 2A→Checkpoint 2B→Checkpoint 3execute sequentially

---

## Scenario5: ArcC Dataset (Two-Stage Mode)

### Dataset Characteristics

- **Task Type**: multiple choice（Scientific reasoning）
- **Field to Rephrase**: `question`（question）
- **Other Fields**: `choices`（choices）、`answerKey`（answer）
- **Data Example**:
  ```json
  {"id": "Mercury_SC_415702",
   "question": "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?",
   "choices": {
     "text": ["dry palms", "wet palms", "palms covered with oil", "palms covered with lotion"],
     "label": ["A", "B", "C", "D"]
   },
   "answerKey": "A"}
  ```

### Key Configuration Modifications

```yaml
experiment:
  batch_id: "batch_20241230_arcc_baseline"
  purpose: "arcc_baseline"
  description: "ARC-ChallengeDatasetbaseline experiment"

task_name: "ArcC"
training_method: "mezo"

Dataset:
  task_name: "arc_challenge"
  Dataset_name: "ArcC"
  input_path: "Data/original/ArcC_Cloze/ARC-Challenge_train.jsonl"
  original_dir: "Data/original/ArcC_Cloze"
  fields:
    - "question"
    - "choices"
    - "answerKey"

generation:
  strategy: "two_stage"
  Model: "gpt-4o"
  temperature: 0.9
  top_p: 1.0
  field_to_rephrase: "question"  # ArcCrephrasedquestionfield

  rephrase_prompt: |
    You are tasked with rephrasing multiple-choice science questions.
    {{REPHRASE_FEWSHOT}}

    **Original question**: "{question}"
    **Choices**: {', '.join([f"{label}: {text}" for label, text in zip(choices['label'], choices['text'])])}
    **Correct answer**: {answerKey}

    **Directly output only one rephrased question**:

validation:
  Model: "gpt-4o"
  temperature: 0.0

  validation_prompt: |
    Task: Verify if the rephrased question maintains the same meaning...
    {{VALIDATION_FEWSHOT}}

    **Original question**: "{original_question}"
    **Rephrased question**: "{rephrased_question}"
    **Choices**: {choices}
    **Correct answer**: {answerKey}

    Is the rephrased question semantically equivalent? Answer "same" or "not the same":

  few_shot_examples: []
```

### Sample Comparison Examples (ArcC specific)

```
【Sample 1】
  Original question:  George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?
  Rephrased question:  To rapidly warm his hands through rubbing, which type of skin surface should George use to generate maximum heat?
  choices: A: dry palms, B: wet palms, C: palms covered with oil, D: palms covered with lotion
  answerKey: A

【Sample 2】
  Original question:  A student wants to look under a heavy rock. Which simple machine would be BEST to use to lift the rock?
  Rephrased question:  What simple machine would be most effective for a student attempting to lift a large, heavy rock?
  choices: A: Wheel and axle, B: Lever, C: Inclined plane, D: Screw
  answerKey: B

【Sample 3】
  Original question:  Which of these do scientists most likely do when studying the interaction of animals in their natural habitat?
  Rephrased question:  When observing animals in their natural environment, which activity would scientists typically perform?
  choices: A: design a mathematical Model, B: perform a controlled experiment, C: collect data, D: formulate a hypothesis
  answerKey: C
```

### Complete Execution Workflow

Workflow andCopacompletely the same（Step1-10），only the rephrased field is`question`。

---

# Part 2: Direct-AllMode（Parametersstudy）

## Scenario6: Copa Dataset (Direct-All Mode)

### Use Case

Already through the first two-stage generation, obtained usable prompt and few-shot examples, now want to quickly explore different temperature parameters (0.5, 0.7, 0.9) impact on synthetic data quality.

### Step1: PrepareDirect-Allconfiguration

```bash
$ cd automation/configs/stage1
$ cp ../examples/stage1_direct_all_copa.yaml temperature_05.yaml
$ vim temperature_05.yaml
```

**Configuration content**：
```yaml
experiment:
  batch_id: "batch_20241230_temperature_study"
  purpose: "temperature_comparison"
  description: "Comparingtemperature=0.5/0.7/0.9 forCopaimpact on synthetic data quality"

task_name: "Copa"
training_method: "mezo"

Dataset:
  task_name: "copa"
  Dataset_name: "Copa"
  input_path: "Data/original/Copa/copa_train.jsonl"
  original_dir: "Data/original/Copa"
  fields:
    - "premise"
    - "choice1"
    - "choice2"
    - "question"
    - "label"

generation:
  # Rewriter APIconfiguration
  api_key: "sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c"
  base_url: "https://api2.aigcbest.top/v1"
  timeout: 120

  strategy: "direct_all"  # 🔥 Direct full generation
  Model: "gpt-4o"
  temperature: 0.5  # 🔬 Parametersvariable
  top_p: 1.0
  field_to_rephrase: "premise"

  # ⚠️ must containcompletefew-shot（from the firsttwo-stageGenerateobtained from）
  rephrase_prompt: |
    You are tasked with rephrasing the given premise...

    ### Few-shot Examples:
    Original premise: "My body cast a shadow over the grass."
    Rephrased premise: "A shadow appeared on the grass beside me."

    Original premise: "The woman tolerated her friend's difficult behavior."
    Rephrased premise: "The woman was patient with her friend's challenging attitude."

    ... (complete17 few-shot examples) ...

    ### Your Task:
    **Original premise**: "{premise}"
    **Choice 1**: "{choice1}"
    **Choice 2**: "{choice2}"
    **Question**: "{question}"
    **Correct answer**: "{choice1 if label == 0 else choice2}"

    **Directly output only one rephrased premise**:

# ⚠️ direct_allModedoes not requirevalidationconfiguration
```

### Step2: Generate3differenttemperature configuration

```bash
# Temperature 0.5
$ cp temperature_05.yaml temperature_05.yaml

# Temperature 0.7
$ sed 's/temperature: 0.5/temperature: 0.7/' temperature_05.yaml > temperature_07.yaml

# Temperature 0.9
$ sed 's/temperature: 0.5/temperature: 0.9/' temperature_05.yaml > temperature_09.yaml
```

### Step3: Generateand run（Temperature 0.5）

```bash
$ cd ../stage1_generation
$ python generator.py ../configs/stage1/temperature_05.yaml

$ cd Data_v2/synthetic/_shared/Copa/temp05_topp10_gpt4o/scripts
$ python rephrase_all.py
```

**System Output**：
```
Loaded 400 original data samples
Output file: ../Copa/copa_train.jsonl

Processing data: 100%|████████████████████| 400/400 [48:30<00:00]

Complete! Output: ../Copa/copa_train.jsonl
```

### Step4: duplicateGenerateothertemperature

```bash
# Temperature 0.7
$ cd ../../stage1_generation
$ python generator.py ../configs/stage1/temperature_07.yaml
$ cd Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/scripts
$ python rephrase_all.py

# Temperature 0.9
$ cd ../../stage1_generation
$ python generator.py ../configs/stage1/temperature_09.yaml
$ cd Data_v2/synthetic/_shared/Copa/temp09_topp10_gpt4o/scripts
$ python rephrase_all.py
```

### Batch systemautomatically managed

```
Data_v2/synthetic/
├── _shared/                              # Physical storage
│   └── Copa/
│       ├── temp05_topp10_gpt4o/         # Temperature 0.5
│       │   └── Copa/copa_train.jsonl
│       ├── temp07_topp10_gpt4o/         # Temperature 0.7
│       │   └── Copa/copa_train.jsonl
│       └── temp09_topp10_gpt4o/         # Temperature 0.9
│           └── Copa/copa_train.jsonl
│
└── batch_20241230_temperature_study/    # Batch view（symbolic link）
    └── Copa/
        ├── temp05_topp10_gpt4o -> ../../_shared/Copa/temp05_topp10_gpt4o/
        ├── temp07_topp10_gpt4o -> ../../_shared/Copa/temp07_topp10_gpt4o/
        └── temp09_topp10_gpt4o -> ../../_shared/Copa/temp09_topp10_gpt4o/
```

---

## Scenario7: BOOLQ Dataset (Direct-All Mode)

### Configuration filekey differences

```yaml
experiment:
  batch_id: "batch_20241230_boolq_topp_study"
  purpose: "boolq_topp_comparison"
  description: "Comparingtop_p=0.8/0.9/1.0 forBOOLQimpact on synthetic data quality"

task_name: "BOOLQ"

Dataset:
  task_name: "boolq"
  Dataset_name: "BOOLQ"
  input_path: "Data/original/BOOLQ/boolq_train.jsonl"
  fields:
    - "passage"
    - "question"
    - "label"

generation:
  strategy: "direct_all"
  temperature: 0.9  # fixed
  top_p: 0.8  # 🔬 studyvariable
  field_to_rephrase: "passage"

  rephrase_prompt: |
    You are tasked with rephrasing the given passage...

    ### Few-shot Examples:
    Original passage: "The Supreme Court of the United States is..."
    Rephrased passage: "As the highest federal court in..."

    ... (17completeexamples) ...

    **Original passage**: "{passage}"
    **Question**: "{question}"
    **Answer**: {"Yes" if label == 1 else "No"}

    **Directly output only one rephrased passage**:
```

### execution flow

```bash
# Generatetop_p=0.8 configuration
$ python generator.py ../configs/stage1/boolq_topp08.yaml
$ cd Data_v2/synthetic/_shared/BOOLQ/temp09_topp08_gpt4o/scripts
$ python rephrase_all.py

# Generatetop_p=0.9 configuration
$ python generator.py ../configs/stage1/boolq_topp09.yaml
$ cd Data_v2/synthetic/_shared/BOOLQ/temp09_topp09_gpt4o/scripts
$ python rephrase_all.py

# Generatetop_p=1.0 configuration
$ python generator.py ../configs/stage1/boolq_topp10.yaml
$ cd Data_v2/synthetic/_shared/BOOLQ/temp09_topp10_gpt4o/scripts
$ python rephrase_all.py
```

---

## Scenario8: CB Dataset (Direct-All Mode)

### Configuration filekey differences

```yaml
experiment:
  batch_id: "batch_20241230_cb_Model_study"
  purpose: "cb_Model_comparison"
  description: "Comparinggpt-4o vs gpt-4o-mini forCBimpact on synthetic data quality"

task_name: "CB"

Dataset:
  task_name: "cb"
  Dataset_name: "CB"
  input_path: "Data/original/CB/cb_train.jsonl"
  fields:
    - "premise"
    - "hypothesis"
    - "label"

generation:
  strategy: "direct_all"
  Model: "gpt-4o"  # 🔬 studyvariable（can be changedasgpt-4o-mini）
  temperature: 0.9
  top_p: 1.0
  field_to_rephrase: "hypothesis"

  rephrase_prompt: |
    You are tasked with rephrasing the given hypothesis...

    ### Few-shot Examples:
    Premise: "It was a complex language. Not written down but handed down."
    Original hypothesis: "the language was written down"
    Rephrased hypothesis: "the language existed in written form"

    ... (17completeexamples) ...

    **Premise**: "{premise}"
    **Original hypothesis**: "{hypothesis}"
    **Label**: {["entailment", "contradiction", "neutral"][label]}

    **Directly output only one rephrased hypothesis**:
```

### execution flow

```bash
# GPT-4o
$ python generator.py ../configs/stage1/cb_gpt4o.yaml
$ cd Data_v2/synthetic/_shared/CB/temp09_topp10_gpt4o/scripts
$ python rephrase_all.py

# GPT-4o-mini
$ python generator.py ../configs/stage1/cb_gpt4o_mini.yaml
$ cd Data_v2/synthetic/_shared/CB/temp09_topp10_gpt4omini/scripts
$ python rephrase_all.py
```

---

## Scenario9: RTE Dataset (Direct-All Mode)

### Configuration filekey differences

```yaml
experiment:
  batch_id: "batch_20241230_rte_temp_study"
  purpose: "rte_temperature_comparison"
  description: "Comparingtemperature=0.5/0.7/0.9 forRTEimpact on synthetic data quality"

task_name: "RTE"

Dataset:
  task_name: "rte"
  Dataset_name: "RTE"
  input_path: "Data/original/RTE/rte_train.jsonl"
  fields:
    - "premise"
    - "hypothesis"
    - "label"

generation:
  strategy: "direct_all"
  Model: "gpt-4o"
  temperature: 0.5  # 🔬 studyvariable
  top_p: 1.0
  field_to_rephrase: "premise"

  rephrase_prompt: |
    You are tasked with rephrasing the given premise for textual entailment...

    ### Few-shot Examples:
    Original premise: "No Weapons of Mass Destruction Found in Iraq Yet."
    Rephrased premise: "Weapons of mass destruction have not been discovered in Iraq so far."

    ... (17completeexamples) ...

    **Original premise**: "{premise}"
    **Hypothesis**: "{hypothesis}"
    **Label**: {["entailment", "not_entailment"][label]}

    **Directly output only one rephrased premise**:
```

### execution flow

```bash
# Temperature 0.5
$ python generator.py ../configs/stage1/rte_temp05.yaml
$ cd Data_v2/synthetic/_shared/RTE/temp05_topp10_gpt4o/scripts
$ python rephrase_all.py

# Temperature 0.7
$ python generator.py ../configs/stage1/rte_temp07.yaml
$ cd Data_v2/synthetic/_shared/RTE/temp07_topp10_gpt4o/scripts
$ python rephrase_all.py

# Temperature 0.9
$ python generator.py ../configs/stage1/rte_temp09.yaml
$ cd Data_v2/synthetic/_shared/RTE/temp09_topp10_gpt4o/scripts
$ python rephrase_all.py
```

---

## Scenario10: ArcC Dataset (Direct-All Mode)

### Configuration filekey differences

```yaml
experiment:
  batch_id: "batch_20241230_arcc_temp_study"
  purpose: "arcc_temperature_comparison"
  description: "Comparingtemperature=0.5/0.7/0.9 forArcCimpact on synthetic data quality"

task_name: "ArcC"

Dataset:
  task_name: "arc_challenge"
  Dataset_name: "ArcC"
  input_path: "Data/original/ArcC_Cloze/ARC-Challenge_train.jsonl"
  fields:
    - "question"
    - "choices"
    - "answerKey"

generation:
  strategy: "direct_all"
  Model: "gpt-4o"
  temperature: 0.5  # 🔬 studyvariable
  top_p: 1.0
  field_to_rephrase: "question"

  rephrase_prompt: |
    You are tasked with rephrasing multiple-choice science questions...

    ### Few-shot Examples:
    Original question: "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?"
    Rephrased question: "To rapidly warm his hands through rubbing, which type of skin surface should George use to generate maximum heat?"

    ... (17completeexamples) ...

    **Original question**: "{question}"
    **Choices**: {', '.join([f"{label}: {text}" for label, text in zip(choices['label'], choices['text'])])}
    **Correct answer**: {answerKey}

    **Directly output only one rephrased question**:
```

### execution flow

```bash
# Temperature 0.5
$ python generator.py ../configs/stage1/arcc_temp05.yaml
$ cd Data_v2/synthetic/_shared/ArcC/temp05_topp10_gpt4o/scripts
$ python rephrase_all.py

# Temperature 0.7
$ python generator.py ../configs/stage1/arcc_temp07.yaml
$ cd Data_v2/synthetic/_shared/ArcC/temp07_topp10_gpt4o/scripts
$ python rephrase_all.py

# Temperature 0.9
$ python generator.py ../configs/stage1/arcc_temp09.yaml
$ cd Data_v2/synthetic/_shared/ArcC/temp09_topp10_gpt4o/scripts
$ python rephrase_all.py
```

---

# Appendix

## Dataset Comparison Table

| Scenario | Dataset | Mode | rephrasedfield | otherfield | checkpointcount | Total time（estimated） |
|------|--------|------|----------|----------|--------|---------------|
| Scenario1 | Copa | Two-Stage | premise | choice1, choice2, question, label | 3 | ~90minutes |
| Scenario2 | BOOLQ | Two-Stage | passage | question, label | 3 | ~90minutes |
| Scenario3 | CB | Two-Stage | hypothesis | premise, label | 3 | ~90minutes |
| Scenario4 | RTE | Two-Stage | premise | hypothesis, label | 3 | ~90minutes |
| Scenario5 | ArcC | Two-Stage | question | choices, answerKey | 3 | ~90minutes |
| Scenario6 | Copa | Direct-All | premise | choice1, choice2, question, label | 0 | ~50minutes |
| Scenario7 | BOOLQ | Direct-All | passage | question, label | 0 | ~50minutes |
| Scenario8 | CB | Direct-All | hypothesis | premise, label | 0 | ~50minutes |
| Scenario9 | RTE | Direct-All | premise | hypothesis, label | 0 | ~50minutes |
| Scenario10 | ArcC | Direct-All | question | choices, answerKey | 0 | ~50minutes |

## manual participation timecomparison

### Two-StageMode（Scenario1-5）:
- **Checkpoint 1review**: review20samples + input numbers ≈ **1minutes**
- **Checkpoint 2Areview**: review20samples + input numbers ≈ **1minutes**
- **Checkpoint 2Breview**: review40samples + input numbers ≈ **2minutes**
- **Totalmanual time**: ~**4minutes**

### Direct-AllMode（Scenario6-10）:
- **no manual participation required** ✅
- completelyautomated，suitable forParametersstudy

## Key pointsSummary

### 1. batch inputMode
- Useronly need toinput unqualifiednumbers（such as：`3,7,12`），no need to confirm one by one
- significantly reducemanual interaction time

### 2. Automatic rejection sampling
- System automaticallyreplace unqualifiedsamplesasoriginal data
- All3checkpoint（1-20, 21-40, 41-80）all executerejection sampling

### 3. Automatic few-shotGenerate
- Checkpoint 1：from17qualifiedsamplesGeneraterephrase few-shot
- Checkpoint 2A：from18qualifiedsamplesGeneratevalidation few-shot

### 4. automatic annotation
- Allsame/not the sameannotation bySystem automaticallycomplete
- Generatetest_setfor testingAI judgeaccuracy

### 5. multipleDatasetzero-code support
- 5Dataset（Copa, BOOLQ, CB, RTE, ArcC）
- only need tomodifyConfiguration file field names andprompt
- no need to modify any code

### 6. Parametersdeduplication（Batch solution3++）
- Automatically detectsameParametersconfiguration
- Physical storagein`_shared/`，avoid duplicationGenerate
- Batch view organizes experiments through symbolic links

### 7. twoGeneratestrategy
- **Two-Stage**: Exploratory experiment，need to determinepromptandfew-shot
- **Direct-All**: promptconfirmed，quicklyParametersstudy

## Usage recommendations

1. **First experiment**：useTwo-StageModedetermine the bestpromptandfew-shot examples
2. **Parameters study**: After obtaining few-shot from Two-Stage, use Direct-All Mode to quickly generate data with different parameter configurations
3. **manual review**：carefully reviewfront80samples，ensureAI judgeaccuracy≥95%
4. **Batch management**：use`list_batches.py`and other tools to view and manage experiments
5. **Data reuse**：Make good use ofBatch systemParametersdeduplication feature，avoid duplicationGenerate
