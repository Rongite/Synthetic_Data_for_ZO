# Stage 1CompleteWorkflowprocess：Manual checkpoint + Parameter tuningRepeat use 

versionDocumentDescriptionNewAutomated System such as whatsupport：
1. **First timeGenerate**：contain3must need Manual checkpointValidate
2. **Parameter tuningExperiment**：base at alreadyValidateprompt，Quick parameter tuningRepeat use 

---

## ✅ ImplementationStatus

### ImplementedTool

| Tool | location | Feature | Status |
|------|------|------|------|
| `generator.py` | `automation/stage1_generation/` | Generaterephrase and validationScript | ✅ **alreadyFix** |
| `review_top20.py` | `automation/stage1_generation/tools/` | Checkpoint1：manual review front 20Sample | ✅ Implemented |
| `extract_samples.py` | `automation/stage1_generation/tools/` | ExtractSpecifyScopeSample | ✅ Implemented |
| `annotate_samples.py` | `automation/stage1_generation/tools/` | Checkpoint2：Manual annotation21-80Sample | ✅ Implemented |
| `generate_validation_test.py` | `automation/stage1_generation/tools/` | GeneratejudgerTestScript | ✅ Implemented |

### 🔧 Key fixes

**generator.py (validate.pyGenerateLogic)**:
- ✅ **alreadyFixExclude21-40SampleLogic**
- Generate`validate.py`current in  will skipSample21-40（Index20-39）
- theseSample use asjudgerfew-shot examples，Should not by judgerValidate（AvoidDataLeak）
- Fixlocation：`generator.py:300-308`

```python
# 🔴 ExcludeSample21-40（Index20-39）
if 20 <= i < 40:
    # DirectlyuseSyntheticData，WithoutjudgerValidate
    out_file.write(json.dumps(synthetic, ensure_ascii=False) + "\n")
    correct_count += 1
    total_count += 1
    continue
```

---

## ⚠️ ImportantDescription

### OnlySynthetictrainData

**PipelineOnly will Synthetic/RephrasetrainingData（train.jsonl），validation and testDataDirectly from originalDataCollectCopy**：

- ✅ **{dataset}_train.jsonl** → SyntheticData（alreadyrephrase + validation + rejection sampling）
- 📋 **{dataset}_validation.jsonl** → originalData（ from  Data/original/ Copy）
- 📋 **{dataset}_test.jsonl** → originalData（ from  Data/original/ Copy）

 this Do thisYes as ：
1. **Maintainevaluationstandardation** - validation and testDataMaintainoriginalStatus，EnsureFairEvaluate
2. **Experiment Results can compare** - DifferentExperimentusesameevaluationData
3. **SymbolResearch conventions** - Only in trainingstageuseSyntheticDataenhance

**Automaticprocess**: `validate.py`  in ValidatetrainData back ， will Automatic from originalDataCollectCopyvalidation and testFile。

---

## 🗂️ BatchSolution3++ - smart can ExperimentManage

### whatYesBatchSolution？

BatchSolution3++pass**Physicalstorage and LogicviewGraphSeparation**，ImplementationMoreParameterExperimentsmart can Manage and AutomaticremoveHeavy。

**coreMechanism**:
- **Physicalstorage (_shared/)**: StoreActualData， according to ParameterFingerprintremoveHeavy
- **LogicviewGraph (batch_*)**: passSymbolIDLinkOrganizeExperiment， according to time/objectiveGroup

**ParameterremoveHeavy**: sameParameter configurationDataOnlyGenerateOnce，DifferentbatchcanRepeat use 

### DirectorystructureExample

```
Data_v2/synthetic/
├── _shared/                                    # PhysicalData（removeHeavy）
│   └── Copa/
│       ├── temp05_topp10_gpt4o/               # ActualData
│       ├── temp07_topp09_gpt4o/
│       └── temp09_topp10_gpt4o/
│
├── batch_20241229_temperature/                 # Batch 1: TemperatureExperiment
│   └── Copa/
│       ├── temp05_topp10_gpt4o -> ../../_shared/...
│       ├── temp07_topp10_gpt4o -> ../../_shared/...
│       └── temp09_topp10_gpt4o -> ../../_shared/...
│
└── batch_20241230_topp/                        # Batch 2: top_pExperiment
    └── Copa/
        ├── temp07_topp08_gpt4o -> ../../_shared/...
        └── temp07_topp09_gpt4o -> ../../_shared/...  # Repeat use ！
```

### ConfigurationFileSettings

 in ConfigurationFile in add `experiment.batch_id`:

```yaml
experiment:
  # Batch ID（ can select，notSpecifythenAutomaticGenerate）
  batch_id: "batch_20241229_temperature"
  purpose: "temperature_study"
  description: "ResearchtemperatureParameter for SyntheticDataqualityImpact"

generation:
  model: "gpt-4o"
  temperature: 0.7  # Experimentvariable
  # ...
```

### AutomaticremoveHeavyOriginalmanage

When youRun `generator.py` Sometimes：

1. **CalculateParameterFingerprint**: base at allImpactDataGenerateParameter（Model、temperature、top_p、prompts etc.）
2. **FindalreadyhasData**:  in  `_shared/{Dataset}/`  in SearchsameFingerprint
3. **Repeat use  or New**:
   - find to sameFingerprint → Repeat use PhysicalData，CreatebatchSymbolIDLink
   - Not found to  → CreateNewPhysicalDirectory，GenerateData

**Sectionsaveresource**: NoneneedRepeatGeneratesameParameterData，SectionsaveAPIadjust use Cost and time

### BatchManageTool

```bash
# Columnoutallbatch
python automation/stage1_generation/batch_tools/list_batches.py --verbose

# ViewbatchDetails
python automation/stage1_generation/batch_tools/list_batch_experiments.py \
    batch_20241229_temperature --verbose

# ViewPhysicalDatauseCase
python automation/stage1_generation/batch_tools/list_shared_experiments.py \
    --dataset Copa --verbose

# compareExperimentParameter
python automation/stage1_generation/batch_tools/compare_experiments.py \
    --shared Copa/temp07_topp10_gpt4o \
    --shared Copa/temp09_topp10_gpt4o
```

**DetailedDescription**: See [BATCH_GUIDE.md](../../BATCH_GUIDE.md)

---

## WorkflowProcess overview

```
First timeGenerate（hasManual checkpoint）              Parameter tuningExperiment（NoneManual checkpoint）
┌────────────────────────┐          ┌────────────────────────┐
│ 1. CreatedraftConfiguration        │          │ 1. base at validatedTemplate    │
│    (ManualWriteinitialprompt) │          │    CreateExperimentConfiguration         │
└───────────┬────────────┘          └───────────┬────────────┘
            │                                   │
            v                                   v
┌────────────────────────┐          ┌────────────────────────┐
│ 2. GenerateScript             │          │ 2. GenerateScript             │
│    (generator.py)      │          │    (generator.py)      │
└───────────┬────────────┘          └───────────┬────────────┘
            │                                   │
            v                                   v
┌────────────────────────┐          ┌────────────────────────┐
│ 🔴 Checkpoint1: Reviewtop20    │          │ 3. DirectlyRun             │
│    → Generatefew-shot      │          │    rephrase_all.py     │
└───────────┬────────────┘          │    (Noneneedmanual review)      │
            │                       └───────────┬────────────┘
            v                                   │
┌────────────────────────┐                     │
│ 3. GeneraterestData         │                     │
└───────────┬────────────┘                     │
            │                                   │
            v                                   │
┌────────────────────────┐                     │
│ 🔴 Checkpoint2: Annotate21-80    │                     │
│    → Generatevalidation    │                     │
│       prompt few-shot  │                     │
└───────────┬────────────┘                     │
            │                                   │
            v                                   v
┌────────────────────────┐          ┌────────────────────────┐
│ 🔴 Checkpoint3: Testprompt   │          │ 4. usealreadyValidate         │
│    → Tunedirect to ≥95%      │          │    validation prompt   │
└───────────┬────────────┘          │    ValidateData             │
            │                       └───────────┬────────────┘
            v                                   │
┌────────────────────────┐                     │
│ 4. batchValidateData         │                     │
└───────────┬────────────┘                     │
            │                                   │
            v                                   v
┌────────────────────────┐          ┌────────────────────────┐
│ 5. Archive as validatedTemplate  │          │ 5. Complete！               │
│    ( can Repeat use )            │          │    comparisonDifferentVersionquality     │
└────────────────────────┘          └────────────────────────┘
```

---

## ScenarioA：First timeGenerate（needManualValidate）

###  front setCondition
- originalDataalreadyAccurateprepare：`Data/original/{Task}/{task}_train.jsonl`
- You haveManualWriteinitialversionprompt（Nonefew-shot）

### Step 1: CreatedraftConfiguration

CreateConfigurationFile：`automation/configs/stage1/drafts/copa_mezo_v1_draft.yaml`

```yaml
task_name: "Copa"
training_method: "mezo"
version: "v1"

dataset:
  task_name: "copa"
  input_path: "Data/original/Copa/copa_train.jsonl"
  fields: ["premise", "choice1", "choice2", "question", "label"]

generation:
  model: "gpt-4o"
  temperature: 0.5
  field_to_rephrase: "premise"

  # ManualWriteinitialprompt（Nonefew-shot）
  rephrase_prompt: |
    You are tasked with rephrasing...
    （ManualWritepromptContent）

validation:
  model: "gpt-4o"
  temperature: 0.0

  # ManualWriteinitialvalidation prompt（Nonefew-shot）
  validation_prompt: |
    Judge if the rephrased premise...
    （ManualWritepromptContent）

  # tempSometimeskeepNull， back continueAutomaticGenerate
  few_shot_examples: []
```

### Step 2: GenerateScript

```bash
python automation/stage1_generation/generator.py \
       automation/configs/stage1/drafts/copa_mezo_v1_draft.yaml
```

**output**：
```
Data_v2/synthetic/Copa_mezo_gpt4o_v1/
├── scripts/
│   ├── rephrase_top20.py
│   ├── rephrase_rest.py
│   ├── rephrase_all.py
│   └── validate.py
├── generation_config.yaml
└── README.md
```

### Step 3: Generate front 20Sample

```bash
cd Data_v2/synthetic/Copa_mezo_gpt4o_v1/scripts/
export OPENAI_API_KEY="your-key"
python rephrase_top20.py
```

**output**：`copa_train_top20.jsonl`（20Sample）

---

### 🔴 **Checkpoint1：manual reviewtop20Sample**

#### 3.1 Side by sideVieworiginalvsSyntheticData

```bash
# usemanual reviewTool（To be implemented）
python review_top20.py
```

**ReviewInterfaceExample**：
```
Sample 1/20:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
originalpremise:
  "My body cast a shadow over the grass."

Syntheticpremise:
  "A shadow from my body fell across the grass."

Choice 1: The sun was rising.
Choice 2: The grass was cut.
Question: cause
Correct answer: Choice 1

This rephraseYesNoqualified？
  [y] qualified - SemanticsConsistent，qualityGood
  [n] unqualified - SemanticsChange or qualitydifference
  [s] skip
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
youJudgment: y

（ContinueReviewSample2-20...）
```

**output**：
```json
// validation_checkpoints/top20_review.json
{
  "total": 20,
  "approved": 18,
  "rejected": 2,
  "annotations": [
    {
      "index": 0,
      "original": "My body cast a shadow over the grass.",
      "rephrased": "A shadow from my body fell across the grass.",
      "judgment": "approved",
      "note": ""
    },
    // ... 19MoreSample
  ]
}
```

#### 3.2 AutomaticGeneratefew-shotAnd inject to rephrase_rest.py

```bash
# base at ReviewresultAutomaticGeneratefew-shot examples
python update_rest_prompt.py
```

**Feature**：
1. read`top20_review.json`
2. Extract`judgment == "approved"`Sample
3. formatIntofew-shot examples
4. AutomaticUpdate`rephrase_rest.py` in prompt

---

### Step 4: GenerateRemaining380Sample

```bash
python rephrase_rest.py
# current in prompt in containmanual reviewpassfew-shot examples
```

**output**：`copa_train_rest.jsonl`（380Sample）

### Step 5: MergeData

```bash
cat copa_train_top20.jsonl copa_train_rest.jsonl > ../copa_train.jsonl
```

---

### 🔴 **Checkpoint2：Manual annotationline21-80Sample**

> **✅ ToolImplemented**: `extract_samples.py`, `annotate_samples.py` Bit at  `automation/stage1_generation/tools/`

#### 5.1 Extractline21-80Sample

```bash
# EnterDataCollectDirectory
cd Data_v2/synthetic/{experiment_purpose}/{experiment_id}/{Dataset}/

# ExtractSample21-80（Total60）
python /path/to/automation/stage1_generation/tools/extract_samples.py \
    --range 21-80 \
    --input Copa/copa_train.jsonl

#  or er in toolsDirectoryDirectlyRun
cd /path/to/automation/stage1_generation/tools/
python extract_samples.py \
    --range 21-80 \
    --input /path/to/Copa/copa_train.jsonl
```

**output**：
```
validation_checkpoints/samples_21_80.jsonl  # 60Sample
```

#### 5.2 Manual annotation

```bash
#  in DataCollectDirectory or toolsDirectoryRun
python annotate_samples.py validation_checkpoints/samples_21_80.jsonl

#  can selectParameter：
# --output validation_checkpoints/custom_name_annotated.json  # CustomoutputFile
# --no-resume                                                  # HeavyNewOnstart，notContinue up timesAnnotate
```

**AnnotateInterfaceExample**：
```
Sample 1/60 (originalDataline21):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
originalpremise:
  "The tenant misplaced his keys."

Syntheticpremise:
  "The tenant lost his apartment keys."

Choice 1: His landlord unlocked the door.
Choice 2: His landlord repaired the door.
Question: effect
Correct answer: Choice 1

SemanticsYesNoConsistent？
  [s] same - SemanticsConsistent
  [n] not the same - SemanticsChange
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
youJudgment: s

（ContinueAnnotateSample2-60...）
```

**output**：
```json
// validation_checkpoints/samples_21_80_annotated.json
{
  "total": 60,
  "same": 57,
  "not_the_same": 3,
  "annotations": [
    {
      "index": 20,  // originalData in Index（line21）
      "original_premise": "The tenant misplaced his keys.",
      "rephrased_premise": "The tenant lost his apartment keys.",
      "choice1": "His landlord unlocked the door.",
      "choice2": "His landlord repaired the door.",
      "question_type": "effect",
      "correct_answer": "Choice 1",
      "judgment": "same",  // ManualJudgment
      "note": ""
    },
    // ... 59MoreSample
  ]
}
```

#### 5.3 AutomaticGeneratevalidation promptTestScript

> **✅ ToolImplemented**: `generate_validation_test.py` Bit at  `automation/stage1_generation/tools/`

```bash
# useDefaultPath
python /path/to/automation/stage1_generation/tools/generate_validation_test.py

#  or SpecifyParameter
python generate_validation_test.py \
    --annotations validation_checkpoints/samples_21_80_annotated.json \
    --fewshot-range 21-40 \
    --test-range 41-80 \
    --output scripts/validate_prompt_test.py \
    --api-key your-api-key \
    --base-url https://api.openai.com/v1
```

**Feature**：
1. read`samples_21_80_annotated.json`
2. **line21-40"same"Sample** → formatIntovalidation promptfew-shot examples
3. **line41-80allSample** → formatIntotest_set（containground truth）
4. AutomaticGenerate`validate_prompt_test.py`

**GenerateTestScript**：
```python
# scripts/validate_prompt_test.py
def generate_validation_prompt(...):
    return f"""
    Judge if the rephrased premise...

    ### Few-shot Examples (Fromline21-40):
    Example 1:
    Original: The tenant misplaced his keys.
    Rephrased: The tenant lost his apartment keys.
    Judgment: same

    ... (Total20few-shot)
    """

# Test set（Fromline41-80，Total40）
test_set = [
    {
        "original_premise": "...",
        "rephrased_premise": "...",
        "ground_truth": "same"  # Manual annotation
    },
    ...
]

# Testpromptaccuracy
for item in test_set:
    response = gpt4o_judge(item)
    if response == item["ground_truth"]:
        correct += 1

accuracy = correct / len(test_set)
print(f"Promptaccuracy: {accuracy:.2%}")
```

---

### 🔴 **Checkpoint3：TestandTunevalidation prompt**

#### 6.1 Testpromptaccuracy

```bash
python validate_prompt_test.py
```

**outputExample**：
```
Testing validation prompt on 40 samples...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test Results:
  Correct: 36 / 40
  Accuracy: 90.0%

✗ PromptNot met standard（need≥95%）

ErrorSample:
  Sample 23: Judgment as same，Actual as not the same
  Sample 45: Judgment as not the same，Actual as same
  Sample 67: Judgment as same，Actual as not the same
  Sample 78: Judgment as not the same，Actual as same

Recommendation:
  1. Checkfew-shot examples in YesNocontainclassSimilar counter examples
  2. adjustmentvalidation_prompt in JudgmentstandardDescribe
  3. Increase for BoundaryCaseDescription

pleaseManualadjustmentConfigurationFile in validation_prompt，natural back HeavyNewRunthisTest。
```

#### 6.2 ManualTuneprompt

EditConfigurationFile：
```bash
vim automation/configs/stage1/drafts/copa_mezo_v1_draft.yaml
```

Modify`validation.validation_prompt`，example such as ：
- addMoreExplicitJudgmentstandard
- SupplementBoundaryCasefew-shot examples
- adjustmentprompt use Word

#### 6.3 HeavyNewGenerateScriptandTest

```bash
# HeavyNewGenerateScript
python automation/stage1_generation/generator.py \
       automation/configs/stage1/drafts/copa_mezo_v1_draft.yaml

# HeavyNewTest
cd Data_v2/synthetic/Copa_mezo_gpt4o_v1/scripts/
python validate_prompt_test.py
```

**Repeat6.1-6.3direct to accuracy≥95%**：
```
Test Results:
  Correct: 39 / 40
  Accuracy: 97.5%

✓ PromptMet the standard！
  CreatepassTag: validation_checkpoints/prompt_test_passed.flag
```

---

### Step 7: batchValidateall400Sample

```bash
python validate.py
```

**Feature**：
1. CheckYesNoexist in `prompt_test_passed.flag`（Gate）
2.  such as If does not exist in ，RejectExecuteandTipfirstRunTest
3.  such as If exists in ，usealreadyValidatevalidation promptValidateall400Sample
4. Rejection sampling：unqualified use originalDataReplace

**output**：
```
ValidateComplete!
Pass rate: 381/400 = 95.25%
outputFile: copa_train_validated.jsonl
```

---

### Step 8: Archive as validatedTemplate

```bash
cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/

python automation/stage1_generation/archive_validated_config.py \
       --source automation/configs/stage1/drafts/copa_mezo_v1_draft.yaml \
       --data-dir Data_v2/synthetic/Copa_mezo_gpt4o_v1/
```

**output**：
```
✓ ConfigurationalreadyArchive!

TemplatePath: automation/configs/stage1/templates/copa_mezo_validated.yaml
ArchivePath: automation/configs/stage1/archive/2024-12/copa_mezo_v1_complete_20241224_153000.yaml

current in canbase at thisTemplateCreateParameter tuningExperimentConfiguration。
```

---

## ScenarioB：Parameter tuningExperiment（base at alreadyValidateprompt）

###  front setCondition
- alreadyhasvalidatedTemplate：`automation/configs/stage1/templates/copa_mezo_validated.yaml`
- PromptPassedallManualValidate
- want need adjustmentGenerateParameterObserve for DataqualityImpact

### Step 1: CreateExperimentConfiguration

```bash
# Experiment1: Raisetemperature
python automation/stage1_generation/create_experiment.py \
       --template automation/configs/stage1/templates/copa_mezo_validated.yaml \
       --version v2 \
       --param generation.temperature=0.7
```

**output**：
```
✓ ConfigurationalreadyCreate: automation/configs/stage1/experiments/copa_mezo_v2_temperature07.yaml

ParameterchangeMore:
  - generation.temperature: 0.5 → 0.7
```

### Step 2: GenerateScript

```bash
python automation/stage1_generation/generator.py \
       automation/configs/stage1/experiments/copa_mezo_v2_temperature07.yaml
```

**output**：
```
Data_v2/synthetic/Copa_mezo_gpt4o_v2/
├── scripts/
│   ├── rephrase_all.py      # containalreadyValidatefew-shot
│   ├── rephrase_top20.py
│   ├── rephrase_rest.py
│   └── validate.py          # containalreadyValidatevalidation prompt
└── ...
```

### Step 3: DirectlyGenerateCompleteDataCollect（Noneneedmanual review）

```bash
cd Data_v2/synthetic/Copa_mezo_gpt4o_v2/scripts/
export OPENAI_API_KEY="your-key"

# DirectlyRunrephrase_all.py（400Sample）
python rephrase_all.py
```

**Key**：
- ✅ usealreadyValidatefew-shot examples
- ✅ OnlyhastemperatureChange（0.5 → 0.7）
- ✅ NoneneedRepeatmanual reviewCheckpoint1-3

### Step 4: usealreadyValidatevalidation promptValidate

```bash
python validate.py
```

**output**：
```
ValidateComplete!
Pass rate: 378/400 = 94.5%
outputFile: copa_train_validated.jsonl
```

### Step 5: comparisonDifferentVersion

```bash
# comparisonv1 and v2Dataquality
python automation/analysis/compare_versions.py \
       --v1 Data_v2/synthetic/Copa_mezo_gpt4o_v1/copa_train_validated.jsonl \
       --v2 Data_v2/synthetic/Copa_mezo_gpt4o_v2/copa_train_validated.jsonl
```

**output**：
```
Versioncomparison:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
metrics                   v1          v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ValidatePass rate           95.25%      94.50%
averageEditDistance         12.3        15.7
WordGatherMoreDiversity (TTR)     0.82        0.87
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Conclusion: v2 (temp=0.7) MoreDiversityHigher，ButPass rateSlightlyLow
```

---

## MoreExperimentparallel

```bash
# CreatemultipleExperimentConfiguration
python automation/stage1_generation/create_experiment.py \
       --template automation/configs/stage1/templates/copa_mezo_validated.yaml \
       --version v3 \
       --param generation.model=gpt-4o-mini

python automation/stage1_generation/create_experiment.py \
       --template automation/configs/stage1/templates/copa_mezo_validated.yaml \
       --version v4 \
       --param generation.temperature=0.9

# parallelGenerate（useDifferentGPU or timeParagraph）
for version in v2 v3 v4; do
  config="automation/configs/stage1/experiments/copa_mezo_${version}_*.yaml"
  python automation/stage1_generation/generator.py $config
  cd Data_v2/synthetic/Copa_mezo_*_${version}/scripts/
  python rephrase_all.py &
  cd -
done
```

---

## DirectorystructureSummary

```
automation/configs/stage1/
├── drafts/                                    # First timeGenerate：pendingValidateConfiguration
│   └── copa_mezo_v1_draft.yaml               # ManualWriteinitialprompt
│
├── templates/                                 # alreadyValidateTemplate（ can Repeat use ）
│   ├── copa_mezo_validated.yaml              # CopataskMeZOmethodTemplate
│   ├── rte_mezo_validated.yaml               # RTEtaskTemplate
│   └── README.md
│
├── experiments/                               # Parameter tuningExperimentConfiguration
│   ├── copa_mezo_v2_temperature07.yaml       # Experiment：temp=0.7
│   ├── copa_mezo_v3_gpt4omini.yaml           # Experiment：changeModel
│   └── copa_mezo_v4_temperature09.yaml       # Experiment：temp=0.9
│
├── archive/                                   # HistoryArchive
│   └── 2024-12/
│       └── copa_mezo_v1_complete_20241224.yaml
│
└── examples/                                  # Example
    └── stage1_example_copa_mezo.yaml

Data_v2/synthetic/Copa_mezo_gpt4o_v1/
├── scripts/
│   ├── rephrase_top20.py
│   ├── rephrase_rest.py
│   ├── rephrase_all.py
│   ├── validate.py
│   ├── review_top20.py                       # 🆕 manual reviewTool
│   ├── annotate_samples.py                   # 🆕 Manual annotationTool
│   ├── validate_prompt_test.py               # 🆕 AutomaticGenerateTestScript
│   ├── extract_samples.py                    # 🆕 SampleExtract
│   ├── update_rest_prompt.py                 # 🆕 AutomaticInjectfew-shot
│   └── generate_validation_test.py           # 🆕 GenerateTestScript
├── validation_checkpoints/                    # 🆕 ManualValidateRecord
│   ├── top20_review.json                     # Checkpoint1Record
│   ├── samples_21_80_annotated.json          # Checkpoint2Record
│   ├── prompt_test_results.json              # Checkpoint3Record
│   └── prompt_test_passed.flag               # passTag
├── copa_train_top20.jsonl
├── copa_train_rest.jsonl
├── copa_train.jsonl                          # Merge back notValidateData
├── copa_train_validated.jsonl                # finalValidatepassData
├── generation_config.yaml
└── README.md
```

---

## KeyOriginalthen

1. **First timeGeneratemustalreadyManual checkpoint**
   - Checkpoint1：Reviewtop20 → Generaterephrase few-shot
   - Checkpoint2：Annotate21-80 → Generatevalidation few-shot and test_set
   - Checkpoint3：Testprompt → Tunedirect to ≥95%

2. **Parameter tuningExperimentRepeat use alreadyValidateprompt**
   - Inheritancetemplates/ in Configuration
   - OnlyModifyGenerateParameter
   - Directlyuserephrase_all.py
   - NoneneedRepeatmanual review

3. **allpromptAll needManualCreate**
   - initialpromptManualWrite
   - Few-shot examplesbymanual reviewAutomaticGenerate
   - PromptTunebyManualIterationComplete

4. **VersionManage**
   - drafts/: First timeGenerateConfiguration
   - templates/: ValidatepassTemplate
   - experiments/: Parameter tuningExperimentConfiguration
   - archive/: HistoryRecord（containCompleteValidateHistory）

---

## To be implementedToolclearSingle

### HighPriority（Manual checkpointRequired）
- [ ] `review_top20.py` - Checkpoint1manual reviewInterface
- [ ] `update_rest_prompt.py` - AutomaticInjectfew-shot to rephrase_rest.py
- [ ] `extract_samples.py` - Extractline21-80Sample
- [ ] `annotate_samples.py` - Checkpoint2Manual annotationInterface
- [ ] `generate_validation_test.py` - AutomaticGeneratevalidationTestScript
- [ ] `validate_prompt_test.py` - Testpromptaccuracy（AutomaticGenerate）
- [ ] Modify`validate.py` - addGateCheck

###  in Priority（ImproveExperience）
- [ ] `tune_validation_prompt.py` - PromptTuneauxiliaryTool
- [ ] `compare_versions.py` - VersioncomparisonAnalysis
- [ ] Modify`generator.py` - supportAutomaticfew-shotInject

### LowPriority（elegant up Icing on cake）
- [ ] WebInterfaceReplaceCLIReview/AnnotateTool
- [ ] AutomaticationpromptTuneRecommendation
- [ ] batchExperimentManageTool
