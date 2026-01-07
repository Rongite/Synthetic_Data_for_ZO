# Batch Solution 3++ User Guide

This document provides detailed information on the design principles, usage methods, and best practices of Batch Solution 3++.

---

## 📖 Table of Contents

1. [Design Principles](#design-principles)
2. [Directory Structure](#directory-structure)
3. [Core Concepts](#core-concepts)
4. [Usage Methods](#usage-methods)
5. [Practical Operation Examples](#practical-operation-examples)
6. [Data Reuse Mechanism](#data-reuse-mechanism)
7. [Batch Management Tools](#batch-management-tools)
8. [FAQ](#faq)

---

## Design Principles

### Why Do We Need Batch Solution?

When conducting multi-parameter experiments (e.g., adjusting temperature, top_p, model, etc.), the following issues are often encountered:

1. **Parameter Combination Duplication**: Different batches of experiments may use the same parameter configuration
2. **Storage Waste**: Data with the same parameters is repeatedly generated and stored
3. **Organizational Chaos**: Difficult to manage and view experiments from different batches
4. **Traceability Difficulty**: Hard to find when a parameter configuration was first generated

### Batch Solution 3++ Approach

**Core Idea**: Separation of physical storage and logical views

- **Physical Storage (_shared/)**: Stores actual data, deduplicated by parameter fingerprint
- **Logical Views (batch_*)**: Organizes experiments through symbolic links, grouped by time/purpose

**Three Dimensions**:
1. **Batch Dimension**: batch_*/ (time + purpose)
2. **Dataset Dimension**: {Dataset}/ (e.g., Copa, CB, BOOLQ)
3. **Parameter Dimension**: {semantic_dirname}/ (e.g., temp07_topp09_gpt4o)

---

## Directory Structure

### Complete Structure Example

```
Data_v2/synthetic/
├── _shared/                                    # Physical data storage (unique real data)
│   ├── Copa/
│   │   ├── temp05_topp10_gpt4o/               # Parameter configuration 1
│   │   │   ├── .fingerprint                   # Parameter fingerprint (MD5)
│   │   │   ├── experiment_metadata.json       # Experiment metadata
│   │   │   ├── generation_config.yaml         # Configuration copy
│   │   │   ├── Copa/                          # Dataset subdirectory
│   │   │   │   ├── copa_train.jsonl
│   │   │   │   ├── copa_validation.jsonl
│   │   │   │   └── copa_test.jsonl
│   │   │   ├── scripts/
│   │   │   │   ├── rephrase_all.py
│   │   │   │   └── validate.py
│   │   │   └── README.md
│   │   │
│   │   ├── temp07_topp09_gpt4o/               # Parameter configuration 2
│   │   └── temp09_topp10_gpt4o/               # Parameter configuration 3
│   │
│   └── CB/
│       └── temp07_topp10_gpt4o/
│
├── batch_20241229_temperature/                 # Batch 1: Temperature experiment
│   ├── Copa/
│   │   ├── temp05_topp10_gpt4o -> ../../_shared/Copa/temp05_topp10_gpt4o/
│   │   ├── temp07_topp10_gpt4o -> ../../_shared/Copa/temp07_topp10_gpt4o/
│   │   └── temp09_topp10_gpt4o -> ../../_shared/Copa/temp09_topp10_gpt4o/
│   └── CB/
│       └── temp07_topp10_gpt4o -> ../../_shared/CB/temp07_topp10_gpt4o/
│
└── batch_20241230_topp/                        # Batch 2: top_p experiment
    └── Copa/
        ├── temp07_topp08_gpt4o -> ../../_shared/Copa/temp07_topp08_gpt4o/
        └── temp07_topp09_gpt4o -> ../../_shared/Copa/temp07_topp09_gpt4o/  # Reused!
```

### Directory Responsibilities

| Directory | Responsibility | Data Type |
|------|------|----------|
| `_shared/` | Physical data storage, deduplicated by parameter fingerprint | Actual data files |
| `batch_*/` | Logical experiment views, organized by time/purpose | Symbolic links |

---

## Core Concepts

### 1. Parameter Fingerprint

The parameter fingerprint is an MD5 hash (first 12 characters) calculated based on **all key parameters that affect data generation**.

**Included Parameters**:
```python
{
    'gen_model': 'gpt-4o',
    'gen_temperature': 0.7,
    'gen_top_p': 1.0,
    'gen_max_tokens': 256,
    'gen_frequency_penalty': 0.0,
    'gen_presence_penalty': 0.0,
    'val_model': 'gpt-4o',
    'val_temperature': 0.0,
    'gen_prompt_hash': 'a1b2c3d4',  # rephrase_prompthash
    'val_prompt_hash': 'e5f6g7h8'   # validation_prompthash
}
```

**Fingerprint use way**:
- **DeduplicationJudgment**: sameFingerprint = sameParameter = Repeat use data
- **UniqueIdentifier**: preciseIdentifyParameter configuration
- **Trace source**: passFingerprintfind to First timeGeneratebatch

### 2. SemanticsationDirectoryname (Semantic Directory Name)

 as personclass can Readability，useSemanticsationDirectorynameWhilenotYesDirectlyusehash。

**Namingformat**:
```
temp{temperature}_topp{top_p}_{model}
```

**Example**:
- `temp07_topp10_gpt4o` → temperature=0.7, top_p=1.0, model=gpt-4o
- `temp09_topp08_gpt4o` → temperature=0.9, top_p=0.8, model=gpt-4o
- `temp05_topp10_gpt35` → temperature=0.5, top_p=1.0, model=gpt-3.5-turbo

**smart can saveSlightly**:
- Defaultvalue top_p=1.0 → Show as  topp10
- NotDefaultvalue top_p=0.9 → Show as  topp09

**preciseMatch**: Directory inside  `.fingerprint` Filestorageprecisehash， use  at ParameterMatch

### 3. Batch ID

Batch ID use  at grouporganizemultiplemutualOffExperiment to Samebatch。

**format**:
```
batch_{date}_{purpose}
```

**Example**:
- `batch_20241229_temperature` → 2024Year12Month29dayTemperatureExperiment
- `batch_20241230_topp` → 2024Year12Month30daytop_pExperiment
- `batch_20250103_model_comparison` → 2025Year1Month3daymodelcomparisonExperiment

**AutomaticGenerate**: IfConfigurationFile in Unspecified `batch_id`，system will According toCurrentdateand `purpose` AutomaticGenerate

---

## usemethod

### ConfigurationFilesetup

 in ConfigurationFile in add `experiment.batch_id` Field：

```yaml
experiment:
  # Batch ID（ can select）
  # format: batch_{date}_{purpose}
  # notSpecifySometimesAutomaticGenerate: batch_{YYYYMMDD}_{purpose}
  batch_id: "batch_20241229_temperature"

  purpose: "temperature_study"
  description: "ResearchtemperatureParameter for synthetic dataqualityImpact"

# OtherConfiguration...
generation:
  model: "gpt-4o"
  temperature: 0.7  # Experimentvariable
  top_p: 1.0
  # ...
```

### Generatescript

use `generator.py` GeneratescriptSometimes，BatchSolution will AutomaticEnable：

```bash
# GenerateExperimentscript
python automation/stage1_generation/generator.py \
    automation/configs/examples/stage1_full_example_copa.yaml
```

**system will Automatic**:
1. CalculateParameter fingerprint
2.  in  `_shared/{Dataset}/`  in FindsameFingerprint
3. If found to  → Repeat use Physicaldata + CreatebatchSymbolic link
4. If not found to  → CreateNewPhysicalDirectory + CreatebatchSymbolic link

### outputInterpret

```
================================================================================
🔧 BatchExperimentManage
================================================================================
Batch ID: batch_20241229_temperature
Dataset: Copa
Parameter fingerprint: a1b2c3d4e5f6
SemanticsSemantic name: temp07_topp09_gpt4o
================================================================================

🔍  in  _shared/Copa/  in SearchFingerprint a1b2c3d4e5f6...
✅ DiscoversameParameteralreadyhasExperiment！
   location: _shared/Copa/temp07_topp09_gpt4o
   Createtime: 2024-12-29 10:30:00
   Originalbatch: batch_20241228_pilot

📂 Repeat use alreadyhasdata
   Physical storage: _shared/Copa/temp07_topp09_gpt4o (Exists in ，Repeat use )
   BatchView: batch_20241229_temperature/Copa/temp07_topp09_gpt4o

✅ alreadyhasData reuseSuccess
   💾 Sectionsaveresource: NoneneedHeavyNewGeneratedata
```

**Keyinformation**:
- ✅ DiscoversameParameter → data will  by Repeat use 
- ✓ Not found to Match → CreateNewExperiment
- 💾 Sectionsaveresource → not will RepeatGeneratedata

---

## ActualOperationExample

### scenarioA: First timebatch - TemperatureExperiment

**goal**: test temperature=0.5, 0.7, 0.9  for CopadataqualityImpact

#### Step1: AccurateprepareConfigurationFile

CreatethreeConfigurationFile（orusescriptbatchGenerate）：

**config_temp05.yaml**:
```yaml
experiment:
  batch_id: "batch_20241229_temperature"
  purpose: "temperature_study"

generation:
  model: "gpt-4o"
  temperature: 0.5  # variable
  top_p: 1.0
```

**config_temp07.yaml**, **config_temp09.yaml** classsimilar，Onlychangetemperaturevalue。

#### Step2: Generatescript

```bash
# GeneratethreeExperimentscript
python automation/stage1_generation/generator.py automation/configs/temp05.yaml
python automation/stage1_generation/generator.py automation/configs/temp07.yaml
python automation/stage1_generation/generator.py automation/configs/temp09.yaml
```

#### Step3: ViewGenerateDirectory Structure

```bash
python automation/stage1_generation/batch_tools/list_batch_experiments.py \
    batch_20241229_temperature --verbose
```

**output**:
```
📊 Copa (3 Experiment)
  🔧 temp05_topp10_gpt4o
     ⚡ Data reuse: No (NewGenerate)
  🔧 temp07_topp10_gpt4o
     ⚡ Data reuse: No (NewGenerate)
  🔧 temp09_topp10_gpt4o
     ⚡ Data reuse: No (NewGenerate)
```

#### Step4: RundataGenerate

```bash
# Way1: ManualIn orderRun
cd Data_v2/synthetic/_shared/Copa/temp05_topp10_gpt4o/scripts/
python rephrase_all.py && python validate.py

cd ../../../temp07_topp10_gpt4o/scripts/
python rephrase_all.py && python validate.py

cd ../../../temp09_topp10_gpt4o/scripts/
python rephrase_all.py && python validate.py

# Way2: usescriptbatchRun（Recommended）
# TODO: Create batch_run.py Tool
```

---

### scenarioB: Secondbatch - top_pExperiment

**goal**:  in  temperature=0.7  down ，test top_p=0.8, 0.9, 1.0 Impact

#### Step1: AccurateprepareConfigurationFile

**config_topp08.yaml**:
```yaml
experiment:
  batch_id: "batch_20241230_topp"  # Newbatch
  purpose: "topp_study"

generation:
  model: "gpt-4o"
  temperature: 0.7  # Fixed
  top_p: 0.8        # variable
```

**config_topp09.yaml**, **config_topp10.yaml** classsimilar。

#### Step2: Generatescript

```bash
python automation/stage1_generation/generator.py automation/configs/topp08.yaml
python automation/stage1_generation/generator.py automation/configs/topp09.yaml
python automation/stage1_generation/generator.py automation/configs/topp10.yaml
```

**Keyoutput**:

 for  at  **config_topp10.yaml** (temperature=0.7, top_p=1.0):
```
🔍  in  _shared/Copa/  in SearchFingerprint a1b2c3d4e5f6...
✅ DiscoversameParameteralreadyhasExperiment！
   location: _shared/Copa/temp07_topp10_gpt4o
   Originalbatch: batch_20241229_temperature

📂 Repeat use alreadyhasdata
   💾 Sectionsaveresource: NoneneedHeavyNewGeneratedata
```

#### Step3: ViewDirectory Structure

```bash
python automation/stage1_generation/batch_tools/list_batch_experiments.py \
    batch_20241230_topp --verbose
```

**output**:
```
📊 Copa (3 Experiment)
  🔧 temp07_topp08_gpt4o
     ⚡ Data reuse: No (NewGenerate)

  🔧 temp07_topp09_gpt4o
     ⚡ Data reuse: No (NewGenerate)

  🔧 temp07_topp10_gpt4o
     ⚡ Data reuse: Yes (Originalbatch: batch_20241229_temperature)
```

**Data reuseSuccess！** temp07_topp10_gpt4o dataDirectlyRepeat use selfFirstbatch。

#### Step4: RundataGenerate

```bash
# OnlyneedGenerateNewParameterdata
cd Data_v2/synthetic/_shared/Copa/temp07_topp08_gpt4o/scripts/
python rephrase_all.py && python validate.py

cd ../../../temp07_topp09_gpt4o/scripts/
python rephrase_all.py && python validate.py

# temp07_topp10_gpt4o Alreadyhasdata，skip！
```

---

### scenarioC: ViewandCompareExperiment

#### Viewallbatch

```bash
python automation/stage1_generation/batch_tools/list_batches.py --verbose
```

**output**:
```
find to  2 batch

📦 batch_20241229_temperature
   ExperimentSeveral: 3
   Copa: 3 Experiment

📦 batch_20241230_topp
   ExperimentSeveral: 3
   Copa: 3 Experiment
```

#### ViewPhysical storageuseCase

```bash
python automation/stage1_generation/batch_tools/list_shared_experiments.py \
    --dataset Copa --verbose
```

**output**:
```
📊 Copa (5 Experiment)  # Onlyhas5Physicaldata，notYes6！

  📦 temp05_topp10_gpt4o
     originalBatch: batch_20241229_temperature

  📦 temp07_topp08_gpt4o
     originalBatch: batch_20241230_topp

  📦 temp07_topp09_gpt4o
     originalBatch: batch_20241230_topp

  📦 temp07_topp10_gpt4o  #  by twobatchTotalshare！
     originalBatch: batch_20241229_temperature

  📦 temp09_topp10_gpt4o
     originalBatch: batch_20241229_temperature
```

#### ComparetwoExperimentParameter

```bash
python automation/stage1_generation/batch_tools/compare_experiments.py \
    --shared Copa/temp07_topp10_gpt4o \
    --shared Copa/temp09_topp10_gpt4o
```

**output**:
```
✅ sameParameter:
  generation.model: gpt-4o
  generation.top_p: 1.0
  validation.model: gpt-4o

⚠️  DifferentParameter:
  generation.temperature:
    Experiment1: 0.7
    Experiment2: 0.9
```

---

## Data reuseMechanism

### Repeat use Condition

**mustsatisfyEnough**: Parameter fingerprintFullysame

Parameter fingerprintinclude：
- Generatemodel、temperature、top_p、max_tokens、frequencyPenalty、exist in Penalty
- validatemodel、temperature
- rephrase_prompt hash
- validation_prompt hash

**Only need hasthisParameterDifferent，FingerprintAs forDifferent，needHeavyNewGeneratedata。**

### Repeat use workflow

1. **GeneratescriptSometimes**:
   - CalculateConfigurationFileParameter fingerprint
   -  in  `_shared/{Dataset}/`  in TraverseallExperimentDirectory
   - readeachDirectory `.fingerprint` File
   - If found to sameFingerprint → Repeat use 

2. **Repeat use Operation**:
   - **notCreateNewPhysicalDirectory**
   - **notGenerateNewdata**
   - Only in  `batch_*/`  in CreateSymbolic linkrefer towards currenthasPhysicalDirectory

3. **metadataRecord**:
   - PhysicalDirectorymetadataMaintainInvariance（RecordFirst timeCreatebatch）
   - batchSymbolic linkNoneadditional outside metadata

### validateRepeat use 

```bash
# CheckSymbolic link
ls -la Data_v2/synthetic/batch_20241230_topp/Copa/

# outputclasssimilar:
# temp07_topp10_gpt4o -> ../../_shared/Copa/temp07_topp10_gpt4o

# CheckPhysicalDirectory
ls -la Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/
# shouldsee to ActualdataFile

# useToolvalidate
python automation/stage1_generation/batch_tools/list_batch_experiments.py \
    batch_20241230_topp --verbose
# shouldsee to  "⚡ Data reuse: Yes"
```

---

## BatchManageTool

See [batch_tools/README.md](stage1_generation/batch_tools/README.md)

### QuickReference

```bash
# Columnoutallbatch
python batch_tools/list_batches.py --verbose

# ViewbatchDetails
python batch_tools/list_batch_experiments.py batch_20241229_temperature --verbose

# ViewPhysicaldata
python batch_tools/list_shared_experiments.py --dataset Copa --verbose

# CompareExperimentParameter
python batch_tools/compare_experiments.py \
    --shared Copa/temp07_topp10_gpt4o \
    --shared Copa/temp09_topp10_gpt4o

# ⭐ Newincrease：FinddataPath（ use  at trainingConfiguration）
python batch_tools/list_data_paths.py --dataset Copa --format yaml

# ⭐ Newincrease：Pathconvert
python batch_tools/resolve_data_path.py "Data_v2/synthetic/batch_xxx/Copa/..."
```

---

## FAQ

### Q1: If IManualmodify_shared/ in data，batch_*/ in Symbolic link will AutomaticUpdate？

**Answer**: Yes！Symbolic linkrefer towards PhysicalPath，modifyPhysicaldata back ，allreference use thisdatabatchall will see to Update。

**Note**:  this  can  can Lead toDifferentbatchtrainingresultnotConsistent，Recommendationnot need ManualmodifyalreadyGeneratedata。

### Q2: If IdeleteSomebatch_*/Directory，_shared/ in Physicaldata will  by delete？

**Answer**: not will 。batch_*/OnlycontainSymbolic link，deletebatchnotImpactPhysicaldata。

**CleanupRecommendation**: If need CleanupnotAgainuseExperimentdata，should:
1. firstdeleteallreference use thisdatabatchSymbolic link
2. Againdelete_shared/ in PhysicalDirectory

### Q3: IcanManualCreatebatch？

**Answer**: can，ButnotRecommended。shouldpassConfigurationFile + generator.py AutomaticManage。

If indeedneedManualOperation：
```bash
mkdir -p Data_v2/synthetic/batch_20241231_manual/Copa
ln -s ../../_shared/Copa/temp07_topp10_gpt4o \
    Data_v2/synthetic/batch_20241231_manual/Copa/temp07_topp10_gpt4o
```

### Q4: Parameter fingerprintYesHowCalculate？Icansee to Detailedcontent？

**Answer**: canView `.fingerprint` Fileand `experiment_metadata.json`:

```bash
# ViewFingerprint
cat Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/.fingerprint

# ViewCompletemetadata（containallParameter）
cat Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/experiment_metadata.json | jq .
```

### Q5: I wantHeavyNewGenerateSomeParameter configurationdata，What to do？

**Answer**:
1. delete_shared/ in  for corresponding physicalDirectory
2. deleteallbatch_*/ in refer towards thisDirectorySymbolic link
3. HeavyNewRun generator.py（ will Detect to dataDoes not exist in andHeavyNewGenerate）

**Example**:
```bash
# 1. deletePhysicaldata
rm -rf Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o

# 2. deleteallSymbolic link
find Data_v2/synthetic/batch_* -name "temp07_topp10_gpt4o" -type l -delete

# 3. HeavyNewGenerate
python automation/stage1_generation/generator.py automation/configs/temp07.yaml
cd Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/scripts/
python rephrase_all.py && python validate.py
```

### Q6: batch_idYesmust？

**Answer**: notYesmust。IfConfigurationFile in Unspecified `batch_id`，system will According toCurrentdateand `purpose` AutomaticGenerate：

```
batch_{YYYYMMDD}_{purpose}
```

for example: `batch_20241229_temperature_study`

### Q7: IcanmultipleDataset（Copa, CB, BOOLQ）put in Samebatch in ？

**Answer**: can！batchYescrossDataset。Only need ConfigurationFile in Specifysame `batch_id`，DifferentDatasetExperimentall will Appear in Samebatch in 。

**Example**:

**copa_config.yaml**:
```yaml
experiment:
  batch_id: "batch_20241229_multi_dataset"
dataset:
  dataset_name: "Copa"
```

**cb_config.yaml**:
```yaml
experiment:
  batch_id: "batch_20241229_multi_dataset"  # samebatch_id
dataset:
  dataset_name: "CB"
```

result：
```
batch_20241229_multi_dataset/
├── Copa/
│   └── temp07_topp10_gpt4o/
└── CB/
    └── temp07_topp10_gpt4o/
```

### Q8: Stillneedusepublish_dataset.py？

**Answer**: **notneed！** trainer.pycanDirectlyuse `Data_v2/` Path。

**Recommended approach**（DirectlyuseData_v2Path）:
```yaml
# trainingConfiguration
data:
  # Recommended：usebatchPath（MoreIntuitive）
  path: "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"

  # orusesharedPath
  # path: "Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa"
```

** can Selection method**（Only use  at CompatibleOldscript）:
```bash
# Only in needCompatibleOldtrainingscriptSometimesuse
python automation/stage1_generation/tools/publish_dataset.py \
    --source Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa \
    --dataset Copa \
    --target Data/rejection_sampling/0_data
```

### Q9: how toQuick find to dataPath use  at trainingConfiguration？

**Answer**: useNewincrease `list_data_paths.py` Tool：

```bash
# outputYAMLformat， can Directlycopy to ConfigurationFile
python automation/stage1_generation/batch_tools/list_data_paths.py --dataset Copa --format yaml
```

**outputExample**:
```yaml
data:
  path: "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"
```

---

## Best Practices

### 1. BatchNamingspecification

- usedate front prefix: `batch_YYYYMMDD_*`
- useDescriptionilitypurpose: `temperature`, `topp`, `model_comparison`
- AvoiduseChineseorSpeciallyspecialCharacterSymbol

### 2. ConfigurationFileManage

```
automation/configs/
├── batches/
│   ├── batch_20241229_temperature/
│   │   ├── copa_temp05.yaml
│   │   ├── copa_temp07.yaml
│   │   └── copa_temp09.yaml
│   └── batch_20241230_topp/
│       ├── copa_topp08.yaml
│       ├── copa_topp09.yaml
│       └── copa_topp10.yaml
```

### 3. RegularCleanup

- RegularView `_shared/` useCase
- deletenotAgainneedExperimentdata
- KeephasvaluevalueExperimentresult

### 4. DocumentRecord

 in eachbatchDirectory in Create `README.md` Record：
- ExperimentPurpose
- Parametersetup
- resultsummary
- trainingEffectcomparison

---

## andtrainingscriptcompatibility

### ✅ Recommended：DirectlyuseData_v2Path

**trainer.pycanDirectlyuse `Data_v2/` Path**，NoneneedpublishStep：

```yaml
# trainingConfiguration - automation/configs/stage2/my_training.yaml
experiment:
  purpose: "temperature_study"

model: "meta-llama/Llama-3.2-1B"
task: "Copa"
method: "zo"

data:
  # Recommended：usebatchPath（ according to ExperimentPurposegrouporganize，MoreIntuitive）
  path: "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"

  # orusesharedPath（Physical storage）
  # path: "Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa"
```

**how toQuick find to dataPath**：
```bash
python automation/stage1_generation/batch_tools/list_data_paths.py --dataset Copa --format yaml
```

###  can select：Release to Data/（Only use  at CompatibleOldscript）

IfneedCompatibleOldtrainingscript（Directlyuse `Data/` Directory），canusepublishTool：

```bash
python automation/stage1_generation/tools/publish_dataset.py \
    --source Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o \
    --dataset Copa \
    --target Data/rejection_sampling/0_data
```

**Note**: Only use  at CompatibleOldprojectstructure，NewprojectRecommendedDirectlyuse `Data_v2/` Path。

---

## summary

BatchSolution3++passPhysical storageandLogicViewSeparation，Implementation：

✅ **ParameterDeduplication**: sameParameter configurationOnlyGenerateOncedata
✅ **storageoptimize**: SectionsavediskspaceandAPIadjust use Cost
✅ **Flexiblegrouporganize**:  according to time/PurposeFlexiblegrouporganizeExperiment
✅ **easy at Trace**: ClearRecordeachExperimentSourceandParameter
✅ **backward compatible**: notImpactcurrenthastrainingscriptandTool

---

**Createdate**: 2024-12-29
**Version**: 1.0
**Maintenance**: Synthetic Data Generation Team
