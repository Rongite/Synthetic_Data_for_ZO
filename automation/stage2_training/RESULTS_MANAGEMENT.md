# trainingResults ManagementSystem

## 🔴 Core design philosophy: Stage 1 and Stage 2 experiment purposes are independently established

### **Why do we need to classify?**

**Stage 1（DataGenerate）Experimentobjective**：
- Answer：" as whatGeneratethisData？"
- Example：`prompt_engineering`, `temperature_study`, `data_quality_optimization`
- storagelocation：`Data_v2/synthetic/{DataGenerateobjective}/`

**Stage 2（Model Training）Experimentobjective**：
- Answer："Why conduct this training?"
- Example：`model_comparison`, `hyperparameter_tuning`, `baseline_comparison`
- storagelocation：`Results_v2/{trainingobjective}/`

### **Typical scenario**

```
【Scenario】：Use the same dataset to conduct multiple types of different training experiments

Dataset（Stage 1）：
Data_v2/synthetic/prompt_engineering/copa_mezo_v1/
↑ DataGenerateobjective：Testprompt for DataqualityImpact

trainingExperiment（Stage 2）：
├── Results_v2/model_comparison/        ← trainingobjective：comparisonDifferentModel
├── Results_v2/hyperparameter_tuning/   ← trainingobjective：adjustmentLearning Rate
├── Results_v2/baseline_comparison/     ← trainingobjective： and originalDatacomparison
└── Results_v2/ablation_study/          ← trainingobjective：ablation experiment
```

**Key points**：
- ✅ SameDataset（`prompt_engineering/copa_mezo_v1`）can use  at multipleDifferenttrainingExperiment
- ✅ Each training experiment has its own purpose, results are classified according to training purpose
- ❌ If not classified, all results will be mixed in the `prompt_engineering` directory, and cannot be distinguishshedsh

---

## 📋 Directorystructure

### **NewResults_v2structure**

```
Results_v2/
└── {experiment_purpose}/           # 🆕 Experiment purpose classification (aligned with Data_v2)
    └── {Model}/
        └── {Task}_{Method}_{DataType}_{LR}/
            └── {Timestamp}/
                ├── experiment_config.yaml  # ExperimentConfiguration
                ├── {lr}_train.out         # trainingoutput
                ├── {lr}_train.err         # Erroroutput
                └── ...                    # Modelcheckpoint etc.
```

### **DirectoryDescription**

1. **experiment_purpose**: Experimentobjectiveclassification
   -  and Data_v2experiment_purpose for should
   - example such as ：`prompt_engineering`, `temperature_study`, `model_comparison`

2. **Model**: ModelName
   - example such as ：`meta-llama/Llama-3.2-1B`, `mistralai/Mistral-Nemo-Base-2407`

3. **Task_Method_DataType_LR**: Experiment identifier
   - Task: taskName（Copa, BOOLQ, CB etc.）
   - Method: Training Method（zo, fo_full, fo_lora）
   - DataType: Dataclasstype（original, synthetic etc.）
   - LR: Learning Rate（format, e.g. `1_7` indicates 1e-7）

4. **Timestamp**: timestamp (format: YYYYMMDD_HHMMSS)
   - Running same configuration multiple times will create different timestamp directories

---

## 🎯 coreFeature

### **1. trainingExperimentobjectiveclassification**

Training Results according to **trainingExperimentobjective**classification（ and DataGenerateobjectiveindependentestablish）：

```yaml
# ConfigurationFile
experiment:
  purpose: "hyperparameter_tuning"  # 🔴 trainingobjective！resultSave to : Results_v2/hyperparameter_tuning/

data:
  path: "Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa"
  #                        ↑ DataGenerateobjective（ and trainingobjectiveDifferent）
```

### **2. Must explicitly specify training purpose**

`experiment.purpose` must be explicitly specified, otherwise results will use default value `uncategorized`：

```yaml
# ✅ Recommended: Explicitly specify
experiment:
  purpose: "model_comparison"

# ⚠️  If not specified, results will be saved to Results_v2/uncategorized/
```

**Recommended training experiment purpose categories**：
- `baseline_comparison` -  and baselinecomparison
- `model_comparison` - comparisonDifferentModel
- `hyperparameter_tuning` - HyperparametersTune
- `ablation_study` - ablation experiment
- `prompt_effectiveness` - TestpromptEffect
- `data_quality_impact` - TestDataqualityImpact
- `scaling_study` - Scalability research

### **3. CompletemetaDataTrace**

eachtrainingExperimentAutomaticSaveCompleteConfiguration：

```yaml
# experiment_config.yaml
timestamp: "20251226_143000"
experiment_purpose: "prompt_engineering"
model: "meta-llama/Llama-3.2-1B"
task: "Copa"
method: "zo"
data:
  path: "Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa"
hyperparameters:
  learning_rate: 1e-6
  batch_size: 16
  steps: 20000
  seed: 0
training_info:
  env_vars: {...}
  command: "..."
  out_file: "..."
  err_file: "..."
```

---

## 📖 useGuide

### **Scenario1：HyperparametersTune（useSyntheticData）**

```yaml
# training_config.yaml
experiment:
  purpose: "hyperparameter_tuning"  # 🔴 trainingobjective：TuneHyperparameters
  description: "usecopa_mezo_v1DataTestDifferentLearning Rate"

model: "meta-llama/Llama-3.2-1B"
task: "Copa"
method: "zo"

data:
  path: "Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa"  # 🆕 Directly specify path

hyperparameters:
  learning_rate: [1e-6, 5e-7, 2e-7, 1e-7]
  batch_size: 16
  steps: 20000
  seed: 0
```

**Runtraining**：
```bash
python automation/stage2_training/trainer.py training_config.yaml
```

**resultSave to **：
```
Results_v2/hyperparameter_tuning/meta-llama/Llama-3.2-1B/
                ↑  according to trainingobjectiveclassification（notYesDataGenerateobjective）
├── Copa_zo_copa_mezo_v1_1_6/
│   └── 20251226_143000/
├── Copa_zo_copa_mezo_v1_5_7/
│   └── 20251226_143000/
├── Copa_zo_copa_mezo_v1_2_7/
│   └── 20251226_143000/
└── Copa_zo_copa_mezo_v1_1_7/
    └── 20251226_143000/
```

### **Scenario2：Modelcomparison（usesameData）**

```yaml
# training_config.yaml
experiment:
  purpose: "model_comparison"  # 🔴 trainingobjective：comparisonDifferentModel
  description: " in copa_mezo_v1Data up comparisonLlama and Mistral"

model: "mistralai/Mistral-Nemo-Base-2407"  # 🔧 TestDifferentModel
task: "Copa"
method: "zo"

data:
  path: "Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa"
  #                        ↑ Datafromselfprompt_engineeringExperiment
  #                        ↑ buttrainingobjectiveYesmodel_comparison

hyperparameters:
  learning_rate: 5e-7  # useKnownBestLearning Rate
  batch_size: 16
  steps: 20000
  seed: 0
```

**System as **：
- DataSource：`Data_v2/synthetic/prompt_engineering/...`
- trainingobjective：`model_comparison`（ and DataGenerateobjectiveDifferent）
- resultSave to ：`Results_v2/model_comparison/`

### **Scenario3：Baselinecomparison（originalData vs SyntheticData）**

```yaml
# training_config.yaml
experiment:
  purpose: "baseline_comparison"  # 🔴 trainingobjective：comparisonbaseline
  description: "comparisonoriginalData and SyntheticDatatrainingEffect"

model: "meta-llama/Llama-3.2-1B"
task: "Copa"
method: "zo"

data:
  path: "Data_v2/original/Copa"  # 🔧 useoriginalDataas as baseline

hyperparameters:
  learning_rate: 5e-7  # use and SyntheticDatasameHyperparameters
  batch_size: 16
  steps: 20000
  seed: 0
```

**resultSave to **：
```
Results_v2/baseline_comparison/meta-llama/Llama-3.2-1B/Copa_zo_original_5_7/20251226_143000/
```

**comparisonAnalysis**：
```
SyntheticDataresult：Results_v2/hyperparameter_tuning/.../Copa_zo_copa_mezo_v1_5_7/...
originalDataresult：Results_v2/baseline_comparison/.../Copa_zo_original_5_7/...
↑ twoExperimentallSave in eachselfExperimentobjectivedirectory ，methodeasycomparison
```

---

## 🔧 ManageTool

### **list_results.py**

listoutandManageallTraining Results。

#### **Viewsummary need **

```bash
python automation/stage2_training/list_results.py
```

**outputExample**：
```
================================================================================
Training Resultssummary need  - Results_v2
================================================================================

📁 Experimentobjective: prompt_engineering
   Experimentcount: 12
   └─ meta-llama/Llama-3.2-1B: 12 Experiment

📁 Experimentobjective: temperature_study
   Experimentcount: 8
   └─ meta-llama/Llama-3.2-1B: 8 Experiment

📁 Experimentobjective: baseline
   Experimentcount: 4
   └─ meta-llama/Llama-3.2-1B: 4 Experiment

================================================================================
total: 3 Experimentobjective, 24 trainingExperiment
================================================================================
```

#### **ViewDetailedinformation**

```bash
# ViewallExperimentDetailedinformation
python automation/stage2_training/list_results.py --detail

# ViewspecialspecifyExperimentobjectiveDetailedinformation
python automation/stage2_training/list_results.py --detail --purpose prompt_engineering
```

**outputExample**：
```
================================================================================
Training ResultsDetails
================================================================================

📁 Experimentobjective: prompt_engineering
--------------------------------------------------------------------------------

  [1] Copa_zo_copa_mezo_v1_1_6
      Model: meta-llama/Llama-3.2-1B
      time: 20251226_143000
      Path: Results_v2/prompt_engineering/meta-llama/Llama-3.2-1B/Copa_zo_copa_mezo_v1_1_6/20251226_143000
      task: Copa
      method: zo
      Hyperparameters:
        - LR: 1e-06
        - BS: 16
        - Steps: 20000
        - Seed: 0
      Data: Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa

  [2] Copa_zo_copa_mezo_v1_5_7
      ...
```

---

## 🔄 Data-result for shouldOffsystem

### **CompleteExperimentTracechain**

```
Stage 1：DataGenerate
Data_v2/synthetic/
└── prompt_engineering/           # Experimentobjective
    └── copa_mezo_v1/              # ExperimentID
        ├── Copa/                  # Dataset
        │   ├── copa_train.jsonl
        │   ├── copa_validation.jsonl
        │   └── copa_test.jsonl
        └── experiment_metadata.json  # DataGeneration parameters

                    ⬇

Stage 2：Model Training
Results_v2/
└── prompt_engineering/           # 🔗 sameExperimentobjective
    └── meta-llama/Llama-3.2-1B/
        └── Copa_zo_copa_mezo_v1_1_6/
            └── 20251226_143000/
                └── experiment_config.yaml  # trainingParameter
```

### ** for shouldOffsystem**

| Dataset | Training Results |
|--------|----------|
| `Data_v2/synthetic/{purpose}/{exp_id}/{Dataset}` | `Results_v2/{purpose}/{Model}/{Task}_{Method}_{exp_id}_{LR}/{Timestamp}` |

**Key points**：
- `{purpose}`  in twosideMaintainConsistent
- `{exp_id}`  in Results Directoryname in bodycurrent
- pass`experiment_config.yaml` in `data.path`canTrace to sourceData

---

## 📊 best practices

### **1. trainingExperimentobjectiveNamingspecification**

**Recommended training experiment purpose categories**（Stage 2）：

- `baseline_comparison` -  and baselinecomparison
- `model_comparison` - ModelcomparisonExperiment
- `hyperparameter_tuning` - HyperparametersTune
- `ablation_study` - ablation experiment
- `prompt_effectiveness` - TestpromptEffect
- `data_quality_impact` - TestDataqualityImpact
- `scaling_study` - Scalability research
- `method_comparison` - Training Methodcomparison（MeZO vs LoRA vs Full FT）

**DataGenerateExperimentobjectiveclasscategory**（Stage 1，onlyprovideReference）：

- `prompt_engineering` - PromptoptimizeExperiment
- `temperature_study` - TemperatureParameterResearch
- `data_quality_optimization` - Dataqualityoptimize
- `few_shot_study` - Few-shotExampleResearch

### **2. ConfigurationFileOrganize**

 according to **trainingExperimentobjective**OrganizeConfigurationFile：

```
automation/configs/stage2/
├── baseline_comparison/
│   ├── copa_original.yaml
│   └── boolq_original.yaml
├── model_comparison/
│   ├── copa_llama_vs_mistral.yaml
│   └── copa_llama_1b_vs_3b.yaml
├── hyperparameter_tuning/
│   ├── copa_lr_sweep.yaml
│   └── copa_bs_sweep.yaml
└── prompt_effectiveness/
    ├── copa_v1_vs_v2.yaml
    └── copa_temp_comparison.yaml
```

**Note**：ConfigurationFile according to trainingobjectiveclassification，notYes according to Datasetclassification

### **3. ExperimentRecord**

eachtimesImportantExperiment back ， in  for shouldExperimentobjectivedirectory Record：

```bash
#  in Results_v2/{trainingobjective}/README.md in Record
echo "## ExperimentRecord

### 2025-12-26: Learning RatescanExperiment
- trainingobjective: hyperparameter_tuning
- Dataset: Data_v2/synthetic/prompt_engineering/copa_mezo_v1/
- Model: Llama-3.2-1B
- Learning Rategrid: [1e-6, 5e-7, 2e-7, 1e-7]
- Bestresult: LR=5e-7, Acc=85.2%
- preparenote: 5e-7YesBestLearning Rate， use  at  back continueExperiment
" >> Results_v2/hyperparameter_tuning/README.md
```

---

## ⚠️ Notematteritem

### **1. Stage 1 and Stage 2ExperimentobjectiveYesindependentestablish！**

🔴 **mostImportantconcept**：

```
❌ ErrorUnderstand：
   Datafromself Data_v2/synthetic/prompt_engineering/...
   → resultshouldSave to  Results_v2/prompt_engineering/

✅ correctUnderstand：
   Datafromself Data_v2/synthetic/prompt_engineering/...  ← DataGenerateobjective
   trainingobjectiveYes hyperparameter_tuning                    ← trainingExperimentobjective
   → resultSave to  Results_v2/hyperparameter_tuning/
```

### **2. mustexplicitlyreferspecifytrainingExperimentobjective**

System**not will ** from DataPathAutomaticInferencetrainingExperimentobjective：

```yaml
# ❌ Error：nohasreferspecifyexperiment.purpose
data:
  path: "Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa"
# → result will Save to  Results_v2/uncategorized/

# ✅ correct：explicitlyreferspecifytrainingobjective
experiment:
  purpose: "hyperparameter_tuning"
data:
  path: "Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa"
# → resultSave to  Results_v2/hyperparameter_tuning/
```

### **3. Oldformatcompatibility**

SystemstillsupportOld`data.type`format，butRecommendeduseNew`data.path`：

```yaml
# ✅ Recommended（Newformat）
data:
  path: "Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa"

# ⚠️  alreadydiscard use （Oldformat）
data:
  type: "synthetic_mezo_gpt4o_v1"
```

### **3. timestampisolated**

sameConfigurationmultipletimesRun will CreateDifferenttimestampDirectory，avoidOverride：

```
Copa_zo_copa_mezo_v1_1_6/
├── 20251226_143000/  # line1timesRun
├── 20251226_153000/  # line2timesRun
└── 20251227_093000/  # line3timesRun
```

---

## 🎉 Summary

### **NewSystemAdvantage**

1. ✅ **Experimentobjectiveclassification**：result according to ExperimentobjectiveAutomaticOrganize
2. ✅ **smart can Inference**： from DataPathAutomaticInferenceExperimentobjective
3. ✅ **CompleteTrace**：Dataset ↔ Training ResultsComplete for should
4. ✅ **metaData management**：AutomaticSaveallExperimentParameter
5. ✅ **ManageTool**：list_results.pyQuickViewresult

### ** and OldSystemcomparison**

| Feature | OldSystem | NewSystem |
|------|--------|--------|
| resultOrganize | ❌ allresultmixed in thisup | ✅  according to Experimentobjectiveclassification |
| ExperimentTrace | ❌ ManualRecord | ✅ AutomaticTrace to Dataset |
| ConfigurationManage | ⚠️  PartialSave | ✅ CompleteSave |
| ViewTool | ❌ None | ✅ list_results.py |

---

**OnstartyoutrainingExperiment！** 🚀
