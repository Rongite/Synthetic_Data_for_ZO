# Complete Pipeline Interaction Flow Simulation

This document simulates the complete interaction process between users and the system in the entire synthetic data generation pipeline, covering **all 5 datasets** and **two generation strategies**.

## 📚 Table of Contents

### Part 1: Two-Stage Mode (Detailed Complete Workflow)
- [Scenario 1: Copa Dataset (Two-Stage Mode)](#scenario-1-copa-dataset-two-stage-mode)
- [Scenario 2: BOOLQ Dataset (Two-Stage Mode)](#scenario-2-boolq-dataset-two-stage-mode)
- [Scenario 3: CB Dataset (Two-Stage Mode)](#scenario-3-cb-dataset-two-stage-mode)
- [Scenario 4: RTE Dataset (Two-Stage Mode)](#scenario-4-rte-dataset-two-stage-mode)
- [Scenario 5: ArcC Dataset (Two-Stage Mode)](#scenario-5-arcc-dataset-two-stage-mode)

### Part 2: Direct-All Mode (Parameter Study)
- [Scenario 6: Copa Dataset (Direct-All Mode)](#scenario-6-copa-dataset-direct-all-mode)
- [Scenario 7: BOOLQ Dataset (Direct-All Mode)](#scenario-7-boolq-dataset-direct-all-mode)
- [Scenario 8: CB Dataset (Direct-All Mode)](#scenario-8-cb-dataset-direct-all-mode)
- [Scenario 9: RTE Dataset (Direct-All Mode)](#scenario-9-rte-dataset-direct-all-mode)
- [Scenario 10: ArcC Dataset (Direct-All Mode)](#scenario-10-arcc-dataset-direct-all-mode)

### Appendix
- [Dataset Comparison Table](#dataset-comparison-table)
- [Key Points Summary](#key-points-summary)

---

# Part 1: Two-Stage Mode (Detailed Complete Workflow)

## Scenario 1: Copa Dataset (Two-Stage Mode)

### Step 0: Prepare Configuration File

```bash
$ cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/automation

# Copy configuration template
$ cp configs/examples/stage1_full_example_copa.yaml configs/stage1/my_copa.yaml

# Edit configuration file
$ vim configs/stage1/my_copa.yaml
```

**Configuration Content**:
```yaml
experiment:
  batch_id: "batch_20241230_copa_baseline"
  purpose: "copa_baseline"
  description: "Copa dataset baseline experiment"

task_name: "Copa"
training_method: "mezo"

dataset:
  task_name: "copa"
  dataset_name: "Copa"
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
  model: "gpt-4o"
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

  model: "gpt-4o"
  temperature: 0.0

  validation_prompt: |
    Task: Verify if the rephrased premise maintains consistency...
    {{VALIDATION_FEWSHOT}}
    ... (complete prompt) ...

  few_shot_examples: []  # Initially empty, auto-generated after checkpoint 2A
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
Experiment Description: Copa dataset baseline experiment
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

Usage (two_stage mode):
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
  原始: /home/ubuntu/.../Data/original/Copa/copa_train.jsonl
  改写: ../Copa/copa_train.jsonl

================================================================================
样本 21-40 对比 - 请仔细查看原始数据与改写数据
================================================================================

【Sample 21】
  Original premise:  The girl received a trophy.
  Rephrased premise:  The young lady was awarded a trophy for her achievement.
  choice1: She won a spelling bee.
  choice2: She made a new friend.
  question: cause
  label: 0

... (样本22-40) ...

================================================================================

请输入不合格样本的序号（21-40），多个序号用逗号分隔
示例: 23,27,35  表示这几个样本不合格
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

【Checkpoint 2A: 第21-40个样本处理】

Performing rejection sampling...
  Sample23: Using original data (unqualified)
  Sample35: Using original data (unqualified)

✓ Rejection sampling complete: ../validation_checkpoints/samples_21_40_validated.jsonl

生成validation few-shot examples...
✓ Generated18个validation few-shot examples
✓ Validation few-shot已保存: ../validation_checkpoints/validation_fewshot.json

✓ 标注记录已保存: ../validation_checkpoints/samples_21_40_annotation.json

================================================================================
✅ 样本 21-40 处理完成！
================================================================================

✅ Checkpoint 2A Complete总结:
  1. Rejection sampling: 18/20 个样本保留改写
  2. Validation few-shot: 生成了 18 个examples

Next step:
  Run: python annotate_samples.py --range 41-80
```

### Step7: Checkpoint 2B - 处理第41-80个样本

```bash
$ python annotate_samples.py --range 41-80
```

**System Output**（显示样本41-80后）：
```
请输入不合格样本的序号（41-80），多个序号用逗号分隔
示例: 43,47,55,72  表示这几个样本不合格
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

【Checkpoint 2B: 第41-80个样本处理】

Performing rejection sampling...
  Sample43: Using original data (unqualified)
  Sample47: Using original data (unqualified)
  Sample72: Using original data (unqualified)

✓ Rejection sampling complete: ../validation_checkpoints/samples_41_80_validated.jsonl

生成test_set...
✓ Generated40个test样本
  Ground Truth标注Statistics:
  - same (合格): 37
  - not the same (不合格): 3
✓ Test set已保存: ../validation_checkpoints/validation_test_set.json
  用途: 测试AI judge validation prompt的准确率

✓ 标注记录已保存: ../validation_checkpoints/samples_41_80_annotation.json

================================================================================
✅ 样本 41-80 处理完成！
================================================================================

✅ Checkpoint 2B Complete总结:
  1. Rejection sampling: 37/40 个样本保留改写
  2. Test set: 生成了 40 个标注样本
  3. Ground Truth: same=37, not the same=3

Next step:
  使用test_set测试validation prompt准确率
  Run: python generate_validation_test.py
```

### Step8: 测试AI Judge准确率

```bash
$ cp ../../../../automation/stage1_generation/tools/generate_validation_test.py .
$ python generate_validation_test.py
```

**System Output**：
```
加载test set...
  文件: ../validation_checkpoints/validation_test_set.json
  Sample数: 40

加载validation配置...
  模型: gpt-4o
  Temperature: 0.0

开始测试AI judge...

测试样本 1/40: 100%|████████████████████| 40/40 [05:30<00:00]

================================================================================
📊 测试结果
================================================================================
总测试样本: 40
AI判断为 same: 38
AI判断为 not the same: 2

与Ground Truth对比:
  ✓ 判断正确: 39
  ✗ 判断错误: 1

准确率: 97.5%

================================================================================
✅ 测试通过！准确率 ≥ 95%
================================================================================

可以继续执行Checkpoint 3（自动验证剩余数据）
```

### Step9: Checkpoint 3 - 自动验证剩余数据

```bash
$ python validate.py
```

**System Output**：
```
加载训练数据...
  文件: ../Copa/copa_train.jsonl
  总样本数: 400

已处理样本（Checkpoint 1和2）: 80
待验证样本: 320 (样本81-400)

加载validation配置...
  模型: gpt-4o
  Temperature: 0.0
  Few-shot examples: 18个

开始自动验证...

验证进度: 100%|████████████████████| 320/320 [38:45<00:00]

================================================================================
📊 验证结果统计
================================================================================
总验证样本: 320
判断为 same: 307 (95.9%)
判断为 not the same: 13 (4.1%)

Performing rejection sampling...
  ✓ 保留改写: 307 条
  ✗ 替换为原始: 13 条

保存最终数据...
✓ Saved: ../Copa/copa_train_final.jsonl

复制validation和test集...
✓ 已复制: ../Copa/copa_validation.jsonl
✓ 已复制: ../Copa/copa_test.jsonl

================================================================================
✅ 数据集生成完成！
================================================================================

最终数据集:
  训练集: ../Copa/copa_train_final.jsonl (400条)
    - 改写数据: 359条 (89.8%)
    - 原始数据: 41条 (10.2%)
  验证集: ../Copa/copa_validation.jsonl
  测试集: ../Copa/copa_test.jsonl

数据集路径: Data_v2/synthetic/_shared/Copa/temp09_topp10_gpt4o/Copa/
可直接用于MeZO训练！
```

---

## Scenario2: BOOLQ Dataset (Two-Stage Mode)

### Dataset Characteristics

- **Task Type**: 布尔问答（Yes/No）
- **Field to Rephrase**: `passage`（段落）
- **Other Fields**: `question`（问题）、`label`（0=No, 1=Yes）

### Key Configuration Modifications

```yaml
experiment:
  purpose: "boolq_baseline"

task_name: "BOOLQ"
training_method: "mezo"

dataset:
  task_name: "boolq"
  dataset_name: "BOOLQ"
  input_path: "Data/original/BOOLQ/boolq_train.jsonl"
  original_dir: "Data/original/BOOLQ"
  fields:
    - "passage"
    - "question"
    - "label"

generation:
  field_to_rephrase: "passage"  # BOOLQ改写passage字段

  rephrase_prompt: |
    You are tasked with rephrasing the given passage...
    {{REPHRASE_FEWSHOT}}

    **Original passage**: "{passage}"
    **Question**: "{question}"
    **Answer**: {"Yes" if label == 1 else "No"}

    **Directly output only one rephrased passage**:
```

### The workflow is the same as Copa

执行步骤1-10与Copa相同，只是字段名从`premise`变为`passage`。

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

- **Task Type**: 自然语言推理（NLI）
- **Field to Rephrase**: `hypothesis`（假设）
- **Other Fields**: `premise`（前提）、`label`（0=entailment, 1=contradiction, 2=neutral）

### Key Configuration Modifications

```yaml
experiment:
  purpose: "cb_baseline"

task_name: "CB"
training_method: "mezo"

dataset:
  task_name: "cb"
  dataset_name: "CB"
  input_path: "Data/original/CB/cb_train.jsonl"
  original_dir: "Data/original/CB"
  fields:
    - "premise"
    - "hypothesis"
    - "label"

generation:
  field_to_rephrase: "hypothesis"  # CB改写hypothesis字段

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

- **Task Type**: 自然语言推理（Recognizing Textual Entailment）
- **Field to Rephrase**: `premise`（前提）
- **Other Fields**: `hypothesis`（假设）、`label`（0=entailment, 1=not_entailment）
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
  description: "RTE数据集基线实验"

task_name: "RTE"
training_method: "mezo"

dataset:
  task_name: "rte"
  dataset_name: "RTE"
  input_path: "Data/original/RTE/rte_train.jsonl"
  original_dir: "Data/original/RTE"
  fields:
    - "premise"
    - "hypothesis"
    - "label"

generation:
  strategy: "two_stage"
  model: "gpt-4o"
  temperature: 0.9
  top_p: 1.0
  field_to_rephrase: "premise"  # RTE改写premise字段

  rephrase_prompt: |
    You are tasked with rephrasing the given premise for a textual entailment task.
    {{REPHRASE_FEWSHOT}}

    **Original premise**: "{premise}"
    **Hypothesis**: "{hypothesis}"
    **Label**: {["entailment", "not_entailment"][label]}

    **Directly output only one rephrased premise**:

validation:
  model: "gpt-4o"
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

流程与Copa完全相同（步骤1-10），只需：
1. 准备RTE配置文件
2. 执行 `python generator.py ../configs/stage1/my_rte.yaml`
3. 按照Checkpoint 1→Checkpoint 2A→Checkpoint 2B→Checkpoint 3依次执行

---

## Scenario5: ArcC Dataset (Two-Stage Mode)

### Dataset Characteristics

- **Task Type**: 多选题（科学推理）
- **Field to Rephrase**: `question`（问题）
- **Other Fields**: `choices`（选项）、`answerKey`（答案）
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
  description: "ARC-Challenge数据集基线实验"

task_name: "ArcC"
training_method: "mezo"

dataset:
  task_name: "arc_challenge"
  dataset_name: "ArcC"
  input_path: "Data/original/ArcC_Cloze/ARC-Challenge_train.jsonl"
  original_dir: "Data/original/ArcC_Cloze"
  fields:
    - "question"
    - "choices"
    - "answerKey"

generation:
  strategy: "two_stage"
  model: "gpt-4o"
  temperature: 0.9
  top_p: 1.0
  field_to_rephrase: "question"  # ArcC改写question字段

  rephrase_prompt: |
    You are tasked with rephrasing multiple-choice science questions.
    {{REPHRASE_FEWSHOT}}

    **Original question**: "{question}"
    **Choices**: {', '.join([f"{label}: {text}" for label, text in zip(choices['label'], choices['text'])])}
    **Correct answer**: {answerKey}

    **Directly output only one rephrased question**:

validation:
  model: "gpt-4o"
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
  choices: A: design a mathematical model, B: perform a controlled experiment, C: collect data, D: formulate a hypothesis
  answerKey: C
```

### Complete Execution Workflow

流程与Copa完全相同（步骤1-10），只是改写字段为`question`。

---

# Part 2: Direct-All模式（参数研究）

## Scenario6: Copa Dataset (Direct-All Mode)

### Use Case

已经通过第一次two-stage生成获得了可用的prompt和few-shot examples，现在想要快速探究不同temperature参数（0.5, 0.7, 0.9）对合成数据质量的影响。

### Step1: 准备Direct-All配置

```bash
$ cd automation/configs/stage1
$ cp ../examples/stage1_direct_all_copa.yaml temperature_05.yaml
$ vim temperature_05.yaml
```

**配置内容**：
```yaml
experiment:
  batch_id: "batch_20241230_temperature_study"
  purpose: "temperature_comparison"
  description: "比较temperature=0.5/0.7/0.9对Copa合成数据质量的影响"

task_name: "Copa"
training_method: "mezo"

dataset:
  task_name: "copa"
  dataset_name: "Copa"
  input_path: "Data/original/Copa/copa_train.jsonl"
  original_dir: "Data/original/Copa"
  fields:
    - "premise"
    - "choice1"
    - "choice2"
    - "question"
    - "label"

generation:
  # Rewriter API配置
  api_key: "sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c"
  base_url: "https://api2.aigcbest.top/v1"
  timeout: 120

  strategy: "direct_all"  # 🔥 直接全量生成
  model: "gpt-4o"
  temperature: 0.5  # 🔬 参数变量
  top_p: 1.0
  field_to_rephrase: "premise"

  # ⚠️ 必须包含完整的few-shot（从第一次two-stage生成中获得）
  rephrase_prompt: |
    You are tasked with rephrasing the given premise...

    ### Few-shot Examples:
    Original premise: "My body cast a shadow over the grass."
    Rephrased premise: "A shadow appeared on the grass beside me."

    Original premise: "The woman tolerated her friend's difficult behavior."
    Rephrased premise: "The woman was patient with her friend's challenging attitude."

    ... (完整的17 few-shot examples) ...

    ### Your Task:
    **Original premise**: "{premise}"
    **Choice 1**: "{choice1}"
    **Choice 2**: "{choice2}"
    **Question**: "{question}"
    **Correct answer**: "{choice1 if label == 0 else choice2}"

    **Directly output only one rephrased premise**:

# ⚠️ direct_all模式不需要validation配置
```

### Step2: 生成3个不同temperature的配置

```bash
# Temperature 0.5
$ cp temperature_05.yaml temperature_05.yaml

# Temperature 0.7
$ sed 's/temperature: 0.5/temperature: 0.7/' temperature_05.yaml > temperature_07.yaml

# Temperature 0.9
$ sed 's/temperature: 0.5/temperature: 0.9/' temperature_05.yaml > temperature_09.yaml
```

### Step3: 生成并运行（Temperature 0.5）

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

### Step4: 重复生成其他temperature

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

### Batch系统自动管理

```
Data_v2/synthetic/
├── _shared/                              # 物理存储
│   └── Copa/
│       ├── temp05_topp10_gpt4o/         # Temperature 0.5
│       │   └── Copa/copa_train.jsonl
│       ├── temp07_topp10_gpt4o/         # Temperature 0.7
│       │   └── Copa/copa_train.jsonl
│       └── temp09_topp10_gpt4o/         # Temperature 0.9
│           └── Copa/copa_train.jsonl
│
└── batch_20241230_temperature_study/    # Batch视图（符号链接）
    └── Copa/
        ├── temp05_topp10_gpt4o -> ../../_shared/Copa/temp05_topp10_gpt4o/
        ├── temp07_topp10_gpt4o -> ../../_shared/Copa/temp07_topp10_gpt4o/
        └── temp09_topp10_gpt4o -> ../../_shared/Copa/temp09_topp10_gpt4o/
```

---

## Scenario7: BOOLQ Dataset (Direct-All Mode)

### 配置文件关键差异

```yaml
experiment:
  batch_id: "batch_20241230_boolq_topp_study"
  purpose: "boolq_topp_comparison"
  description: "比较top_p=0.8/0.9/1.0对BOOLQ合成数据质量的影响"

task_name: "BOOLQ"

dataset:
  task_name: "boolq"
  dataset_name: "BOOLQ"
  input_path: "Data/original/BOOLQ/boolq_train.jsonl"
  fields:
    - "passage"
    - "question"
    - "label"

generation:
  strategy: "direct_all"
  temperature: 0.9  # 固定
  top_p: 0.8  # 🔬 研究变量
  field_to_rephrase: "passage"

  rephrase_prompt: |
    You are tasked with rephrasing the given passage...

    ### Few-shot Examples:
    Original passage: "The Supreme Court of the United States is..."
    Rephrased passage: "As the highest federal court in..."

    ... (17个完整examples) ...

    **Original passage**: "{passage}"
    **Question**: "{question}"
    **Answer**: {"Yes" if label == 1 else "No"}

    **Directly output only one rephrased passage**:
```

### 执行流程

```bash
# 生成top_p=0.8的配置
$ python generator.py ../configs/stage1/boolq_topp08.yaml
$ cd Data_v2/synthetic/_shared/BOOLQ/temp09_topp08_gpt4o/scripts
$ python rephrase_all.py

# 生成top_p=0.9的配置
$ python generator.py ../configs/stage1/boolq_topp09.yaml
$ cd Data_v2/synthetic/_shared/BOOLQ/temp09_topp09_gpt4o/scripts
$ python rephrase_all.py

# 生成top_p=1.0的配置
$ python generator.py ../configs/stage1/boolq_topp10.yaml
$ cd Data_v2/synthetic/_shared/BOOLQ/temp09_topp10_gpt4o/scripts
$ python rephrase_all.py
```

---

## Scenario8: CB Dataset (Direct-All Mode)

### 配置文件关键差异

```yaml
experiment:
  batch_id: "batch_20241230_cb_model_study"
  purpose: "cb_model_comparison"
  description: "比较gpt-4o vs gpt-4o-mini对CB合成数据质量的影响"

task_name: "CB"

dataset:
  task_name: "cb"
  dataset_name: "CB"
  input_path: "Data/original/CB/cb_train.jsonl"
  fields:
    - "premise"
    - "hypothesis"
    - "label"

generation:
  strategy: "direct_all"
  model: "gpt-4o"  # 🔬 研究变量（可改为gpt-4o-mini）
  temperature: 0.9
  top_p: 1.0
  field_to_rephrase: "hypothesis"

  rephrase_prompt: |
    You are tasked with rephrasing the given hypothesis...

    ### Few-shot Examples:
    Premise: "It was a complex language. Not written down but handed down."
    Original hypothesis: "the language was written down"
    Rephrased hypothesis: "the language existed in written form"

    ... (17个完整examples) ...

    **Premise**: "{premise}"
    **Original hypothesis**: "{hypothesis}"
    **Label**: {["entailment", "contradiction", "neutral"][label]}

    **Directly output only one rephrased hypothesis**:
```

### 执行流程

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

### 配置文件关键差异

```yaml
experiment:
  batch_id: "batch_20241230_rte_temp_study"
  purpose: "rte_temperature_comparison"
  description: "比较temperature=0.5/0.7/0.9对RTE合成数据质量的影响"

task_name: "RTE"

dataset:
  task_name: "rte"
  dataset_name: "RTE"
  input_path: "Data/original/RTE/rte_train.jsonl"
  fields:
    - "premise"
    - "hypothesis"
    - "label"

generation:
  strategy: "direct_all"
  model: "gpt-4o"
  temperature: 0.5  # 🔬 研究变量
  top_p: 1.0
  field_to_rephrase: "premise"

  rephrase_prompt: |
    You are tasked with rephrasing the given premise for textual entailment...

    ### Few-shot Examples:
    Original premise: "No Weapons of Mass Destruction Found in Iraq Yet."
    Rephrased premise: "Weapons of mass destruction have not been discovered in Iraq so far."

    ... (17个完整examples) ...

    **Original premise**: "{premise}"
    **Hypothesis**: "{hypothesis}"
    **Label**: {["entailment", "not_entailment"][label]}

    **Directly output only one rephrased premise**:
```

### 执行流程

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

### 配置文件关键差异

```yaml
experiment:
  batch_id: "batch_20241230_arcc_temp_study"
  purpose: "arcc_temperature_comparison"
  description: "比较temperature=0.5/0.7/0.9对ArcC合成数据质量的影响"

task_name: "ArcC"

dataset:
  task_name: "arc_challenge"
  dataset_name: "ArcC"
  input_path: "Data/original/ArcC_Cloze/ARC-Challenge_train.jsonl"
  fields:
    - "question"
    - "choices"
    - "answerKey"

generation:
  strategy: "direct_all"
  model: "gpt-4o"
  temperature: 0.5  # 🔬 研究变量
  top_p: 1.0
  field_to_rephrase: "question"

  rephrase_prompt: |
    You are tasked with rephrasing multiple-choice science questions...

    ### Few-shot Examples:
    Original question: "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?"
    Rephrased question: "To rapidly warm his hands through rubbing, which type of skin surface should George use to generate maximum heat?"

    ... (17个完整examples) ...

    **Original question**: "{question}"
    **Choices**: {', '.join([f"{label}: {text}" for label, text in zip(choices['label'], choices['text'])])}
    **Correct answer**: {answerKey}

    **Directly output only one rephrased question**:
```

### 执行流程

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

# 附录

## 数据集对比表

| 场景 | 数据集 | 模式 | 改写字段 | 其他字段 | 断点数 | 总耗时（估算） |
|------|--------|------|----------|----------|--------|---------------|
| 场景1 | Copa | Two-Stage | premise | choice1, choice2, question, label | 3个 | ~90分钟 |
| 场景2 | BOOLQ | Two-Stage | passage | question, label | 3个 | ~90分钟 |
| 场景3 | CB | Two-Stage | hypothesis | premise, label | 3个 | ~90分钟 |
| 场景4 | RTE | Two-Stage | premise | hypothesis, label | 3个 | ~90分钟 |
| 场景5 | ArcC | Two-Stage | question | choices, answerKey | 3个 | ~90分钟 |
| 场景6 | Copa | Direct-All | premise | choice1, choice2, question, label | 0个 | ~50分钟 |
| 场景7 | BOOLQ | Direct-All | passage | question, label | 0个 | ~50分钟 |
| 场景8 | CB | Direct-All | hypothesis | premise, label | 0个 | ~50分钟 |
| 场景9 | RTE | Direct-All | premise | hypothesis, label | 0个 | ~50分钟 |
| 场景10 | ArcC | Direct-All | question | choices, answerKey | 0个 | ~50分钟 |

## 人工参与时间对比

### Two-Stage模式（场景1-5）:
- **Checkpoint 1审核**: 浏览20个样本 + 输入序号 ≈ **1分钟**
- **Checkpoint 2A审核**: 浏览20个样本 + 输入序号 ≈ **1分钟**
- **Checkpoint 2B审核**: 浏览40个样本 + 输入序号 ≈ **2分钟**
- **总计人工时间**: ~**4分钟**

### Direct-All模式（场景6-10）:
- **无需人工参与** ✅
- 完全自动化，适合参数研究

## 关键要点总结

### 1. 批量输入模式
- 用户只需输入不合格序号（如：`3,7,12`），无需逐个确认
- 大幅减少人工交互时间

### 2. 自动Rejection Sampling
- 系统自动替换不合格样本为原始数据
- 所有3个断点（1-20, 21-40, 41-80）都执行rejection sampling

### 3. 自动Few-shot生成
- Checkpoint 1：从17个合格样本生成rephrase few-shot
- Checkpoint 2A：从18个合格样本生成validation few-shot

### 4. 自动标注
- 所有same/not the same标注由系统自动完成
- 生成test_set用于测试AI judge准确率

### 5. 多数据集零代码支持
- 5个数据集（Copa, BOOLQ, CB, RTE, ArcC）
- 只需修改配置文件中的字段名和prompt
- 无需修改任何代码

### 6. 参数去重（Batch方案3++）
- 自动检测相同参数配置
- 物理存储在`_shared/`，避免重复生成
- Batch视图通过符号链接组织实验

### 7. 两种生成策略
- **Two-Stage**: 探索性实验，需要确定prompt和few-shot
- **Direct-All**: prompt已确定，快速参数研究

## 使用建议

1. **首次实验**：使用Two-Stage模式确定最佳prompt和few-shot examples
2. **参数研究**：从Two-Stage获得few-shot后，使用Direct-All模式快速生成不同参数配置的数据
3. **人工审核**：认真审核前80个样本，确保AI judge准确率≥95%
4. **Batch管理**：使用`list_batches.py`等工具查看和管理实验
5. **数据复用**：善用Batch系统的参数去重功能，避免重复生成
