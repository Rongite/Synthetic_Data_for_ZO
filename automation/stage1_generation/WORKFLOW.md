# 阶段1完整工作流程：人工断点 + 调参复用

本文档说明新自动化系统如何支持：
1. **首次生成**：包含3个必要的人工断点验证
2. **调参实验**：基于已验证prompt，快速调参复用

---

## ✅ 实现状态

### 已实现工具

| 工具 | 位置 | 功能 | 状态 |
|------|------|------|------|
| `generator.py` | `automation/stage1_generation/` | 生成rephrase和validation脚本 | ✅ **已修复** |
| `review_top20.py` | `automation/stage1_generation/tools/` | 断点1：人工审核前20个样本 | ✅ 已实现 |
| `extract_samples.py` | `automation/stage1_generation/tools/` | 提取指定范围样本 | ✅ 已实现 |
| `annotate_samples.py` | `automation/stage1_generation/tools/` | 断点2：人工标注21-80样本 | ✅ 已实现 |
| `generate_validation_test.py` | `automation/stage1_generation/tools/` | 生成judger测试脚本 | ✅ 已实现 |

### 🔧 关键修复

**generator.py (validate.py生成逻辑)**:
- ✅ **已修复排除21-40样本的逻辑**
- 生成的`validate.py`现在会跳过样本21-40（索引20-39）
- 这些样本用作judger的few-shot examples，不应被judger验证（避免数据泄露）
- 修复位置：`generator.py:300-308`

```python
# 🔴 排除样本21-40（索引20-39）
if 20 <= i < 40:
    # 直接使用合成数据，不经过judger验证
    out_file.write(json.dumps(synthetic, ensure_ascii=False) + "\n")
    correct_count += 1
    total_count += 1
    continue
```

---

## ⚠️ 重要说明

### 只合成train数据

**Pipeline只会合成/改写训练数据（train.jsonl），validation和test数据直接从原始数据集复制**：

- ✅ **{dataset}_train.jsonl** → 合成数据（经过rephrase + validation + rejection sampling）
- 📋 **{dataset}_validation.jsonl** → 原始数据（从 Data/original/ 复制）
- 📋 **{dataset}_test.jsonl** → 原始数据（从 Data/original/ 复制）

这样做是为了：
1. **保持evaluation标准化** - validation和test数据保持原始状态，确保公平评估
2. **实验结果可比较** - 不同实验使用相同的evaluation数据
3. **符合研究惯例** - 只在训练阶段使用合成数据增强

**自动处理**: `validate.py` 在验证train数据后，会自动从原始数据集复制validation和test文件。

---

## 🗂️ Batch方案3++ - 智能实验管理

### 什么是Batch方案？

Batch方案3++通过**物理存储与逻辑视图分离**，实现多参数实验的智能管理和自动去重。

**核心机制**:
- **物理存储 (_shared/)**: 存放实际数据，按参数指纹去重
- **逻辑视图 (batch_*)**: 通过符号链接组织实验，按时间/目的分组

**参数去重**: 相同参数配置的数据只生成一次，不同batch可以复用

### 目录结构示例

```
Data_v2/synthetic/
├── _shared/                                    # 物理数据（去重）
│   └── Copa/
│       ├── temp05_topp10_gpt4o/               # 实际数据
│       ├── temp07_topp09_gpt4o/
│       └── temp09_topp10_gpt4o/
│
├── batch_20241229_temperature/                 # Batch 1: 温度实验
│   └── Copa/
│       ├── temp05_topp10_gpt4o -> ../../_shared/...
│       ├── temp07_topp10_gpt4o -> ../../_shared/...
│       └── temp09_topp10_gpt4o -> ../../_shared/...
│
└── batch_20241230_topp/                        # Batch 2: top_p实验
    └── Copa/
        ├── temp07_topp08_gpt4o -> ../../_shared/...
        └── temp07_topp09_gpt4o -> ../../_shared/...  # 复用！
```

### 配置文件设置

在配置文件中添加 `experiment.batch_id`:

```yaml
experiment:
  # Batch ID（可选，不指定则自动生成）
  batch_id: "batch_20241229_temperature"
  purpose: "temperature_study"
  description: "研究temperature参数对合成数据质量的影响"

generation:
  model: "gpt-4o"
  temperature: 0.7  # 实验变量
  # ...
```

### 自动去重原理

当你运行 `generator.py` 时：

1. **计算参数指纹**: 基于所有影响数据生成的参数（模型、temperature、top_p、prompts等）
2. **查找已有数据**: 在 `_shared/{Dataset}/` 中搜索相同指纹
3. **复用或新建**:
   - 找到相同指纹 → 复用物理数据，创建batch符号链接
   - 未找到 → 创建新物理目录，生成数据

**节省资源**: 无需重复生成相同参数的数据，节省API调用成本和时间

### Batch管理工具

```bash
# 列出所有batch
python automation/stage1_generation/batch_tools/list_batches.py --verbose

# 查看batch详情
python automation/stage1_generation/batch_tools/list_batch_experiments.py \
    batch_20241229_temperature --verbose

# 查看物理数据使用情况
python automation/stage1_generation/batch_tools/list_shared_experiments.py \
    --dataset Copa --verbose

# 比较实验参数
python automation/stage1_generation/batch_tools/compare_experiments.py \
    --shared Copa/temp07_topp10_gpt4o \
    --shared Copa/temp09_topp10_gpt4o
```

**详细说明**: 参见 [BATCH_GUIDE.md](../../BATCH_GUIDE.md)

---

## 工作流程概览

```
首次生成（有人工断点）              调参实验（无人工断点）
┌────────────────────────┐          ┌────────────────────────┐
│ 1. 创建draft配置        │          │ 1. 基于validated模板    │
│    (人工编写初始prompt) │          │    创建实验配置         │
└───────────┬────────────┘          └───────────┬────────────┘
            │                                   │
            v                                   v
┌────────────────────────┐          ┌────────────────────────┐
│ 2. 生成脚本             │          │ 2. 生成脚本             │
│    (generator.py)      │          │    (generator.py)      │
└───────────┬────────────┘          └───────────┬────────────┘
            │                                   │
            v                                   v
┌────────────────────────┐          ┌────────────────────────┐
│ 🔴 断点1: 审核top20    │          │ 3. 直接运行             │
│    → 生成few-shot      │          │    rephrase_all.py     │
└───────────┬────────────┘          │    (无需人工审核)      │
            │                       └───────────┬────────────┘
            v                                   │
┌────────────────────────┐                     │
│ 3. 生成rest数据         │                     │
└───────────┬────────────┘                     │
            │                                   │
            v                                   │
┌────────────────────────┐                     │
│ 🔴 断点2: 标注21-80    │                     │
│    → 生成validation    │                     │
│       prompt few-shot  │                     │
└───────────┬────────────┘                     │
            │                                   │
            v                                   v
┌────────────────────────┐          ┌────────────────────────┐
│ 🔴 断点3: 测试prompt   │          │ 4. 使用已验证的         │
│    → 调优直到≥95%      │          │    validation prompt   │
└───────────┬────────────┘          │    验证数据             │
            │                       └───────────┬────────────┘
            v                                   │
┌────────────────────────┐                     │
│ 4. 批量验证数据         │                     │
└───────────┬────────────┘                     │
            │                                   │
            v                                   v
┌────────────────────────┐          ┌────────────────────────┐
│ 5. 存档为validated模板  │          │ 5. 完成！               │
│    (可复用)            │          │    对比不同版本质量     │
└────────────────────────┘          └────────────────────────┘
```

---

## 场景A：首次生成（需要人工验证）

### 前置条件
- 原始数据已准备：`Data/original/{Task}/{task}_train.jsonl`
- 您已人工编写初始版prompt（无few-shot）

### Step 1: 创建draft配置

创建配置文件：`automation/configs/stage1/drafts/copa_mezo_v1_draft.yaml`

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

  # 人工编写的初始prompt（无few-shot）
  rephrase_prompt: |
    You are tasked with rephrasing...
    （人工编写的prompt内容）

validation:
  model: "gpt-4o"
  temperature: 0.0

  # 人工编写的初始validation prompt（无few-shot）
  validation_prompt: |
    Judge if the rephrased premise...
    （人工编写的prompt内容）

  # 暂时留空，后续自动生成
  few_shot_examples: []
```

### Step 2: 生成脚本

```bash
python automation/stage1_generation/generator.py \
       automation/configs/stage1/drafts/copa_mezo_v1_draft.yaml
```

**输出**：
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

### Step 3: 生成前20个样本

```bash
cd Data_v2/synthetic/Copa_mezo_gpt4o_v1/scripts/
export OPENAI_API_KEY="your-key"
python rephrase_top20.py
```

**输出**：`copa_train_top20.jsonl`（20个样本）

---

### 🔴 **断点1：人工审核top20样本**

#### 3.1 并排查看原始vs合成数据

```bash
# 使用人工审核工具（待实现）
python review_top20.py
```

**审核界面示例**：
```
样本 1/20:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原始premise:
  "My body cast a shadow over the grass."

合成premise:
  "A shadow from my body fell across the grass."

Choice 1: The sun was rising.
Choice 2: The grass was cut.
Question: cause
Correct answer: Choice 1

此改写是否合格？
  [y] 合格 - 语义一致，质量良好
  [n] 不合格 - 语义改变或质量差
  [s] 跳过
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
您的判断: y

（继续审核样本2-20...）
```

**输出**：
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
    // ... 19个更多样本
  ]
}
```

#### 3.2 自动生成few-shot并注入到rephrase_rest.py

```bash
# 基于审核结果自动生成few-shot examples
python update_rest_prompt.py
```

**功能**：
1. 读取`top20_review.json`
2. 提取`judgment == "approved"`的样本
3. 格式化成few-shot examples
4. 自动更新`rephrase_rest.py`中的prompt

---

### Step 4: 生成剩余380个样本

```bash
python rephrase_rest.py
# 现在prompt中包含了人工审核通过的few-shot examples
```

**输出**：`copa_train_rest.jsonl`（380个样本）

### Step 5: 合并数据

```bash
cat copa_train_top20.jsonl copa_train_rest.jsonl > ../copa_train.jsonl
```

---

### 🔴 **断点2：人工标注第21-80个样本**

> **✅ 工具已实现**: `extract_samples.py`, `annotate_samples.py` 位于 `automation/stage1_generation/tools/`

#### 5.1 提取第21-80个样本

```bash
# 进入数据集目录
cd Data_v2/synthetic/{experiment_purpose}/{experiment_id}/{Dataset}/

# 提取样本21-80（共60个）
python /path/to/automation/stage1_generation/tools/extract_samples.py \
    --range 21-80 \
    --input Copa/copa_train.jsonl

# 或者在tools目录直接运行
cd /path/to/automation/stage1_generation/tools/
python extract_samples.py \
    --range 21-80 \
    --input /path/to/Copa/copa_train.jsonl
```

**输出**：
```
validation_checkpoints/samples_21_80.jsonl  # 60个样本
```

#### 5.2 人工标注

```bash
# 在数据集目录或tools目录运行
python annotate_samples.py validation_checkpoints/samples_21_80.jsonl

# 可选参数：
# --output validation_checkpoints/custom_name_annotated.json  # 自定义输出文件
# --no-resume                                                  # 重新开始，不继续上次标注
```

**标注界面示例**：
```
样本 1/60 (原始数据第21个):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原始premise:
  "The tenant misplaced his keys."

合成premise:
  "The tenant lost his apartment keys."

Choice 1: His landlord unlocked the door.
Choice 2: His landlord repaired the door.
Question: effect
Correct answer: Choice 1

语义是否一致？
  [s] same - 语义一致
  [n] not the same - 语义改变
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
您的判断: s

（继续标注样本2-60...）
```

**输出**：
```json
// validation_checkpoints/samples_21_80_annotated.json
{
  "total": 60,
  "same": 57,
  "not_the_same": 3,
  "annotations": [
    {
      "index": 20,  // 原始数据中的索引（第21个）
      "original_premise": "The tenant misplaced his keys.",
      "rephrased_premise": "The tenant lost his apartment keys.",
      "choice1": "His landlord unlocked the door.",
      "choice2": "His landlord repaired the door.",
      "question_type": "effect",
      "correct_answer": "Choice 1",
      "judgment": "same",  // 人工判断
      "note": ""
    },
    // ... 59个更多样本
  ]
}
```

#### 5.3 自动生成validation prompt测试脚本

> **✅ 工具已实现**: `generate_validation_test.py` 位于 `automation/stage1_generation/tools/`

```bash
# 使用默认路径
python /path/to/automation/stage1_generation/tools/generate_validation_test.py

# 或指定参数
python generate_validation_test.py \
    --annotations validation_checkpoints/samples_21_80_annotated.json \
    --fewshot-range 21-40 \
    --test-range 41-80 \
    --output scripts/validate_prompt_test.py \
    --api-key your-api-key \
    --base-url https://api.openai.com/v1
```

**功能**：
1. 读取`samples_21_80_annotated.json`
2. **第21-40个"same"样本** → 格式化成validation prompt的few-shot examples
3. **第41-80个所有样本** → 格式化成test_set（包含ground truth）
4. 自动生成`validate_prompt_test.py`

**生成的测试脚本**：
```python
# scripts/validate_prompt_test.py
def generate_validation_prompt(...):
    return f"""
    Judge if the rephrased premise...

    ### Few-shot Examples (来自第21-40个):
    Example 1:
    Original: The tenant misplaced his keys.
    Rephrased: The tenant lost his apartment keys.
    Judgment: same

    ... (共20个few-shot)
    """

# Test set（来自第41-80个，共40个）
test_set = [
    {
        "original_premise": "...",
        "rephrased_premise": "...",
        "ground_truth": "same"  # 人工标注
    },
    ...
]

# 测试prompt准确率
for item in test_set:
    response = gpt4o_judge(item)
    if response == item["ground_truth"]:
        correct += 1

accuracy = correct / len(test_set)
print(f"Prompt准确率: {accuracy:.2%}")
```

---

### 🔴 **断点3：测试并调优validation prompt**

#### 6.1 测试prompt准确率

```bash
python validate_prompt_test.py
```

**输出示例**：
```
Testing validation prompt on 40 samples...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Test Results:
  Correct: 36 / 40
  Accuracy: 90.0%

✗ Prompt未达标（需要≥95%）

错误样本:
  Sample 23: 判断为same，实际为not the same
  Sample 45: 判断为not the same，实际为same
  Sample 67: 判断为same，实际为not the same
  Sample 78: 判断为not the same，实际为same

建议:
  1. 检查few-shot examples中是否包含类似的反例
  2. 调整validation_prompt中的判断标准描述
  3. 增加对边界情况的说明

请手动调整配置文件中的validation_prompt，然后重新运行此测试。
```

#### 6.2 手动调优prompt

编辑配置文件：
```bash
vim automation/configs/stage1/drafts/copa_mezo_v1_draft.yaml
```

修改`validation.validation_prompt`，例如：
- 添加更明确的判断标准
- 补充边界情况的few-shot examples
- 调整prompt用词

#### 6.3 重新生成脚本并测试

```bash
# 重新生成脚本
python automation/stage1_generation/generator.py \
       automation/configs/stage1/drafts/copa_mezo_v1_draft.yaml

# 重新测试
cd Data_v2/synthetic/Copa_mezo_gpt4o_v1/scripts/
python validate_prompt_test.py
```

**重复6.1-6.3直到准确率≥95%**：
```
Test Results:
  Correct: 39 / 40
  Accuracy: 97.5%

✓ Prompt已达标！
  创建通过标记: validation_checkpoints/prompt_test_passed.flag
```

---

### Step 7: 批量验证所有400个样本

```bash
python validate.py
```

**功能**：
1. 检查是否存在`prompt_test_passed.flag`（门禁）
2. 如果不存在，拒绝执行并提示先运行测试
3. 如果存在，使用已验证的validation prompt验证所有400个样本
4. Rejection sampling：不合格的用原始数据替换

**输出**：
```
验证完成!
通过率: 381/400 = 95.25%
输出文件: copa_train_validated.jsonl
```

---

### Step 8: 存档为validated模板

```bash
cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/

python automation/stage1_generation/archive_validated_config.py \
       --source automation/configs/stage1/drafts/copa_mezo_v1_draft.yaml \
       --data-dir Data_v2/synthetic/Copa_mezo_gpt4o_v1/
```

**输出**：
```
✓ 配置已存档!

模板路径: automation/configs/stage1/templates/copa_mezo_validated.yaml
归档路径: automation/configs/stage1/archive/2024-12/copa_mezo_v1_complete_20241224_153000.yaml

现在可以基于此模板创建调参实验配置。
```

---

## 场景B：调参实验（基于已验证prompt）

### 前置条件
- 已有validated模板：`automation/configs/stage1/templates/copa_mezo_validated.yaml`
- Prompt已通过所有人工验证
- 想要调整生成参数观察对数据质量的影响

### Step 1: 创建实验配置

```bash
# 实验1: 提高temperature
python automation/stage1_generation/create_experiment.py \
       --template automation/configs/stage1/templates/copa_mezo_validated.yaml \
       --version v2 \
       --param generation.temperature=0.7
```

**输出**：
```
✓ 配置已创建: automation/configs/stage1/experiments/copa_mezo_v2_temperature07.yaml

参数变更:
  - generation.temperature: 0.5 → 0.7
```

### Step 2: 生成脚本

```bash
python automation/stage1_generation/generator.py \
       automation/configs/stage1/experiments/copa_mezo_v2_temperature07.yaml
```

**输出**：
```
Data_v2/synthetic/Copa_mezo_gpt4o_v2/
├── scripts/
│   ├── rephrase_all.py      # 包含已验证的few-shot
│   ├── rephrase_top20.py
│   ├── rephrase_rest.py
│   └── validate.py          # 包含已验证的validation prompt
└── ...
```

### Step 3: 直接生成完整数据集（无需人工审核）

```bash
cd Data_v2/synthetic/Copa_mezo_gpt4o_v2/scripts/
export OPENAI_API_KEY="your-key"

# 直接运行rephrase_all.py（400个样本）
python rephrase_all.py
```

**关键**：
- ✅ 使用已验证的few-shot examples
- ✅ 只有temperature改变（0.5 → 0.7）
- ✅ 无需重复人工审核断点1-3

### Step 4: 使用已验证的validation prompt验证

```bash
python validate.py
```

**输出**：
```
验证完成!
通过率: 378/400 = 94.5%
输出文件: copa_train_validated.jsonl
```

### Step 5: 对比不同版本

```bash
# 对比v1和v2的数据质量
python automation/analysis/compare_versions.py \
       --v1 Data_v2/synthetic/Copa_mezo_gpt4o_v1/copa_train_validated.jsonl \
       --v2 Data_v2/synthetic/Copa_mezo_gpt4o_v2/copa_train_validated.jsonl
```

**输出**：
```
版本对比:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
指标                   v1          v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
验证通过率           95.25%      94.50%
平均编辑距离         12.3        15.7
词汇多样性 (TTR)     0.82        0.87
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

结论: v2 (temp=0.7) 多样性更高，但通过率略低
```

---

## 多实验并行

```bash
# 创建多个实验配置
python automation/stage1_generation/create_experiment.py \
       --template automation/configs/stage1/templates/copa_mezo_validated.yaml \
       --version v3 \
       --param generation.model=gpt-4o-mini

python automation/stage1_generation/create_experiment.py \
       --template automation/configs/stage1/templates/copa_mezo_validated.yaml \
       --version v4 \
       --param generation.temperature=0.9

# 并行生成（使用不同GPU或时间段）
for version in v2 v3 v4; do
  config="automation/configs/stage1/experiments/copa_mezo_${version}_*.yaml"
  python automation/stage1_generation/generator.py $config
  cd Data_v2/synthetic/Copa_mezo_*_${version}/scripts/
  python rephrase_all.py &
  cd -
done
```

---

## 目录结构总结

```
automation/configs/stage1/
├── drafts/                                    # 首次生成：待验证配置
│   └── copa_mezo_v1_draft.yaml               # 人工编写初始prompt
│
├── templates/                                 # 已验证模板（可复用）
│   ├── copa_mezo_validated.yaml              # Copa任务MeZO方法模板
│   ├── rte_mezo_validated.yaml               # RTE任务模板
│   └── README.md
│
├── experiments/                               # 调参实验配置
│   ├── copa_mezo_v2_temperature07.yaml       # 实验：temp=0.7
│   ├── copa_mezo_v3_gpt4omini.yaml           # 实验：换模型
│   └── copa_mezo_v4_temperature09.yaml       # 实验：temp=0.9
│
├── archive/                                   # 历史存档
│   └── 2024-12/
│       └── copa_mezo_v1_complete_20241224.yaml
│
└── examples/                                  # 示例
    └── stage1_example_copa_mezo.yaml

Data_v2/synthetic/Copa_mezo_gpt4o_v1/
├── scripts/
│   ├── rephrase_top20.py
│   ├── rephrase_rest.py
│   ├── rephrase_all.py
│   ├── validate.py
│   ├── review_top20.py                       # 🆕 人工审核工具
│   ├── annotate_samples.py                   # 🆕 人工标注工具
│   ├── validate_prompt_test.py               # 🆕 自动生成测试脚本
│   ├── extract_samples.py                    # 🆕 样本提取
│   ├── update_rest_prompt.py                 # 🆕 自动注入few-shot
│   └── generate_validation_test.py           # 🆕 生成测试脚本
├── validation_checkpoints/                    # 🆕 人工验证记录
│   ├── top20_review.json                     # 断点1记录
│   ├── samples_21_80_annotated.json          # 断点2记录
│   ├── prompt_test_results.json              # 断点3记录
│   └── prompt_test_passed.flag               # 通过标记
├── copa_train_top20.jsonl
├── copa_train_rest.jsonl
├── copa_train.jsonl                          # 合并后的未验证数据
├── copa_train_validated.jsonl                # 最终验证通过数据
├── generation_config.yaml
└── README.md
```

---

## 关键原则

1. **首次生成必须经过人工断点**
   - 断点1：审核top20 → 生成rephrase few-shot
   - 断点2：标注21-80 → 生成validation few-shot和test_set
   - 断点3：测试prompt → 调优直到≥95%

2. **调参实验复用已验证prompt**
   - 继承templates/中的配置
   - 只修改生成参数
   - 直接使用rephrase_all.py
   - 无需重复人工审核

3. **所有prompt都需人工制作**
   - 初始prompt人工编写
   - Few-shot examples由人工审核自动生成
   - Prompt调优由人工迭代完成

4. **版本管理**
   - drafts/: 首次生成的配置
   - templates/: 验证通过的模板
   - experiments/: 调参实验配置
   - archive/: 历史记录（包含完整验证历史）

---

## 待实现工具清单

### 高优先级（人工断点必需）
- [ ] `review_top20.py` - 断点1人工审核界面
- [ ] `update_rest_prompt.py` - 自动注入few-shot到rephrase_rest.py
- [ ] `extract_samples.py` - 提取第21-80个样本
- [ ] `annotate_samples.py` - 断点2人工标注界面
- [ ] `generate_validation_test.py` - 自动生成validation测试脚本
- [ ] `validate_prompt_test.py` - 测试prompt准确率（自动生成）
- [ ] 修改`validate.py` - 添加门禁检查

### 中优先级（提升体验）
- [ ] `tune_validation_prompt.py` - Prompt调优辅助工具
- [ ] `compare_versions.py` - 版本对比分析
- [ ] 修改`generator.py` - 支持自动few-shot注入

### 低优先级（锦上添花）
- [ ] Web界面替代CLI审核/标注工具
- [ ] 自动化prompt调优建议
- [ ] 批量实验管理工具
