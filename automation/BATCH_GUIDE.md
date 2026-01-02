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
    'gen_prompt_hash': 'a1b2c3d4',  # rephrase_prompt的hash
    'val_prompt_hash': 'e5f6g7h8'   # validation_prompt的hash
}
```

**指纹用途**:
- **去重判断**: 相同指纹 = 相同参数 = 复用数据
- **唯一标识**: 精确识别参数配置
- **追溯来源**: 通过指纹找到首次生成的batch

### 2. 语义化目录名 (Semantic Directory Name)

为了人类可读性，使用语义化的目录名而不是直接使用hash。

**命名格式**:
```
temp{temperature}_topp{top_p}_{model}
```

**示例**:
- `temp07_topp10_gpt4o` → temperature=0.7, top_p=1.0, model=gpt-4o
- `temp09_topp08_gpt4o` → temperature=0.9, top_p=0.8, model=gpt-4o
- `temp05_topp10_gpt35` → temperature=0.5, top_p=1.0, model=gpt-3.5-turbo

**智能省略**:
- 默认值 top_p=1.0 → 显示为 topp10
- 非默认值 top_p=0.9 → 显示为 topp09

**精确匹配**: 目录内的 `.fingerprint` 文件存储精确hash，用于参数匹配

### 3. Batch ID

Batch ID用于组织多个相关实验到同一批次。

**格式**:
```
batch_{date}_{purpose}
```

**示例**:
- `batch_20241229_temperature` → 2024年12月29日的温度实验
- `batch_20241230_topp` → 2024年12月30日的top_p实验
- `batch_20250103_model_comparison` → 2025年1月3日的模型对比实验

**自动生成**: 如果配置文件中未指定 `batch_id`，系统会根据当前日期和 `purpose` 自动生成

---

## 使用方法

### 配置文件设置

在配置文件中添加 `experiment.batch_id` 字段：

```yaml
experiment:
  # Batch ID（可选）
  # 格式: batch_{date}_{purpose}
  # 不指定时自动生成: batch_{YYYYMMDD}_{purpose}
  batch_id: "batch_20241229_temperature"

  purpose: "temperature_study"
  description: "研究temperature参数对合成数据质量的影响"

# 其他配置...
generation:
  model: "gpt-4o"
  temperature: 0.7  # 实验变量
  top_p: 1.0
  # ...
```

### 生成脚本

使用 `generator.py` 生成脚本时，Batch方案会自动启用：

```bash
# 生成实验脚本
python automation/stage1_generation/generator.py \
    automation/configs/examples/stage1_full_example_copa.yaml
```

**系统会自动**:
1. 计算参数指纹
2. 在 `_shared/{Dataset}/` 中查找相同指纹
3. 如果找到 → 复用物理数据 + 创建batch符号链接
4. 如果未找到 → 创建新物理目录 + 创建batch符号链接

### 输出解读

```
================================================================================
🔧 Batch实验管理
================================================================================
Batch ID: batch_20241229_temperature
数据集: Copa
参数指纹: a1b2c3d4e5f6
语义化名称: temp07_topp09_gpt4o
================================================================================

🔍 在 _shared/Copa/ 中搜索指纹 a1b2c3d4e5f6...
✅ 发现相同参数的已有实验！
   位置: _shared/Copa/temp07_topp09_gpt4o
   创建时间: 2024-12-29 10:30:00
   原batch: batch_20241228_pilot

📂 复用已有数据
   物理存储: _shared/Copa/temp07_topp09_gpt4o (已存在，复用)
   Batch视图: batch_20241229_temperature/Copa/temp07_topp09_gpt4o

✅ 已有数据复用成功
   💾 节省资源: 无需重新生成数据
```

**关键信息**:
- ✅ 发现相同参数 → 数据会被复用
- ✓ 未找到匹配 → 创建新实验
- 💾 节省资源 → 不会重复生成数据

---

## 实际操作示例

### 场景A: 首次batch - 温度实验

**目标**: 测试 temperature=0.5, 0.7, 0.9 对Copa数据质量的影响

#### 步骤1: 准备配置文件

创建三个配置文件（或使用脚本批量生成）：

**config_temp05.yaml**:
```yaml
experiment:
  batch_id: "batch_20241229_temperature"
  purpose: "temperature_study"

generation:
  model: "gpt-4o"
  temperature: 0.5  # 变量
  top_p: 1.0
```

**config_temp07.yaml**, **config_temp09.yaml** 类似，只改temperature值。

#### 步骤2: 生成脚本

```bash
# 生成三个实验的脚本
python automation/stage1_generation/generator.py automation/configs/temp05.yaml
python automation/stage1_generation/generator.py automation/configs/temp07.yaml
python automation/stage1_generation/generator.py automation/configs/temp09.yaml
```

#### 步骤3: 查看生成的目录结构

```bash
python automation/stage1_generation/batch_tools/list_batch_experiments.py \
    batch_20241229_temperature --verbose
```

**输出**:
```
📊 Copa (3 个实验)
  🔧 temp05_topp10_gpt4o
     ⚡ 数据复用: 否 (新生成)
  🔧 temp07_topp10_gpt4o
     ⚡ 数据复用: 否 (新生成)
  🔧 temp09_topp10_gpt4o
     ⚡ 数据复用: 否 (新生成)
```

#### 步骤4: 运行数据生成

```bash
# 方式1: 手动依次运行
cd Data_v2/synthetic/_shared/Copa/temp05_topp10_gpt4o/scripts/
python rephrase_all.py && python validate.py

cd ../../../temp07_topp10_gpt4o/scripts/
python rephrase_all.py && python validate.py

cd ../../../temp09_topp10_gpt4o/scripts/
python rephrase_all.py && python validate.py

# 方式2: 使用脚本批量运行（推荐）
# TODO: 创建 batch_run.py 工具
```

---

### 场景B: 第二个batch - top_p实验

**目标**: 在 temperature=0.7 下，测试 top_p=0.8, 0.9, 1.0 的影响

#### 步骤1: 准备配置文件

**config_topp08.yaml**:
```yaml
experiment:
  batch_id: "batch_20241230_topp"  # 新的batch
  purpose: "topp_study"

generation:
  model: "gpt-4o"
  temperature: 0.7  # 固定
  top_p: 0.8        # 变量
```

**config_topp09.yaml**, **config_topp10.yaml** 类似。

#### 步骤2: 生成脚本

```bash
python automation/stage1_generation/generator.py automation/configs/topp08.yaml
python automation/stage1_generation/generator.py automation/configs/topp09.yaml
python automation/stage1_generation/generator.py automation/configs/topp10.yaml
```

**关键输出**:

对于 **config_topp10.yaml** (temperature=0.7, top_p=1.0):
```
🔍 在 _shared/Copa/ 中搜索指纹 a1b2c3d4e5f6...
✅ 发现相同参数的已有实验！
   位置: _shared/Copa/temp07_topp10_gpt4o
   原batch: batch_20241229_temperature

📂 复用已有数据
   💾 节省资源: 无需重新生成数据
```

#### 步骤3: 查看目录结构

```bash
python automation/stage1_generation/batch_tools/list_batch_experiments.py \
    batch_20241230_topp --verbose
```

**输出**:
```
📊 Copa (3 个实验)
  🔧 temp07_topp08_gpt4o
     ⚡ 数据复用: 否 (新生成)

  🔧 temp07_topp09_gpt4o
     ⚡ 数据复用: 否 (新生成)

  🔧 temp07_topp10_gpt4o
     ⚡ 数据复用: 是 (原batch: batch_20241229_temperature)
```

**数据复用成功！** temp07_topp10_gpt4o 的数据直接复用自第一个batch。

#### 步骤4: 运行数据生成

```bash
# 只需要生成新参数的数据
cd Data_v2/synthetic/_shared/Copa/temp07_topp08_gpt4o/scripts/
python rephrase_all.py && python validate.py

cd ../../../temp07_topp09_gpt4o/scripts/
python rephrase_all.py && python validate.py

# temp07_topp10_gpt4o 已经有数据，跳过！
```

---

### 场景C: 查看和比较实验

#### 查看所有batch

```bash
python automation/stage1_generation/batch_tools/list_batches.py --verbose
```

**输出**:
```
找到 2 个batch

📦 batch_20241229_temperature
   实验数: 3
   Copa: 3 个实验

📦 batch_20241230_topp
   实验数: 3
   Copa: 3 个实验
```

#### 查看物理存储使用情况

```bash
python automation/stage1_generation/batch_tools/list_shared_experiments.py \
    --dataset Copa --verbose
```

**输出**:
```
📊 Copa (5 个实验)  # 只有5个物理数据，不是6个！

  📦 temp05_topp10_gpt4o
     原始Batch: batch_20241229_temperature

  📦 temp07_topp08_gpt4o
     原始Batch: batch_20241230_topp

  📦 temp07_topp09_gpt4o
     原始Batch: batch_20241230_topp

  📦 temp07_topp10_gpt4o  # 被两个batch共享！
     原始Batch: batch_20241229_temperature

  📦 temp09_topp10_gpt4o
     原始Batch: batch_20241229_temperature
```

#### 比较两个实验参数

```bash
python automation/stage1_generation/batch_tools/compare_experiments.py \
    --shared Copa/temp07_topp10_gpt4o \
    --shared Copa/temp09_topp10_gpt4o
```

**输出**:
```
✅ 相同参数:
  generation.model: gpt-4o
  generation.top_p: 1.0
  validation.model: gpt-4o

⚠️  不同参数:
  generation.temperature:
    实验1: 0.7
    实验2: 0.9
```

---

## 数据复用机制

### 复用条件

**必须满足**: 参数指纹完全相同

参数指纹包括：
- 生成模型、temperature、top_p、max_tokens、频率惩罚、存在惩罚
- 验证模型、temperature
- rephrase_prompt 的hash
- validation_prompt 的hash

**只要有一个参数不同，指纹就不同，需要重新生成数据。**

### 复用流程

1. **生成脚本时**:
   - 计算配置文件的参数指纹
   - 在 `_shared/{Dataset}/` 中遍历所有实验目录
   - 读取每个目录的 `.fingerprint` 文件
   - 如果找到相同指纹 → 复用

2. **复用操作**:
   - **不创建新的物理目录**
   - **不生成新的数据**
   - 只在 `batch_*/` 中创建符号链接指向现有物理目录

3. **元数据记录**:
   - 物理目录的元数据保持不变（记录首次创建的batch）
   - batch符号链接无额外元数据

### 验证复用

```bash
# 检查符号链接
ls -la Data_v2/synthetic/batch_20241230_topp/Copa/

# 输出类似:
# temp07_topp10_gpt4o -> ../../_shared/Copa/temp07_topp10_gpt4o

# 检查物理目录
ls -la Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/
# 应该看到实际的数据文件

# 使用工具验证
python automation/stage1_generation/batch_tools/list_batch_experiments.py \
    batch_20241230_topp --verbose
# 应该看到 "⚡ 数据复用: 是"
```

---

## Batch管理工具

详见 [batch_tools/README.md](stage1_generation/batch_tools/README.md)

### 快速参考

```bash
# 列出所有batch
python batch_tools/list_batches.py --verbose

# 查看batch详情
python batch_tools/list_batch_experiments.py batch_20241229_temperature --verbose

# 查看物理数据
python batch_tools/list_shared_experiments.py --dataset Copa --verbose

# 比较实验参数
python batch_tools/compare_experiments.py \
    --shared Copa/temp07_topp10_gpt4o \
    --shared Copa/temp09_topp10_gpt4o

# ⭐ 新增：查找数据路径（用于训练配置）
python batch_tools/list_data_paths.py --dataset Copa --format yaml

# ⭐ 新增：路径转换
python batch_tools/resolve_data_path.py "Data_v2/synthetic/batch_xxx/Copa/..."
```

---

## FAQ

### Q1: 如果我手动修改了_shared/中的数据，batch_*/中的符号链接会自动更新吗？

**回答**: 是的！符号链接指向物理路径，修改物理数据后，所有引用该数据的batch都会看到更新。

**注意**: 这可能导致不同batch的训练结果不一致，建议不要手动修改已生成的数据。

### Q2: 如果我删除了某个batch_*/目录，_shared/中的物理数据会被删除吗？

**回答**: 不会。batch_*/只包含符号链接，删除batch不影响物理数据。

**清理建议**: 如果要清理不再使用的实验数据，应该:
1. 先删除所有引用该数据的batch符号链接
2. 再删除_shared/中的物理目录

### Q3: 我可以手动创建batch吗？

**回答**: 可以，但不推荐。应该通过配置文件 + generator.py 自动管理。

如果确实需要手动操作：
```bash
mkdir -p Data_v2/synthetic/batch_20241231_manual/Copa
ln -s ../../_shared/Copa/temp07_topp10_gpt4o \
    Data_v2/synthetic/batch_20241231_manual/Copa/temp07_topp10_gpt4o
```

### Q4: 参数指纹是怎么计算的？我可以看到详细内容吗？

**回答**: 可以查看 `.fingerprint` 文件和 `experiment_metadata.json`:

```bash
# 查看指纹
cat Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/.fingerprint

# 查看完整元数据（包含所有参数）
cat Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/experiment_metadata.json | jq .
```

### Q5: 我想重新生成某个参数配置的数据，怎么办？

**回答**:
1. 删除_shared/中对应的物理目录
2. 删除所有batch_*/中指向该目录的符号链接
3. 重新运行 generator.py（会检测到数据不存在并重新生成）

**示例**:
```bash
# 1. 删除物理数据
rm -rf Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o

# 2. 删除所有符号链接
find Data_v2/synthetic/batch_* -name "temp07_topp10_gpt4o" -type l -delete

# 3. 重新生成
python automation/stage1_generation/generator.py automation/configs/temp07.yaml
cd Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/scripts/
python rephrase_all.py && python validate.py
```

### Q6: batch_id是必须的吗？

**回答**: 不是必须的。如果配置文件中未指定 `batch_id`，系统会根据当前日期和 `purpose` 自动生成：

```
batch_{YYYYMMDD}_{purpose}
```

例如: `batch_20241229_temperature_study`

### Q7: 我可以把多个数据集（Copa, CB, BOOLQ）放在同一个batch中吗？

**回答**: 可以！batch是跨数据集的。只要配置文件中指定相同的 `batch_id`，不同数据集的实验都会出现在同一个batch中。

**示例**:

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
  batch_id: "batch_20241229_multi_dataset"  # 相同batch_id
dataset:
  dataset_name: "CB"
```

结果：
```
batch_20241229_multi_dataset/
├── Copa/
│   └── temp07_topp10_gpt4o/
└── CB/
    └── temp07_topp10_gpt4o/
```

### Q8: 还需要使用publish_dataset.py吗？

**回答**: **不需要！** trainer.py可以直接使用 `Data_v2/` 路径。

**推荐方式**（直接使用Data_v2路径）:
```yaml
# 训练配置
data:
  # 推荐：使用batch路径（更直观）
  path: "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"

  # 或使用shared路径
  # path: "Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa"
```

**可选方式**（仅用于兼容旧脚本）:
```bash
# 仅在需要兼容旧训练脚本时使用
python automation/stage1_generation/tools/publish_dataset.py \
    --source Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa \
    --dataset Copa \
    --target Data/rejection_sampling/0_data
```

### Q9: 如何快速找到数据路径用于训练配置？

**回答**: 使用新增的 `list_data_paths.py` 工具：

```bash
# 输出YAML格式，可直接复制到配置文件
python automation/stage1_generation/batch_tools/list_data_paths.py --dataset Copa --format yaml
```

**输出示例**:
```yaml
data:
  path: "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"
```

---

## 最佳实践

### 1. Batch命名规范

- 使用日期前缀: `batch_YYYYMMDD_*`
- 使用描述性purpose: `temperature`, `topp`, `model_comparison`
- 避免使用中文或特殊字符

### 2. 配置文件管理

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

### 3. 定期清理

- 定期查看 `_shared/` 使用情况
- 删除不再需要的实验数据
- 保留有价值的实验结果

### 4. 文档记录

在每个batch目录中创建 `README.md` 记录：
- 实验目的
- 参数设置
- 结果总结
- 训练效果对比

---

## 与训练脚本的兼容性

### ✅ 推荐：直接使用Data_v2路径

**trainer.py可以直接使用 `Data_v2/` 路径**，无需publish步骤：

```yaml
# 训练配置 - automation/configs/stage2/my_training.yaml
experiment:
  purpose: "temperature_study"

model: "meta-llama/Llama-3.2-1B"
task: "Copa"
method: "zo"

data:
  # 推荐：使用batch路径（按实验目的组织，更直观）
  path: "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"

  # 或使用shared路径（物理存储）
  # path: "Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa"
```

**如何快速找到数据路径**：
```bash
python automation/stage1_generation/batch_tools/list_data_paths.py --dataset Copa --format yaml
```

### 可选：发布到Data/（仅用于兼容旧脚本）

如果需要兼容旧的训练脚本（直接使用 `Data/` 目录），可以使用publish工具：

```bash
python automation/stage1_generation/tools/publish_dataset.py \
    --source Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o \
    --dataset Copa \
    --target Data/rejection_sampling/0_data
```

**注意**: 仅用于兼容旧项目结构，新项目推荐直接使用 `Data_v2/` 路径。

---

## 总结

Batch方案3++通过物理存储与逻辑视图分离，实现了：

✅ **参数去重**: 相同参数配置只生成一次数据
✅ **存储优化**: 节省磁盘空间和API调用成本
✅ **灵活组织**: 按时间/目的灵活组织实验
✅ **易于追溯**: 清晰记录每个实验的来源和参数
✅ **向后兼容**: 不影响现有训练脚本和工具

---

**创建日期**: 2024-12-29
**版本**: 1.0
**维护**: Synthetic Data Generation Team
