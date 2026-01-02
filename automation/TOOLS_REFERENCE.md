# Tools Reference Manual

**Version**: 2.1
**Last Updated**: 2026-01-01

This document provides complete reference for all automation tools.

---

## 📋 Table of Contents

1. [Stage 1 Tools (Data Generation)](#stage-1-tools)
2. [Stage 2 Tools (Model Training)](#stage-2-tools)
3. [Batch Management Tools](#batch-management-tools)
4. [Data Path Tools (New)](#data-path-tools)
5. [Configuration and Diagnostic Tools](#configuration-and-diagnostic-tools)

---

## Stage 1 Tools

### generator.py

**Purpose**: Generate data synthesis scripts

**Location**: `automation/stage1_generation/generator.py`

**Usage**:
```bash
python automation/stage1_generation/generator.py <config.yaml>
```

**Parameters**:
- `config.yaml`: Stage 1 configuration file path (required)

**Output**:
- Creates generation scripts in `Data_v2/synthetic/_shared/{Dataset}/{exp_name}/scripts/`
- If batch_id specified, creates symbolic links in `Data_v2/synthetic/{batch_id}/`

**Example**:
```bash
python automation/stage1_generation/generator.py \
    automation/configs/examples/stage1_full_example_copa.yaml
```

**Features**:
- ✅ Automatic parameter deduplication (Batch Solution 3++)
- ✅ Automatically detect and reuse existing data
- ✅ Multi-dataset support
- ✅ Fully configuration-driven

---

### review_top20.py

**用途**: 交互式审核前20个样本

**位置**: `automation/stage1_generation/tools/review_top20.py`

**用法**:
```bash
# 复制到实验目录
cp automation/stage1_generation/tools/review_top20.py <exp_dir>/scripts/

# 运行（在生成前20个样本后）
cd <exp_dir>/scripts/
python review_top20.py
```

**交互流程**:
```
样本 1/20:
原文: The man broke his toe because...
改写: The individual fractured his toe due to...

Is this a good rephrase? (y/n): y

样本 2/20:
...
```

**输出**:
- `../copa_train_top20_annotated.jsonl` - 标注的前20个样本
- 优质样本自动提取用于few-shot

---

### annotate_samples.py

**用途**: 标注样本21-40是否与原文相同

**位置**: `automation/stage1_generation/tools/annotate_samples.py`

**用法**:
```bash
cd <exp_dir>/scripts/
python annotate_samples.py
```

**交互**:
```
请标注样本是否与原文相同:
样本 21: same/not the same? same
样本 22: same/not the same? not the same
...
```

**输出**:
- `../copa_train_samples21to40_annotated.jsonl`

---

### extract_samples.py

**用途**: 从标注数据提取优质样本

**位置**: `automation/stage1_generation/tools/extract_samples.py`

**用法**:
```bash
python automation/stage1_generation/tools/extract_samples.py \
    <annotated.jsonl> \
    --output <output.jsonl> \
    --num-samples 5
```

---

### publish_dataset.py

**用途**: 发布数据到训练目录（可选）

**位置**: `automation/stage1_generation/tools/publish_dataset.py`

**用法**:
```bash
python automation/stage1_generation/tools/publish_dataset.py \
    --source Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa \
    --dataset Copa \
    --target Data/rejection_sampling/0_data
```

**注意**:
- ⚠️ 训练脚本可以直接使用 `Data_v2/` 路径，此工具是可选的
- 仅用于兼容旧的训练脚本结构

---

## 阶段2工具

### trainer.py

**用途**: 自动化模型训练

**位置**: `automation/stage2_training/trainer.py`

**用法**:
```bash
# 执行训练
python automation/stage2_training/trainer.py <config.yaml>

# 预览（不实际训练）
python automation/stage2_training/trainer.py <config.yaml> --dry-run
```

**支持的训练方法**:
- `zo` - MeZO (零阶优化)
- `fo_full` - 全参数微调
- `fo_lora` - LoRA微调

**配置示例**:
```yaml
experiment:
  purpose: "hyperparameter_tuning"
  description: "测试不同学习率"

model: "meta-llama/Llama-3.2-1B"
task: "Copa"
method: "fo_lora"

data:
  path: "Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa"

hyperparameters:
  learning_rate: [1e-6, 5e-7]  # 网格搜索
  batch_size: 16
  steps: 20000
  seed: 0
  lora_rank: 8  # LoRA专用

cuda_devices: "0"
```

**输出**:
```
Results_v2/
└── hyperparameter_tuning/  # 按实验目的组织
    └── Llama-3.2-1B/
        └── Copa_fo_lora_temp07_1e-6/
            └── 20260101_120000/
                ├── train.out
                ├── train.err
                └── experiment_config.yaml
```

---

### list_results.py

**用途**: 查看训练结果摘要

**位置**: `automation/stage2_training/list_results.py`

**用法**:
```bash
# 查看摘要
python automation/stage2_training/list_results.py

# 查看详细结果
python automation/stage2_training/list_results.py --detail

# 筛选特定实验目的
python automation/stage2_training/list_results.py --purpose hyperparameter_tuning
```

**输出示例**:
```
================================================================================
训练结果摘要 - Results_v2
================================================================================

📁 实验目的: hyperparameter_tuning
   实验数量: 5
   └─ meta-llama: 5 个实验

📁 实验目的: baseline_comparison
   实验数量: 3
   └─ meta-llama: 3 个实验

总计: 2 个实验目的, 8 个训练实验
```

---

## Batch管理工具

### list_batches.py

**用途**: 列出所有batch

**位置**: `automation/stage1_generation/batch_tools/list_batches.py`

**用法**:
```bash
# 列出所有batch
python automation/stage1_generation/batch_tools/list_batches.py

# 显示详细信息
python automation/stage1_generation/batch_tools/list_batches.py --verbose
```

**输出示例**:
```
================================================================================
找到 2 个batch
================================================================================

📦 batch_20241229_temperature
   实验数: 3
   Copa: 3 个实验

📦 batch_20241230_topp
   实验数: 2
   Copa: 2 个实验
```

---

### list_batch_experiments.py

**用途**: 查看batch中的实验

**位置**: `automation/stage1_generation/batch_tools/list_batch_experiments.py`

**用法**:
```bash
# 列出batch中的实验
python automation/stage1_generation/batch_tools/list_batch_experiments.py <batch_id>

# 只看特定数据集
python automation/stage1_generation/batch_tools/list_batch_experiments.py <batch_id> --dataset Copa

# 显示详细信息
python automation/stage1_generation/batch_tools/list_batch_experiments.py <batch_id> --verbose
```

**输出示例**:
```
📊 Copa (3 个实验)

  🔧 temp05_topp10_gpt4o
     ⚡ 数据复用: 否 (新生成)

  🔧 temp07_topp10_gpt4o
     ⚡ 数据复用: 是 (原batch: batch_20241228_pilot)
```

---

### list_shared_experiments.py

**用途**: 查看物理存储的所有实验

**位置**: `automation/stage1_generation/batch_tools/list_shared_experiments.py`

**用法**:
```bash
# 列出所有物理实验
python automation/stage1_generation/batch_tools/list_shared_experiments.py

# 只看特定数据集
python automation/stage1_generation/batch_tools/list_shared_experiments.py --dataset Copa

# 显示详细信息
python automation/stage1_generation/batch_tools/list_shared_experiments.py --verbose
```

**用途**:
- 查看哪些参数配置已生成过数据
- 避免重复生成相同参数的数据
- 了解物理存储使用情况

---

### compare_experiments.py

**用途**: 比较实验参数

**位置**: `automation/stage1_generation/batch_tools/compare_experiments.py`

**用法**:
```bash
# 比较两个物理实验
python automation/stage1_generation/batch_tools/compare_experiments.py \
    --shared Copa/temp07_topp10_gpt4o \
    --shared Copa/temp09_topp10_gpt4o

# 比较batch中的实验
python automation/stage1_generation/batch_tools/compare_experiments.py \
    --batch1 batch_20241229_temperature \
    --batch2 batch_20241230_topp \
    --dataset1 Copa \
    --dataset2 Copa
```

**输出示例**:
```
✅ 相同参数:
  generation.model: gpt-4o
  generation.top_p: 1.0

⚠️  不同参数:
  generation.temperature:
    实验1: 0.7
    实验2: 0.9
```

---

## 数据路径工具

### list_data_paths.py ⭐ 新增

**用途**: 列出所有可用的训练数据路径

**位置**: `automation/stage1_generation/batch_tools/list_data_paths.py`

**用法**:
```bash
# 列出所有数据
python automation/stage1_generation/batch_tools/list_data_paths.py

# 只看某个batch
python automation/stage1_generation/batch_tools/list_data_paths.py \
    --batch batch_20241229_temperature

# 只看某个数据集
python automation/stage1_generation/batch_tools/list_data_paths.py --dataset Copa

# 输出YAML格式（可直接复制到配置文件）
python automation/stage1_generation/batch_tools/list_data_paths.py --format yaml
```

**输出示例（简洁模式）**:
```
====================================================================================================
📊 可用的训练数据路径
====================================================================================================

====================================================================================================
🗂️  Batch: batch_20241229_temperature
====================================================================================================

📁 Copa / temp07_topp10_gpt4o
   📝 描述: 研究temperature参数对合成数据质量的影响

   ✅ Batch路径（推荐 - 按实验目的组织）:
      Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa

   📦 Shared路径（物理存储）:
      Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa
```

**输出示例（YAML模式）**:
```yaml
# 可直接复制到训练配置文件
data:
  # Batch路径（推荐）
  path: "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"
  # 或使用 Shared路径
  # path: "Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa"
```

**使用场景**:
- 编写训练配置时查找数据路径
- 快速浏览所有可用数据
- 生成配置文件模板

---

### resolve_data_path.py ⭐ 新增

**用途**: 转换batch路径和shared路径

**位置**: `automation/stage1_generation/batch_tools/resolve_data_path.py`

**用法**:
```bash
# Batch路径 → Shared路径
python automation/stage1_generation/batch_tools/resolve_data_path.py \
    "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"

# Shared路径 → 所有引用的batch
python automation/stage1_generation/batch_tools/resolve_data_path.py \
    "Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa"
```

**输出示例**:
```
====================================================================================================
🔍 路径解析：Batch → Shared
====================================================================================================

输入路径: batch_20241229_temperature/Copa/temp07_topp10_gpt4o
物理路径: _shared/Copa/temp07_topp10_gpt4o

✅ 此数据被以下batch引用:
  • batch_20241229_temperature
  • batch_20241230_model_comparison
```

**使用场景**:
- 查找物理数据位置
- 了解数据复用情况
- 验证符号链接是否正确

---

## 配置和诊断工具

### config.py

**用途**: 验证项目路径配置

**位置**: `automation/config.py`

**用法**:
```bash
# 验证配置
python automation/config.py
```

**输出**:
```
================================================================================
🔧 项目配置
================================================================================
PROJECT_ROOT:         /path/to/Synthetic_Data_for_ZO
AUTOMATION_DIR:       /path/to/Synthetic_Data_for_ZO/automation
DATA_V2_DIR:          /path/to/Synthetic_Data_for_ZO/Data_v2
RESULTS_V2_DIR:       /path/to/Synthetic_Data_for_ZO/Results_v2
================================================================================
✅ 所有关键路径验证通过
================================================================================
```

**环境变量覆盖**:
```bash
export SYNTHETIC_DATA_PROJECT_ROOT="/your/custom/path"
python automation/config.py
```

---

### fix_hardcoded_paths.py

**用途**: 批量修复硬编码路径（维护工具）

**位置**: `automation/fix_hardcoded_paths.py`

**用法**:
```bash
cd automation
python fix_hardcoded_paths.py
```

**说明**:
- 已在v2.1版本中执行
- 修复了10个文件的硬编码路径问题
- 一般用户无需使用此工具

---

## 工具使用流程

### 完整数据生成流程

```bash
# 1. 生成脚本
python automation/stage1_generation/generator.py config.yaml

# 2. 查看生成的batch
python automation/stage1_generation/batch_tools/list_batches.py --verbose

# 3. 执行数据生成（见 USER_GUIDE.md）
cd Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/scripts/
python rephrase_top20.py
python review_top20.py
...

# 4. 查找数据路径（用于训练配置）
python automation/stage1_generation/batch_tools/list_data_paths.py --dataset Copa --format yaml
```

### 完整训练流程

```bash
# 1. 编写训练配置（使用list_data_paths获取的路径）
vim automation/configs/stage2/my_training.yaml

# 2. 预览训练
python automation/stage2_training/trainer.py my_training.yaml --dry-run

# 3. 执行训练
python automation/stage2_training/trainer.py my_training.yaml

# 4. 查看结果
python automation/stage2_training/list_results.py --detail
```

---

## 常见问题

### Q: 如何快速找到数据路径用于训练？

A: 使用 `list_data_paths.py`:
```bash
python automation/stage1_generation/batch_tools/list_data_paths.py \
    --dataset Copa --format yaml
```

### Q: 如何检查是否已生成过某个参数配置？

A: 使用 `list_shared_experiments.py`:
```bash
python automation/stage1_generation/batch_tools/list_shared_experiments.py \
    --dataset Copa --verbose
```

### Q: 训练配置应该使用batch路径还是shared路径？

A: **都可以！** 推荐使用batch路径（更直观）：
```yaml
data:
  path: "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"
```

### Q: 还需要用publish_dataset.py吗？

A: **不需要！** trainer.py可以直接使用 `Data_v2/` 路径。`publish_dataset.py` 仅用于兼容旧代码。

---

## 更新日志

### v2.1 (2026-01-01)
- ✅ 新增 `list_data_paths.py` - 数据路径查找工具
- ✅ 新增 `resolve_data_path.py` - 路径转换工具
- ✅ 修复LoRA训练脚本名称错误
- ✅ 修复LoRA环境变量错误
- ✅ 移除所有硬编码路径，使用统一config.py
- ✅ 修复batch_tools路径导入问题

### v2.0 (2024-12-30)
- ✅ 实现Batch方案3++
- ✅ 完全配置驱动的系统
- ✅ 多数据集支持

---

**完整的工具生态系统！使用这些工具提升效率！** 🚀
