# 数据参考手册

> 本文档整合了数据位置说明、验证报告和快速迁移参考，为您提供完整的数据相关信息。

---

## 📑 目录

1. [数据位置说明](#数据位置说明)
   - 核心问题解答
   - 旧项目数据位置
   - 新项目数据位置
   - 实际存在的目录结构

2. [数据验证报告](#数据验证报告)
   - 旧项目数据分类
   - 新项目数据位置
   - 数据对比总结
   - 改写质量对比

3. [快速迁移参考](#快速迁移参考)
   - 一键命令
   - 文件命名规范
   - 目录结构
   - 常见问题

---

# 数据位置说明

## 📌 核心问题解答

### Q1: 以前生成的合成数据是否都暂存到 `Pending_Manual_Classification/` 下了？

**答案：否，这个目录不存在。**

经过检查，`Pending_Manual_Classification/` 目录在新项目中**并不存在**：

```bash
$ find /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO -name "*Pending*"
# 无结果

$ ls /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Pending_Manual_Classification
# ls: cannot access '...': No such file or directory
```

**实际情况**：
- `Pending_Manual_Classification/` 目录是**计划中的目录**，但尚未创建
- 旧项目的合成数据**仍然在 Backup 项目**中，还没有迁移到新项目

---

### Q2: 旧合成数据的实际位置

**旧项目（Backup）中的合成数据位置**：

```
/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/
├── Copa/
│   ├── copa_train.jsonl
│   ├── copa_validation.jsonl
│   ├── copa_test.jsonl
│   └── mezo_gpt/          # 多个版本
│       ├── version_1/
│       ├── version_2/
│       └── ...
├── BOOLQ/
│   ├── boolq_train.jsonl
│   └── boolq_validation.jsonl
├── CB/
│   ├── cb_train.jsonl
│   ├── cb_validation.jsonl
│   └── cb_test.jsonl
├── RTE/
│   ├── rte_train.jsonl
│   ├── rte_validation.jsonl
│   └── rte_test.jsonl
└── ArcC_Cloze/
    ├── ARC-Challenge_train.jsonl
    ├── ARC-Challenge_validation.jsonl
    └── ARC-Challenge_test.jsonl
```

**新项目中的数据位置**：

```
/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/
└── original/              # ← 只有原始数据，没有合成数据
    ├── Copa/
    ├── BOOLQ/
    ├── CB/
    ├── RTE/
    ├── ArcC_Cloze/
    └── ArcC_MC/
```

**结论**：旧的合成数据**还在 Backup 项目中，尚未迁移到新项目**。

---

## 📊 results 目录分析

### 新项目 results 目录

**位置**：`/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/results/`

**结构**：
```
results/
├── Llama-3.2-1B/
│   ├── Copa/
│   │   ├── zo/
│   │   │   ├── original/           # 原始数据训练结果
│   │   │   │   ├── 1e-6_original.out
│   │   │   │   ├── 1e-6_original.err
│   │   │   │   └── ...
│   │   │   └── rejection_sampling/  # 合成数据训练结果
│   │   ├── fo_full/
│   │   └── fo_lora/
│   ├── CB/
│   ├── BOOLQ/
│   ├── RTE/
│   └── ArcC_Cloze/
├── Llama-3.2-3B/
├── Mistral-7B-v0.1/
└── back_up/                # 早期备份
```

**内容说明**：
- **original/** - 使用原始数据训练的结果（.out 和 .err 文件）
- **rejection_sampling/** - 使用合成数据训练的结果
- 每个文件命名格式：`{learning_rate}_{data_type}.out/err`

### Backup 项目 results 目录

**位置**：`/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/results/`

**结构**：完全相同（与新项目 results 结构一致）

### 新项目 results 与 Backup results 的关系

**文件校验**：
```bash
# 检查两个项目中相同文件的 MD5 值
$ md5sum {新项目}/results/Llama-3.2-1B/Copa/zo/original/1e-6_original.out
24387ca5e3c5719c5ce2a961a544d16a

$ md5sum {Backup}/results/Llama-3.2-1B/Copa/zo/original/1e-6_original.out
24387ca5e3c5719c5ce2a961a544d16a
```

**MD5 值完全相同** → 文件内容一致

**结论**：
- **内容完全相同** - MD5 值一致
- **可能的情况**：新项目运行训练后产生结果，后来被复制到 Backup 作为备份

---

## 📁 实际存在的目录结构（新项目）

```
/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/
├── Data/
│   └── original/              # ✓ 存在：原始数据
│       ├── Copa/
│       ├── BOOLQ/
│       ├── CB/
│       ├── RTE/
│       ├── ArcC_Cloze/
│       └── ArcC_MC/
│
├── Data_v2/                   # ✓ 存在：新的数据组织结构
│   ├── original/              # 原始数据
│   └── synthetic/             # 新生成的合成数据
│
├── results/                   # ✓ 存在：训练结果
│   ├── Llama-3.2-1B/
│   ├── Llama-3.2-3B/
│   ├── Mistral-7B-v0.1/
│   └── back_up/
│
├── Results_v2/                # ✓ 存在：新的结果组织结构
│
├── automation/                # ✓ 存在：自动化系统
│   ├── stage1_generation/
│   ├── stage2_training/
│   └── batch_tools/
│
├── running_scripts/           # ✓ 存在：135 个手动训练脚本
│   ├── Llama-3.2-1B/
│   ├── Llama-3.2-3B/
│   └── Mistral-7B-v0.1/
│
└── PromptZO/                  # ✓ 存在：MeZO 训练框架
    └── MeZO/
```

**不存在的目录**：
- ✗ `Data/rejection_sampling/` - 需要创建（用于迁移旧合成数据）
- ✗ `Pending_Manual_Classification/` - 计划目录，尚未创建

---

## 🎯 迁移任务总结

### 当前状态

| 数据/结果类型 | Backup 项目 | 新项目 | 迁移状态 |
|-------------|-----------|--------|---------|
| **原始数据** | ✓ 存在 | ✓ 存在 | ✅ 已迁移 |
| **合成数据** | ✓ 存在 (`Data/rejection_sampling/0_data/`) | ✗ 不存在 | ❌ 未迁移 |
| **训练结果** | ✓ 存在 | ✓ 存在（内容相同） | ✅ 已同步 |
| **训练脚本** | ✓ 135 个脚本 | ✓ 135 个脚本（相同） | ✅ 已迁移 |

### 需要执行的迁移

使用迁移脚本：

```bash
cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/automation
bash migrate_synthetic_data.sh link  # 创建符号链接
# 或
bash migrate_synthetic_data.sh copy  # 完整复制
```

这将创建：
```
Data/rejection_sampling/0_data/
├── Copa/
├── BOOLQ/
├── CB/
├── RTE/
├── ArcC_Cloze/
└── ArcC_MC/
```

---

# 数据验证报告

## 🔍 旧项目（Backup）数据分类

### 路径：`/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/`

### 1. **original/** - ✅ 原始数据（已验证）

**验证方法**：与 HuggingFace 在线数据集逐行对比

**验证结果**：

| 数据集 | 本地样本数 | 在线样本数 | 验证状态 | 匹配率 |
|--------|----------|----------|---------|--------|
| **COPA** | 400 (train), 100 (val) | 400 (train), 100 (val) | ✅ 完全匹配 | 100% |
| **CB** | 250 (train) | 250 (train) | ✅ 完全匹配 | 100% |
| **BOOLQ** | 1000 (train) | 9427 (train) | ✅ 前1000个匹配 | 前1000/9427 |
| **RTE** | 1000 (train) | 2490 (train) | ✅ 采样子集 | 前1000/2490 |

**说明**：
- COPA 和 CB 是完整的训练集
- BOOLQ 和 RTE 是采样的子集（可能为了训练效率）
- **所有数据均为原始数据，未经改写**

**内容示例**（COPA）：
```json
{
  "premise": "My body cast a shadow over the grass.",
  "choice1": "The sun was rising.",
  "choice2": "The grass was cut.",
  "question": "cause",
  "idx": 0,
  "label": 0
}
```

---

### 2. **rejection_sampling/0_data/** - ✅ 合成数据（已验证）

**生成方法**：Rejection Sampling（拒绝采样）

**改写统计**（以 COPA 为例）：
- 总样本数：400
- 改写样本：296 (74.0%)
- 保留原始：104 (26.0%)

**内容示例**（COPA，对比原始数据）：
```json
// 原始数据
{
  "premise": "My body cast a shadow over the grass.",
  ...
}

// rejection_sampling 改写后
{
  "premise": "A shadow appeared on the grass beside me.",
  ...
}
```

**改写特点**：
- 语义保留，表达方式改变
- 部分样本保留原始（拒绝采样机制：如果改写质量不佳则保留原始）
- 字段结构与原始数据完全相同

---

### 3. **synthetic/mezo/** - ✅ 合成数据（已验证）

**生成方法**：MeZO + GPT 改写

**改写统计**（以 COPA 为例）：
- 总样本数：400
- 改写样本：400 (100.0%)
- 保留原始：0 (0%)

**版本管理**：
- 多个版本目录：version_1, version_2, ..., version_13-2
- 每个版本是不同参数或提示的改写结果

**内容示例**（COPA，对比原始数据）：
```json
// 原始数据
{
  "premise": "My body cast a shadow over the grass.",
  ...
}

// synthetic/mezo/version_1 改写后
{
  "premise": "A shadow from my body was cast across the grass.",
  ...
}
```

**改写特点**：
- 100% 改写，无保留原始样本
- 改写质量较高，语义完全保留
- 多个版本可用于对比实验

---

## 🆕 新项目数据位置

### 路径：`/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/`

### 存在的数据目录

#### 1. **Data/original/** - ✅ 原始数据

**状态**：✅ 已迁移，与 Backup/Data/original 完全相同

**验证结果**：
```
新项目 Copa train: 400 样本
旧项目 Copa train: 400 样本
所有样本匹配: True
```

**数据集列表**：
- Copa (400 train, 100 val, 500 test)
- BOOLQ (1000 train, 3270 val)
- CB (250 train, 56 val, 250 test)
- RTE (1000 train, 277 val, 3000 test)
- ArcC_Cloze (1119 train, 299 val, 1172 test)
- ArcC_MC (1119 train, 299 val, 1172 test)

**结论**：新项目的原始数据已完整迁移

---

#### 2. **Data_v2/original/** - ✅ 原始数据（重复）

**状态**：✅ 存在，内容与 Data/original 相同

**说明**：Data_v2 是新的数据组织结构，包含：
- `Data_v2/original/` - 原始数据
- `Data_v2/synthetic/` - 新生成的合成数据

---

#### 3. **Data_v2/synthetic/** - ✅ 新生成的合成数据

**状态**：✅ 包含使用 automation 系统新生成的数据

**结构**：
```
Data_v2/synthetic/
├── _shared/                                  # 共享数据池
│   └── Copa/
│       ├── temp07_topp10_gpt4o/             # 不同参数生成的数据
│       └── temp09_topp10_gpt4o/
├── batch_20241229_temperature/               # Batch实验1
│   └── Copa/
│       └── temp09_topp10_gpt4o -> ...       # 符号链接到共享池
└── batch_20241230_temperature_study/        # Batch实验2
    └── Copa/
        └── temp07_topp10_gpt4o -> ...       # 符号链接到共享池
```

**特点**：
- 使用 Batch 方案3++（参数指纹去重）
- 符号链接机制避免重复数据
- 按实验目的组织（batch_YYYYMMDD_purpose）

---

### 不存在的数据目录

#### ❌ **Data/rejection_sampling/** - 旧合成数据（未迁移）

**状态**：❌ 不存在

**原因**：旧项目的合成数据尚未迁移到新项目

**位置**：仍在 Backup 项目中
- `Backup/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/`

**迁移方案**：使用 `migrate_synthetic_data.sh` 脚本

---

## 📊 数据对比总结表

| 数据类型 | Backup 项目位置 | 新项目位置 | 迁移状态 | 验证状态 |
|---------|---------------|----------|---------|---------|
| **原始数据** | `Data/original/` | `Data/original/`<br>`Data_v2/original/` | ✅ 已迁移 | ✅ 已验证（100%匹配） |
| **Rejection Sampling 合成数据** | `Data/rejection_sampling/0_data/` | ❌ 不存在 | ❌ 未迁移 | ✅ 已验证（74%改写） |
| **MeZO 合成数据** | `Data/synthetic/mezo/` | ❌ 不存在 | ❌ 未迁移 | ✅ 已验证（100%改写） |
| **新生成的合成数据** | - | `Data_v2/synthetic/` | ✅ 新生成 | ✅ 使用automation系统生成 |

---

## 🎯 改写质量对比

### COPA 数据集改写示例（前5个样本）

| 样本 | 原始 premise | rejection_sampling | synthetic/mezo/v1 |
|-----|-------------|-------------------|-------------------|
| 0 | My body cast a shadow over the grass. | A shadow appeared on the grass beside me. | A shadow from my body was cast across the grass. |
| 1 | The woman tolerated her friend's difficult behavior. | The woman was patient with her friend's challenging attitude. | The woman put up with her friend's challenging behavior. |
| 2 | The women met for coffee. | The women gathered at a cafe. | The women gathered for a coffee. |
| 3 | The runner wore shorts. | Shorts were the runner's attire. | The athlete had on a pair of shorts. |
| 4 | The guests of the party hid behind the couch. | During the gathering, the attendees positioned themselves out of sight behind the sofa. | The party attendees concealed themselves behind the couch. |

### 改写特点分析

**rejection_sampling**:
- ✅ 语义保留良好
- ✅ 表达方式多样化
- ⚠️ 26%样本未改写（质量控制机制）
- 🎯 适合需要保守改写的场景

**synthetic/mezo**:
- ✅ 100%改写，无原始样本
- ✅ 改写更自然流畅
- ✅ 词汇替换更丰富
- 🎯 适合需要完全新数据的场景

---

# 快速迁移参考

## TL;DR - 一键命令

运行此命令自动迁移所有旧合成数据：

```bash
cd /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/automation
bash migrate_synthetic_data.sh link  # 创建符号链接（推荐）
# 或
bash migrate_synthetic_data.sh copy  # 创建完整副本（使用更多空间）
```

---

## 📋 文件命名规范（按数据集）

| 数据集 | Train 文件 | Validation 文件 |
|---------|-----------|----------------|
| Copa | `copa_train.jsonl` | `copa_validation.jsonl` |
| BOOLQ | `boolq_train.jsonl` | `boolq_validation.jsonl` |
| CB | `cb_train.jsonl` | `cb_validation.jsonl` |
| RTE | `rte_train.jsonl` | `rte_validation.jsonl` |
| ArcC_Cloze | `ARC-Challenge_train.jsonl` | `ARC-Challenge_validation.jsonl` |
| ArcC_MC | `ARC-Challenge_train.jsonl` | `ARC-Challenge_validation.jsonl` |

**⚠️ 注意：文件名区分大小写！**

---

## 📂 目标目录结构

迁移后应该形成的结构：

```
/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/
├── Copa/
│   ├── copa_train.jsonl
│   └── copa_validation.jsonl
├── BOOLQ/
│   ├── boolq_train.jsonl
│   └── boolq_validation.jsonl
├── CB/
│   ├── cb_train.jsonl
│   └── cb_validation.jsonl
├── RTE/
│   ├── rte_train.jsonl
│   └── rte_validation.jsonl
├── ArcC_Cloze/
│   ├── ARC-Challenge_train.jsonl
│   └── ARC-Challenge_validation.jsonl
└── ArcC_MC/
    ├── ARC-Challenge_train.jsonl
    └── ARC-Challenge_validation.jsonl
```

---

## 🔧 手动迁移（3个命令）

如果不想使用自动脚本：

```bash
# 1. 创建目录
mkdir -p /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/{Copa,BOOLQ,CB,RTE,ArcC_Cloze,ArcC_MC}

# 2. 设置路径变量
OLD=/home/ubuntu/LLM-inference/jikai-project/Backup/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data
NEW=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data

# 3. 链接文件（以 Copa 为例）
ln -s ${OLD}/Copa/copa_train.jsonl ${NEW}/Copa/
ln -s ${OLD}/Copa/copa_validation.jsonl ${NEW}/Copa/
# 对其他数据集重复此步骤...
```

---

## 🔍 验证命令

### 检查目录结构

```bash
tree -L 2 /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data
```

### 验证 Copa 文件

```bash
ls -lh /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/Copa/
```

### 测试文件可读

```bash
head -n 1 /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/Copa/copa_train.jsonl | python3 -m json.tool
```

---

## ⚠️ 常见问题

### 问题 1: "File not found" 错误

**原因**: 文件名大小写不匹配

**解决**:
```bash
# 检查实际文件名（注意大小写）
ls /home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/Copa/

# 确保使用正确的大小写：
# copa_train.jsonl  ✓  正确
# Copa_train.jsonl  ✗  错误
# COPA_train.jsonl  ✗  错误
```

### 问题 2: "Broken symbolic link" 错误

**原因**: 源文件被移动或删除

**解决**: 使用 `copy` 模式而非 `link` 模式
```bash
bash migrate_synthetic_data.sh copy
```

### 问题 3: "Dataset is empty" 错误

**原因**: JSONL 格式问题

**解决**: 验证文件格式
```bash
head -n 1 copa_train.jsonl | python3 -m json.tool
```

### 问题 4: 训练脚本找不到数据

**原因**: 路径设置不正确

**检查**: 训练脚本中的 TASK 路径
```bash
# 应该是：
TASK=/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO/Data/rejection_sampling/0_data/Copa

# 而不是：
TASK=/home/ubuntu/.../Backup/.../Data/rejection_sampling/0_data/Copa
```

---

## 📚 相关文档

- **完整迁移指南**: `SYNTHETIC_DATA_MIGRATION_GUIDE.md`
- **训练脚本**: `../running_scripts/`
- **数据加载器**: `../PromptZO/MeZO/large_models/tasks.py`

---

## 💡 温馨提示

1. **优先使用 `link` 模式**：节省磁盘空间
2. **迁移前备份**：虽然迁移不会修改源文件，但建议先备份
3. **验证迁移结果**：使用上面的验证命令确保文件正确
4. **测试训练脚本**：迁移后运行一个训练脚本测试数据可访问性
