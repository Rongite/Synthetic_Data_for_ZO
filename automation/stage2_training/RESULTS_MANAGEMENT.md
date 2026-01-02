# 训练结果管理系统

## 🔴 核心设计理念：阶段1和阶段2的实验目的独立

### **为什么要分开？**

**阶段1（数据生成）的实验目的**：
- 回答："为什么生成这个数据？"
- 示例：`prompt_engineering`, `temperature_study`, `data_quality_optimization`
- 存储位置：`Data_v2/synthetic/{数据生成目的}/`

**阶段2（模型训练）的实验目的**：
- 回答："为什么进行这个训练？"
- 示例：`model_comparison`, `hyperparameter_tuning`, `baseline_comparison`
- 存储位置：`Results_v2/{训练目的}/`

### **典型场景**

```
【场景】：使用同一个数据集进行多种不同的训练实验

数据集（阶段1）：
Data_v2/synthetic/prompt_engineering/copa_mezo_v1/
↑ 数据生成目的：测试prompt对数据质量的影响

训练实验（阶段2）：
├── Results_v2/model_comparison/        ← 训练目的：对比不同模型
├── Results_v2/hyperparameter_tuning/   ← 训练目的：调整学习率
├── Results_v2/baseline_comparison/     ← 训练目的：与原始数据对比
└── Results_v2/ablation_study/          ← 训练目的：消融实验
```

**关键点**：
- ✅ 同一个数据集（`prompt_engineering/copa_mezo_v1`）可以用于多个不同的训练实验
- ✅ 每个训练实验有自己的目的，结果按训练目的分类
- ❌ 如果不分开，所有结果都会混在`prompt_engineering`目录下，无法区分

---

## 📋 目录结构

### **新的Results_v2结构**

```
Results_v2/
└── {experiment_purpose}/           # 🆕 实验目的分类（与Data_v2对齐）
    └── {Model}/
        └── {Task}_{Method}_{DataType}_{LR}/
            └── {Timestamp}/
                ├── experiment_config.yaml  # 实验配置
                ├── {lr}_train.out         # 训练输出
                ├── {lr}_train.err         # 错误输出
                └── ...                    # 模型checkpoint等
```

### **目录说明**

1. **experiment_purpose**: 实验目的分类
   - 与Data_v2的experiment_purpose对应
   - 例如：`prompt_engineering`, `temperature_study`, `model_comparison`

2. **Model**: 模型名称
   - 例如：`meta-llama/Llama-3.2-1B`, `mistralai/Mistral-Nemo-Base-2407`

3. **Task_Method_DataType_LR**: 实验标识
   - Task: 任务名称（Copa, BOOLQ, CB等）
   - Method: 训练方法（zo, fo_full, fo_lora）
   - DataType: 数据类型（original, synthetic等）
   - LR: 学习率（格式化，如`1_7`表示1e-7）

4. **Timestamp**: 时间戳（格式：YYYYMMDD_HHMMSS）
   - 同一配置的多次运行会创建不同的时间戳目录

---

## 🎯 核心功能

### **1. 训练实验目的分类**

训练结果按**训练实验目的**分类（与数据生成目的独立）：

```yaml
# 配置文件
experiment:
  purpose: "hyperparameter_tuning"  # 🔴 训练目的！结果保存到: Results_v2/hyperparameter_tuning/

data:
  path: "Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa"
  #                        ↑ 数据生成目的（与训练目的不同）
```

### **2. 必须显式指定训练目的**

`experiment.purpose`必须显式指定，如果未指定则使用默认值`uncategorized`：

```yaml
# ✅ 推荐：显式指定
experiment:
  purpose: "model_comparison"

# ⚠️  如果不指定，结果会保存到 Results_v2/uncategorized/
```

**推荐的训练实验目的类别**：
- `baseline_comparison` - 与baseline对比
- `model_comparison` - 对比不同模型
- `hyperparameter_tuning` - 超参数调优
- `ablation_study` - 消融实验
- `prompt_effectiveness` - 测试prompt效果
- `data_quality_impact` - 测试数据质量影响
- `scaling_study` - 扩展性研究

### **3. 完整元数据追溯**

每个训练实验自动保存完整配置：

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

## 📖 使用指南

### **场景1：超参数调优（使用合成数据）**

```yaml
# training_config.yaml
experiment:
  purpose: "hyperparameter_tuning"  # 🔴 训练目的：调优超参数
  description: "使用copa_mezo_v1数据测试不同学习率"

model: "meta-llama/Llama-3.2-1B"
task: "Copa"
method: "zo"

data:
  path: "Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa"  # 🆕 直接指定路径

hyperparameters:
  learning_rate: [1e-6, 5e-7, 2e-7, 1e-7]
  batch_size: 16
  steps: 20000
  seed: 0
```

**运行训练**：
```bash
python automation/stage2_training/trainer.py training_config.yaml
```

**结果保存到**：
```
Results_v2/hyperparameter_tuning/meta-llama/Llama-3.2-1B/
                ↑ 按训练目的分类（不是数据生成目的）
├── Copa_zo_copa_mezo_v1_1_6/
│   └── 20251226_143000/
├── Copa_zo_copa_mezo_v1_5_7/
│   └── 20251226_143000/
├── Copa_zo_copa_mezo_v1_2_7/
│   └── 20251226_143000/
└── Copa_zo_copa_mezo_v1_1_7/
    └── 20251226_143000/
```

### **场景2：模型对比（使用相同数据）**

```yaml
# training_config.yaml
experiment:
  purpose: "model_comparison"  # 🔴 训练目的：对比不同模型
  description: "在copa_mezo_v1数据上对比Llama和Mistral"

model: "mistralai/Mistral-Nemo-Base-2407"  # 🔧 测试不同模型
task: "Copa"
method: "zo"

data:
  path: "Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa"
  #                        ↑ 数据来自prompt_engineering实验
  #                        ↑ 但训练目的是model_comparison

hyperparameters:
  learning_rate: 5e-7  # 使用已知最佳学习率
  batch_size: 16
  steps: 20000
  seed: 0
```

**系统行为**：
- 数据来源：`Data_v2/synthetic/prompt_engineering/...`
- 训练目的：`model_comparison`（与数据生成目的不同）
- 结果保存到：`Results_v2/model_comparison/`

### **场景3：Baseline对比（原始数据 vs 合成数据）**

```yaml
# training_config.yaml
experiment:
  purpose: "baseline_comparison"  # 🔴 训练目的：对比baseline
  description: "对比原始数据和合成数据的训练效果"

model: "meta-llama/Llama-3.2-1B"
task: "Copa"
method: "zo"

data:
  path: "Data_v2/original/Copa"  # 🔧 使用原始数据作为baseline

hyperparameters:
  learning_rate: 5e-7  # 使用与合成数据相同的超参数
  batch_size: 16
  steps: 20000
  seed: 0
```

**结果保存到**：
```
Results_v2/baseline_comparison/meta-llama/Llama-3.2-1B/Copa_zo_original_5_7/20251226_143000/
```

**对比分析**：
```
合成数据结果：Results_v2/hyperparameter_tuning/.../Copa_zo_copa_mezo_v1_5_7/...
原始数据结果：Results_v2/baseline_comparison/.../Copa_zo_original_5_7/...
↑ 两个实验都保存在各自的实验目的目录下，方便对比
```

---

## 🔧 管理工具

### **list_results.py**

列出并管理所有训练结果。

#### **查看摘要**

```bash
python automation/stage2_training/list_results.py
```

**输出示例**：
```
================================================================================
训练结果摘要 - Results_v2
================================================================================

📁 实验目的: prompt_engineering
   实验数量: 12
   └─ meta-llama/Llama-3.2-1B: 12 个实验

📁 实验目的: temperature_study
   实验数量: 8
   └─ meta-llama/Llama-3.2-1B: 8 个实验

📁 实验目的: baseline
   实验数量: 4
   └─ meta-llama/Llama-3.2-1B: 4 个实验

================================================================================
总计: 3 个实验目的, 24 个训练实验
================================================================================
```

#### **查看详细信息**

```bash
# 查看所有实验的详细信息
python automation/stage2_training/list_results.py --detail

# 查看特定实验目的的详细信息
python automation/stage2_training/list_results.py --detail --purpose prompt_engineering
```

**输出示例**：
```
================================================================================
训练结果详情
================================================================================

📁 实验目的: prompt_engineering
--------------------------------------------------------------------------------

  [1] Copa_zo_copa_mezo_v1_1_6
      模型: meta-llama/Llama-3.2-1B
      时间: 20251226_143000
      路径: Results_v2/prompt_engineering/meta-llama/Llama-3.2-1B/Copa_zo_copa_mezo_v1_1_6/20251226_143000
      任务: Copa
      方法: zo
      超参数:
        - LR: 1e-06
        - BS: 16
        - Steps: 20000
        - Seed: 0
      数据: Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa

  [2] Copa_zo_copa_mezo_v1_5_7
      ...
```

---

## 🔄 数据-结果对应关系

### **完整的实验追溯链**

```
阶段1：数据生成
Data_v2/synthetic/
└── prompt_engineering/           # 实验目的
    └── copa_mezo_v1/              # 实验ID
        ├── Copa/                  # 数据集
        │   ├── copa_train.jsonl
        │   ├── copa_validation.jsonl
        │   └── copa_test.jsonl
        └── experiment_metadata.json  # 数据生成参数

                    ⬇

阶段2：模型训练
Results_v2/
└── prompt_engineering/           # 🔗 相同的实验目的
    └── meta-llama/Llama-3.2-1B/
        └── Copa_zo_copa_mezo_v1_1_6/
            └── 20251226_143000/
                └── experiment_config.yaml  # 训练参数
```

### **对应关系**

| 数据集 | 训练结果 |
|--------|----------|
| `Data_v2/synthetic/{purpose}/{exp_id}/{Dataset}` | `Results_v2/{purpose}/{Model}/{Task}_{Method}_{exp_id}_{LR}/{Timestamp}` |

**关键点**：
- `{purpose}` 在两边保持一致
- `{exp_id}` 在结果目录名中体现
- 通过`experiment_config.yaml`中的`data.path`可以追溯到源数据

---

## 📊 最佳实践

### **1. 训练实验目的命名规范**

**推荐的训练实验目的类别**（阶段2）：

- `baseline_comparison` - 与baseline对比
- `model_comparison` - 模型对比实验
- `hyperparameter_tuning` - 超参数调优
- `ablation_study` - 消融实验
- `prompt_effectiveness` - 测试prompt效果
- `data_quality_impact` - 测试数据质量影响
- `scaling_study` - 扩展性研究
- `method_comparison` - 训练方法对比（MeZO vs LoRA vs Full FT）

**数据生成实验目的类别**（阶段1，仅供参考）：

- `prompt_engineering` - Prompt优化实验
- `temperature_study` - 温度参数研究
- `data_quality_optimization` - 数据质量优化
- `few_shot_study` - Few-shot示例研究

### **2. 配置文件组织**

按**训练实验目的**组织配置文件：

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

**注意**：配置文件按训练目的分类，不是按数据集分类

### **3. 实验记录**

每次重要实验后，在对应的实验目的目录下记录：

```bash
# 在Results_v2/{训练目的}/README.md中记录
echo "## 实验记录

### 2025-12-26: 学习率扫描实验
- 训练目的: hyperparameter_tuning
- 数据集: Data_v2/synthetic/prompt_engineering/copa_mezo_v1/
- 模型: Llama-3.2-1B
- 学习率网格: [1e-6, 5e-7, 2e-7, 1e-7]
- 最佳结果: LR=5e-7, Acc=85.2%
- 备注: 5e-7是最佳学习率，用于后续实验
" >> Results_v2/hyperparameter_tuning/README.md
```

---

## ⚠️ 注意事项

### **1. 阶段1和阶段2的实验目的是独立的！**

🔴 **最重要的概念**：

```
❌ 错误理解：
   数据来自 Data_v2/synthetic/prompt_engineering/...
   → 结果应该保存到 Results_v2/prompt_engineering/

✅ 正确理解：
   数据来自 Data_v2/synthetic/prompt_engineering/...  ← 数据生成目的
   训练目的是 hyperparameter_tuning                    ← 训练实验目的
   → 结果保存到 Results_v2/hyperparameter_tuning/
```

### **2. 必须显式指定训练实验目的**

系统**不会**从数据路径自动推断训练实验目的：

```yaml
# ❌ 错误：没有指定experiment.purpose
data:
  path: "Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa"
# → 结果会保存到 Results_v2/uncategorized/

# ✅ 正确：显式指定训练目的
experiment:
  purpose: "hyperparameter_tuning"
data:
  path: "Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa"
# → 结果保存到 Results_v2/hyperparameter_tuning/
```

### **3. 旧格式兼容性**

系统仍支持旧的`data.type`格式，但推荐使用新的`data.path`：

```yaml
# ✅ 推荐（新格式）
data:
  path: "Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa"

# ⚠️  已弃用（旧格式）
data:
  type: "synthetic_mezo_gpt4o_v1"
```

### **3. 时间戳隔离**

相同配置的多次运行会创建不同的时间戳目录，避免覆盖：

```
Copa_zo_copa_mezo_v1_1_6/
├── 20251226_143000/  # 第1次运行
├── 20251226_153000/  # 第2次运行
└── 20251227_093000/  # 第3次运行
```

---

## 🎉 总结

### **新系统优势**

1. ✅ **实验目的分类**：结果按实验目的自动组织
2. ✅ **智能推断**：从数据路径自动推断实验目的
3. ✅ **完整追溯**：数据集 ↔ 训练结果完整对应
4. ✅ **元数据管理**：自动保存所有实验参数
5. ✅ **管理工具**：list_results.py快速查看结果

### **与旧系统对比**

| 功能 | 旧系统 | 新系统 |
|------|--------|--------|
| 结果组织 | ❌ 所有结果混在一起 | ✅ 按实验目的分类 |
| 实验追溯 | ❌ 手动记录 | ✅ 自动追溯到数据集 |
| 配置管理 | ⚠️  部分保存 | ✅ 完整保存 |
| 查看工具 | ❌ 无 | ✅ list_results.py |

---

**开始您的训练实验！** 🚀
