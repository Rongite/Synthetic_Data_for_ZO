# Copa 合成数据生成

**生成时间**: 2025-12-30 19:06:38

## 实验信息

- **实验目的**: temperature_comparison
- **实验ID**: N/A
- **实验描述**: 比较temperature=0.5/0.7/0.9对Copa合成数据质量的影响
- **参数指纹**: a5df2df31852

## 配置信息

- **生成策略**: direct_all
- **任务**: Copa
- **训练方法**: mezo
- **数据集**: copa
- **生成模型**: gpt-4o
- **Temperature**: 0.7
- **版本**: v1

## 目录结构

```
temp07_topp10_gpt4o/
├── Copa/     # 🆕 数据集目录（MeZO可直接使用）
│   ├── copa_train.jsonl              # 合成+验证后的训练集
│   ├── copa_validation.jsonl         # 验证集（复制自原始）
│   └── copa_test.jsonl               # 测试集（复制自原始）
├── scripts/
│   ├── rephrase_all.py      # 改写全部数据
│   ├── rephrase_top20.py    # 改写前20个困难样本
│   ├── rephrase_rest.py     # 改写剩余样本
│   └── validate.py          # 验证脚本（拒绝采样+数据集最终化）
├── generation_config.yaml   # 配置文件副本
├── experiment_metadata.json # 实验元数据
└── README.md               # 本文件
```

## 使用方法

### 1. 设置环境变量

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"  # 可选
```

### 2. 生成合成数据

```bash
# direct_all 模式：直接生成全部数据
python scripts/rephrase_all.py

```

### 3. 使用数据集训练模型

```bash
# 使用MeZO训练
python PromptZO/MeZO/large_models/run.py \
    --task Copa \
    --model meta-llama/Llama-3.2-1B \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4
```

## 最终数据集结构

```
Copa/
├── copa_train.jsonl       # 合成数据
├── copa_validation.jsonl  # 验证集（来自原始数据）
└── copa_test.jsonl        # 测试集（来自原始数据）
```

此目录可以直接传递给MeZO训练脚本使用。

## Prompt 信息

### 改写 Prompt

```
You are tasked with rephrasing the given premise while preserving its original meaning. Your goal is to create rephrased data optimized for enhancing gradient estimation in training with a memory-effi...
```

详见 `generation_config.yaml`
