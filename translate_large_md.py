#!/usr/bin/env python3
"""
Comprehensive translation script for large markdown files.
This script translates all Chinese text to English while preserving code blocks and structure.
"""

import re
import sys

# Comprehensive translation dictionary
TRANSLATIONS = {
    # Headers and sections
    "步骤": "Step",
    "场景": "Scenario",
    "断点": "Checkpoint",
    "附录": "Appendix",
    "目录": "Table of Contents",
    "阶段": "Stage",
    "完整工作流程": "Complete Workflow",

    # Actions
    "生成": "Generate",
    "处理": "Process",
    "加载": "Load",
    "保存": "Save",
    "复制": "Copy",
    "合并": "Merge",
    "创建": "Create",
    "执行": "Execute",
    "运行": "Run",
    "审核": "Review",
    "标注": "Annotate",
    "验证": "Validate",
    "测试": "Test",
    "调优": "Optimize",
    "编辑": "Edit",
    "修改": "Modify",
    "注入": "Inject",
    "提取": "Extract",
    "查看": "View",
    "列出": "List",
    "对比": "Compare",
    "分析": "Analyze",

    # Data and files
    "数据": "data",
    "样本": "sample",
    "文件": "file",
    "脚本": "script",
    "配置": "configuration",
    "模板": "template",
    "原始": "original",
    "改写": "rephrased",
    "合成": "synthetic",
    "训练": "training",
    "验证集": "validation set",
    "测试集": "test set",
    "数据集": "dataset",

    # Status and results
    "完成": "Complete",
    "成功": "Success",
    "失败": "Failed",
    "通过": "Pass",
    "合格": "Qualified",
    "不合格": "Unqualified",
    "正确": "Correct",
    "错误": "Error",

    # Descriptions
    "系统输出": "System Output",
    "用户输入": "User Input",
    "示例": "Example",
    "说明": "Description",
    "注意": "Note",
    "重要": "Important",
    "建议": "Recommendation",
    "提示": "Tip",
    "警告": "Warning",

    # Common phrases for file 1 (COMPLETE_PIPELINE_SIMULATION.md)
    "模式": "Mode",
    "详细完整流程": "Detailed Complete Workflow",
    "参数研究": "Parameter Study",
    "数据集基线实验": "dataset baseline experiment",
    "准备配置文件": "Prepare Configuration File",
    "配置模板": "configuration template",
    "配置内容": "Configuration Content",
    "API配置": "API Configuration",
    "默认，探索性实验": "Default, exploratory experiment",
    "完整prompt": "complete prompt",
    "初始为空": "Initially empty",
    "后自动生成": "auto-generated after",
    "生成脚本": "Generate Scripts",
    "合成数据生成脚本自动生成器": "Synthetic Data Generation Script Auto-Generator",
    "生成策略": "Generation Strategy",
    "实验目的": "Experiment Purpose",
    "实验描述": "Experiment Description",
    "任务": "Task",
    "训练方法": "Training Method",
    "生成模型": "Generation Model",
    "验证模型": "Validation Model",
    "实验管理": "Experiment Management",
    "参数指纹": "Parameter Fingerprint",
    "语义化名称": "Semantic Name",
    "在": "in",
    "中搜索指纹": "searching for fingerprint",
    "未找到匹配的实验": "No matching experiment found",
    "将创建新实验": "will create new experiment",
    "创建新实验": "Creating New Experiment",
    "物理存储": "Physical Storage",
    "视图": "View",
    "输出目录": "Output Directory",
    "数据集目录": "Dataset Directory",
    "生成改写脚本": "Generating rephrase scripts",
    "生成验证脚本": "Generating validation scripts",
    "保存配置": "Saving configuration",
    "配置副本": "Config Copy",
    "实验元数据": "Experiment Metadata",
    "生成完成": "Generation Complete",
    "脚本位置": "Script Location",
    "使用方法": "Usage",
    "API配置已从配置文件读取": "API configuration loaded from config file",
    "无需设置环境变量": "no need to set environment variables",
    "前20个样本": "First 20 Samples",
    "条原始数据": "original data samples",
    "输出文件": "Output file",
    "处理数据": "Processing data",
    "人工审核前20个样本": "Manual Review of First 20 Samples",
    "加载数据": "Loading data",
    "对比 - 请仔细查看原始数据与改写数据": "Comparison - Please carefully review original vs rephrased data",
    "请输入不合格样本的序号": "Enter the sample numbers that are unqualified",
    "多个序号用逗号分隔": "separated by commas",
    "表示第": "means samples",
    "个样本不合格": "are unqualified",
    "如果全部合格，直接按回车": "If all are qualified, press Enter directly",
    "不合格样本序号": "Unqualified sample numbers",
    "系统继续输出": "System Continues Output",
    "统计": "Statistics",
    "合格样本": "Qualified samples",
    "不合格样本": "Unqualified samples",
    "执行rejection sampling": "Performing rejection sampling",
    "使用原始数据": "Using original data",
    "替换被拒绝的样本": "replaced rejected samples",
    "质量良好": "good quality",
    "保存结果": "Saving results",
    "已保存": "Saved",
    "生成few-shot examples": "Generating few-shot examples",
    "生成了": "Generated",
    "注入few-shot到": "Injecting few-shot into",
    "已注入到": "injected to",
    "备份已保存": "Backup saved",
    "完成": "Complete",
    "下一步": "Next step",
    "生成剩余数据": "Generate Remaining Data",
    "剩余": "remaining",
    "合并数据": "Merge Data",
    "处理第": "Process Samples",
    "个样本": "samples",
    "并排查看": "view side-by-side",
    "执行rejection sampling": "Execute rejection sampling",
    "自动生成": "Auto-generate",
    "并注入到": "and inject into",
    "基于审核结果": "Based on review results",
    "读取": "Read",
    "提取": "Extract",
    "格式化成": "Format as",
    "自动更新": "Automatically update",
    "中的prompt": "prompt in",
    "功能": "Function",
    "格式化成": "Format as",
    "人工标注第": "Manual Annotation of Samples",
    "提取第": "Extract Samples",
    "共": "total",
    "人工标注": "Manual Annotation",
    "可选参数": "Optional parameters",
    "自定义输出文件": "custom output file",
    "重新开始，不继续上次标注": "restart, don't continue previous annotation",
    "标注界面示例": "Annotation Interface Example",
    "原始数据第": "sample #",
    "语义是否一致": "Is semantics consistent",
    "语义一致": "semantics consistent",
    "语义改变": "semantics changed",
    "您的判断": "Your judgment",
    "继续标注样本": "Continue annotating samples",
    "原始数据中的索引": "Index in original data",
    "第": "#",
    "人工判断": "Human judgment",
    "备注": "Note",
    "自动生成validation prompt测试脚本": "Auto-generate validation prompt test script",
    "使用默认路径": "Using default paths",
    "或指定参数": "or specify parameters",
    "样本": "samples",
    "格式化成test_set": "Format as test_set",
    "包含ground truth": "including ground truth",
    "生成的测试脚本": "Generated test script",
    "来自第": "from samples",
    "个": "",
    "用途": "Purpose",
    "测试": "test",
    "准确率": "accuracy",
    "测试prompt准确率": "Test prompt accuracy",
    "在": "on",
    "上": "on",
    "测试结果": "Test Results",
    "判断为": "Judged as",
    "与Ground Truth对比": "Compared with Ground Truth",
    "判断正确": "Correct judgment",
    "判断错误": "Wrong judgment",
    "准确率": "Accuracy",
    "测试通过": "Test passed",
    "可以继续执行": "Can continue to execute",
    "自动验证剩余数据": "Auto-validate remaining data",
    "批量验证所有": "Batch validate all",
    "加载训练数据": "Loading training data",
    "总样本数": "Total samples",
    "已处理样本": "Processed samples",
    "待验证样本": "Samples to validate",
    "加载validation配置": "Loading validation configuration",
    "开始自动验证": "Starting auto-validation",
    "验证进度": "Validation progress",
    "验证结果统计": "Validation Result Statistics",
    "总验证样本": "Total validated samples",
    "保留改写": "Keep rephrased",
    "替换为原始": "Replace with original",
    "保存最终数据": "Saving final data",
    "复制validation和test集": "Copying validation and test sets",
    "已复制": "Copied",
    "数据集生成完成": "Dataset generation complete",
    "最终数据集": "Final dataset",
    "训练集": "Training set",
    "条": "samples",
    "改写数据": "rephrased data",
    "原始数据": "original data",
    "数据集路径": "Dataset path",
    "可直接用于": "Can be used directly for",
    "训练": "training",

    # Dataset-specific
    "数据集特点": "Dataset Characteristics",
    "任务类型": "Task Type",
    "改写字段": "Field to Rephrase",
    "其他字段": "Other Fields",
    "数据示例": "Data Example",
    "配置文件关键修改": "Key Configuration Modifications",
    "流程与": "The workflow is the same as",
    "完全相同": "completely the same",
    "只是字段名从": "except the field name changes from",
    "变为": "to",
    "样本对比示例": "Sample Comparison Examples",
    "特有": "specific",
    "完整执行流程": "Complete Execution Workflow",
    "相同": "same",
    "只需": "only need",
    "按照": "Follow",
    "依次执行": "execute in sequence",

    # Part 2 - Direct-All Mode
    "使用场景": "Use Case",
    "已经通过第一次": "After the first",
    "生成获得了可用的": "generation obtained usable",
    "现在想要快速探究": "now want to quickly explore",
    "对合成数据质量的影响": "impact on synthetic data quality",
    "配置": "Configuration",
    "直接全量生成": "Direct full generation",
    "参数变量": "parameter variable",
    "必须包含完整的few-shot": "Must include complete few-shot",
    "从第一次": "from the first",
    "生成中获得": "obtained from generation",
    "模式不需要": "mode doesn't need",
    "个不同temperature的配置": "configurations with different temperature",
    "重复生成其他": "Repeat for other",
    "自动管理": "Automatic Management",
    "关键差异": "Key Differences",
    "固定": "fixed",
    "研究变量": "research variable",
    "执行流程": "Execution Workflow",
    "可改为": "can change to",

    # Appendix
    "数据集对比表": "Dataset Comparison Table",
    "改写字段": "Field to Rephrase",
    "其他字段": "Other Fields",
    "断点数": "Checkpoints",
    "总耗时": "Total Time",
    "估算": "estimated",
    "分钟": "minutes",
    "人工参与时间对比": "Manual Participation Time Comparison",
    "浏览": "Browse",
    "输入序号": "input numbers",
    "总计人工时间": "Total manual time",
    "无需人工参与": "No manual participation needed",
    "完全自动化": "Fully automated",
    "适合参数研究": "suitable for parameter studies",
    "关键要点总结": "Key Points Summary",
    "批量输入模式": "Batch Input Mode",
    "用户只需输入不合格序号": "Users only need to input unqualified numbers",
    "无需逐个确认": "No need to confirm one by one",
    "大幅减少人工交互时间": "Significantly reduces manual interaction time",
    "自动Rejection Sampling": "Automatic Rejection Sampling",
    "系统自动替换不合格样本为原始数据": "System automatically replaces unqualified samples with original data",
    "所有": "all",
    "都执行rejection sampling": "all execute rejection sampling",
    "自动Few-shot生成": "Automatic Few-shot Generation",
    "从": "from",
    "个合格样本生成": "qualified samples generate",
    "自动标注": "Automatic Annotation",
    "标注由系统自动完成": "annotation automatically completed by system",
    "用于测试AI judge准确率": "used to test AI judge accuracy",
    "多数据集零代码支持": "Zero-code Support for Multiple Datasets",
    "只需修改配置文件中的字段名和prompt": "Only need to modify field names and prompts in config file",
    "无需修改任何代码": "No code changes needed",
    "参数去重": "Parameter Deduplication",
    "方案": "Solution",
    "自动检测相同参数配置": "Automatically detect same parameter configuration",
    "避免重复生成": "Avoid duplicate generation",
    "通过符号链接组织实验": "Organize experiments through symbolic links",
    "两种生成策略": "Two Generation Strategies",
    "探索性实验": "Exploratory experiment",
    "需要确定prompt和few-shot": "need to determine prompt and few-shot",
    "prompt已确定": "prompt already determined",
    "快速参数研究": "fast parameter study",
    "使用建议": "Usage Recommendations",
    "首次实验": "First experiment",
    "使用": "Use",
    "模式确定最佳prompt和few-shot examples": "mode to determine best prompt and few-shot examples",
    "获得few-shot后": "After obtaining few-shot",
    "快速生成不同参数配置的数据": "quickly generate data with different parameter configurations",
    "认真审核前": "Carefully review the first",
    "等工具查看和管理实验": "and other tools to view and manage experiments",
    "数据复用": "Data Reuse",
    "善用Batch系统的参数去重功能": "Make good use of Batch system's parameter deduplication feature",
}

def translate_text(text):
    """Translate Chinese text to English."""
    for cn, en in TRANSLATIONS.items():
        # Use word boundaries to avoid partial matches
        text = re.sub(r'\b' + re.escape(cn) + r'\b', en, text)
    return text

def process_file(input_file, output_file):
    """Process markdown file and translate Chinese to English."""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    translated_lines = []
    in_code_block = False

    for line in lines:
        # Check if we're entering/leaving a code block
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            translated_lines.append(line)
            continue

        # Don't translate code blocks
        if in_code_block:
            translated_lines.append(line)
            continue

        # Translate the line
        translated_line = translate_text(line)
        translated_lines.append(translated_line)

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(translated_lines))

    print(f"Translated {input_file} -> {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python translate_large_md.py <input_file> [output_file]")
        print("\nThis will translate Chinese text to English in markdown files")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file + ".translated.md"

    process_file(input_file, output_file)
