#!/usr/bin/env python3
"""
Stage 1: Synthetic Data Generation Automation
Auto-generate data generation and validation scripts from configuration files
"""

import os
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add automation directory to path, import unified config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROJECT_ROOT, SYNTHETIC_BASE, ORIGINAL_DATA_DIR

# Add utils directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

# Import experiment manager (Batch solution)
from experiment_manager_batch import (
    BatchExperimentManager,
    compute_parameter_fingerprint,
    save_experiment_metadata
)

# Import API config loader
try:
    from api_config_loader import load_api_config, get_openai_client_code
except ImportError:
    # If not found, provide default implementation
    def load_api_config(config_name="default"):
        return {
            'base_url': 'https://api2.aigcbest.top/v1',
            'timeout': 120
        }

class SyntheticDataGenerator:
    def __init__(self, config_path: str, auto_resolve_conflicts: bool = False):
        """
        Initialize generator

        Args:
            config_path: Configuration file path (YAML)
            auto_resolve_conflicts: Auto-resolve directory conflicts (don't prompt user)
        """
        self.config = self.load_config(config_path)
        self.validate_config()

        self.project_root = PROJECT_ROOT
        self.output_base = SYNTHETIC_BASE
        self.auto_resolve_conflicts = auto_resolve_conflicts

        # Initialize experiment manager (Batch solution)
        self.experiment_manager = BatchExperimentManager(self.output_base)

    def load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def validate_config(self):
        """Validate required fields in configuration file"""
        required_fields = [
            'task_name',
            'dataset',
            'generation'
        ]

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field in config file: {field}")

        # Validate generation section
        gen_fields = ['model', 'temperature', 'rephrase_prompt', 'field_to_rephrase']
        for field in gen_fields:
            if field not in self.config['generation']:
                raise ValueError(f"Missing required field in generation config: {field}")

        # Check generation strategy
        strategy = self.config['generation'].get('strategy', 'two_stage')

        # Validate validation section (optional in direct_all mode)
        if strategy == 'two_stage':
            if 'validation' not in self.config:
                raise ValueError(f"Missing required field in two_stage mode: validation")

            val_fields = ['model', 'validation_prompt', 'few_shot_examples']
            for field in val_fields:
                if field not in self.config['validation']:
                    raise ValueError(f"Missing required field in validation config: {field}")
        elif strategy == 'direct_all':
            # direct_all mode: validation config is optional
            pass
        else:
            raise ValueError(f"Unknown generation strategy: {strategy}. Supported strategies: 'two_stage', 'direct_all'")

    def _get_api_config_code(self, config_name: str = "generation") -> str:
        """
        Get API configuration code (for generated scripts)

        Args:
            config_name: Config name ("generation" or "validation")

        Returns:
            API client initialization code
        """
        # Read API config from generation or validation section
        if config_name == "generation" and 'generation' in self.config:
            api_key = self.config['generation'].get('api_key', 'sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c')
            base_url = self.config['generation'].get('base_url', 'https://api2.aigcbest.top/v1')
            timeout = self.config['generation'].get('timeout', 120)
        elif config_name == "validation" and 'validation' in self.config:
            api_key = self.config['validation'].get('api_key', 'sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c')
            base_url = self.config['validation'].get('base_url', 'https://api2.aigcbest.top/v1')
            timeout = self.config['validation'].get('timeout', 120)
        else:
            # Default values
            api_key = 'sk-eWSYPo0CvhRYgcJs55B0C3F00aC74f6e95F47c1f4772292c'
            base_url = 'https://api2.aigcbest.top/v1'
            timeout = 120

        return f'''# API configuration (read from {config_name} section in config file)
API_KEY = "{api_key}"
API_BASE = "{base_url}"

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
    timeout={timeout}
)'''

    def get_output_dir_name(self) -> Tuple[Path, str]:
        """
        Generate output directory path based on config (Batch solution 3++)

        Directory structure:
        - _shared/{Dataset}/{semantic_dirname}/  # Physical storage
        - batch_{date}_{purpose}/{Dataset}/{semantic_dirname}/  # Logical view (symlink)

        Examples:
        - _shared/Copa/temp07_topp09_gpt4o/  # Actual data
        - batch_20241229_temperature/Copa/temp07_topp09_gpt4o/  # Symlink

        Returns:
            (physical_dir, fingerprint)
            physical_dir: Physical directory path in _shared/ (actual data generation location)
            fingerprint: Parameter fingerprint (for deduplication)
        """
        # Use experiment manager to prepare directory
        output_dir, fingerprint = self.experiment_manager.prepare_experiment_dir(
            self.config,
            auto_resolve=self.auto_resolve_conflicts
        )

        return output_dir, fingerprint

    def generate_rephrase_script(self, output_dir: Path, strategy: str = "all") -> str:
        """
        Generate data rephrasing script

        Args:
            output_dir: Output directory path
            strategy: "all", "top20", "rest"

        Returns:
            Script content
        """
        cfg = self.config
        gen_cfg = cfg['generation']
        dataset_cfg = cfg['dataset']

        # 确定数据量限制
        if strategy == "all":
            limit_code = ""
            output_suffix = ""
        elif strategy == "top20":
            limit_code = """
    if progress > 20:
        break
"""
            output_suffix = "_top20"
        elif strategy == "rest":
            limit_code = """
    if progress <= 20:
        continue
"""
            output_suffix = "_rest"
        else:
            raise ValueError(f"未知策略: {strategy}")

        # 生成脚本
        script = f'''#!/usr/bin/env python3
"""
自动生成的合成数据生成脚本

任务: {cfg['task_name']}
训练方法: {cfg.get('training_method', 'general')}
生成模型: {gen_cfg['model']}
策略: {strategy}
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from tqdm import tqdm
import os
import json
from openai import OpenAI

{self._get_api_config_code("generation")}

# ⭐ Few-shot examples placeholder (will be injected by review_top20.py)
# FEWSHOT_EXAMPLES = [...]

def generate_prompt({', '.join(dataset_cfg['fields'])}):
    """生成改写提示词"""

    # ⭐ 构建few-shot文本（如果存在FEWSHOT_EXAMPLES）
    fewshot_text = ""
    if 'FEWSHOT_EXAMPLES' in globals() and len(FEWSHOT_EXAMPLES) > 0:
        for i, ex in enumerate(FEWSHOT_EXAMPLES, 1):
            fewshot_text += f"Example {{i}}:\\n"
            fewshot_text += f"Original {gen_cfg['field_to_rephrase']}: {{ex['original']}}\\n"
            fewshot_text += f"Rephrased {gen_cfg['field_to_rephrase']}: {{ex['rephrased']}}\\n"
            # 添加其他字段作为上下文
            for key in ex:
                if key not in ['original', 'rephrased']:
                    fewshot_text += f"{{key}}: {{ex[key]}}\\n"
            fewshot_text += "\\n"

    # ⭐ 原始prompt模板
    prompt_template = """\\
{gen_cfg['rephrase_prompt']}
"""

    # ⭐ 替换{{{{REPHRASE_FEWSHOT}}}}占位符
    prompt = prompt_template.replace("{{{{REPHRASE_FEWSHOT}}}}", fewshot_text)

    # ⭐ 替换字段值（使用.format()）
    return prompt.format({', '.join([f'{f}={f}' for f in dataset_cfg['fields']])})

# 加载原始数据
data = []
input_file = "{self.project_root / dataset_cfg['input_path']}"
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

print(f"加载了 {{len(data)}} 条原始数据")

# 准备输出
# 🆕 创建数据集子目录
dataset_dir = os.path.join("{output_dir}", "{dataset_cfg.get('dataset_name', cfg.get('task_name', 'Dataset'))}")
os.makedirs(dataset_dir, exist_ok=True)

output_file = os.path.join(dataset_dir, "{dataset_cfg['task_name']}_train{output_suffix}.jsonl")
out_file = open(output_file, "w", encoding='utf-8')

print(f"输出文件: {{output_file}}")

# 处理数据
progress = 0
for i in tqdm(range(len(data))):
    progress += 1{limit_code}

    # 构造提示词
    prompt_args = {{{', '.join([f'"{f}": data[i]["{f}"]' for f in dataset_cfg['fields']])}}}
    prompt = generate_prompt(**prompt_args)

    # 调用 API
    try:
        response = client.chat.completions.create(
            model="{gen_cfg['model']}",
            messages=[
                {{"role": "user", "content": prompt}}
            ],
            temperature={gen_cfg['temperature']}
        )

        # 提取结果
        rephrased_text = response.choices[0].message.content.strip()

        # 构造输出
        result = data[i].copy()
        result["{gen_cfg['field_to_rephrase']}"] = rephrased_text

        # 写入文件
        out_file.write(json.dumps(result, ensure_ascii=False) + "\\n")
        out_file.flush()

    except Exception as e:
        print(f"\\n处理第 {{i}} 条数据时出错: {{e}}")
        # 出错时使用原始数据
        out_file.write(json.dumps(data[i], ensure_ascii=False) + "\\n")
        out_file.flush()

out_file.close()
print(f"\\n完成! 输出: {{output_file}}")
'''
        return script

    def generate_validation_script(self, output_dir: Path) -> str:
        """生成验证脚本"""
        cfg = self.config
        val_cfg = cfg['validation']
        dataset_cfg = cfg['dataset']
        gen_cfg = cfg['generation']

        # 从配置读取field_to_rephrase
        field_to_rephrase = gen_cfg['field_to_rephrase']

        script = f'''#!/usr/bin/env python3
"""
自动生成的合成数据验证脚本（拒绝采样）

任务: {cfg['task_name']}
训练方法: {cfg.get('training_method', 'general')}
验证模型: {val_cfg['model']}
Field to rephrase: {field_to_rephrase}
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from tqdm import tqdm
import os
import json
from openai import OpenAI

{self._get_api_config_code("validation")}

# ⭐ 尝试从validation_checkpoints加载自动生成的few-shot
# （由annotate_samples.py生成）
VALIDATION_FEWSHOT_EXAMPLES = []
try:
    import sys
    from pathlib import Path
    checkpoint_file = Path(__file__).parent.parent / "validation_checkpoints" / "validation_fewshot.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            fewshot_data = json.load(f)
            VALIDATION_FEWSHOT_EXAMPLES = fewshot_data.get('examples', [])
        print(f"✓ 加载了 {{len(VALIDATION_FEWSHOT_EXAMPLES)}} 个自动生成的validation few-shot examples")
except Exception as e:
    print(f"⚠️  未找到自动生成的few-shot，将使用配置文件中的few-shot: {{e}}")

def generate_validation_prompt({', '.join(['original_' + f for f in dataset_cfg['fields']] + ['rephrased_' + field_to_rephrase])}):
    """生成验证提示词"""

    # ⭐ 构建few-shot文本
    fewshot_text = ""

    # 优先使用自动生成的few-shot（来自样本21-40）
    if len(VALIDATION_FEWSHOT_EXAMPLES) > 0:
        for i, ex in enumerate(VALIDATION_FEWSHOT_EXAMPLES, 1):
            fewshot_text += f"Example {{i}}:\\n"
            fewshot_text += f"Original {field_to_rephrase}: {{ex.get('original_{field_to_rephrase}', 'N/A')}}\\n"
            fewshot_text += f"Rephrased {field_to_rephrase}: {{ex.get('rephrased_{field_to_rephrase}', 'N/A')}}\\n"
            # 添加其他字段
            for key in ex:
                if not key.startswith('original_') and not key.startswith('rephrased_') and key != 'evaluation':
                    fewshot_text += f"{{key}}: {{ex[key]}}\\n"
            fewshot_text += f"Evaluation: {{ex.get('evaluation', 'same')}}\\n\\n"
    else:
        # 备用：使用配置文件中手动提供的few-shot
        manual_examples = {val_cfg.get('few_shot_examples', [])}
        for i, ex in enumerate(manual_examples, 1):
            if isinstance(ex, dict):
                fewshot_text += f"Example {{i}}:\\n"
                for k, v in ex.items():
                    fewshot_text += f"{{k}}: {{v}}\\n"
                fewshot_text += "\\n"

    # ⭐ 原始prompt模板
    prompt_template = """\\
{val_cfg['validation_prompt']}
"""

    # ⭐ 替换{{{{VALIDATION_FEWSHOT}}}}占位符
    prompt = prompt_template.replace("{{{{VALIDATION_FEWSHOT}}}}", fewshot_text)

    # ⭐ 构建字段字典用于format
    format_dict = {{}}
    for field in {dataset_cfg['fields']}:
        format_dict[f'original_{{field}}'] = locals().get(f'original_{{field}}', '')
    format_dict['rephrased_{field_to_rephrase}'] = locals().get('rephrased_{field_to_rephrase}', '')

    # ⭐ 替换字段值
    return prompt.format(**format_dict)
"""

# 加载原始数据
original_data = []
with open("{self.project_root / dataset_cfg['input_path']}", 'r', encoding='utf-8') as f:
    for line in f:
        original_data.append(json.loads(line.strip()))

# 加载合成数据
# 🆕 从数据集子目录读取
dataset_dir = os.path.join("{output_dir}", "{dataset_cfg.get('dataset_name', cfg.get('task_name', 'Dataset'))}")
synthetic_data = []
synthetic_file = os.path.join(dataset_dir, "{dataset_cfg['task_name']}_train.jsonl")
with open(synthetic_file, 'r', encoding='utf-8') as f:
    for line in f:
        synthetic_data.append(json.loads(line.strip()))

print(f"原始数据: {{len(original_data)}} 条")
print(f"合成数据: {{len(synthetic_data)}} 条")

if len(original_data) != len(synthetic_data):
    print("⚠ 警告: 数据量不匹配!")

# 准备输出（临时文件）
temp_output_file = os.path.join(dataset_dir, "{dataset_cfg['task_name']}_train_validated.jsonl")
out_file = open(temp_output_file, "w", encoding='utf-8')

correct_count = 0
total_count = 0

# 验证每条数据
for i in tqdm(range(min(len(original_data), len(synthetic_data)))):
    original = original_data[i]
    synthetic = synthetic_data[i]

    # 🔴 排除样本21-40（索引20-39）
    # 这些样本用作judger的few-shot examples，不应被judger验证（避免数据泄露）
    if 20 <= i < 40:
        # 直接使用合成数据，不经过judger验证
        out_file.write(json.dumps(synthetic, ensure_ascii=False) + "\\n")
        correct_count += 1
        total_count += 1
        out_file.flush()
        continue

    # 构造验证提示词
    prompt_args = {{}}
    for field in {dataset_cfg['fields']}:
        prompt_args[f'original_{{field}}'] = original[field]
    prompt_args['rephrased_{gen_cfg['field_to_rephrase']}'] = synthetic['{gen_cfg['field_to_rephrase']}']

    prompt = generate_validation_prompt(**prompt_args)

    try:
        response = client.chat.completions.create(
            model="{val_cfg['model']}",
            messages=[
                {{"role": "system", "content": "You are a helpful judge."}},
                {{"role": "user", "content": prompt}}
            ],
            temperature=0.0
        )

        result = response.choices[0].message.content.strip().lower()

        # 判断是否通过验证
        if 'not the same' in result or 'not same' in result:
            # 验证失败，使用原始数据
            out_file.write(json.dumps(original, ensure_ascii=False) + "\\n")
        else:
            # 验证成功，使用合成数据
            out_file.write(json.dumps(synthetic, ensure_ascii=False) + "\\n")
            correct_count += 1

        total_count += 1
        out_file.flush()

    except Exception as e:
        print(f"\\n验证第 {{i}} 条数据时出错: {{e}}")
        # 出错时使用原始数据
        out_file.write(json.dumps(original, ensure_ascii=False) + "\\n")
        total_count += 1
        out_file.flush()

out_file.close()

accuracy = correct_count / total_count if total_count > 0 else 0
print(f"\\n验证完成!")
print(f"通过率: {{correct_count}}/{{total_count}} = {{accuracy:.2%}}")
print(f"临时输出文件: {{temp_output_file}}")

# 🆕 最终化数据集：重命名validated文件 + 复制validation/test
print("\\n最终化数据集...")
import shutil

# 1. 将validated文件重命名为正式的train文件
final_train_file = os.path.join(dataset_dir, "{dataset_cfg['task_name']}_train.jsonl")
if os.path.exists(final_train_file):
    os.remove(final_train_file)  # 删除原始的未验证文件
shutil.move(temp_output_file, final_train_file)
print(f"✓ 训练集: {{final_train_file}}")

# 2. 复制validation和test文件from原始数据集
original_dir = "{self.project_root / dataset_cfg.get('original_dir', dataset_cfg['input_path'].rsplit('/', 1)[0])}"
files_config = {dataset_cfg.get('files', {})}

# 复制validation文件
if 'validation' in files_config:
    val_file = files_config['validation']
    src_val = os.path.join(original_dir, val_file)
    dst_val = os.path.join(dataset_dir, val_file)
    if os.path.exists(src_val):
        shutil.copy2(src_val, dst_val)
        print(f"✓ 验证集: {{dst_val}}")
    else:
        print(f"⚠  警告: 验证集文件不存在: {{src_val}}")

# 复制test文件（如果有）
if 'test' in files_config:
    test_file = files_config['test']
    src_test = os.path.join(original_dir, test_file)
    dst_test = os.path.join(dataset_dir, test_file)
    if os.path.exists(src_test):
        shutil.copy2(src_test, dst_test)
        print(f"✓ 测试集: {{dst_test}}")

print(f"\\n✅ 数据集已完成！可用于MeZO训练：")
print(f"   python PromptZO/MeZO/large_models/run.py --task {{dataset_dir}}")
'''
        return script

    def create_config_copy(self, output_dir: Path):
        """保存配置文件副本到输出目录"""
        config_copy_path = output_dir / "generation_config.yaml"
        with open(config_copy_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
        print(f"✓ 配置副本: {config_copy_path}")

    def generate_all(self):
        """生成所有脚本"""
        print("\n" + "="*80)
        print(f"合成数据生成脚本自动生成器")
        print("="*80)

        # 获取实验信息
        experiment_cfg = self.config.get('experiment', {})
        experiment_purpose = experiment_cfg.get('purpose', 'general')
        experiment_desc = experiment_cfg.get('description', '')

        # ⭐ 获取生成策略
        gen_strategy = self.config['generation'].get('strategy', 'two_stage')

        print(f"生成策略: {gen_strategy}")
        print(f"实验目的: {experiment_purpose}")
        if experiment_desc:
            print(f"实验描述: {experiment_desc}")
        print(f"任务: {self.config['task_name']}")
        print(f"训练方法: {self.config.get('training_method', 'general')}")
        print(f"生成模型: {self.config['generation']['model']}")

        # ⭐ validation模型信息（仅在two_stage模式下显示）
        if gen_strategy == 'two_stage' and 'validation' in self.config:
            print(f"验证模型: {self.config['validation']['model']}")

        print("="*80)

        # 使用实验管理器准备输出目录（包含冲突检测）
        output_dir, fingerprint = self.get_output_dir_name()
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n输出目录: {output_dir.relative_to(self.project_root)}")
        print(f"参数指纹: {fingerprint}")

        # 🆕 创建数据集子目录（用于存放数据文件）
        dataset_cfg = self.config['dataset']
        dataset_name = dataset_cfg.get('dataset_name', self.config.get('task_name', 'Dataset'))
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        print(f"数据集目录: {dataset_dir.relative_to(self.project_root)}")

        # 创建 scripts 子目录
        scripts_dir = output_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        # ⭐ 根据生成策略生成不同的脚本
        print("\n生成改写脚本...")

        if gen_strategy == 'direct_all':
            # 🔥 direct_all 模式：只生成 rephrase_all.py
            script_path = scripts_dir / "rephrase_all.py"
            script_content = self.generate_rephrase_script(output_dir, "all")

            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)

            os.chmod(script_path, 0o755)
            print(f"  ✓ {script_path.name}")
            print(f"  (direct_all 模式：跳过 top20 和 rest 脚本)")

        else:  # two_stage 模式（默认）
            # 生成 all, top20, rest 三个脚本
            for strategy in ["all", "top20", "rest"]:
                script_path = scripts_dir / f"rephrase_{strategy}.py"
                script_content = self.generate_rephrase_script(output_dir, strategy)

                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(script_content)

                os.chmod(script_path, 0o755)
                print(f"  ✓ {script_path.name}")

        # ⭐ 生成验证脚本（仅在 two_stage 模式且配置了 validation 时）
        if gen_strategy == 'two_stage' and 'validation' in self.config:
            print("\n生成验证脚本...")
            val_script_path = scripts_dir / "validate.py"
            val_script_content = self.generate_validation_script(output_dir)

            with open(val_script_path, 'w', encoding='utf-8') as f:
                f.write(val_script_content)

            os.chmod(val_script_path, 0o755)
            print(f"  ✓ {val_script_path.name}")
        elif gen_strategy == 'direct_all':
            print("\n跳过验证脚本生成（direct_all 模式）")

        # 保存配置副本
        print("\n保存配置...")
        self.create_config_copy(output_dir)

        # 保存实验元数据
        metadata_path = save_experiment_metadata(output_dir, self.config, fingerprint)
        if metadata_path:
            print(f"✓ 实验元数据: {metadata_path.relative_to(self.project_root)}")

        # 生成 README
        readme_content = self.generate_readme(output_dir, fingerprint)
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"✓ README: {readme_path.relative_to(self.project_root)}")

        print("\\n" + "="*80)
        print("生成完成！")
        print("="*80)
        print(f"\\n脚本位置: {scripts_dir}")

        # ⭐ 根据策略显示不同的使用说明
        if gen_strategy == 'direct_all':
            print(f"\\n使用方法 (direct_all 模式):")
            print(f"  1. 设置环境变量: export OPENAI_API_KEY=your-key")
            print(f"  2. 直接运行生成: python {scripts_dir}/rephrase_all.py")
            print(f"  3. 生成完成后，数据保存在: {dataset_dir}")
        else:
            print(f"\\n使用方法 (two_stage 模式):")
            print(f"  1. 设置环境变量: export OPENAI_API_KEY=your-key")
            print(f"  2. 运行生成: python {scripts_dir}/rephrase_all.py")
            print(f"  3. 运行验证: python {scripts_dir}/validate.py")

    def generate_readme(self, output_dir: Path, fingerprint: str) -> str:
        """生成 README 文档"""
        cfg = self.config
        experiment_cfg = cfg.get('experiment', {})
        gen_strategy = cfg['generation'].get('strategy', 'two_stage')

        # ⭐ 构建validation模型信息（如果有）
        val_model_info = ""
        if 'validation' in cfg:
            val_model_info = f"\n- **验证模型**: {cfg['validation']['model']}"

        return f"""# {cfg['task_name']} 合成数据生成

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 实验信息

- **实验目的**: {experiment_cfg.get('purpose', 'general')}
- **实验ID**: {experiment_cfg.get('experiment_id', 'N/A')}
- **实验描述**: {experiment_cfg.get('description', 'N/A')}
- **参数指纹**: {fingerprint}

## 配置信息

- **生成策略**: {gen_strategy}
- **任务**: {cfg['task_name']}
- **训练方法**: {cfg.get('training_method', 'general')}
- **数据集**: {cfg['dataset']['task_name']}
- **生成模型**: {cfg['generation']['model']}
- **Temperature**: {cfg['generation']['temperature']}{val_model_info}
- **版本**: {cfg.get('version', 'v1')}

## 目录结构

```
{output_dir.name}/
├── {cfg['dataset'].get('dataset_name', cfg['task_name'])}/     # 🆕 数据集目录（MeZO可直接使用）
│   ├── {cfg['dataset']['task_name']}_train.jsonl              # 合成+验证后的训练集
│   ├── {cfg['dataset']['task_name']}_validation.jsonl         # 验证集（复制自原始）
│   └── {cfg['dataset']['task_name']}_test.jsonl               # 测试集（复制自原始）
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
{"# direct_all 模式：直接生成全部数据" if gen_strategy == 'direct_all' else "# 方式1: 改写全部数据"}
python scripts/rephrase_all.py
{"" if gen_strategy == 'direct_all' else '''
# 方式2: 分别改写困难样本和剩余样本
python scripts/rephrase_top20.py
python scripts/rephrase_rest.py'''}
```
{"" if gen_strategy == 'direct_all' else '''
### 3. 验证数据质量并最终化数据集

```bash
python scripts/validate.py
```

此脚本会：
1. 使用rejection sampling验证合成数据质量
2. 将验证通过的数据重命名为正式训练集
3. 从原始数据集复制validation和test文件
4. 生成完整的MeZO可用数据集
'''}
### {"3" if gen_strategy == 'direct_all' else "4"}. 使用数据集训练模型

```bash
# 使用MeZO训练
python PromptZO/MeZO/large_models/run.py \\
    --task {cfg['dataset'].get('dataset_name', cfg['task_name'])} \\
    --model meta-llama/Llama-3.2-1B \\
    --num_train_epochs 3 \\
    --per_device_train_batch_size 4
```

## 最终数据集结构

```
{cfg['dataset'].get('dataset_name', cfg['task_name'])}/
├── {cfg['dataset']['task_name']}_train.jsonl       # {"合成数据" if gen_strategy == 'direct_all' else "合成+验证后的训练集"}
├── {cfg['dataset']['task_name']}_validation.jsonl  # 验证集（来自原始数据）
└── {cfg['dataset']['task_name']}_test.jsonl        # 测试集（来自原始数据）
```

此目录可以直接传递给MeZO训练脚本使用。

## Prompt 信息

### 改写 Prompt

```
{cfg['generation']['rephrase_prompt'][:200]}...
```
{"" if gen_strategy == 'direct_all' else f'''
### 验证 Prompt

```
{cfg.get('validation', {}).get('validation_prompt', 'N/A')[:200]}...
```
'''}
详见 `generation_config.yaml`
"""

def main():
    import argparse

    parser = argparse.ArgumentParser(description="合成数据生成脚本自动生成器")
    parser.add_argument("config", help="配置文件路径 (YAML)")
    parser.add_argument(
        "--auto-resolve",
        action="store_true",
        help="自动解决目录冲突（不提示用户）"
    )
    args = parser.parse_args()

    generator = SyntheticDataGenerator(args.config, auto_resolve_conflicts=args.auto_resolve)
    generator.generate_all()

if __name__ == "__main__":
    main()
