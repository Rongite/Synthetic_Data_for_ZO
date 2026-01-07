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
    BatchExperiment managementr,
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
        self.experiment_manager = BatchExperiment managementr(self.output_base)

    def load_config(self, config_path: str) -> Dict:
        """Loaded configuration file"""
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

        # Determine data quantity limit
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
            raise ValueError(f"UnknownStrategy: {strategy}")

        # generatescript
        script = f'''#!/usr/bin/env python3
"""
Automatically generated synthetic data generation script

task: {cfg['task_name']}
Training method: {cfg.get('training_method', 'general')}
Generation model: {gen_cfg['model']}
Strategy: {strategy}
Generation time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from tqdm import tqdm
import os
import json
from openai import OpenAI

{self._get_api_config_code("generation")}

# ⭐ Few-shot examples placeholder (will be injected by review_top20.py)
# FEWSHOT_EXAMPLES = [...]

def generate_prompt({', '.join(dataset_cfg['fields'])}):
    """Generate rephrase prompt"""

    # ⭐ Build few-shot text (if FEWSHOT_EXAMPLES exists)
    fewshot_text = ""
    if 'FEWSHOT_EXAMPLES' in globals() and len(FEWSHOT_EXAMPLES) > 0:
        for i, ex in enumerate(FEWSHOT_EXAMPLES, 1):
            fewshot_text += f"Example {{i}}:\\n"
            fewshot_text += f"Original {gen_cfg['field_to_rephrase']}: {{ex['original']}}\\n"
            fewshot_text += f"Rephrased {gen_cfg['field_to_rephrase']}: {{ex['rephrased']}}\\n"
            # add other fields as context
            for key in ex:
                if key not in ['original', 'rephrased']:
                    fewshot_text += f"{{key}}: {{ex[key]}}\\n"
            fewshot_text += "\\n"

    # ⭐ Original prompt template
    prompt_template = """\\
{gen_cfg['rephrase_prompt']}
"""

    # ⭐ Replace{{{{REPHRASE_FEWSHOT}}}}placeholder
    prompt = prompt_template.replace("{{{{REPHRASE_FEWSHOT}}}}", fewshot_text)

    # ⭐ Replace field values (using .format())
    return prompt.format({', '.join([f'{f}={f}' for f in dataset_cfg['fields']])})

# Loaded original data
data = []
input_file = "{self.project_root / dataset_cfg['input_path']}"
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

print(f"Loaded {{len(data)}} original data samples")

# Prepare output
# 🆕 create dataset subdirectory
dataset_dir = os.path.join("{output_dir}", "{dataset_cfg.get('dataset_name', cfg.get('task_name', 'Dataset'))}")
os.makedirs(dataset_dir, exist_ok=True)

output_file = os.path.join(dataset_dir, "{dataset_cfg['task_name']}_train{output_suffix}.jsonl")
out_file = open(output_file, "w", encoding='utf-8')

print(f"output file: {{output_file}}")

# process data
progress = 0
for i in tqdm(range(len(data))):
    progress += 1{limit_code}

    # Construct prompt
    prompt_args = {{{', '.join([f'"{f}": data[i]["{f}"]' for f in dataset_cfg['fields']])}}}
    prompt = generate_prompt(**prompt_args)

    # Call API
    try:
        response = client.chat.completions.create(
            model="{gen_cfg['model']}",
            messages=[
                {{"role": "user", "content": prompt}}
            ],
            temperature={gen_cfg['temperature']}
        )

        # Extract result
        rephrased_text = response.choices[0].message.content.strip()

        # Construct output
        result = data[i].copy()
        result["{gen_cfg['field_to_rephrase']}"] = rephrased_text

        # write to file
        out_file.write(json.dumps(result, ensure_ascii=False) + "\\n")
        out_file.flush()

    except Exception as e:
        print(f"\\nError processing sample {{i}} : {{e}}")
        # Use original data on error
        out_file.write(json.dumps(data[i], ensure_ascii=False) + "\\n")
        out_file.flush()

out_file.close()
print(f"\\nComplete! output: {{output_file}}")
'''
        return script

    def generate_validation_script(self, output_dir: Path) -> str:
        """generatevalidatescript"""
        cfg = self.config
        val_cfg = cfg['validation']
        dataset_cfg = cfg['dataset']
        gen_cfg = cfg['generation']

        #  from configurereadfield_to_rephrase
        field_to_rephrase = gen_cfg['field_to_rephrase']

        script = f'''#!/usr/bin/env python3
"""
Automaticgeneratesynthetic datavalidatescript（rejection sampling）

task: {cfg['task_name']}
Training method: {cfg.get('training_method', 'general')}
validatemodel: {val_cfg['model']}
Field to rephrase: {field_to_rephrase}
Generation time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from tqdm import tqdm
import os
import json
from openai import OpenAI

{self._get_api_config_code("validation")}

# ⭐ Try to load automatically generated few-shot from validation_checkpoints
# （Generated by annotate_samples.py）
VALIDATION_FEWSHOT_EXAMPLES = []
try:
    import sys
    from pathlib import Path
    checkpoint_file = Path(__file__).parent.parent / "validation_checkpoints" / "validation_fewshot.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            fewshot_data = json.load(f)
            VALIDATION_FEWSHOT_EXAMPLES = fewshot_data.get('examples', [])
        print(f"✓ Loaded {{len(VALIDATION_FEWSHOT_EXAMPLES)}} Automaticgeneratevalidation few-shot examples")
except Exception as e:
    print(f"⚠️  Automatically generated few-shot not found, will use few-shot from configuration file: {{e}}")

def generate_validation_prompt({', '.join(['original_' + f for f in dataset_cfg['fields']] + ['rephrased_' + field_to_rephrase])}):
    """generatevalidatePrompt"""

    # ⭐ Build few-shot text
    fewshot_text = ""

    # Prefer to use automatically generated few-shot (from samples 21-40)
    if len(VALIDATION_FEWSHOT_EXAMPLES) > 0:
        for i, ex in enumerate(VALIDATION_FEWSHOT_EXAMPLES, 1):
            fewshot_text += f"Example {{i}}:\\n"
            fewshot_text += f"Original {field_to_rephrase}: {{ex.get('original_{field_to_rephrase}', 'N/A')}}\\n"
            fewshot_text += f"Rephrased {field_to_rephrase}: {{ex.get('rephrased_{field_to_rephrase}', 'N/A')}}\\n"
            # add other fields
            for key in ex:
                if not key.startswith('original_') and not key.startswith('rephrased_') and key != 'evaluation':
                    fewshot_text += f"{{key}}: {{ex[key]}}\\n"
            fewshot_text += f"Evaluation: {{ex.get('evaluation', 'same')}}\\n\\n"
    else:
        # Fallback: use manually provided few-shot from configuration file
        manual_examples = {val_cfg.get('few_shot_examples', [])}
        for i, ex in enumerate(manual_examples, 1):
            if isinstance(ex, dict):
                fewshot_text += f"Example {{i}}:\\n"
                for k, v in ex.items():
                    fewshot_text += f"{{k}}: {{v}}\\n"
                fewshot_text += "\\n"

    # ⭐ Original prompt template
    prompt_template = """\\
{val_cfg['validation_prompt']}
"""

    # ⭐ Replace{{{{VALIDATION_FEWSHOT}}}}placeholder
    prompt = prompt_template.replace("{{{{VALIDATION_FEWSHOT}}}}", fewshot_text)

    # ⭐ buildFielddictionary use  at format
    format_dict = {{}}
    for field in {dataset_cfg['fields']}:
        format_dict[f'original_{{field}}'] = locals().get(f'original_{{field}}', '')
    format_dict['rephrased_{field_to_rephrase}'] = locals().get('rephrased_{field_to_rephrase}', '')

    # ⭐ ReplaceFieldvalue
    return prompt.format(**format_dict)
"""

# Loaded original data
original_data = []
with open("{self.project_root / dataset_cfg['input_path']}", 'r', encoding='utf-8') as f:
    for line in f:
        original_data.append(json.loads(line.strip()))

# Loadsynthetic data
# 🆕  read from dataset subdirectory
dataset_dir = os.path.join("{output_dir}", "{dataset_cfg.get('dataset_name', cfg.get('task_name', 'Dataset'))}")
synthetic_data = []
synthetic_file = os.path.join(dataset_dir, "{dataset_cfg['task_name']}_train.jsonl")
with open(synthetic_file, 'r', encoding='utf-8') as f:
    for line in f:
        synthetic_data.append(json.loads(line.strip()))

print(f"original data: {{len(original_data)}}  samples")
print(f"synthetic data: {{len(synthetic_data)}}  samples")

if len(original_data) != len(synthetic_data):
    print("⚠ warning: data count mismatch!")

# Prepare output（temporaryfile）
temp_output_file = os.path.join(dataset_dir, "{dataset_cfg['task_name']}_train_validated.jsonl")
out_file = open(temp_output_file, "w", encoding='utf-8')

correct_count = 0
total_count = 0

# Validate each data sample
for i in tqdm(range(min(len(original_data), len(synthetic_data)))):
    original = original_data[i]
    synthetic = synthetic_data[i]

    # 🔴 Excludesample21-40（Index20-39）
    # These samples are used as judger few-shot examples, should not be validated by judger (avoid data leakage)
    if 20 <= i < 40:
        # Directly use synthetic data without judger validation
        out_file.write(json.dumps(synthetic, ensure_ascii=False) + "\\n")
        correct_count += 1
        total_count += 1
        out_file.flush()
        continue

    # ConstructionvalidatePrompt
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

        # Determine if validation passes
        if 'not the same' in result or 'not same' in result:
            # validatefail，useoriginal data
            out_file.write(json.dumps(original, ensure_ascii=False) + "\\n")
        else:
            # validatesuccess，usesynthetic data
            out_file.write(json.dumps(synthetic, ensure_ascii=False) + "\\n")
            correct_count += 1

        total_count += 1
        out_file.flush()

    except Exception as e:
        print(f"\\nvalidateline {{i}} : {{e}}")
        # Use original data on error
        out_file.write(json.dumps(original, ensure_ascii=False) + "\\n")
        total_count += 1
        out_file.flush()

out_file.close()

accuracy = correct_count / total_count if total_count > 0 else 0
print(f"\\nvalidateComplete!")
print(f"Pass rate: {{correct_count}}/{{total_count}} = {{accuracy:.2%}}")
print(f"temporaryoutput file: {{temp_output_file}}")

# 🆕 finalationdataset：Renamevalidatedfile + Copyvalidation/test
print("\\nfinalationdataset...")
import shutil

# 1.  Rename validated file as official train file
final_train_file = os.path.join(dataset_dir, "{dataset_cfg['task_name']}_train.jsonl")
if os.path.exists(final_train_file):
    os.remove(final_train_file)  # Delete original unvalidated file
shutil.move(temp_output_file, final_train_file)
print(f"✓ training set: {{final_train_file}}")

# 2. Copyvalidation and testfilefromoriginal dataset
original_dir = "{self.project_root / dataset_cfg.get('original_dir', dataset_cfg['input_path'].rsplit('/', 1)[0])}"
files_config = {dataset_cfg.get('files', {})}

# Copyvalidationfile
if 'validation' in files_config:
    val_file = files_config['validation']
    src_val = os.path.join(original_dir, val_file)
    dst_val = os.path.join(dataset_dir, val_file)
    if os.path.exists(src_val):
        shutil.copy2(src_val, dst_val)
        print(f"✓ validation set: {{dst_val}}")
    else:
        print(f"⚠  warning: validation setfiledoes not exist: {{src_val}}")

# Copytestfile（Ifhas）
if 'test' in files_config:
    test_file = files_config['test']
    src_test = os.path.join(original_dir, test_file)
    dst_test = os.path.join(dataset_dir, test_file)
    if os.path.exists(src_test):
        shutil.copy2(src_test, dst_test)
        print(f"✓ Testset: {{dst_test}}")

print(f"\\n✅ Dataset is complete! Can be used for MeZO training:")
print(f"   python PromptZO/MeZO/large_models/run.py --task {{dataset_dir}}")
'''
        return script

    def create_config_copy(self, output_dir: Path):
        """Saveconfiguration fileReplica to outputdirectory"""
        config_copy_path = output_dir / "generation_config.yaml"
        with open(config_copy_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
        print(f"✓ configureReplica: {config_copy_path}")

    def generate_all(self):
        """generatethathasscript"""
        print("\n" + "="*80)
        print(f"Automatically generated synthetic data generation script")
        print("="*80)

        # Get experiment information
        experiment_cfg = self.config.get('experiment', {})
        experiment_purpose = experiment_cfg.get('purpose', 'general')
        experiment_desc = experiment_cfg.get('description', '')

        # ⭐ Get generation strategy
        gen_strategy = self.config['generation'].get('strategy', 'two_stage')

        print(f"generateStrategy: {gen_strategy}")
        print(f"Experimentpurpose: {experiment_purpose}")
        if experiment_desc:
            print(f"Experiment description: {experiment_desc}")
        print(f"task: {self.config['task_name']}")
        print(f"Training method: {self.config.get('training_method', 'general')}")
        print(f"Generation model: {self.config['generation']['model']}")

        # ⭐ validationmodelinformation（only in two_stageMode down Show）
        if gen_strategy == 'two_stage' and 'validation' in self.config:
            print(f"validatemodel: {self.config['validation']['model']}")

        print("="*80)

        # useExperiment managementtoolPrepare outputdirectory（containconflictDetect）
        output_dir, fingerprint = self.get_output_dir_name()
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\noutputdirectory: {output_dir.relative_to(self.project_root)}")
        print(f"parameterfingerprint: {fingerprint}")

        # 🆕 create dataset subdirectory（ use  at Storedatafile）
        dataset_cfg = self.config['dataset']
        dataset_name = dataset_cfg.get('dataset_name', self.config.get('task_name', 'Dataset'))
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        print(f"datasetdirectory: {dataset_dir.relative_to(self.project_root)}")

        # create scripts Subdirectory
        scripts_dir = output_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        # ⭐ according togenerateStrategygenerateDifferentscript
        print("\ngenerateRephrasescript...")

        if gen_strategy == 'direct_all':
            # 🔥 direct_all Mode：onlygenerate rephrase_all.py
            script_path = scripts_dir / "rephrase_all.py"
            script_content = self.generate_rephrase_script(output_dir, "all")

            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)

            os.chmod(script_path, 0o755)
            print(f"  ✓ {script_path.name}")
            print(f"  (direct_all Mode：skip top20  and  rest script)")

        else:  # two_stage Mode（Default）
            # generate all, top20, rest threescript
            for strategy in ["all", "top20", "rest"]:
                script_path = scripts_dir / f"rephrase_{strategy}.py"
                script_content = self.generate_rephrase_script(output_dir, strategy)

                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(script_content)

                os.chmod(script_path, 0o755)
                print(f"  ✓ {script_path.name}")

        # ⭐ generatevalidatescript（only in  two_stage Modeandconfigure validation time）
        if gen_strategy == 'two_stage' and 'validation' in self.config:
            print("\ngeneratevalidatescript...")
            val_script_path = scripts_dir / "validate.py"
            val_script_content = self.generate_validation_script(output_dir)

            with open(val_script_path, 'w', encoding='utf-8') as f:
                f.write(val_script_content)

            os.chmod(val_script_path, 0o755)
            print(f"  ✓ {val_script_path.name}")
        elif gen_strategy == 'direct_all':
            print("\nskipvalidatescriptgenerate（direct_all Mode）")

        # SaveconfigureReplica
        print("\nSaveconfigure...")
        self.create_config_copy(output_dir)

        # SaveExperimentmetadata
        metadata_path = save_experiment_metadata(output_dir, self.config, fingerprint)
        if metadata_path:
            print(f"✓ Experimentmetadata: {metadata_path.relative_to(self.project_root)}")

        # generate README
        readme_content = self.generate_readme(output_dir, fingerprint)
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"✓ README: {readme_path.relative_to(self.project_root)}")

        print("\\n" + "="*80)
        print("generatecomplete！")
        print("="*80)
        print(f"\\nscriptlocation: {scripts_dir}")

        # ⭐ according toStrategyShowDifferentuseDescription
        if gen_strategy == 'direct_all':
            print(f"\\nusemethod (direct_all Mode):")
            print(f"  1. Settingsenvironment variable: export OPENAI_API_KEY=your-key")
            print(f"  2. directlyRungenerate: python {scripts_dir}/rephrase_all.py")
            print(f"  3. generatecomplete back ，dataSave in : {dataset_dir}")
        else:
            print(f"\\nusemethod (two_stage Mode):")
            print(f"  1. Settingsenvironment variable: export OPENAI_API_KEY=your-key")
            print(f"  2. Rungenerate: python {scripts_dir}/rephrase_all.py")
            print(f"  3. Runvalidate: python {scripts_dir}/validate.py")

    def generate_readme(self, output_dir: Path, fingerprint: str) -> str:
        """generate README Document"""
        cfg = self.config
        experiment_cfg = cfg.get('experiment', {})
        gen_strategy = cfg['generation'].get('strategy', 'two_stage')

        # ⭐ buildvalidationmodelinformation（Ifhas）
        val_model_info = ""
        if 'validation' in cfg:
            val_model_info = f"\n- **validatemodel**: {cfg['validation']['model']}"

        return f"""# {cfg['task_name']} synthetic datagenerate

**Generation time**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Experimentinformation

- **Experimentpurpose**: {experiment_cfg.get('purpose', 'general')}
- **ExperimentID**: {experiment_cfg.get('experiment_id', 'N/A')}
- **Experiment description**: {experiment_cfg.get('description', 'N/A')}
- **parameterfingerprint**: {fingerprint}

## configureinformation

- **generateStrategy**: {gen_strategy}
- **task**: {cfg['task_name']}
- **Training method**: {cfg.get('training_method', 'general')}
- **dataset**: {cfg['dataset']['task_name']}
- **Generation model**: {cfg['generation']['model']}
- **Temperature**: {cfg['generation']['temperature']}{val_model_info}
- **Version**: {cfg.get('version', 'v1')}

## directorystructure

```
{output_dir.name}/
├── {cfg['dataset'].get('dataset_name', cfg['task_name'])}/     # 🆕 datasetdirectory（MeZO can directlyuse）
│   ├── {cfg['dataset']['task_name']}_train.jsonl              # synthetic+validate back training set
│   ├── {cfg['dataset']['task_name']}_validation.jsonl         # validation set（Copyselforiginal）
│   └── {cfg['dataset']['task_name']}_test.jsonl               # Testset（Copyselforiginal）
├── scripts/
│   ├── rephrase_all.py      # RephraseAlldata
│   ├── rephrase_top20.py    # Rephrase front 20difficultsample
│   ├── rephrase_rest.py     # Rephraseremainingsample
│   └── validate.py          # validatescript（rejection sampling+datasetfinalation）
├── generation_config.yaml   # configuration fileReplica
├── experiment_metadata.json # Experimentmetadata
└── README.md               # versionfile
```

## usemethod

### 1. Settingsenvironment variable

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"  #  can select
```

### 2. generatesynthetic data

```bash
{"# direct_all Mode：directlygenerateAlldata" if gen_strategy == 'direct_all' else "# method1: RephraseAlldata"}
python scripts/rephrase_all.py
{"" if gen_strategy == 'direct_all' else '''
# method2: classifycategoryRephrasedifficultsample and remainingsample
python scripts/rephrase_top20.py
python scripts/rephrase_rest.py'''}
```
{"" if gen_strategy == 'direct_all' else '''
### 3. validatedataqualityandfinalationdataset

```bash
python scripts/validate.py
```

thisscript will ：
1. userejection samplingvalidatesynthetic dataquality
2.  will validatepassdataRename as officialtraining set
3.  from original datasetCopyvalidation and testfile
4. generateCompleteMeZO can  use dataset
'''}
### {"3" if gen_strategy == 'direct_all' else "4"}. usedatasettrainingmodel

```bash
# useMeZOtraining
python PromptZO/MeZO/large_models/run.py \\
    --task {cfg['dataset'].get('dataset_name', cfg['task_name'])} \\
    --model meta-llama/Llama-3.2-1B \\
    --num_train_epochs 3 \\
    --per_device_train_batch_size 4
```

## finaldatasetstructure

```
{cfg['dataset'].get('dataset_name', cfg['task_name'])}/
├── {cfg['dataset']['task_name']}_train.jsonl       # {"synthetic data" if gen_strategy == 'direct_all' else "synthetic+validate back training set"}
├── {cfg['dataset']['task_name']}_validation.jsonl  # validation set（fromoriginal data）
└── {cfg['dataset']['task_name']}_test.jsonl        # Testset（fromoriginal data）
```

thisdirectorycandirectlypass to MeZOtrainingscriptuse。

## Prompt information

### Rephrase Prompt

```
{cfg['generation']['rephrase_prompt'][:200]}...
```
{"" if gen_strategy == 'direct_all' else f'''
### validate Prompt

```
{cfg.get('validation', {}).get('validation_prompt', 'N/A')[:200]}...
```
'''}
See details `generation_config.yaml`
"""

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Automatically generated synthetic data generation script")
    parser.add_argument("config", help="configuration filepath (YAML)")
    parser.add_argument(
        "--auto-resolve",
        action="store_true",
        help="AutomaticSolvedirectoryconflict（nottipUser）"
    )
    args = parser.parse_args()

    generator = SyntheticDataGenerator(args.config, auto_resolve_conflicts=args.auto_resolve)
    generator.generate_all()

if __name__ == "__main__":
    main()
