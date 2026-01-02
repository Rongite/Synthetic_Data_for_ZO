#!/usr/bin/env python3
"""
Stage 2: Training Pipeline Automation
Automatically executes training and manages results based on configuration files
"""

import os
import sys
import yaml
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import itertools

# Add automation directory to path, import unified configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROJECT_ROOT, RESULTS_V2_DIR, TRAINING_SCRIPTS_DIR

class TrainingPipeline:
    def __init__(self, config_path: str):
        """
        Initialize training pipeline

        Args:
            config_path: Configuration file path (YAML)
        """
        self.config = self.load_config(config_path)
        self.validate_config()

        self.project_root = PROJECT_ROOT
        self.results_base = RESULTS_V2_DIR
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # NEW: Get training experiment purpose (independent from data generation experiment purpose)
        # Note: This is the "training" experiment purpose, not the "data generation" experiment purpose
        # Example: Data comes from Data_v2/synthetic/prompt_engineering/...
        #          But training purpose might be model_comparison, hyperparameter_tuning, etc.
        self.experiment_purpose = self.config.get('experiment', {}).get('purpose', 'uncategorized')

        # Training script path
        self.training_scripts_dir = TRAINING_SCRIPTS_DIR

    def load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def validate_config(self):
        """Validate configuration file"""
        required_fields = ['model', 'task', 'method', 'data', 'hyperparameters']

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Configuration file missing required field: {field}")

        # Validate hyperparameters
        hp_fields = ['learning_rate', 'batch_size', 'steps', 'seed']
        for field in hp_fields:
            if field not in self.config['hyperparameters']:
                raise ValueError(f"hyperparameters missing required field: {field}")

    def get_method_script_name(self, method: str, data_type: str) -> str:
        """
        Determine training script name based on method and data type

        Args:
            method: zo, fo_full, fo_lora
            data_type: original, synthetic_*, mixed_*

        Returns:
            Script filename
        """
        is_synthetic = data_type.startswith('synthetic') or data_type.startswith('mixed')

        method_map = {
            'zo': ('mezo_finetune_original.sh', 'mezo_finetune_synthetic.sh'),
            'fo_full': ('fo_full_finetune_original.sh', 'fo_full_finetune_synthetic.sh'),
            'fo_lora': ('fo_lora_finetune_original.sh', 'fo_lora_finetune_synthetic.sh')
        }

        if method not in method_map:
            raise ValueError(f"Unknown training method: {method}")

        return method_map[method][1 if is_synthetic else 0]

    def infer_data_type_from_path(self, data_path: str) -> str:
        """
        NEW: Infer data type from data path

        Args:
            data_path: Data path

        Returns:
            Data type: "original" or string containing dataset ID
        """
        path = Path(data_path)
        parts = path.parts

        # Check if it's original data
        if 'original' in parts:
            return 'original'

        # Check if it's synthetic data
        if 'synthetic' in parts:
            synthetic_idx = parts.index('synthetic')
            # Path format: Data_v2/synthetic/{purpose}/{exp_id}/{Dataset}
            # exp_id is typically: copa_mezo_v1, boolq_mezo_temp07, etc.
            if synthetic_idx + 2 < len(parts):
                exp_id = parts[synthetic_idx + 2]  # Get experiment ID
                return exp_id  # Return experiment ID as data type identifier

        # Default to original
        return 'original'

    def get_data_path(self, data_config: Dict) -> str:
        """
        Determine data path based on data configuration

        Args:
            data_config: Data configuration dictionary

        Returns:
            Data path

        Supports two formats:
        1. Old format: type: "synthetic_mezo_gpt4o_v1" (deprecated)
        2. NEW format: path: "Data_v2/synthetic/{purpose}/{exp_id}/{Dataset}"
        """
        # NEW: Prioritize new format - directly specified path
        if 'path' in data_config:
            data_path = data_config['path']
            # Support relative and absolute paths
            if not Path(data_path).is_absolute():
                data_path = str(self.project_root / data_path)
            return data_path

        # Old format: infer path from type (maintain compatibility)
        data_type = data_config['type']
        task = self.config['task']

        if data_type == 'original':
            return str(self.project_root / "Data_v2" / "original" / task)
        elif data_type.startswith('synthetic'):
            # synthetic format: synthetic_{method}_{model}_{version}
            # Example: synthetic_mezo_gpt4o_v1
            parts = data_type.split('_')
            if len(parts) < 4:
                raise ValueError(f"synthetic data type format error: {data_type}, should be synthetic_{{method}}_{{model}}_{{version}}")

            method, model, version = parts[1], parts[2], parts[3]
            dir_name = f"{task}_{method}_{model}_{version}"
            return str(self.project_root / "Data_v2" / "synthetic" / dir_name)
        elif data_type.startswith('mixed'):
            # mixed format: mixed_{method}_{model}_{version}
            parts = data_type.split('_')
            if len(parts) < 4:
                raise ValueError(f"mixed data type format error: {data_type}")

            method, model, version = parts[1], parts[2], parts[3]
            dir_name = f"{task}_{method}_{model}_{version}_mixed"
            return str(self.project_root / "Data_v2" / "mixed" / dir_name)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def get_result_dir_name(self, lr: str, data_type: str) -> str:
        """
        Generate result directory name

        Format: {task}_{method}_{data_type}_{lr}

        Args:
            lr: Learning rate
            data_type: Data type

        Returns:
            Directory name
        """
        task = self.config['task']
        method = self.config['method']

        # Format learning rate (remove 'e' from scientific notation, replace with underscore)
        lr_str = str(lr).replace('e-', '_').replace('.', '')

        return f"{task}_{method}_{data_type}_{lr_str}"

    def prepare_result_directory(self, lr: str, data_type: str) -> Path:
        """
        Prepare result output directory

        Structure: Results_v2/{experiment_purpose}/{Model}/{Task}_{Method}_{DataType}_{LR}/{Timestamp}/

        Args:
            lr: Learning rate
            data_type: Data type

        Returns:
            Result directory path
        """
        model = self.config['model']
        dir_name = self.get_result_dir_name(lr, data_type)

        # NEW: Add experiment purpose level
        result_dir = self.results_base / self.experiment_purpose / model / dir_name / self.timestamp
        result_dir.mkdir(parents=True, exist_ok=True)

        return result_dir

    def build_training_command(
        self,
        lr: float,
        bs: int,
        steps: int,
        seed: int,
        data_path: str,
        result_dir: Path,
        cuda_devices: str,
        data_type: str = None  # NEW: Add data_type parameter
    ) -> Dict[str, str]:
        """
        Build training command

        Args:
            data_type: Data type, if None will fetch from configuration

        Returns:
            Dictionary containing environment variables and command
        """
        method = self.config['method']
        model = self.config['model']

        # NEW: Support both old and new formats
        if data_type is None:
            if 'type' in self.config['data']:
                data_type = self.config['data']['type']
            else:
                data_type = self.infer_data_type_from_path(data_path)

        # Get corresponding training script
        script_name = self.get_method_script_name(method, data_type)
        script_path = self.training_scripts_dir / script_name

        if not script_path.exists():
            raise FileNotFoundError(f"Training script does not exist: {script_path}")

        # Build environment variables
        env_vars = {
            "CUDA_VISIBLE_DEVICES": cuda_devices,
            "MODEL": model,
            "MODE": "ft",
            "TASK": data_path,
            "LR": str(lr),
            "BS": str(bs),
            "STEPS": str(steps),
            "SEED": str(seed)
        }

        # Add method-specific parameters
        if method == 'zo':
            env_vars["EPS"] = str(self.config['hyperparameters'].get('zo_eps', 1e-3))

        if method == 'fo_lora':
            # Training script expects RANK variable (not LORA_RANK)
            # Reference: running_scripts/Llama-3.2-1B/1_2_fo_lora_orig_copa_rk8n16.sh
            env_vars["RANK"] = str(self.config['hyperparameters'].get('lora_rank', 8))
            # Note: lora_alpha is not currently used by training script (no LORA_ALPHA variable in script)

        # Output files
        out_file = result_dir / f"{lr}_train.out"
        err_file = result_dir / f"{lr}_train.err"

        # Build complete command
        env_str = " ".join([f"{k}={v}" for k, v in env_vars.items()])
        command = f"{env_str} bash {script_path} 1>>{out_file} 2>>{err_file}"

        return {
            "env_vars": env_vars,
            "command": command,
            "out_file": str(out_file),
            "err_file": str(err_file),
            "script": str(script_path)
        }

    def save_experiment_config(self, result_dir: Path, training_info: Dict):
        """Save experiment configuration to result directory"""
        config_path = result_dir / "experiment_config.yaml"

        experiment_config = {
            "timestamp": self.timestamp,
            "experiment_purpose": self.experiment_purpose,  # NEW: Save experiment purpose
            "model": self.config['model'],
            "task": self.config['task'],
            "method": self.config['method'],
            "data": self.config['data'],
            "hyperparameters": self.config['hyperparameters'],
            "training_info": training_info
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(experiment_config, f, allow_unicode=True, default_flow_style=False)

        print(f"  ✓ Configuration file: {config_path}")

    def run_training(
        self,
        lr: float,
        bs: int,
        steps: int,
        seed: int,
        cuda_devices: str,
        dry_run: bool = False
    ):
        """
        Execute single training task

        Args:
            lr: Learning rate
            bs: Batch size
            steps: Training steps
            seed: Random seed
            cuda_devices: CUDA devices
            dry_run: Whether to only display command without executing
        """
        # Prepare data path
        data_path = self.get_data_path(self.config['data'])

        # NEW: Get data type (support both old and new formats)
        if 'type' in self.config['data']:
            data_type = self.config['data']['type']  # Old format
        else:
            data_type = self.infer_data_type_from_path(data_path)  # New format: infer from path

        print(f"\n{'='*80}")
        print(f"Training configuration: LR={lr}, BS={bs}, Steps={steps}, Seed={seed}")
        print(f"CUDA devices: {cuda_devices}")
        print(f"{'='*80}")

        print(f"Data path: {data_path}")
        print(f"Data type: {data_type}")

        # Prepare result directory
        result_dir = self.prepare_result_directory(lr, data_type)
        print(f"Result directory: {result_dir}")

        # Build training command
        training_info = self.build_training_command(
            lr, bs, steps, seed, data_path, result_dir, cuda_devices, data_type  # NEW: Pass data_type
        )

        # Save experiment configuration
        if not dry_run:
            self.save_experiment_config(result_dir, training_info)

        # Display command
        print(f"\nTraining script: {training_info['script']}")
        print(f"\nEnvironment variables:")
        for k, v in training_info['env_vars'].items():
            print(f"  {k}={v}")

        print(f"\nOutput files:")
        print(f"  stdout: {training_info['out_file']}")
        print(f"  stderr: {training_info['err_file']}")

        if dry_run:
            print(f"\n[DRY RUN] Command to execute:")
            print(f"  {training_info['command']}")
            return

        # Execute training
        print(f"\nStarting training...")
        try:
            # Use subprocess to execute command
            process = subprocess.Popen(
                training_info['command'],
                shell=True,
                cwd=str(self.training_scripts_dir),
                env={**os.environ, **training_info['env_vars']}
            )

            print(f"Training process PID: {process.pid}")
            print(f"Monitor logs: tail -f {training_info['out_file']}")

            if self.config.get('wait_for_completion', False):
                print("Waiting for training to complete...")
                process.wait()
                if process.returncode == 0:
                    print("✓ Training completed")
                else:
                    print(f"✗ Training failed, return code: {process.returncode}")
            else:
                print("Training started in background")

        except Exception as e:
            print(f"✗ Failed to start training: {e}")
            raise

    def run_grid_search(self, dry_run: bool = False):
        """
        Execute hyperparameter grid search

        Args:
            dry_run: Whether to only display commands without executing
        """
        hp = self.config['hyperparameters']

        # Get parameter lists
        lrs = hp['learning_rate'] if isinstance(hp['learning_rate'], list) else [hp['learning_rate']]
        batch_sizes = hp['batch_size'] if isinstance(hp['batch_size'], list) else [hp['batch_size']]
        steps_list = hp['steps'] if isinstance(hp['steps'], list) else [hp['steps']]
        seeds = hp['seed'] if isinstance(hp['seed'], list) else [hp['seed']]

        # Get CUDA device configuration
        cuda_devices = self.config.get('cuda_devices', '0')

        print("\n" + "="*80)
        print("Training Pipeline - Grid Search")
        print("="*80)
        print(f"Experiment purpose: {self.experiment_purpose}")  # NEW: Display experiment purpose
        print(f"Model: {self.config['model']}")
        print(f"Task: {self.config['task']}")
        print(f"Method: {self.config['method']}")

        # NEW: Improved data info display
        if 'path' in self.config['data']:
            print(f"Data path: {self.config['data']['path']}")
        else:
            print(f"Data type: {self.config['data']['type']}")

        print("\nHyperparameter combinations:")
        print(f"  Learning rate: {lrs}")
        print(f"  Batch Size: {batch_sizes}")
        print(f"  Steps: {steps_list}")
        print(f"  Seed: {seeds}")
        print(f"\nTotal experiments: {len(list(itertools.product(lrs, batch_sizes, steps_list, seeds)))}")
        print(f"\nResults saved to: Results_v2/{self.experiment_purpose}/")
        print("="*80)

        # Execute all combinations
        for lr, bs, steps, seed in itertools.product(lrs, batch_sizes, steps_list, seeds):
            try:
                self.run_training(lr, bs, steps, seed, cuda_devices, dry_run)
            except Exception as e:
                print(f"\n✗ Experiment failed: LR={lr}, BS={bs}, Steps={steps}, Seed={seed}")
                print(f"  Error: {e}")

                if not self.config.get('continue_on_error', True):
                    raise

        print("\n" + "="*80)
        print("All training tasks submitted!")
        print("="*80)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Training Pipeline Automation Tool")
    parser.add_argument("config", help="Configuration file path (YAML)")
    parser.add_argument("--dry-run", action="store_true", help="Only display commands without executing")
    args = parser.parse_args()

    pipeline = TrainingPipeline(args.config)
    pipeline.run_grid_search(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
