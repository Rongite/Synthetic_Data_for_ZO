#!/usr/bin/env python3
"""
Experiment Management Module - Batch Solution

Features:
1. 3-tier directory structure: _shared/ + batch_*/ + dataset/
2. Parameter fingerprint deduplication (across batches)
3. Symlink management (physical/logical separation)
4. Semantic directory names
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


def compute_parameter_fingerprint(config: Dict) -> str:
    """
    Compute parameter fingerprint

    Only includes key parameters that affect data generation

    Returns:
        MD5 hash string (first 12 characters)
    """
    params = {
        'gen_model': config['generation']['model'],
        'gen_temperature': config['generation']['temperature'],
        'gen_top_p': config['generation'].get('top_p', 1.0),
        'gen_max_tokens': config['generation'].get('max_tokens', 256),
        'gen_frequency_penalty': config['generation'].get('frequency_penalty', 0.0),
        'gen_presence_penalty': config['generation'].get('presence_penalty', 0.0),
    }

    # Note: validation config is optional (may not exist in direct_all mode)
    if 'validation' in config:
        params['val_model'] = config['validation']['model']
        params['val_temperature'] = config['validation'].get('temperature', 0.0)

    # Hash of prompt
    gen_prompt = config['generation'].get('rephrase_prompt', '')
    params['gen_prompt_hash'] = hashlib.md5(gen_prompt.encode()).hexdigest()[:8]

    # Note: validation prompt is optional
    if 'validation' in config:
        val_prompt = config['validation'].get('validation_prompt', '')
        params['val_prompt_hash'] = hashlib.md5(val_prompt.encode()).hexdigest()[:8]

    # Compute overall hash
    params_str = json.dumps(params, sort_keys=True)
    fingerprint = hashlib.md5(params_str.encode()).hexdigest()[:12]

    return fingerprint


def generate_semantic_dirname(config: Dict) -> str:
    """
    Generate semantic directory name

    Format: temp{temperature}_topp{top_p}_{model}
    Example: temp07_topp09_gpt4o

    Args:
        config: Configuration dictionary

    Returns:
        Semantic directory name
    """
    gen_cfg = config['generation']

    # Temperature (remove decimal point)
    temp = gen_cfg['temperature']
    temp_str = f"temp{str(temp).replace('.', '')}"

    # Top_p (remove decimal point, only show non-default values)
    top_p = gen_cfg.get('top_p', 1.0)
    if top_p != 1.0:
        topp_str = f"_topp{str(top_p).replace('.', '')}"
    else:
        topp_str = "_topp10"

    # Model (simplified name)
    model = gen_cfg['model'].lower()
    if 'gpt-4o' in model:
        model_str = 'gpt4o'
    elif 'gpt-4' in model:
        model_str = 'gpt4'
    elif 'gpt-3.5' in model:
        model_str = 'gpt35'
    else:
        # Remove special characters
        model_str = model.replace('/', '_').replace('-', '')[:10]

    dirname = f"{temp_str}{topp_str}_{model_str}"

    return dirname


def find_existing_by_fingerprint(
    shared_dir: Path,
    dataset_name: str,
    fingerprint: str
) -> Optional[Path]:
    """
    Find existing experiments with same fingerprint in _shared/

    Args:
        shared_dir: _shared root directory
        dataset_name: Dataset name (e.g. Copa)
        fingerprint: Parameter fingerprint

    Returns:
        Matching directory path, or None if not exists
    """
    dataset_dir = shared_dir / dataset_name

    if not dataset_dir.exists():
        return None

    # Iterate through all experiment directories under this dataset
    for exp_dir in dataset_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        fingerprint_file = exp_dir / ".fingerprint"
        if not fingerprint_file.exists():
            continue

        try:
            with open(fingerprint_file, 'r') as f:
                existing_fingerprint = f.read().strip()

            if existing_fingerprint == fingerprint:
                return exp_dir
        except Exception:
            continue

    return None


def save_fingerprint(exp_dir: Path, fingerprint: str):
    """Save parameter fingerprint to .fingerprint file"""
    fingerprint_file = exp_dir / ".fingerprint"
    with open(fingerprint_file, 'w') as f:
        f.write(fingerprint)


def save_experiment_metadata(exp_dir: Path, config: Dict, fingerprint: str, is_shared: bool = True):
    """
    Save experiment metadata

    Args:
        exp_dir: Experiment directory
        config: Configuration dictionary
        fingerprint: Parameter fingerprint
        is_shared: Whether it's a shared directory (for recording physical/logical location)
    """
    experiment_cfg = config.get('experiment', {})

    metadata = {
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'is_shared': is_shared,
        'batch_id': experiment_cfg.get('batch_id', ''),
        'experiment_purpose': experiment_cfg.get('purpose', 'general'),
        'experiment_id': experiment_cfg.get('experiment_id', ''),
        'experiment_description': experiment_cfg.get('description', ''),
        'task_name': config.get('task_name'),
        'dataset_name': config.get('dataset', {}).get('dataset_name', ''),
        'training_method': config.get('training_method'),
        'parameter_fingerprint': fingerprint,
        'generation': {
            'model': config['generation']['model'],
            'temperature': config['generation']['temperature'],
            'top_p': config['generation'].get('top_p', 1.0),
            'max_tokens': config['generation'].get('max_tokens', 256),
            'strategy': config['generation'].get('strategy', 'two_stage'),
        }
    }

    # Note: validation config is optional (may not exist in direct_all mode)
    if 'validation' in config:
        metadata['validation'] = {
            'model': config['validation']['model'],
            'temperature': config['validation'].get('temperature', 0.0),
        }

    metadata_path = exp_dir / "experiment_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def create_symlink(target: Path, link: Path):
    """
    Create symbolic link

    Args:
        target: Target path (actual data location)
        link: Link path
    """
    # Delete existing link
    if link.exists() or link.is_symlink():
        link.unlink()

    # Create parent directory
    link.parent.mkdir(parents=True, exist_ok=True)

    # Create symbolic link using relative path
    try:
        rel_target = os.path.relpath(target, link.parent)
        link.symlink_to(rel_target)
    except Exception as e:
        # If relative path fails, use absolute path
        link.symlink_to(target.absolute())


class BatchExperimentManager:
    """Batch Experiment Manager"""

    def __init__(self, base_output_dir: Path):
        """
        Initialize experiment manager

        Args:
            base_output_dir: Base output directory (e.g. Data_v2/synthetic/)
        """
        self.base_output_dir = base_output_dir
        self.shared_dir = base_output_dir / "_shared"

    def prepare_experiment_dir(
        self,
        config: Dict,
        auto_resolve: bool = False
    ) -> Tuple[Path, str]:
        """
        Prepare experiment directory (Batch solution)

        Workflow:
        1. Compute parameter fingerprint
        2. Search for existing fingerprint in _shared/{Dataset}/
        3. If exists → Reuse, only create batch symlink
        4. If not exists → Create new physical storage + batch symlink

        Args:
            config: Configuration dictionary
            auto_resolve: Automatically resolve conflicts

        Returns:
            (physical_dir, fingerprint)
            physical_dir: Physical directory path in _shared/
        """
        # Extract configuration info
        experiment_cfg = config.get('experiment', {})
        batch_id = experiment_cfg.get('batch_id', '')
        dataset_name = config.get('dataset', {}).get('dataset_name', 'Dataset')

        # Auto-generate batch_id if not specified
        if not batch_id:
            today = datetime.now().strftime("%Y%m%d")
            purpose = experiment_cfg.get('purpose', 'experiment')
            batch_id = f"batch_{today}_{purpose}"
            print(f"ℹ️  batch_id not specified, auto-generated: {batch_id}")

        # Compute parameter fingerprint
        fingerprint = compute_parameter_fingerprint(config)

        # Generate semantic directory name
        semantic_dirname = generate_semantic_dirname(config)

        print("\n" + "=" * 80)
        print("🔧 Batch Experiment Management")
        print("=" * 80)
        print(f"Batch ID: {batch_id}")
        print(f"Dataset: {dataset_name}")
        print(f"Parameter fingerprint: {fingerprint}")
        print(f"Semantic name: {semantic_dirname}")
        print("=" * 80)

        # Search for existing experiments in _shared/
        print(f"\n🔍 Searching for fingerprint {fingerprint} in _shared/{dataset_name}/...")

        existing_dir = find_existing_by_fingerprint(
            self.shared_dir,
            dataset_name,
            fingerprint
        )

        if existing_dir:
            # Found existing experiment with same parameters, reuse!
            print(f"✅ Found existing experiment with same parameters!")
            print(f"   Location: {existing_dir.relative_to(self.base_output_dir)}")

            # Read existing experiment metadata
            metadata_file = existing_dir / "experiment_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
                print(f"   Created at: {existing_metadata.get('created_at', 'N/A')}")
                print(f"   Original batch: {existing_metadata.get('batch_id', 'N/A')}")

            print(f"\n📂 Reusing existing data")
            print(f"   Physical storage: {existing_dir.relative_to(self.base_output_dir)} (exists, reusing)")

            physical_dir = existing_dir
            is_new_experiment = False

        else:
            # Not found, create new experiment
            print("✓ No matching experiment found, will create new experiment")

            physical_dir = self.shared_dir / dataset_name / semantic_dirname

            print(f"\n📂 Creating new experiment")
            print(f"   Physical storage: {physical_dir.relative_to(self.base_output_dir)}")

            # Create physical directory
            physical_dir.mkdir(parents=True, exist_ok=True)

            # Save fingerprint
            save_fingerprint(physical_dir, fingerprint)

            # Save metadata
            save_experiment_metadata(physical_dir, config, fingerprint, is_shared=True)

            is_new_experiment = True

        # Create batch view (symbolic link)
        batch_dir = self.base_output_dir / batch_id / dataset_name / semantic_dirname

        print(f"   Batch view: {batch_dir.relative_to(self.base_output_dir)}")

        # Create symbolic link
        create_symlink(physical_dir, batch_dir)

        if is_new_experiment:
            print(f"\n✅ New experiment created successfully")
        else:
            print(f"\n✅ Existing data reused successfully")
            print(f"   💾 Saving resources: No need to regenerate data")

        print(f"\nPhysical storage: {physical_dir}")
        print(f"Batch view: {batch_dir}")

        return physical_dir, fingerprint

    def list_batches(self) -> List[str]:
        """List all batches"""
        batches = []

        for item in self.base_output_dir.iterdir():
            if item.is_dir() and item.name.startswith('batch_'):
                batches.append(item.name)

        return sorted(batches)

    def list_experiments_in_batch(self, batch_id: str, dataset_name: Optional[str] = None) -> List[Dict]:
        """
        List experiments in specified batch

        Args:
            batch_id: batch ID
            dataset_name: Optional, only list experiments of this dataset

        Returns:
            List of experiment information
        """
        batch_dir = self.base_output_dir / batch_id

        if not batch_dir.exists():
            return []

        experiments = []

        if dataset_name:
            dataset_dirs = [batch_dir / dataset_name]
        else:
            dataset_dirs = [d for d in batch_dir.iterdir() if d.is_dir()]

        for dataset_dir in dataset_dirs:
            if not dataset_dir.exists():
                continue

            for exp_link in dataset_dir.iterdir():
                if not exp_link.is_symlink():
                    continue

                # Resolve symbolic link
                target = exp_link.resolve()

                info = {
                    'batch_id': batch_id,
                    'dataset': dataset_dir.name,
                    'exp_name': exp_link.name,
                    'link_path': str(exp_link.relative_to(self.base_output_dir)),
                    'physical_path': str(target.relative_to(self.base_output_dir)),
                    'is_reused': False  # Can determine later by checking metadata
                }

                # Read metadata (if exists)
                metadata_file = target / "experiment_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        info['metadata'] = metadata
                        info['created_at'] = metadata.get('created_at', 'N/A')
                        info['fingerprint'] = metadata.get('parameter_fingerprint', 'N/A')

                        # Determine if reused (original batch is not current batch)
                        original_batch = metadata.get('batch_id', '')
                        if original_batch and original_batch != batch_id:
                            info['is_reused'] = True
                            info['original_batch'] = original_batch
                    except Exception:
                        pass

                experiments.append(info)

        return experiments

    def list_shared_experiments(self, dataset_name: Optional[str] = None) -> List[Dict]:
        """
        List all physical experiments in _shared/

        Args:
            dataset_name: Optional, only list experiments of this dataset

        Returns:
            List of experiment information
        """
        if not self.shared_dir.exists():
            return []

        experiments = []

        if dataset_name:
            dataset_dirs = [self.shared_dir / dataset_name]
        else:
            dataset_dirs = [d for d in self.shared_dir.iterdir() if d.is_dir()]

        for dataset_dir in dataset_dirs:
            if not dataset_dir.exists():
                continue

            for exp_dir in dataset_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                info = {
                    'dataset': dataset_dir.name,
                    'exp_name': exp_dir.name,
                    'physical_path': str(exp_dir.relative_to(self.base_output_dir)),
                }

                # Read fingerprint
                fingerprint_file = exp_dir / ".fingerprint"
                if fingerprint_file.exists():
                    with open(fingerprint_file, 'r') as f:
                        info['fingerprint'] = f.read().strip()

                # Read metadata
                metadata_file = exp_dir / "experiment_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        info['metadata'] = metadata
                        info['created_at'] = metadata.get('created_at', 'N/A')
                        info['batch_id'] = metadata.get('batch_id', 'N/A')
                    except Exception:
                        pass

                experiments.append(info)

        return experiments
