#!/usr/bin/env python3
"""
Experiment Management Module

Features:
1. Generate parameter fingerprint (to determine if experiments have same parameters)
2. Detect directory conflicts (same experiment purpose + same parameters)
3. Manage experiment metadata
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


def compute_parameter_fingerprint(config: Dict) -> str:
    """
    Compute parameter fingerprint

    Only includes key parameters that affect data generation:
    - generation.model
    - generation.temperature
    - generation.top_p (if exists)
    - generation.max_tokens (if exists)
    - generation.rephrase_prompt (hash of prompt)
    - validation.model
    - validation.validation_prompt (hash of prompt)

    Args:
        config: Configuration dictionary

    Returns:
        MD5 hash string (first 12 characters)
    """
    # Extract key parameters
    params = {
        'gen_model': config['generation']['model'],
        'gen_temperature': config['generation']['temperature'],
        'gen_top_p': config['generation'].get('top_p', 1.0),
        'gen_max_tokens': config['generation'].get('max_tokens', 256),
        'gen_frequency_penalty': config['generation'].get('frequency_penalty', 0.0),
        'gen_presence_penalty': config['generation'].get('presence_penalty', 0.0),
        'val_model': config['validation']['model'],
        'val_temperature': config['validation'].get('temperature', 0.0),
    }

    # Hash of prompts (avoid storing full text)
    gen_prompt = config['generation'].get('rephrase_prompt', '')
    val_prompt = config['validation'].get('validation_prompt', '')

    params['gen_prompt_hash'] = hashlib.md5(gen_prompt.encode()).hexdigest()[:8]
    params['val_prompt_hash'] = hashlib.md5(val_prompt.encode()).hexdigest()[:8]

    # Compute overall hash
    params_str = json.dumps(params, sort_keys=True)
    fingerprint = hashlib.md5(params_str.encode()).hexdigest()[:12]

    return fingerprint


def get_experiment_metadata_path(output_dir: Path) -> Path:
    """Get experiment metadata file path"""
    return output_dir / "experiment_metadata.json"


def save_experiment_metadata(output_dir: Path, config: Dict, fingerprint: str):
    """
    Save experiment metadata

    Args:
        output_dir: Output directory
        config: Configuration dictionary
        fingerprint: Parameter fingerprint
    """
    metadata = {
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'experiment_purpose': config.get('experiment', {}).get('purpose', 'general'),
        'experiment_id': config.get('experiment', {}).get('experiment_id', ''),
        'experiment_description': config.get('experiment', {}).get('description', ''),
        'task_name': config.get('task_name'),
        'training_method': config.get('training_method'),
        'version': config.get('version'),
        'parameter_fingerprint': fingerprint,
        'generation': {
            'model': config['generation']['model'],
            'temperature': config['generation']['temperature'],
            'top_p': config['generation'].get('top_p'),
            'max_tokens': config['generation'].get('max_tokens'),
        },
        'validation': {
            'model': config['validation']['model'],
            'temperature': config['validation'].get('temperature', 0.0),
        }
    }

    metadata_path = get_experiment_metadata_path(output_dir)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata_path


def find_existing_experiments(
    base_dir: Path,
    experiment_purpose: str,
    fingerprint: str
) -> List[Path]:
    """
    Find existing experiments with same purpose and parameter fingerprint

    Args:
        base_dir: Base directory (e.g. Data_v2/synthetic/)
        experiment_purpose: Experiment purpose
        fingerprint: Parameter fingerprint

    Returns:
        List of matching experiment directories
    """
    purpose_dir = base_dir / experiment_purpose
    if not purpose_dir.exists():
        return []

    matching_experiments = []

    for exp_dir in purpose_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        metadata_path = get_experiment_metadata_path(exp_dir)
        if not metadata_path.exists():
            continue

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            if metadata.get('parameter_fingerprint') == fingerprint:
                matching_experiments.append(exp_dir)
        except Exception:
            continue

    return matching_experiments


def check_overwrite_conflict(
    output_dir: Path,
    config: Dict,
    fingerprint: str
) -> Tuple[bool, Optional[Path], Optional[Dict]]:
    """
    Check for overwrite conflicts

    Args:
        output_dir: Planned output directory
        config: Configuration dictionary
        fingerprint: Parameter fingerprint

    Returns:
        (has_conflict, existing_dir, existing_metadata)
        - has_conflict: Whether a conflict exists
        - existing_dir: Existing directory (if any)
        - existing_metadata: Metadata of existing experiment (if any)
    """
    # Check if planned output directory already exists
    if not output_dir.exists():
        return False, None, None

    metadata_path = get_experiment_metadata_path(output_dir)
    if not metadata_path.exists():
        # Directory exists but no metadata, might be old or incomplete experiment
        return True, output_dir, None

    # Read existing experiment metadata
    try:
        with open(metadata_path, 'r') as f:
            existing_metadata = json.load(f)

        existing_fingerprint = existing_metadata.get('parameter_fingerprint', '')

        # If fingerprints match → this is the same tuning experiment
        if existing_fingerprint == fingerprint:
            return True, output_dir, existing_metadata

        # Different fingerprint → this is a different tuning experiment, should not overwrite
        return False, None, None

    except Exception:
        return True, output_dir, None


def suggest_alternative_dir(
    base_output_dir: Path,
    experiment_purpose: str,
    experiment_id: str
) -> Path:
    """
    Suggest a non-conflicting directory name (add version suffix)

    Args:
        base_output_dir: Base output directory (e.g. Data_v2/synthetic/)
        experiment_purpose: Experiment purpose
        experiment_id: Experiment ID

    Returns:
        New directory path
    """
    purpose_dir = base_output_dir / experiment_purpose

    # Try adding version number suffix
    version = 2
    while True:
        new_dir_name = f"{experiment_id}_v{version}"
        new_dir = purpose_dir / new_dir_name

        if not new_dir.exists():
            return new_dir

        version += 1

        # Prevent infinite loop
        if version > 100:
            # Use timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_dir_name = f"{experiment_id}_{timestamp}"
            return purpose_dir / new_dir_name


def format_metadata_diff(metadata: Dict) -> str:
    """Format metadata as readable text"""
    lines = []
    lines.append(f"  Experiment purpose: {metadata.get('experiment_purpose', 'N/A')}")
    lines.append(f"  Experiment ID: {metadata.get('experiment_id', 'N/A')}")
    lines.append(f"  Created at: {metadata.get('created_at', 'N/A')}")
    lines.append(f"  Generation model: {metadata.get('generation', {}).get('model', 'N/A')}")
    lines.append(f"  Temperature: {metadata.get('generation', {}).get('temperature', 'N/A')}")
    lines.append(f"  Parameter fingerprint: {metadata.get('parameter_fingerprint', 'N/A')}")

    return "\n".join(lines)


class ExperimentManager:
    """Experiment Manager"""

    def __init__(self, base_output_dir: Path):
        """
        Initialize experiment manager

        Args:
            base_output_dir: Base output directory (e.g. Data_v2/synthetic/)
        """
        self.base_output_dir = base_output_dir

    def prepare_experiment_dir(
        self,
        config: Dict,
        auto_resolve: bool = False
    ) -> Tuple[Path, str]:
        """
        Prepare experiment directory

        Handle conflict detection and overwrite strategy

        Args:
            config: Configuration dictionary
            auto_resolve: Automatically resolve conflicts (don't prompt user)

        Returns:
            (output_dir, fingerprint)

        Raises:
            ValueError: User rejected overwrite or configuration error
        """
        # Extract experiment information
        experiment_cfg = config.get('experiment', {})
        experiment_purpose = experiment_cfg.get('purpose', 'general')
        experiment_id = experiment_cfg.get('experiment_id', '')

        if not experiment_id:
            # Auto-generate experiment_id
            task = config.get('task_name', 'task').lower()
            method = config.get('training_method', 'method').lower()
            model = config['generation']['model'].replace('/', '_').replace('-', '').lower()
            experiment_id = f"{task}_{method}_{model}"

        overwrite_strategy = experiment_cfg.get('overwrite_strategy', 'prompt')

        # Compute parameter fingerprint
        fingerprint = compute_parameter_fingerprint(config)

        # Construct output directory
        purpose_dir = self.base_output_dir / experiment_purpose
        output_dir = purpose_dir / experiment_id

        # Check for conflicts
        has_conflict, existing_dir, existing_metadata = check_overwrite_conflict(
            output_dir, config, fingerprint
        )

        if not has_conflict:
            # No conflict, use directly
            return output_dir, fingerprint

        # Conflict exists, handle based on strategy
        if overwrite_strategy == 'auto':
            # Auto overwrite
            print(f"⚠️  Detected existing experiment: {output_dir.relative_to(self.base_output_dir)}")
            print(f"   Overwrite strategy: auto, will overwrite existing data")
            return output_dir, fingerprint

        elif overwrite_strategy == 'never':
            # Auto create new version
            new_dir = suggest_alternative_dir(
                self.base_output_dir,
                experiment_purpose,
                experiment_id
            )
            print(f"⚠️  Detected existing experiment: {output_dir.relative_to(self.base_output_dir)}")
            print(f"   Overwrite strategy: never, auto-creating new version")
            print(f"   New directory: {new_dir.relative_to(self.base_output_dir)}")
            return new_dir, fingerprint

        else:  # overwrite_strategy == 'prompt' or default
            # Prompt user
            print("\n" + "=" * 80)
            print("⚠️  Detected existing experiment")
            print("=" * 80)
            print(f"Directory: {output_dir.relative_to(self.base_output_dir)}")

            if existing_metadata:
                print("\nExisting experiment info:")
                print(format_metadata_diff(existing_metadata))

                if existing_metadata.get('parameter_fingerprint') == fingerprint:
                    print("\n✓ Fingerprint match: This is the same tuning experiment")
                else:
                    print("\n✗ Fingerprint mismatch: This is a different tuning experiment (should not overwrite)")

            print("\nPlease choose:")
            print("  [o] Overwrite - Delete existing data and regenerate")
            print("  [n] Create new version - Auto-add version suffix")
            print("  [c] Cancel - Exit program")

            if auto_resolve:
                choice = 'n'
            else:
                choice = input("\nYour choice: ").lower().strip()

            if choice == 'o':
                print(f"\nWill overwrite: {output_dir.relative_to(self.base_output_dir)}")
                return output_dir, fingerprint

            elif choice == 'n':
                new_dir = suggest_alternative_dir(
                    self.base_output_dir,
                    experiment_purpose,
                    experiment_id
                )
                print(f"\nCreating new version: {new_dir.relative_to(self.base_output_dir)}")
                return new_dir, fingerprint

            else:
                raise ValueError("User cancelled operation")

    def list_experiments(self, experiment_purpose: Optional[str] = None) -> List[Dict]:
        """
        List all experiments

        Args:
            experiment_purpose: If specified, only list experiments with this purpose

        Returns:
            List of experiment metadata
        """
        experiments = []

        if experiment_purpose:
            purpose_dirs = [self.base_output_dir / experiment_purpose]
        else:
            purpose_dirs = [d for d in self.base_output_dir.iterdir() if d.is_dir()]

        for purpose_dir in purpose_dirs:
            if not purpose_dir.exists():
                continue

            for exp_dir in purpose_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                metadata_path = get_experiment_metadata_path(exp_dir)
                if not metadata_path.exists():
                    continue

                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    metadata['path'] = str(exp_dir.relative_to(self.base_output_dir))
                    experiments.append(metadata)
                except Exception:
                    continue

        return experiments
