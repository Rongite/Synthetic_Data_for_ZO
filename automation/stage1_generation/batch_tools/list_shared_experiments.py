#!/usr/bin/env python3
"""
List all physical experiment data in _shared/

Usage:
    python list_shared_experiments.py
    python list_shared_experiments.py --dataset Copa
    python list_shared_experiments.py --verbose
"""

import argparse
import json
import sys
from pathlib import Path

# addParentdirectory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_manager_batch import BatchExperimentManager

# Importsystemthisconfiguration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(
        description='List all physical experiment data in _shared/',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all physical experiments
  python list_shared_experiments.py

  # List only Copa dataset experiments
  python list_shared_experiments.py --dataset Copa

  # Show detailed information (including metadata, parameter fingerprint, etc.)
  python list_shared_experiments.py --verbose
        """
    )
    parser.add_argument(
        '--dataset',
        help='Show only experiments for specified dataset'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information (including metadata, parameter config, etc.)'
    )
    args = parser.parse_args()

    base_output_dir = Path(PROJECT_ROOT) / "Data_v2" / "synthetic"
    manager = BatchExperimentManager(base_output_dir)

    experiments = manager.list_shared_experiments(args.dataset)

    if not experiments:
        print(f"\nNo experiments found in _shared/")
        if args.dataset:
            print(f"(Dataset: {args.dataset})")
        return 0

    print("\n" + "=" * 80)
    print("Physical Experiment Data in _shared/")
    if args.dataset:
        print(f"Dataset: {args.dataset}")
    print(f"Number of experiments: {len(experiments)}")
    print("=" * 80)

    # Group by dataset
    by_dataset = {}
    for exp in experiments:
        dataset = exp['dataset']
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append(exp)

    for dataset, exps in by_dataset.items():
        print(f"\n📊 {dataset} ({len(exps)} experiments)")
        print("-" * 80)

        for exp in exps:
            print(f"\n  📦 {exp['exp_name']}")
            print(f"     Physical path: {exp['physical_path']}")
            print(f"     Parameter fingerprint: {exp.get('fingerprint', 'N/A')[:12]}...")
            print(f"     Created at: {exp.get('created_at', 'N/A')}")
            print(f"     Original Batch: {exp.get('batch_id', 'N/A')}")

            if args.verbose and 'metadata' in exp:
                metadata = exp['metadata']

                # Show generation config
                if 'generation' in metadata:
                    gen = metadata['generation']
                    print(f"\n     Generation config:")
                    print(f"       Model: {gen.get('model', 'N/A')}")
                    print(f"       Temperature: {gen.get('temperature', 'N/A')}")
                    print(f"       Top_p: {gen.get('top_p', 'N/A')}")
                    print(f"       Max tokens: {gen.get('max_tokens', 'N/A')}")

                # Show validation config
                if 'validation' in metadata:
                    val = metadata['validation']
                    print(f"\n     Validation config:")
                    print(f"       Model: {val.get('model', 'N/A')}")
                    print(f"       Temperature: {val.get('temperature', 'N/A')}")

                # Show experiment info
                if 'experiment_purpose' in metadata:
                    print(f"\n     Experiment purpose: {metadata.get('experiment_purpose', 'N/A')}")
                if 'experiment_description' in metadata:
                    print(f"     Experiment description: {metadata.get('experiment_description', 'N/A')}")

    print()
    return 0


if __name__ == '__main__':
    exit(main())
