#!/usr/bin/env python3
"""
List all experiments in specified batch

Usage:
    python list_batch_experiments.py batch_20241229_temperature
    python list_batch_experiments.py batch_20241229_temperature --dataset Copa
    python list_batch_experiments.py batch_20241229_temperature --verbose
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
        description='List all experiments in specified batch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments in batch
  python list_batch_experiments.py batch_20241229_temperature

  # List only Copa dataset experiments
  python list_batch_experiments.py batch_20241229_temperature --dataset Copa

  # Show detailed information (including metadata)
  python list_batch_experiments.py batch_20241229_temperature --verbose
        """
    )
    parser.add_argument(
        'batch_id',
        help='Batch ID (e.g., batch_20241229_temperature)'
    )
    parser.add_argument(
        '--dataset',
        help='Show only experiments for specified dataset'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information (including metadata, physical path, etc.)'
    )
    args = parser.parse_args()

    base_output_dir = Path(PROJECT_ROOT) / "Data_v2" / "synthetic"
    manager = BatchExperimentManager(base_output_dir)

    # Check if batch exists
    batch_dir = base_output_dir / args.batch_id
    if not batch_dir.exists():
        print(f"❌ Batch does not exist: {args.batch_id}")
        print(f"\nAvailable batches:")
        for batch in manager.list_batches():
            print(f"  - {batch}")
        return 1

    experiments = manager.list_experiments_in_batch(args.batch_id, args.dataset)

    if not experiments:
        print(f"\nNo experiments found in {args.batch_id}")
        if args.dataset:
            print(f"(Dataset: {args.dataset})")
        return 0

    print("\n" + "=" * 80)
    print(f"Batch: {args.batch_id}")
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
            print(f"\n  🔧 {exp['exp_name']}")
            print(f"     Batch view: {exp['link_path']}")

            if args.verbose:
                print(f"     Physical storage: {exp['physical_path']}")
                print(f"     Created at: {exp.get('created_at', 'N/A')}")
                print(f"     Parameter fingerprint: {exp.get('fingerprint', 'N/A')}")

                if exp.get('is_reused'):
                    print(f"     ⚡ Data reuse: Yes (Original batch: {exp.get('original_batch', 'N/A')})")
                else:
                    print(f"     ⚡ Data reuse: No (Newly generated)")

                # Show key information from metadata
                if 'metadata' in exp:
                    metadata = exp['metadata']
                    if 'generation' in metadata:
                        gen = metadata['generation']
                        print(f"     Generation model: {gen.get('model', 'N/A')}")
                        print(f"     Temperature: {gen.get('temperature', 'N/A')}")
                        print(f"     Top_p: {gen.get('top_p', 'N/A')}")

    print()
    return 0


if __name__ == '__main__':
    exit(main())
