#!/usr/bin/env python3
"""
List all batches

Usage:
    python list_batches.py
    python list_batches.py --verbose
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_manager_batch import BatchExperimentManager

# Import unified config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(
        description='List all batch experiment batches'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information (including number of experiments in each batch)'
    )
    args = parser.parse_args()

    base_output_dir = Path(PROJECT_ROOT) / "Data_v2" / "synthetic"
    manager = BatchExperimentManager(base_output_dir)

    batches = manager.list_batches()

    if not batches:
        print("No batches found")
        return 0

    print("\n" + "=" * 80)
    print(f"Found {len(batches)} batches")
    print("=" * 80)

    for batch_id in batches:
        print(f"\n📦 {batch_id}")

        if args.verbose:
            # Get experiments in this batch
            experiments = manager.list_experiments_in_batch(batch_id)
            if experiments:
                datasets = {}
                for exp in experiments:
                    dataset = exp['dataset']
                    if dataset not in datasets:
                        datasets[dataset] = []
                    datasets[dataset].append(exp['exp_name'])

                print(f"   Number of experiments: {len(experiments)}")
                for dataset, exp_names in datasets.items():
                    print(f"   {dataset}: {len(exp_names)} experiments")
                    for name in exp_names:
                        print(f"      - {name}")
            else:
                print("   (Empty batch)")

    print()
    return 0


if __name__ == '__main__':
    exit(main())
