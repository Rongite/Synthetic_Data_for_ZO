#!/usr/bin/env python3
"""
List all available data paths - convenient for configuring training scripts

Features:
  1. List datasets under all batches
  2. Display batch paths and _shared paths (both can be used)
  3. Support filtering (by batch, by dataset)
  4. Copy paths to clipboard (optional)

Usage:
    # List all data
    python list_data_paths.py

    # View specific batch only
    python list_data_paths.py --batch batch_20241229_temperature

    # View specific dataset only
    python list_data_paths.py --dataset Copa

    # Filter by both
    python list_data_paths.py --batch batch_20241229_temperature --dataset Copa

    # Output in YAML format (can be directly copied to config file)
    python list_data_paths.py --format yaml

    # Output as table
    python list_data_paths.py --format table
"""

import argparse
from pathlib import Path
import sys

# addParentdirectory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_manager_batch import BatchExperimentManager

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def list_all_data_paths(batch_filter=None, dataset_filter=None, output_format='simple'):
    """List all available data paths"""

    synthetic_base = PROJECT_ROOT / "Data_v2" / "synthetic"
    manager = BatchExperimentManager(synthetic_base)

    # Get all batch information
    all_batch_ids = manager.list_batches()

    if not all_batch_ids:
        print("❌ No batch experiments found")
        return 1

    # Filter batch
    if batch_filter:
        all_batch_ids = [b for b in all_batch_ids if b == batch_filter]
        if not all_batch_ids:
            print(f"❌ Batch not found: {batch_filter}")
            return 1

    # Collect all data paths
    data_paths = []

    for batch_id in all_batch_ids:
        # Get all experiments in this batch
        experiments = manager.list_experiments_in_batch(batch_id, dataset_name=dataset_filter)

        for exp_info in experiments:
            dataset_name = exp_info['dataset']
            exp_name = exp_info['exp_name']

            # Filter dataset
            if dataset_filter and dataset_name != dataset_filter:
                continue

            # Batch path (symbolic link)
            batch_path = f"Data_v2/synthetic/{batch_id}/{dataset_name}/{exp_name}/{dataset_name}"

            # Shared path (physical files)
            shared_path = f"Data_v2/synthetic/_shared/{dataset_name}/{exp_name}/{dataset_name}"

            # Get metadata
            metadata_path = synthetic_base / "_shared" / dataset_name / exp_name / "experiment_metadata.json"

            import json
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                description = metadata.get('config', {}).get('experiment', {}).get('description', 'N/A')
            except:
                description = 'N/A'

            data_paths.append({
                'batch_id': batch_id,
                'dataset': dataset_name,
                'experiment': exp_name,
                'batch_path': batch_path,
                'shared_path': shared_path,
                'description': description,
                'is_reused': exp_info.get('is_reused', False)
            })

    if not data_paths:
        print(f"❌ No matching data found")
        return 1

    # Output
    if output_format == 'simple':
        print_simple(data_paths)
    elif output_format == 'yaml':
        print_yaml(data_paths)
    elif output_format == 'table':
        print_table(data_paths)

    return 0


def print_simple(data_paths):
    """Simple format output"""
    print("=" * 100)
    print("📊 Available Training Data Paths")
    print("=" * 100)

    current_batch = None

    for item in data_paths:
        # Print batch title
        if item['batch_id'] != current_batch:
            current_batch = item['batch_id']
            print(f"\n{'=' * 100}")
            print(f"🗂️  Batch: {current_batch}")
            print(f"{'=' * 100}")

        print(f"\n📁 {item['dataset']} / {item['experiment']}")

        if item['is_reused']:
            print(f"   ⚡ Data reuse (from other batch)")

        print(f"   📝 Description: {item['description'][:80]}...")
        print(f"\n   ✅ Batch path (recommended - organized by experiment purpose):")
        print(f"      {item['batch_path']}")
        print(f"\n   📦 Shared path (physical storage):")
        print(f"      {item['shared_path']}")


def print_yaml(data_paths):
    """YAML format output - can be directly copied to config file"""
    print("# =====================================================")
    print("# Available Training Data Paths - YAML Format")
    print("# Copy the paths below to data.path in training config file")
    print("# =====================================================")
    print()

    current_batch = None

    for item in data_paths:
        if item['batch_id'] != current_batch:
            current_batch = item['batch_id']
            print(f"\n# Batch: {current_batch}")
            print(f"# " + "=" * 70)

        print(f"\n# {item['dataset']} / {item['experiment']}")
        print(f"# Description: {item['description'][:70]}")
        print(f"data:")
        print(f'  path: "{item["batch_path"]}"')
        print(f'  # Or: "{item["shared_path"]}"')


def print_table(data_paths):
    """Table format output"""
    print("=" * 150)
    print(f"{'Batch':<30} | {'Dataset':<10} | {'Experiment':<25} | {'Reused':<6} | {'Path':<70}")
    print("=" * 150)

    for item in data_paths:
        reused = "Yes" if item['is_reused'] else "No"
        print(f"{item['batch_id']:<30} | {item['dataset']:<10} | {item['experiment']:<25} | {reused:<6} | {item['batch_path']:<70}")

    print("=" * 150)


def main():
    parser = argparse.ArgumentParser(
        description='List all available training data paths',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all data
  python list_data_paths.py

  # View specific batch's data only
  python list_data_paths.py --batch batch_20241229_temperature

  # View Copa dataset only
  python list_data_paths.py --dataset Copa

  # Output in YAML format (can be directly copied to config file)
  python list_data_paths.py --format yaml

  # Output as table format
  python list_data_paths.py --format table
        """
    )

    parser.add_argument('--batch', help='Filter specific batch')
    parser.add_argument('--dataset', help='Filter specific dataset (e.g., Copa, CB, BOOLQ)')
    parser.add_argument('--format', choices=['simple', 'yaml', 'table'], default='simple',
                        help='Output format (default: simple)')

    args = parser.parse_args()

    return list_all_data_paths(
        batch_filter=args.batch,
        dataset_filter=args.dataset,
        output_format=args.format
    )


if __name__ == '__main__':
    sys.exit(main())
