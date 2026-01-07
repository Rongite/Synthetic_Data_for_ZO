#!/usr/bin/env python3
"""
Compare parameter configurations of different experiments

Usage:
    # Compare all experiments in two batches
    python compare_experiments.py \
        --batch1 batch_20241229_temperature \
        --batch2 batch_20241230_topp

    # Compare different datasets in the same batch
    python compare_experiments.py \
        --batch1 batch_20241229_temperature \
        --dataset1 Copa \
        --batch2 batch_20241229_temperature \
        --dataset2 CB

    # Compare two specific experiments in _shared/
    python compare_experiments.py \
        --shared Copa/temp07_topp10_gpt4o \
        --shared Copa/temp09_topp10_gpt4o
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


def compare_params(exp1, exp2):
    """Compare parameters of two experiments"""
    differences = []
    similarities = []

    # Get metadata
    meta1 = exp1.get('metadata', {})
    meta2 = exp2.get('metadata', {})

    # Compare generation config
    gen1 = meta1.get('generation', {})
    gen2 = meta2.get('generation', {})

    gen_fields = ['model', 'temperature', 'top_p', 'max_tokens']
    for field in gen_fields:
        val1 = gen1.get(field, 'N/A')
        val2 = gen2.get(field, 'N/A')

        if val1 != val2:
            differences.append({
                'field': f'generation.{field}',
                'exp1': val1,
                'exp2': val2
            })
        else:
            similarities.append({
                'field': f'generation.{field}',
                'value': val1
            })

    # Compare validation config
    val1 = meta1.get('validation', {})
    val2 = meta2.get('validation', {})

    val_fields = ['model', 'temperature']
    for field in val_fields:
        v1 = val1.get(field, 'N/A')
        v2 = val2.get(field, 'N/A')

        if v1 != v2:
            differences.append({
                'field': f'validation.{field}',
                'exp1': v1,
                'exp2': v2
            })
        else:
            similarities.append({
                'field': f'validation.{field}',
                'value': v1
            })

    return differences, similarities


def main():
    parser = argparse.ArgumentParser(
        description='Compare parameter configurations of different experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--batch1',
        help='First batch ID'
    )
    parser.add_argument(
        '--dataset1',
        help='Dataset name in first batch'
    )
    parser.add_argument(
        '--batch2',
        help='Second batch ID'
    )
    parser.add_argument(
        '--dataset2',
        help='Dataset name in second batch'
    )
    parser.add_argument(
        '--shared',
        nargs=2,
        metavar=('EXP1', 'EXP2'),
        help='Directly compare two experiment paths in _shared/ (e.g., Copa/temp07_topp10_gpt4o)'
    )

    args = parser.parse_args()

    base_output_dir = Path(PROJECT_ROOT) / "Data_v2" / "synthetic"
    manager = BatchExperimentManager(base_output_dir)

    # Get experiment list
    if args.shared:
        # Read directly from _shared/
        shared_exps = manager.list_shared_experiments()

        exp1_path = args.shared[0]
        exp2_path = args.shared[1]

        exp1 = next((e for e in shared_exps if e['physical_path'] == f"_shared/{exp1_path}"), None)
        exp2 = next((e for e in shared_exps if e['physical_path'] == f"_shared/{exp2_path}"), None)

        if not exp1:
            print(f"❌ Experiment 1 does not exist: {exp1_path}")
            return 1
        if not exp2:
            print(f"❌ Experiment 2 does not exist: {exp2_path}")
            return 1

        exp1_name = exp1_path
        exp2_name = exp2_path

    elif args.batch1 and args.batch2:
        # Read from batch
        exps1 = manager.list_experiments_in_batch(args.batch1, args.dataset1)
        exps2 = manager.list_experiments_in_batch(args.batch2, args.dataset2)

        if not exps1:
            print(f"❌ No experiments found in Batch1: {args.batch1}")
            return 1
        if not exps2:
            print(f"❌ No experiments found in Batch2: {args.batch2}")
            return 1

        print("\nSelect experiments to compare:")
        print(f"\nBatch1: {args.batch1}")
        for i, e in enumerate(exps1):
            print(f"  {i+1}. {e['dataset']}/{e['exp_name']}")

        print(f"\nBatch2: {args.batch2}")
        for i, e in enumerate(exps2):
            print(f"  {i+1}. {e['dataset']}/{e['exp_name']}")

        # Simplified: if only one experiment, auto-select
        if len(exps1) == 1 and len(exps2) == 1:
            exp1 = exps1[0]
            exp2 = exps2[0]
        else:
            print("\nFeature not yet implemented: please use --shared to specify experiments directly")
            return 1

        exp1_name = f"{args.batch1}/{exp1['dataset']}/{exp1['exp_name']}"
        exp2_name = f"{args.batch2}/{exp2['dataset']}/{exp2['exp_name']}"

        # Read metadata from physical path
        exp1_metadata_path = base_output_dir / exp1['physical_path'].replace('_shared/', '') / 'experiment_metadata.json'
        exp2_metadata_path = base_output_dir / exp2['physical_path'].replace('_shared/', '') / 'experiment_metadata.json'

        if exp1_metadata_path.exists():
            with open(exp1_metadata_path, 'r') as f:
                exp1['metadata'] = json.load(f)

        if exp2_metadata_path.exists():
            with open(exp2_metadata_path, 'r') as f:
                exp2['metadata'] = json.load(f)

    else:
        parser.print_help()
        return 1

    # Compare parameters
    differences, similarities = compare_params(exp1, exp2)

    print("\n" + "=" * 80)
    print("Experiment Parameter Comparison")
    print("=" * 80)
    print(f"\nExperiment 1: {exp1_name}")
    print(f"Experiment 2: {exp2_name}")

    # Display same parameters
    if similarities:
        print("\n✅ Same parameters:")
        for item in similarities:
            print(f"  {item['field']}: {item['value']}")

    # Display different parameters
    if differences:
        print("\n⚠️  Different parameters:")
        for item in differences:
            print(f"  {item['field']}:")
            print(f"    Experiment 1: {item['exp1']}")
            print(f"    Experiment 2: {item['exp2']}")
    else:
        print("\n✅ All parameters are exactly the same")
        print(f"Parameter fingerprint: {exp1.get('fingerprint', 'N/A')}")

    # Determine if should share physical data
    if exp1.get('fingerprint') == exp2.get('fingerprint'):
        print("\n💾 Hint: These two experiments have the same parameter fingerprint, should share physical data")
    else:
        print(f"\n🔧 Hint: Different parameters, need to generate data separately")
        print(f"  Experiment 1 fingerprint: {exp1.get('fingerprint', 'N/A')}")
        print(f"  Experiment 2 fingerprint: {exp2.get('fingerprint', 'N/A')}")

    print()
    return 0


if __name__ == '__main__':
    exit(main())
